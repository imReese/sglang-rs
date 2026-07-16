use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cache::{KvBlockPrefixIndex, KvCacheWorkerId, KvCacheWorkerSnapshot};
use crate::cli::ServerArgs;
use crate::engine::{Engine, RuntimeError};
use crate::model_artifacts::{HfModelConfig, LocalModelArtifacts, RoutedExpertCheckpointCoverage};
use crate::scheduler::SchedulerError;
use crate::tokenizer::{Tokenizer, TokenizerError};
use crate::types::{
    BootstrapRoom, DisaggregatedParams, RequestId, SamplingParams, TokenGenerateOutput,
    TokenGenerateRequest,
};
use crate::worker::WorkerExecutor;
use crate::worker::WorkerWeightUpdateRequest;

pub const DEFAULT_MAX_NEW_TOKENS: usize = 128;
static ROUTER_REQUEST_ID_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RouterValidationConfig {
    pub max_context_tokens: Option<usize>,
    pub max_request_input_tokens: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouterSamplingParams {
    pub max_new_tokens: Option<i32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub stop_token_id: Option<i32>,
    pub stop_token_ids: Vec<i32>,
    pub ignore_eos: Option<bool>,
    pub n: Option<i32>,
    pub best_of: Option<i32>,
}

impl RouterSamplingParams {
    fn into_sampling_params(self) -> Result<SamplingParams, RouterProtocolError> {
        validate_optional_non_negative_float("temperature", self.temperature)?;
        validate_optional_positive_unit_float("top_p", self.top_p)?;
        validate_optional_unit_float("min_p", self.min_p)?;
        validate_optional_non_negative_float("frequency_penalty", self.frequency_penalty)?;
        validate_optional_non_negative_float("presence_penalty", self.presence_penalty)?;
        validate_optional_positive_float("repetition_penalty", self.repetition_penalty)?;
        validate_optional_unbounded_or_positive_i32("top_k", self.top_k)?;
        validate_optional_non_negative_i32("stop_token_id", self.stop_token_id)?;
        validate_non_negative_i32s("stop_token_ids", &self.stop_token_ids)?;
        validate_optional_positive_i32("n", self.n)?;
        validate_optional_positive_i32("best_of", self.best_of)?;

        let max_new_tokens = match self.max_new_tokens {
            Some(value) if value <= 0 => {
                return Err(RouterProtocolError::InvalidIntegerSamplingParam {
                    field: "max_new_tokens",
                    value,
                    expected: "positive",
                });
            }
            Some(value) => value as usize,
            None => DEFAULT_MAX_NEW_TOKENS,
        };

        Ok(SamplingParams {
            max_new_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            stop_token_ids: merged_stop_token_ids(self.stop_token_id, self.stop_token_ids),
            ignore_eos: self.ignore_eos.unwrap_or(false),
        })
    }
}

fn merged_stop_token_ids(stop_token_id: Option<i32>, stop_token_ids: Vec<i32>) -> Vec<u32> {
    let mut merged = stop_token_id
        .map(|stop_token_id| vec![stop_token_id as u32])
        .unwrap_or_default();
    merged.extend(
        stop_token_ids
            .into_iter()
            .map(|stop_token_id| stop_token_id as u32),
    );
    merged.sort_unstable();
    merged.dedup();
    merged
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RouterTokenizedInput {
    pub original_text: String,
    pub input_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouterDisaggregatedParams {
    pub bootstrap_host: String,
    pub bootstrap_port: u16,
    pub bootstrap_room: BootstrapRoom,
}

impl From<RouterDisaggregatedParams> for DisaggregatedParams {
    fn from(value: RouterDisaggregatedParams) -> Self {
        Self {
            bootstrap_host: value.bootstrap_host,
            bootstrap_port: value.bootstrap_port,
            bootstrap_room: value.bootstrap_room,
        }
    }
}

fn validate_optional_non_negative_float(
    field: &'static str,
    value: Option<f32>,
) -> Result<(), RouterProtocolError> {
    let Some(value) = value else {
        return Ok(());
    };
    if !value.is_finite() || value < 0.0 {
        return Err(RouterProtocolError::InvalidFloatSamplingParam {
            field,
            value,
            expected: "finite and non-negative",
        });
    }

    Ok(())
}

fn validate_optional_positive_float(
    field: &'static str,
    value: Option<f32>,
) -> Result<(), RouterProtocolError> {
    let Some(value) = value else {
        return Ok(());
    };
    if !value.is_finite() || value <= 0.0 {
        return Err(RouterProtocolError::InvalidFloatSamplingParam {
            field,
            value,
            expected: "finite and positive",
        });
    }

    Ok(())
}

fn validate_optional_unit_float(
    field: &'static str,
    value: Option<f32>,
) -> Result<(), RouterProtocolError> {
    let Some(value) = value else {
        return Ok(());
    };
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(RouterProtocolError::InvalidFloatSamplingParam {
            field,
            value,
            expected: "finite and in [0, 1]",
        });
    }

    Ok(())
}

fn validate_optional_positive_unit_float(
    field: &'static str,
    value: Option<f32>,
) -> Result<(), RouterProtocolError> {
    let Some(value) = value else {
        return Ok(());
    };
    if !value.is_finite() || !(0.0..=1.0).contains(&value) || value == 0.0 {
        return Err(RouterProtocolError::InvalidFloatSamplingParam {
            field,
            value,
            expected: "finite and in (0, 1]",
        });
    }

    Ok(())
}

fn validate_optional_unbounded_or_positive_i32(
    field: &'static str,
    value: Option<i32>,
) -> Result<(), RouterProtocolError> {
    let Some(value) = value else {
        return Ok(());
    };
    if value != -1 && value < 1 {
        return Err(RouterProtocolError::InvalidIntegerSamplingParam {
            field,
            value,
            expected: "-1 or positive",
        });
    }

    Ok(())
}

fn validate_optional_non_negative_i32(
    field: &'static str,
    value: Option<i32>,
) -> Result<(), RouterProtocolError> {
    let Some(value) = value else {
        return Ok(());
    };
    if value < 0 {
        return Err(RouterProtocolError::InvalidIntegerSamplingParam {
            field,
            value,
            expected: "non-negative",
        });
    }

    Ok(())
}

fn validate_non_negative_i32s(
    field: &'static str,
    values: &[i32],
) -> Result<(), RouterProtocolError> {
    if let Some(value) = values.iter().copied().find(|value| *value < 0) {
        return Err(RouterProtocolError::InvalidIntegerSamplingParam {
            field,
            value,
            expected: "non-negative",
        });
    }

    Ok(())
}

fn validate_optional_positive_i32(
    field: &'static str,
    value: Option<i32>,
) -> Result<(), RouterProtocolError> {
    let Some(value) = value else {
        return Ok(());
    };
    if value <= 0 {
        return Err(RouterProtocolError::InvalidIntegerSamplingParam {
            field,
            value,
            expected: "positive",
        });
    }

    Ok(())
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouterGenerateRequest {
    pub request_id: String,
    pub tokenized: Option<RouterTokenizedInput>,
    pub sampling_params: Option<RouterSamplingParams>,
    pub disaggregated_params: Option<RouterDisaggregatedParams>,
    pub stream: bool,
    pub data_parallel_rank: i32,
    pub trace_headers: BTreeMap<String, String>,
}

impl RouterGenerateRequest {
    pub fn select_cache_aware_prefill_worker(
        &self,
        index: &KvBlockPrefixIndex,
        candidates: &[KvCacheWorkerSnapshot],
        block_size: usize,
        cache_threshold: f32,
    ) -> Result<Option<KvCacheWorkerId>, RouterProtocolError> {
        let tokenized = self
            .tokenized
            .as_ref()
            .ok_or(RouterProtocolError::MissingTokenizedInput)?;
        if tokenized.input_ids.is_empty() {
            return Err(RouterProtocolError::EmptyTokenizedInput);
        }

        Ok(index.select_cache_aware_worker_for_tokens(
            candidates,
            &tokenized.input_ids,
            block_size,
            cache_threshold,
        ))
    }

    pub fn try_into_token_generate_request(
        self,
    ) -> Result<TokenGenerateRequest, RouterProtocolError> {
        self.try_into_token_generate_request_with_config(RouterValidationConfig::default())
    }

    pub fn try_into_token_generate_request_with_config(
        self,
        validation_config: RouterValidationConfig,
    ) -> Result<TokenGenerateRequest, RouterProtocolError> {
        Ok(self
            .try_into_validated_token_request(validation_config)?
            .request)
    }

    fn try_into_validated_token_request(
        self,
        validation_config: RouterValidationConfig,
    ) -> Result<ValidatedTokenGenerateRequest, RouterProtocolError> {
        let request_id = if self.request_id.is_empty() {
            next_router_request_id()
        } else {
            self.request_id
        };
        let tokenized = self
            .tokenized
            .ok_or(RouterProtocolError::MissingTokenizedInput)?;
        let input_tokens = tokenized.input_ids.len();
        if input_tokens == 0 {
            return Err(RouterProtocolError::EmptyTokenizedInput);
        }

        let sampling = self
            .sampling_params
            .unwrap_or_default()
            .into_sampling_params()?;

        validate_token_budget(input_tokens, sampling.max_new_tokens, validation_config)?;

        Ok(ValidatedTokenGenerateRequest {
            prompt_tokens: input_tokens,
            stream: self.stream,
            request: TokenGenerateRequest {
                request_id: RequestId::from(request_id.as_str()),
                input_ids: tokenized.input_ids,
                sampling,
                disaggregated_params: self.disaggregated_params.map(Into::into),
                data_parallel_rank: self.data_parallel_rank,
            },
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouterTextGenerateRequest {
    pub request_id: String,
    pub text: String,
    pub sampling_params: Option<RouterSamplingParams>,
    pub disaggregated_params: Option<RouterDisaggregatedParams>,
    pub stream: bool,
    pub data_parallel_rank: i32,
    pub trace_headers: BTreeMap<String, String>,
}

fn next_router_request_id() -> String {
    let sequence = ROUTER_REQUEST_ID_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let timestamp_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();

    format!("sglang-rs-{timestamp_nanos:x}-{sequence:x}")
}

#[derive(Clone, Debug, PartialEq)]
struct ValidatedTokenGenerateRequest {
    prompt_tokens: usize,
    stream: bool,
    request: TokenGenerateRequest,
}

fn validate_token_budget(
    input_tokens: usize,
    max_new_tokens: usize,
    validation_config: RouterValidationConfig,
) -> Result<(), RouterProtocolError> {
    if let Some(max_request_input_tokens) = validation_config.max_request_input_tokens
        && input_tokens > max_request_input_tokens
    {
        return Err(RouterProtocolError::InputTooLong {
            input_tokens,
            max_request_input_tokens,
        });
    }

    if let Some(max_context_tokens) = validation_config.max_context_tokens
        && input_tokens.saturating_add(max_new_tokens) > max_context_tokens
    {
        return Err(RouterProtocolError::ContextOverflow {
            input_tokens,
            max_new_tokens,
            max_context_tokens,
        });
    }

    Ok(())
}

fn usize_to_i32(value: usize) -> Option<i32> {
    i32::try_from(value).ok()
}

fn u32_to_i32(value: u32) -> Option<i32> {
    i32::try_from(value).ok()
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RouterGenerateResponse {
    pub request_id: String,
    pub body: RouterGenerateResponseBody,
}

impl RouterGenerateResponse {
    pub fn from_token_generate_output(output: TokenGenerateOutput, prompt_tokens: i32) -> Self {
        let request_id = output.request_id.as_str().to_string();
        let completion_tokens = output.output_ids.len() as i32;
        let cached_tokens = output.cached_tokens as i32;
        let body = if output.finished {
            RouterGenerateResponseBody::Complete(RouterGenerateComplete {
                output_ids: output.output_ids,
                text: String::new(),
                finish_reason: "stop".to_string(),
                prompt_tokens,
                completion_tokens,
                cached_tokens,
                index: 0,
            })
        } else {
            RouterGenerateResponseBody::Chunk(RouterGenerateStreamChunk {
                token_ids: output.output_ids,
                text: String::new(),
                prompt_tokens,
                completion_tokens,
                cached_tokens,
                index: 0,
            })
        };

        Self { request_id, body }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RouterGenerateResponseBody {
    Chunk(RouterGenerateStreamChunk),
    Complete(RouterGenerateComplete),
    Error(RouterGenerateError),
}

impl Default for RouterGenerateResponseBody {
    fn default() -> Self {
        Self::Complete(RouterGenerateComplete::default())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RouterGenerateStreamChunk {
    pub token_ids: Vec<u32>,
    pub text: String,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub cached_tokens: i32,
    pub index: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RouterGenerateComplete {
    pub output_ids: Vec<u32>,
    pub text: String,
    pub finish_reason: String,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub cached_tokens: i32,
    pub index: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RouterGenerateError {
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouterHealthCheckResponse {
    pub healthy: bool,
    pub message: String,
}

impl RouterHealthCheckResponse {
    pub fn healthy() -> Self {
        Self {
            healthy: true,
            message: "ready".to_string(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouterFlushCacheResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouterControlResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouterTransferPollResponse {
    pub completed_batches: usize,
    pub pending_batches: usize,
    pub completed_descriptor_checksums: Vec<String>,
    pub pending_descriptor_checksums: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouterLoadResponse {
    pub waiting_queue_depth: usize,
    pub decode_queue_depth: usize,
    pub available_cache_pages: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouterTokenizeResponse {
    pub token_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouterDetokenizeResponse {
    pub text: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouterGetModelInfoResponse {
    pub model_path: String,
    pub tokenizer_path: String,
    pub is_generation: bool,
    pub preferred_sampling_params: String,
    pub weight_version: String,
    pub served_model_name: String,
    pub max_context_length: i32,
    pub vocab_size: i32,
    pub supports_vision: bool,
    pub model_type: String,
    pub architectures: Vec<String>,
    pub eos_token_ids: Vec<i32>,
    pub pad_token_id: i32,
    pub bos_token_id: i32,
    pub max_req_input_len: i32,
    pub routed_expert_expected_group_count: i32,
    pub routed_expert_actual_group_count: i32,
    pub routed_expert_expected_weight_count: i32,
    pub routed_expert_actual_weight_count: i32,
    pub tp_size: u32,
    pub dp_size: u32,
    pub max_running_requests: u32,
    pub max_num_reqs: u32,
    pub max_prefill_tokens: u32,
    pub max_total_tokens: u32,
}

impl RouterGetModelInfoResponse {
    pub fn from_server_args(args: &ServerArgs) -> Self {
        let tokenizer_path = args
            .tokenizer_path
            .clone()
            .unwrap_or_else(|| args.model_path.clone());
        let served_model_name = args
            .served_model_name
            .clone()
            .unwrap_or_else(|| args.model_path.clone());

        let mut response = Self {
            model_path: args.model_path.clone(),
            tokenizer_path,
            is_generation: true,
            preferred_sampling_params: "{}".to_string(),
            weight_version: String::new(),
            served_model_name,
            max_context_length: 0,
            vocab_size: 0,
            supports_vision: false,
            model_type: String::new(),
            architectures: Vec::new(),
            eos_token_ids: Vec::new(),
            pad_token_id: 0,
            bos_token_id: 0,
            max_req_input_len: 0,
            routed_expert_expected_group_count: 0,
            routed_expert_actual_group_count: 0,
            routed_expert_expected_weight_count: 0,
            routed_expert_actual_weight_count: 0,
            tp_size: usize_to_model_info_u32(args.tp_size),
            dp_size: usize_to_model_info_u32(args.dp_size),
            max_running_requests: optional_usize_to_model_info_u32(args.max_running_requests),
            max_num_reqs: optional_usize_to_model_info_u32(args.max_running_requests),
            max_prefill_tokens: optional_usize_to_model_info_u32(args.max_prefill_tokens),
            max_total_tokens: optional_usize_to_model_info_u32(args.max_total_tokens),
        };

        if let Ok(config) = HfModelConfig::from_model_path(&args.model_path) {
            response.apply_model_config(config);
        }
        if let Ok(artifacts) = LocalModelArtifacts::from_model_path(&args.model_path)
            && let Ok(coverage) = artifacts.validate_routed_expert_checkpoint_coverage()
        {
            response.apply_routed_expert_checkpoint_coverage(coverage);
        }

        response
    }

    pub fn from_local_model_artifacts(
        artifacts: &LocalModelArtifacts,
        served_model_name: String,
        tokenizer_path: String,
        weight_version: String,
    ) -> Self {
        let mut response = Self {
            model_path: artifacts.model_path().to_string_lossy().to_string(),
            tokenizer_path,
            is_generation: true,
            preferred_sampling_params: "{}".to_string(),
            weight_version,
            served_model_name,
            max_context_length: 0,
            vocab_size: 0,
            supports_vision: false,
            model_type: String::new(),
            architectures: Vec::new(),
            eos_token_ids: Vec::new(),
            pad_token_id: 0,
            bos_token_id: 0,
            max_req_input_len: 0,
            routed_expert_expected_group_count: 0,
            routed_expert_actual_group_count: 0,
            routed_expert_expected_weight_count: 0,
            routed_expert_actual_weight_count: 0,
            tp_size: 0,
            dp_size: 0,
            max_running_requests: 0,
            max_num_reqs: 0,
            max_prefill_tokens: 0,
            max_total_tokens: 0,
        };

        response.apply_model_config(artifacts.config().clone());
        if let Ok(coverage) = artifacts.validate_routed_expert_checkpoint_coverage() {
            response.apply_routed_expert_checkpoint_coverage(coverage);
        }

        response
    }

    fn apply_model_config(&mut self, config: HfModelConfig) {
        if let Some(model_type) = config.model_type {
            self.model_type = model_type;
        }
        self.architectures = config.architectures;
        self.eos_token_ids = config
            .eos_token_ids
            .into_iter()
            .filter_map(u32_to_i32)
            .collect();
        if let Some(vocab_size) = config.vocab_size.and_then(usize_to_i32) {
            self.vocab_size = vocab_size;
        }
        if let Some(max_context_length) = config.max_position_embeddings.and_then(usize_to_i32) {
            self.max_context_length = max_context_length;
            self.max_req_input_len = max_context_length;
        }
    }

    fn apply_routed_expert_checkpoint_coverage(
        &mut self,
        coverage: RoutedExpertCheckpointCoverage,
    ) {
        if let Some(value) = usize_to_i32(coverage.expected_group_count) {
            self.routed_expert_expected_group_count = value;
        }
        if let Some(value) = usize_to_i32(coverage.actual_group_count) {
            self.routed_expert_actual_group_count = value;
        }
        if let Some(value) = usize_to_i32(coverage.expected_weight_count) {
            self.routed_expert_expected_weight_count = value;
        }
        if let Some(value) = usize_to_i32(coverage.actual_weight_count) {
            self.routed_expert_actual_weight_count = value;
        }
    }
}

fn optional_usize_to_model_info_u32(value: Option<usize>) -> u32 {
    value.map(usize_to_model_info_u32).unwrap_or_default()
}

fn usize_to_model_info_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

pub struct RouterRuntime<T, W> {
    engine: Engine<T, W>,
    validation_config: RouterValidationConfig,
    default_stop_token_ids: Vec<u32>,
    generation_paused: bool,
}

impl<T, W> RouterRuntime<T, W> {
    pub fn new(engine: Engine<T, W>) -> Self {
        Self::with_validation_config(engine, RouterValidationConfig::default())
    }

    pub fn with_validation_config(
        engine: Engine<T, W>,
        validation_config: RouterValidationConfig,
    ) -> Self {
        Self {
            engine,
            validation_config,
            default_stop_token_ids: Vec::new(),
            generation_paused: false,
        }
    }

    pub fn with_default_stop_token_ids(mut self, default_stop_token_ids: Vec<u32>) -> Self {
        self.default_stop_token_ids = default_stop_token_ids;
        self.default_stop_token_ids.sort_unstable();
        self.default_stop_token_ids.dedup();
        self
    }

    pub fn engine(&self) -> &Engine<T, W> {
        &self.engine
    }

    pub fn load(&self) -> RouterLoadResponse {
        let scheduler = self.engine.scheduler();
        RouterLoadResponse {
            waiting_queue_depth: scheduler.waiting_queue_depth(),
            decode_queue_depth: scheduler.decode_queue_depth(),
            available_cache_pages: scheduler.available_cache_pages(),
        }
    }

    pub fn pause_generation(&mut self) -> RouterControlResponse {
        self.generation_paused = true;
        RouterControlResponse {
            success: true,
            message: "generation paused".to_string(),
        }
    }

    pub fn continue_generation(&mut self) -> RouterControlResponse {
        self.generation_paused = false;
        RouterControlResponse {
            success: true,
            message: "generation continued".to_string(),
        }
    }

    pub fn abort_request(
        &mut self,
        request_id: &str,
    ) -> Result<RouterControlResponse, RouterProtocolError>
    where
        W: WorkerExecutor,
    {
        if request_id.is_empty() {
            return Err(RouterProtocolError::MissingRequestId);
        }

        if self.engine.abort_request(&RequestId::from(request_id)) {
            return Ok(RouterControlResponse {
                success: true,
                message: "request aborted".to_string(),
            });
        }

        Ok(RouterControlResponse {
            success: false,
            message: "request not found".to_string(),
        })
    }

    pub fn abort_all_requests(&mut self) -> RouterControlResponse
    where
        W: WorkerExecutor,
    {
        let aborted = self.engine.abort_all_requests();
        RouterControlResponse {
            success: true,
            message: format!("aborted {aborted} request(s)"),
        }
    }

    fn ensure_generation_ready(&self) -> Result<(), RouterProtocolError> {
        if self.generation_paused {
            return Err(RouterProtocolError::GenerationPaused);
        }

        Ok(())
    }

    fn apply_default_stop_token_ids(
        &self,
        mut request: TokenGenerateRequest,
    ) -> TokenGenerateRequest {
        if request.sampling.ignore_eos {
            return request;
        }
        request
            .sampling
            .stop_token_ids
            .extend_from_slice(&self.default_stop_token_ids);
        request.sampling.stop_token_ids.sort_unstable();
        request.sampling.stop_token_ids.dedup();
        request
    }
}

impl<T, W> RouterRuntime<T, W>
where
    W: WorkerExecutor,
{
    pub fn poll_transfers(&mut self) -> Result<RouterTransferPollResponse, RouterRuntimeError> {
        let summary = self.engine.poll_transfers()?;
        Ok(RouterTransferPollResponse {
            completed_batches: summary.completed_batches(),
            pending_batches: summary.pending_batches(),
            completed_descriptor_checksums: summary.completed_descriptor_checksums().to_vec(),
            pending_descriptor_checksums: summary.pending_descriptor_checksums().to_vec(),
        })
    }

    pub fn update_weights_from_disk(
        &mut self,
        request: WorkerWeightUpdateRequest,
    ) -> Result<(), RouterRuntimeError> {
        self.engine.update_weights_from_disk(&request)?;
        Ok(())
    }
}

impl<T, W> RouterRuntime<T, W>
where
    T: Tokenizer,
{
    pub fn tokenize(&self, text: &str) -> RouterTokenizeResponse {
        RouterTokenizeResponse {
            token_ids: self.engine.tokenize(text),
        }
    }

    pub fn detokenize(
        &self,
        token_ids: &[u32],
    ) -> Result<RouterDetokenizeResponse, TokenizerError> {
        Ok(RouterDetokenizeResponse {
            text: self.engine.detokenize(token_ids)?,
        })
    }
}

impl<T, W> RouterRuntime<T, W>
where
    T: Tokenizer,
    W: WorkerExecutor,
{
    pub fn generate(
        &mut self,
        request: RouterGenerateRequest,
    ) -> Result<RouterGenerateResponse, RouterRuntimeError> {
        let validated_request = request.try_into_validated_token_request(self.validation_config)?;
        self.ensure_generation_ready()?;
        let prompt_tokens = validated_request.prompt_tokens as i32;
        let output = self
            .engine
            .generate_tokens(self.apply_default_stop_token_ids(validated_request.request))?;

        Ok(RouterGenerateResponse::from_token_generate_output(
            output,
            prompt_tokens,
        ))
    }

    pub fn generate_stream(
        &mut self,
        request: RouterGenerateRequest,
    ) -> Result<Vec<RouterGenerateResponse>, RouterRuntimeError> {
        let validated_request = request.try_into_validated_token_request(self.validation_config)?;
        self.ensure_generation_ready()?;
        let prompt_tokens = validated_request.prompt_tokens as i32;
        let stream = validated_request.stream;
        let outputs = self
            .engine
            .generate_token_stream(self.apply_default_stop_token_ids(validated_request.request))?;
        Ok(token_outputs_to_generate_responses(
            outputs,
            prompt_tokens,
            stream,
        ))
    }

    pub fn generate_batch_stream(
        &mut self,
        requests: Vec<RouterGenerateRequest>,
    ) -> Result<Vec<Vec<RouterGenerateResponse>>, RouterRuntimeError> {
        let mut prompt_tokens = Vec::with_capacity(requests.len());
        let mut streams = Vec::with_capacity(requests.len());
        let mut token_requests = Vec::with_capacity(requests.len());
        for request in requests {
            let validated_request =
                request.try_into_validated_token_request(self.validation_config)?;
            prompt_tokens.push(validated_request.prompt_tokens as i32);
            streams.push(validated_request.stream);
            token_requests.push(self.apply_default_stop_token_ids(validated_request.request));
        }

        self.ensure_generation_ready()?;
        let batch_outputs = self.engine.generate_token_batch_stream(token_requests)?;
        Ok(batch_outputs
            .into_iter()
            .zip(prompt_tokens)
            .zip(streams)
            .map(|((outputs, prompt_tokens), stream)| {
                token_outputs_to_generate_responses(outputs, prompt_tokens, stream)
            })
            .collect())
    }

    pub fn generate_text_stream(
        &mut self,
        request: RouterTextGenerateRequest,
    ) -> Result<Vec<RouterGenerateResponse>, RouterRuntimeError> {
        if request.text.is_empty() {
            return Err(RouterProtocolError::EmptyTextInput.into());
        }

        let input_ids = self.engine.tokenize(&request.text);
        let mut responses = self.generate_stream(RouterGenerateRequest {
            request_id: request.request_id,
            tokenized: Some(RouterTokenizedInput {
                original_text: request.text,
                input_ids,
            }),
            sampling_params: request.sampling_params,
            disaggregated_params: request.disaggregated_params,
            stream: request.stream,
            data_parallel_rank: request.data_parallel_rank,
            trace_headers: request.trace_headers,
        })?;

        for response in &mut responses {
            self.fill_generate_response_text(response)?;
        }

        Ok(responses)
    }

    pub fn generate_text_batch_stream(
        &mut self,
        requests: Vec<RouterTextGenerateRequest>,
    ) -> Result<Vec<Vec<RouterGenerateResponse>>, RouterRuntimeError> {
        let mut tokenized_requests = Vec::with_capacity(requests.len());
        for request in requests {
            if request.text.is_empty() {
                return Err(RouterProtocolError::EmptyTextInput.into());
            }

            let input_ids = self.engine.tokenize(&request.text);
            tokenized_requests.push(RouterGenerateRequest {
                request_id: request.request_id,
                tokenized: Some(RouterTokenizedInput {
                    original_text: request.text,
                    input_ids,
                }),
                sampling_params: request.sampling_params,
                disaggregated_params: request.disaggregated_params,
                stream: request.stream,
                data_parallel_rank: request.data_parallel_rank,
                trace_headers: request.trace_headers,
            });
        }

        let mut batch_responses = self.generate_batch_stream(tokenized_requests)?;
        for responses in &mut batch_responses {
            for response in responses {
                self.fill_generate_response_text(response)?;
            }
        }

        Ok(batch_responses)
    }

    fn fill_generate_response_text(
        &self,
        response: &mut RouterGenerateResponse,
    ) -> Result<(), RouterRuntimeError> {
        match &mut response.body {
            RouterGenerateResponseBody::Chunk(chunk) => {
                chunk.text = self
                    .engine
                    .detokenize(&chunk.token_ids)
                    .map_err(RuntimeError::from)?;
            }
            RouterGenerateResponseBody::Complete(complete) => {
                complete.text = self
                    .engine
                    .detokenize(&complete.output_ids)
                    .map_err(RuntimeError::from)?;
            }
            RouterGenerateResponseBody::Error(_) => {}
        }

        Ok(())
    }

    pub fn flush_cache(&mut self) -> RouterFlushCacheResponse {
        if self.engine.flush_cache() {
            RouterFlushCacheResponse {
                success: true,
                message: "cache flushed".to_string(),
            }
        } else {
            RouterFlushCacheResponse {
                success: false,
                message: String::new(),
            }
        }
    }
}

impl<T, W> RouterRuntime<T, W>
where
    T: Tokenizer,
    W: WorkerExecutor,
{
    pub fn generate_stream_with_transfer_polling(
        &mut self,
        request: RouterGenerateRequest,
        max_transfer_polls: usize,
    ) -> Result<Vec<RouterGenerateResponse>, RouterRuntimeError> {
        let validated_request = request.try_into_validated_token_request(self.validation_config)?;
        self.ensure_generation_ready()?;
        let prompt_tokens = validated_request.prompt_tokens as i32;
        let stream = validated_request.stream;
        let outputs = self.engine.generate_token_stream_with_transfer_polling(
            self.apply_default_stop_token_ids(validated_request.request),
            max_transfer_polls,
        )?;
        Ok(token_outputs_to_generate_responses(
            outputs,
            prompt_tokens,
            stream,
        ))
    }

    pub fn generate_batch_stream_with_transfer_polling(
        &mut self,
        requests: Vec<RouterGenerateRequest>,
        max_transfer_polls: usize,
    ) -> Result<Vec<Vec<RouterGenerateResponse>>, RouterRuntimeError> {
        let mut prompt_tokens = Vec::with_capacity(requests.len());
        let mut streams = Vec::with_capacity(requests.len());
        let mut token_requests = Vec::with_capacity(requests.len());
        for request in requests {
            let validated_request =
                request.try_into_validated_token_request(self.validation_config)?;
            prompt_tokens.push(validated_request.prompt_tokens as i32);
            streams.push(validated_request.stream);
            token_requests.push(self.apply_default_stop_token_ids(validated_request.request));
        }

        self.ensure_generation_ready()?;
        let batch_outputs = self
            .engine
            .generate_token_batch_stream_with_transfer_polling(
                token_requests,
                max_transfer_polls,
            )?;
        Ok(batch_outputs
            .into_iter()
            .zip(prompt_tokens)
            .zip(streams)
            .map(|((outputs, prompt_tokens), stream)| {
                token_outputs_to_generate_responses(outputs, prompt_tokens, stream)
            })
            .collect())
    }

    pub fn generate_text_stream_with_transfer_polling(
        &mut self,
        request: RouterTextGenerateRequest,
        max_transfer_polls: usize,
    ) -> Result<Vec<RouterGenerateResponse>, RouterRuntimeError> {
        if request.text.is_empty() {
            return Err(RouterProtocolError::EmptyTextInput.into());
        }

        let input_ids = self.engine.tokenize(&request.text);
        let mut responses = self.generate_stream_with_transfer_polling(
            RouterGenerateRequest {
                request_id: request.request_id,
                tokenized: Some(RouterTokenizedInput {
                    original_text: request.text,
                    input_ids,
                }),
                sampling_params: request.sampling_params,
                disaggregated_params: request.disaggregated_params,
                stream: request.stream,
                data_parallel_rank: request.data_parallel_rank,
                trace_headers: request.trace_headers,
            },
            max_transfer_polls,
        )?;

        for response in &mut responses {
            self.fill_generate_response_text(response)?;
        }

        Ok(responses)
    }

    pub fn generate_text_batch_stream_with_transfer_polling(
        &mut self,
        requests: Vec<RouterTextGenerateRequest>,
        max_transfer_polls: usize,
    ) -> Result<Vec<Vec<RouterGenerateResponse>>, RouterRuntimeError> {
        let mut tokenized_requests = Vec::with_capacity(requests.len());
        for request in requests {
            if request.text.is_empty() {
                return Err(RouterProtocolError::EmptyTextInput.into());
            }

            let input_ids = self.engine.tokenize(&request.text);
            tokenized_requests.push(RouterGenerateRequest {
                request_id: request.request_id,
                tokenized: Some(RouterTokenizedInput {
                    original_text: request.text,
                    input_ids,
                }),
                sampling_params: request.sampling_params,
                disaggregated_params: request.disaggregated_params,
                stream: request.stream,
                data_parallel_rank: request.data_parallel_rank,
                trace_headers: request.trace_headers,
            });
        }

        let mut batch_responses = self
            .generate_batch_stream_with_transfer_polling(tokenized_requests, max_transfer_polls)?;
        for responses in &mut batch_responses {
            for response in responses {
                self.fill_generate_response_text(response)?;
            }
        }

        Ok(batch_responses)
    }
}

fn token_outputs_to_generate_responses(
    outputs: Vec<TokenGenerateOutput>,
    prompt_tokens: i32,
    stream: bool,
) -> Vec<RouterGenerateResponse> {
    let mut output_ids = Vec::new();
    let mut responses = Vec::with_capacity(outputs.len());

    for output in outputs {
        let request_id = output.request_id.as_str().to_string();
        output_ids.extend_from_slice(&output.output_ids);
        let completion_tokens = output_ids.len() as i32;
        let cached_tokens = output.cached_tokens as i32;

        let body = if output.finished {
            RouterGenerateResponseBody::Complete(RouterGenerateComplete {
                output_ids: output_ids.clone(),
                text: String::new(),
                finish_reason: "stop".to_string(),
                prompt_tokens,
                completion_tokens,
                cached_tokens,
                index: 0,
            })
        } else {
            RouterGenerateResponseBody::Chunk(RouterGenerateStreamChunk {
                token_ids: output.output_ids,
                text: String::new(),
                prompt_tokens,
                completion_tokens,
                cached_tokens,
                index: 0,
            })
        };

        responses.push(RouterGenerateResponse { request_id, body });
    }

    if !stream {
        responses
            .retain(|response| matches!(response.body, RouterGenerateResponseBody::Complete(_)));
    }

    responses
}

#[derive(Clone, Debug, PartialEq)]
pub enum RouterProtocolError {
    MissingRequestId,
    MissingTokenizedInput,
    EmptyTokenizedInput,
    EmptyTextInput,
    GenerationPaused,
    InvalidIntegerSamplingParam {
        field: &'static str,
        value: i32,
        expected: &'static str,
    },
    InvalidFloatSamplingParam {
        field: &'static str,
        value: f32,
        expected: &'static str,
    },
    InputTooLong {
        input_tokens: usize,
        max_request_input_tokens: usize,
    },
    ContextOverflow {
        input_tokens: usize,
        max_new_tokens: usize,
        max_context_tokens: usize,
    },
    RunningRequestLimitReached {
        max_running_requests: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouterStatusCode {
    InvalidArgument,
    ResourceExhausted,
    FailedPrecondition,
}

impl RouterProtocolError {
    pub fn status_code(&self) -> RouterStatusCode {
        match self {
            Self::MissingRequestId
            | Self::MissingTokenizedInput
            | Self::EmptyTokenizedInput
            | Self::EmptyTextInput
            | Self::InvalidIntegerSamplingParam { .. }
            | Self::InvalidFloatSamplingParam { .. } => RouterStatusCode::InvalidArgument,
            Self::InputTooLong { .. }
            | Self::ContextOverflow { .. }
            | Self::RunningRequestLimitReached { .. } => RouterStatusCode::ResourceExhausted,
            Self::GenerationPaused => RouterStatusCode::FailedPrecondition,
        }
    }
}

impl fmt::Display for RouterProtocolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRequestId => formatter.write_str("missing router request id"),
            Self::MissingTokenizedInput => formatter.write_str("missing router tokenized input"),
            Self::EmptyTokenizedInput => formatter.write_str("empty router tokenized input"),
            Self::EmptyTextInput => formatter.write_str("empty router text input"),
            Self::GenerationPaused => formatter.write_str("router generation is paused"),
            Self::InvalidIntegerSamplingParam {
                field,
                value,
                expected,
            } => {
                write!(
                    formatter,
                    "router sampling param {field} must be {expected}: {value}"
                )
            }
            Self::InvalidFloatSamplingParam {
                field,
                value,
                expected,
            } => {
                write!(
                    formatter,
                    "router sampling param {field} must be {expected}: {value}"
                )
            }
            Self::InputTooLong {
                input_tokens,
                max_request_input_tokens,
            } => {
                write!(
                    formatter,
                    "router input token count {input_tokens} exceeds max request input token limit {max_request_input_tokens}"
                )
            }
            Self::ContextOverflow {
                input_tokens,
                max_new_tokens,
                max_context_tokens,
            } => {
                write!(
                    formatter,
                    "router input token count {input_tokens} plus max_new_tokens {max_new_tokens} exceeds context token limit {max_context_tokens}"
                )
            }
            Self::RunningRequestLimitReached {
                max_running_requests,
            } => {
                write!(
                    formatter,
                    "router running request limit reached: {max_running_requests}"
                )
            }
        }
    }
}

impl std::error::Error for RouterProtocolError {}

#[derive(Debug)]
pub enum RouterRuntimeError {
    Protocol(RouterProtocolError),
    Runtime(RuntimeError),
}

impl fmt::Display for RouterRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Protocol(error) => write!(formatter, "router protocol error: {error}"),
            Self::Runtime(error) => write!(formatter, "router runtime error: {error}"),
        }
    }
}

impl std::error::Error for RouterRuntimeError {}

impl From<RouterProtocolError> for RouterRuntimeError {
    fn from(value: RouterProtocolError) -> Self {
        Self::Protocol(value)
    }
}

impl From<RuntimeError> for RouterRuntimeError {
    fn from(value: RuntimeError) -> Self {
        if let RuntimeError::Scheduler(SchedulerError::RunningRequestLimitReached {
            max_running_requests,
        }) = value
        {
            return Self::Protocol(RouterProtocolError::RunningRequestLimitReached {
                max_running_requests,
            });
        }

        Self::Runtime(value)
    }
}
