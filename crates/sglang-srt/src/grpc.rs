use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::future::{Future, pending};
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use tonic::{Code, Request, Response, Status};

use crate::cli::ServerArgs;
use crate::engine::Engine;
use crate::proto::sglang::runtime::v1::generate_response::Body as ProtoGenerateResponseBody;
use crate::proto::sglang::runtime::v1::sglang_service_server::{
    SglangService, SglangServiceServer,
};
use crate::proto::sglang::runtime::v1::{
    AbortRequest, ClassifyRequest, ClassifyResponse, ContinueGenerationRequest, ControlResponse,
    DetokenizeRequest, DetokenizeResponse, EmbedRequest, EmbedResponse, FlushCacheRequest,
    GenerateComplete as ProtoGenerateComplete, GenerateError as ProtoGenerateError,
    GenerateRequest as ProtoGenerateRequest, GenerateResponse as ProtoGenerateResponse,
    GenerateStreamChunk as ProtoGenerateStreamChunk, GetLoadRequest, GetModelInfoRequest,
    GetServerInfoRequest, HealthCheckRequest, HealthCheckResponse, ListModelsRequest,
    ListModelsResponse, LoadResponse, ModelInfoResponse, OpenAiJsonRequest, OpenAiJsonResponse,
    PauseGenerationRequest, SamplingParams as ProtoSamplingParams, ServerInfoResponse,
    StartProfileRequest, StopProfileRequest, TextEmbedRequest, TextGenerateRequest,
    TokenizeRequest, TokenizeResponse, UpdateWeightsFromDiskRequest,
};
use crate::router::{
    RouterDisaggregatedParams, RouterGenerateComplete, RouterGenerateError, RouterGenerateRequest,
    RouterGenerateResponse, RouterGenerateResponseBody, RouterGenerateStreamChunk,
    RouterGetModelInfoResponse, RouterProtocolError, RouterRuntime, RouterRuntimeError,
    RouterSamplingParams, RouterStatusCode, RouterTextGenerateRequest,
};
use crate::tokenizer::Tokenizer;
use crate::worker::WorkerExecutor;

type GenerateResponseStream = Pin<
    Box<
        dyn tonic::codegen::tokio_stream::Stream<Item = Result<ProtoGenerateResponse, Status>>
            + Send,
    >,
>;
type OpenAiJsonResponseStream = Pin<
    Box<dyn tonic::codegen::tokio_stream::Stream<Item = Result<OpenAiJsonResponse, Status>> + Send>,
>;

pub const SGLANG_RUNTIME_FILE_DESCRIPTOR_SET: &[u8] =
    tonic::include_file_descriptor_set!("sglang_runtime_descriptor");

#[derive(Clone)]
pub struct GrpcRouterService<T, W> {
    runtime: Arc<Mutex<RouterRuntime<T, W>>>,
    model_info: Option<RouterGetModelInfoResponse>,
    max_transfer_polls: usize,
    server_info_attributes: HashMap<String, String>,
}

#[derive(Debug)]
pub enum GrpcServeError {
    Reflection(tonic_reflection::server::Error),
    Transport(tonic::transport::Error),
}

impl fmt::Display for GrpcServeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reflection(error) => write!(formatter, "gRPC reflection error: {error}"),
            Self::Transport(error) => write!(formatter, "gRPC transport error: {error}"),
        }
    }
}

impl std::error::Error for GrpcServeError {}

impl From<tonic_reflection::server::Error> for GrpcServeError {
    fn from(value: tonic_reflection::server::Error) -> Self {
        Self::Reflection(value)
    }
}

impl From<tonic::transport::Error> for GrpcServeError {
    fn from(value: tonic::transport::Error) -> Self {
        Self::Transport(value)
    }
}

impl<T, W> GrpcRouterService<T, W> {
    pub fn new(runtime: RouterRuntime<T, W>) -> Self {
        Self {
            runtime: Arc::new(Mutex::new(runtime)),
            model_info: None,
            max_transfer_polls: 0,
            server_info_attributes: default_server_info_attributes(),
        }
    }

    pub fn with_model_info(
        runtime: RouterRuntime<T, W>,
        model_info: RouterGetModelInfoResponse,
    ) -> Self {
        Self {
            runtime: Arc::new(Mutex::new(runtime)),
            model_info: Some(model_info),
            max_transfer_polls: 0,
            server_info_attributes: default_server_info_attributes(),
        }
    }

    pub fn with_max_transfer_polls(mut self, max_transfer_polls: usize) -> Self {
        self.max_transfer_polls = max_transfer_polls;
        self
    }

    pub fn with_server_args(runtime: RouterRuntime<T, W>, args: &ServerArgs) -> Self {
        Self::with_model_info(runtime, RouterGetModelInfoResponse::from_server_args(args))
            .with_server_info_attributes(server_info_attributes_from_args(args))
    }

    pub fn with_server_info_attributes(mut self, attributes: HashMap<String, String>) -> Self {
        self.server_info_attributes = attributes;
        self
    }

    pub fn from_engine(engine: Engine<T, W>) -> Self {
        Self::new(RouterRuntime::new(engine))
    }

    pub fn from_engine_with_model_info(
        engine: Engine<T, W>,
        model_info: RouterGetModelInfoResponse,
    ) -> Self {
        Self::with_model_info(RouterRuntime::new(engine), model_info)
    }

    pub fn runtime(&self) -> &Arc<Mutex<RouterRuntime<T, W>>> {
        &self.runtime
    }

    fn model_info(&self) -> Result<RouterGetModelInfoResponse, Status> {
        self.model_info
            .clone()
            .ok_or_else(|| Status::failed_precondition("model info is not configured"))
    }
}

pub fn router_status_code_to_grpc_code(status_code: RouterStatusCode) -> Code {
    match status_code {
        RouterStatusCode::InvalidArgument => Code::InvalidArgument,
        RouterStatusCode::ResourceExhausted => Code::ResourceExhausted,
        RouterStatusCode::FailedPrecondition => Code::FailedPrecondition,
    }
}

pub fn router_protocol_error_to_status(error: RouterProtocolError) -> Status {
    Status::new(
        router_status_code_to_grpc_code(error.status_code()),
        error.to_string(),
    )
}

pub async fn serve_grpc_router<T, W>(
    addr: SocketAddr,
    service: GrpcRouterService<T, W>,
    enable_reflection: bool,
) -> Result<(), GrpcServeError>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    serve_grpc_router_with_shutdown(addr, service, enable_reflection, pending()).await
}

pub async fn serve_grpc_router_with_shutdown<T, W, F>(
    addr: SocketAddr,
    service: GrpcRouterService<T, W>,
    enable_reflection: bool,
    shutdown: F,
) -> Result<(), GrpcServeError>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
    F: Future<Output = ()> + Send + 'static,
{
    let sglang_service = SglangServiceServer::new(service);

    if enable_reflection {
        let reflection_service = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(SGLANG_RUNTIME_FILE_DESCRIPTOR_SET)
            .build_v1()?;

        tonic::transport::Server::builder()
            .add_service(sglang_service)
            .add_service(reflection_service)
            .serve_with_shutdown(addr, shutdown)
            .await?;

        return Ok(());
    }

    tonic::transport::Server::builder()
        .add_service(sglang_service)
        .serve_with_shutdown(addr, shutdown)
        .await?;

    Ok(())
}

fn router_runtime_error_to_status(error: RouterRuntimeError) -> Status {
    match error {
        RouterRuntimeError::Protocol(error) => router_protocol_error_to_status(error),
        RouterRuntimeError::Runtime(error) => Status::internal(error.to_string()),
    }
}

fn unimplemented_rpc(name: &'static str) -> Status {
    Status::unimplemented(format!(
        "{name} is not implemented in the Rust gRPC runtime yet"
    ))
}

#[tonic::async_trait]
impl<T, W> SglangService for GrpcRouterService<T, W>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    type TextGenerateStream = GenerateResponseStream;
    type GenerateStream = GenerateResponseStream;
    type ChatCompleteStream = OpenAiJsonResponseStream;
    type CompleteStream = OpenAiJsonResponseStream;

    async fn text_generate(
        &self,
        request: Request<TextGenerateRequest>,
    ) -> Result<Response<Self::TextGenerateStream>, Status> {
        let request = proto_text_generate_request_to_router_request(request.into_inner())?;
        let responses = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .generate_text_stream_with_transfer_polling(request, self.max_transfer_polls)
            .map_err(router_runtime_error_to_status)?
            .into_iter()
            .map(router_generate_response_to_proto_response)
            .map(Ok)
            .collect::<Vec<_>>();

        Ok(Response::new(Box::pin(tonic::codegen::tokio_stream::iter(
            responses,
        ))))
    }

    async fn generate(
        &self,
        request: Request<ProtoGenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let request = proto_generate_request_to_router_request(request.into_inner())?;
        let responses = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .generate_stream_with_transfer_polling(request, self.max_transfer_polls)
            .map_err(router_runtime_error_to_status)?
            .into_iter()
            .map(router_generate_response_to_proto_response)
            .map(Ok)
            .collect::<Vec<_>>();

        Ok(Response::new(Box::pin(tonic::codegen::tokio_stream::iter(
            responses,
        ))))
    }

    async fn text_embed(
        &self,
        _request: Request<TextEmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        Err(unimplemented_rpc("TextEmbed"))
    }

    async fn embed(
        &self,
        _request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        Err(unimplemented_rpc("Embed"))
    }

    async fn classify(
        &self,
        _request: Request<ClassifyRequest>,
    ) -> Result<Response<ClassifyResponse>, Status> {
        Err(unimplemented_rpc("Classify"))
    }

    async fn tokenize(
        &self,
        request: Request<TokenizeRequest>,
    ) -> Result<Response<TokenizeResponse>, Status> {
        let request = request.into_inner();
        let response = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .tokenize(&request.text);

        Ok(Response::new(TokenizeResponse {
            count: usize_to_u32(response.token_ids.len())?,
            token_ids: response.token_ids,
        }))
    }

    async fn detokenize(
        &self,
        request: Request<DetokenizeRequest>,
    ) -> Result<Response<DetokenizeResponse>, Status> {
        let request = request.into_inner();
        let response = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .detokenize(&request.token_ids)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;

        Ok(Response::new(DetokenizeResponse {
            text: response.text,
        }))
    }

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        Ok(Response::new(HealthCheckResponse {
            healthy: true,
            message: "ready".to_string(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<GetModelInfoRequest>,
    ) -> Result<Response<ModelInfoResponse>, Status> {
        Ok(Response::new(router_model_info_to_proto_response(
            self.model_info()?,
        )))
    }

    async fn get_server_info(
        &self,
        _request: Request<GetServerInfoRequest>,
    ) -> Result<Response<ServerInfoResponse>, Status> {
        Ok(Response::new(ServerInfoResponse {
            version: env!("CARGO_PKG_VERSION").to_string(),
            runtime: "sglang-rs".to_string(),
            attributes: self.server_info_attributes.clone(),
        }))
    }

    async fn list_models(
        &self,
        _request: Request<ListModelsRequest>,
    ) -> Result<Response<ListModelsResponse>, Status> {
        Ok(Response::new(ListModelsResponse {
            models: vec![router_model_info_to_proto_response(self.model_info()?)],
        }))
    }

    async fn get_load(
        &self,
        _request: Request<GetLoadRequest>,
    ) -> Result<Response<LoadResponse>, Status> {
        let response = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .load();

        Ok(Response::new(LoadResponse {
            waiting_queue_depth: usize_to_u32(response.waiting_queue_depth)?,
            decode_queue_depth: usize_to_u32(response.decode_queue_depth)?,
            available_cache_pages: response
                .available_cache_pages
                .map(usize_to_u32)
                .transpose()?,
        }))
    }

    async fn abort(
        &self,
        request: Request<AbortRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        let response = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .abort_request(&request.into_inner().request_id)
            .map_err(router_protocol_error_to_status)?;

        Ok(Response::new(ControlResponse {
            success: response.success,
            message: response.message,
        }))
    }

    async fn flush_cache(
        &self,
        _request: Request<FlushCacheRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        let response = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .flush_cache();

        Ok(Response::new(ControlResponse {
            success: response.success,
            message: response.message,
        }))
    }

    async fn pause_generation(
        &self,
        _request: Request<PauseGenerationRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        let response = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .pause_generation();

        Ok(Response::new(ControlResponse {
            success: response.success,
            message: response.message,
        }))
    }

    async fn continue_generation(
        &self,
        _request: Request<ContinueGenerationRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        let response = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .continue_generation();

        Ok(Response::new(ControlResponse {
            success: response.success,
            message: response.message,
        }))
    }

    async fn chat_complete(
        &self,
        _request: Request<OpenAiJsonRequest>,
    ) -> Result<Response<Self::ChatCompleteStream>, Status> {
        Err(unimplemented_rpc("ChatComplete"))
    }

    async fn complete(
        &self,
        _request: Request<OpenAiJsonRequest>,
    ) -> Result<Response<Self::CompleteStream>, Status> {
        Err(unimplemented_rpc("Complete"))
    }

    async fn open_ai_embed(
        &self,
        _request: Request<OpenAiJsonRequest>,
    ) -> Result<Response<OpenAiJsonResponse>, Status> {
        Err(unimplemented_rpc("OpenAIEmbed"))
    }

    async fn open_ai_classify(
        &self,
        _request: Request<OpenAiJsonRequest>,
    ) -> Result<Response<OpenAiJsonResponse>, Status> {
        Err(unimplemented_rpc("OpenAIClassify"))
    }

    async fn score(
        &self,
        _request: Request<OpenAiJsonRequest>,
    ) -> Result<Response<OpenAiJsonResponse>, Status> {
        Err(unimplemented_rpc("Score"))
    }

    async fn rerank(
        &self,
        _request: Request<OpenAiJsonRequest>,
    ) -> Result<Response<OpenAiJsonResponse>, Status> {
        Err(unimplemented_rpc("Rerank"))
    }

    async fn start_profile(
        &self,
        _request: Request<StartProfileRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        Err(unimplemented_rpc("StartProfile"))
    }

    async fn stop_profile(
        &self,
        _request: Request<StopProfileRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        Err(unimplemented_rpc("StopProfile"))
    }

    async fn update_weights_from_disk(
        &self,
        _request: Request<UpdateWeightsFromDiskRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        Err(unimplemented_rpc("UpdateWeightsFromDisk"))
    }
}

fn default_server_info_attributes() -> HashMap<String, String> {
    HashMap::from([("transport".to_string(), "tonic-grpc".to_string())])
}

fn server_info_attributes_from_args(args: &ServerArgs) -> HashMap<String, String> {
    let mut attributes = default_server_info_attributes();
    attributes.insert(
        "served_model_name".to_string(),
        args.served_model_name
            .clone()
            .unwrap_or_else(|| args.model_path.clone()),
    );
    attributes.insert(
        "disaggregation_mode".to_string(),
        args.disaggregation_mode.clone(),
    );

    if args.disaggregation_mode == "prefill" {
        attributes.insert(
            "disaggregation_bootstrap_port".to_string(),
            args.disaggregation_bootstrap_port.to_string(),
        );
    }

    if let Some(ports) = args.disaggregation_zmq_ports {
        attributes.insert("kv_events.publisher".to_string(), "zmq".to_string());
        attributes.insert(
            "kv_events.endpoint_host".to_string(),
            grpc_kv_events_endpoint_host(args).to_string(),
        );
        attributes.insert(
            "kv_events.endpoint_port_base".to_string(),
            ports.start.to_string(),
        );
        attributes.insert("kv_events.topic".to_string(), String::new());
        attributes.insert(
            "kv_events.block_size".to_string(),
            args.page_size.to_string(),
        );
        attributes.insert("kv_events.dp_size".to_string(), args.dp_size.to_string());
    }

    attributes
}

fn grpc_kv_events_endpoint_host(args: &ServerArgs) -> &str {
    if matches!(args.host.as_str(), "0.0.0.0" | "::" | "[::]") {
        if let Some(host) = args.dist_init_addr.as_deref().and_then(host_from_addr) {
            return host;
        }
    }
    &args.host
}

fn host_from_addr(addr: &str) -> Option<&str> {
    if let Some(rest) = addr.strip_prefix('[') {
        let (host, _) = rest.split_once(']')?;
        return Some(host);
    }
    addr.rsplit_once(':')
        .map(|(host, _)| host)
        .filter(|host| !host.is_empty())
}

fn proto_generate_request_to_router_request(
    request: ProtoGenerateRequest,
) -> Result<RouterGenerateRequest, Status> {
    let options = request.options.unwrap_or_default();

    Ok(RouterGenerateRequest {
        request_id: options.request_id.unwrap_or_default(),
        tokenized: Some(crate::router::RouterTokenizedInput {
            original_text: request.original_text,
            input_ids: request.input_ids,
        }),
        sampling_params: request.sampling_params.map(proto_sampling_params_to_router),
        disaggregated_params: request
            .disaggregated_params
            .map(proto_disaggregated_params_to_router)
            .transpose()?,
        stream: options.stream,
        data_parallel_rank: options.data_parallel_rank,
        trace_headers: options
            .trace_headers
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    })
}

fn proto_text_generate_request_to_router_request(
    request: TextGenerateRequest,
) -> Result<RouterTextGenerateRequest, Status> {
    let options = request.options.unwrap_or_default();

    Ok(RouterTextGenerateRequest {
        request_id: options.request_id.unwrap_or_default(),
        text: request.text,
        sampling_params: request.sampling_params.map(proto_sampling_params_to_router),
        disaggregated_params: request
            .disaggregated_params
            .map(proto_disaggregated_params_to_router)
            .transpose()?,
        stream: options.stream,
        data_parallel_rank: options.data_parallel_rank,
        trace_headers: options
            .trace_headers
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    })
}

fn proto_sampling_params_to_router(params: ProtoSamplingParams) -> RouterSamplingParams {
    RouterSamplingParams {
        max_new_tokens: params.max_new_tokens,
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: params.top_k,
        min_p: params.min_p,
        frequency_penalty: params.frequency_penalty,
        presence_penalty: params.presence_penalty,
        repetition_penalty: params.repetition_penalty,
        stop_token_id: params.stop_token_id,
        stop_token_ids: params
            .stop_token_ids
            .into_iter()
            .map(|stop_token_id| stop_token_id as i32)
            .collect(),
        ignore_eos: params.ignore_eos,
        n: params.n,
        best_of: params.best_of,
    }
}

fn proto_disaggregated_params_to_router(
    params: crate::proto::sglang::runtime::v1::DisaggregatedParams,
) -> Result<RouterDisaggregatedParams, Status> {
    let bootstrap_port = u16::try_from(params.bootstrap_port).map_err(|_| {
        Status::invalid_argument(format!(
            "bootstrap_port must fit in u16: {}",
            params.bootstrap_port
        ))
    })?;

    Ok(RouterDisaggregatedParams {
        bootstrap_host: params.bootstrap_host,
        bootstrap_port,
        bootstrap_room: params.bootstrap_room,
    })
}

fn router_generate_response_to_proto_response(
    response: RouterGenerateResponse,
) -> ProtoGenerateResponse {
    ProtoGenerateResponse {
        request_id: response.request_id,
        body: Some(match response.body {
            RouterGenerateResponseBody::Chunk(chunk) => {
                ProtoGenerateResponseBody::Chunk(router_chunk_to_proto(chunk))
            }
            RouterGenerateResponseBody::Complete(complete) => {
                ProtoGenerateResponseBody::Complete(router_complete_to_proto(complete))
            }
            RouterGenerateResponseBody::Error(error) => {
                ProtoGenerateResponseBody::Error(router_error_to_proto(error))
            }
        }),
    }
}

fn router_chunk_to_proto(chunk: RouterGenerateStreamChunk) -> ProtoGenerateStreamChunk {
    ProtoGenerateStreamChunk {
        token_ids: chunk.token_ids,
        text: chunk.text,
        prompt_tokens: chunk.prompt_tokens,
        completion_tokens: chunk.completion_tokens,
        cached_tokens: chunk.cached_tokens,
        index: chunk.index,
    }
}

fn router_complete_to_proto(complete: RouterGenerateComplete) -> ProtoGenerateComplete {
    ProtoGenerateComplete {
        output_ids: complete.output_ids,
        text: complete.text,
        finish_reason: complete.finish_reason,
        prompt_tokens: complete.prompt_tokens,
        completion_tokens: complete.completion_tokens,
        cached_tokens: complete.cached_tokens,
        index: complete.index,
    }
}

fn router_error_to_proto(error: RouterGenerateError) -> ProtoGenerateError {
    ProtoGenerateError {
        message: error.message,
        code: String::new(),
    }
}

fn router_model_info_to_proto_response(
    model_info: RouterGetModelInfoResponse,
) -> ModelInfoResponse {
    ModelInfoResponse {
        model_path: model_info.model_path,
        tokenizer_path: model_info.tokenizer_path,
        is_generation: model_info.is_generation,
        preferred_sampling_params_json: model_info.preferred_sampling_params,
        weight_version: model_info.weight_version,
        served_model_name: model_info.served_model_name,
        max_context_length: model_info.max_context_length,
        vocab_size: model_info.vocab_size,
        supports_vision: model_info.supports_vision,
        model_type: model_info.model_type,
        eos_token_ids: model_info.eos_token_ids,
        pad_token_id: model_info.pad_token_id,
        bos_token_id: model_info.bos_token_id,
        max_request_input_length: model_info.max_req_input_len,
        routed_expert_expected_group_count: model_info.routed_expert_expected_group_count,
        routed_expert_actual_group_count: model_info.routed_expert_actual_group_count,
        routed_expert_expected_weight_count: model_info.routed_expert_expected_weight_count,
        routed_expert_actual_weight_count: model_info.routed_expert_actual_weight_count,
    }
}

fn usize_to_u32(value: usize) -> Result<u32, Status> {
    u32::try_from(value).map_err(|_| Status::internal("load metric overflowed uint32"))
}
