use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use tonic::{Code, Request, Response, Status};

use crate::engine::Engine;
use crate::proto::sglang::runtime::v1::generate_response::Body as ProtoGenerateResponseBody;
use crate::proto::sglang::runtime::v1::sglang_service_server::SglangService;
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
    RouterProtocolError, RouterRuntime, RouterRuntimeError, RouterSamplingParams, RouterStatusCode,
    RouterTextGenerateRequest,
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

pub struct GrpcRouterService<T, W> {
    runtime: Arc<Mutex<RouterRuntime<T, W>>>,
}

impl<T, W> GrpcRouterService<T, W> {
    pub fn new(runtime: RouterRuntime<T, W>) -> Self {
        Self {
            runtime: Arc::new(Mutex::new(runtime)),
        }
    }

    pub fn from_engine(engine: Engine<T, W>) -> Self {
        Self::new(RouterRuntime::new(engine))
    }

    pub fn runtime(&self) -> &Arc<Mutex<RouterRuntime<T, W>>> {
        &self.runtime
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
            .generate_text_stream(request)
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
            .generate_stream(request)
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
        Err(unimplemented_rpc("GetModelInfo"))
    }

    async fn get_server_info(
        &self,
        _request: Request<GetServerInfoRequest>,
    ) -> Result<Response<ServerInfoResponse>, Status> {
        Err(unimplemented_rpc("GetServerInfo"))
    }

    async fn list_models(
        &self,
        _request: Request<ListModelsRequest>,
    ) -> Result<Response<ListModelsResponse>, Status> {
        Err(unimplemented_rpc("ListModels"))
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
        _request: Request<AbortRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        Err(unimplemented_rpc("Abort"))
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
        Err(unimplemented_rpc("PauseGeneration"))
    }

    async fn continue_generation(
        &self,
        _request: Request<ContinueGenerationRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        Err(unimplemented_rpc("ContinueGeneration"))
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

fn usize_to_u32(value: usize) -> Result<u32, Status> {
    u32::try_from(value).map_err(|_| Status::internal("load metric overflowed uint32"))
}
