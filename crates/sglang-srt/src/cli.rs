use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CliCommand {
    Serve,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ZmqPortRange {
    pub start: u16,
    pub end: u16,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServerArgs {
    pub command: CliCommand,
    pub model_path: String,
    pub host: String,
    pub port: u16,
    pub log_level: Option<String>,
    pub tp_size: usize,
    pub dp_size: usize,
    pub kv_cache_dtype: String,
    pub kv_cache_num_layers: Option<usize>,
    pub kv_cache_kv_heads: Option<usize>,
    pub kv_cache_head_dim: Option<usize>,
    pub page_size: usize,
    pub base_gpu_id: usize,
    pub gpu_id_step: usize,
    pub nnodes: usize,
    pub node_rank: usize,
    pub dist_init_addr: Option<String>,
    pub trust_remote_code: bool,
    pub enable_dp_attention: bool,
    pub enable_dp_lm_head: bool,
    pub disable_cuda_graph: bool,
    pub moe_a2a_backend: Option<String>,
    pub moe_dense_tp_size: Option<usize>,
    pub mem_fraction_static: Option<f32>,
    pub max_running_requests: Option<usize>,
    pub max_prefill_tokens: Option<usize>,
    pub max_total_tokens: Option<usize>,
    pub grpc_mode: bool,
    pub served_model_name: Option<String>,
    pub tokenizer_path: Option<String>,
    pub disaggregation_mode: String,
    pub disaggregation_transfer_backend: String,
    pub disaggregation_bootstrap_port: u16,
    pub disaggregation_mooncake_rpc_port: Option<u16>,
    pub disaggregation_ib_device: Option<String>,
    pub disaggregation_zmq_ports: Option<ZmqPortRange>,
    pub disaggregation_decode_enable_radix_cache: bool,
    pub disaggregation_decode_enable_offload_kvcache: bool,
    pub num_reserved_decode_tokens: usize,
    pub disaggregation_decode_polling_interval: usize,
    pub deepep_config: Option<serde_json::Value>,
    pub deepep_mode: Option<String>,
    pub attention_backend: Option<String>,
    pub enable_nsa_prefill_context_parallel: bool,
    pub nsa_prefill_backend: Option<String>,
    pub nsa_prefill_cp_mode: Option<String>,
    pub speculative_algorithm: Option<String>,
    pub speculative_eagle_topk: Option<usize>,
    pub speculative_num_draft_tokens: Option<usize>,
    pub speculative_num_steps: Option<usize>,
    pub chunked_prefill_size: Option<usize>,
    pub decode_log_interval: Option<usize>,
    pub disable_overlap_schedule: bool,
    pub model_loader_extra_config: Option<serde_json::Value>,
    pub tokenizer_worker_num: Option<usize>,
    pub allow_auto_truncate: bool,
    pub collect_tokens_histogram: bool,
    pub enable_cache_report: bool,
    pub enable_metrics: bool,
    pub disable_radix_cache: bool,
    pub tool_call_parser: Option<String>,
    pub extra_args: Vec<String>,
}

impl ServerArgs {
    pub fn parse_from<I, S>(args: I) -> Result<Self, CliParseError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut parser = ArgParser::new(args);
        parser.parse()
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum CliParseError {
    InvalidCommand(String),
    MissingModelPath,
    MissingValue(&'static str),
    InvalidPort(String),
    InvalidUsize {
        flag: &'static str,
        value: String,
    },
    InvalidFloat {
        flag: &'static str,
        value: String,
    },
    InvalidJson {
        flag: &'static str,
        value: String,
        message: String,
    },
    InvalidZmqPortRange(String),
}

impl fmt::Display for CliParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCommand(command) => write!(formatter, "invalid command: {command}"),
            Self::MissingModelPath => formatter.write_str("missing --model-path"),
            Self::MissingValue(flag) => write!(formatter, "missing value for {flag}"),
            Self::InvalidPort(value) => write!(formatter, "invalid port: {value}"),
            Self::InvalidUsize { flag, value } => write!(formatter, "invalid {flag}: {value}"),
            Self::InvalidFloat { flag, value } => write!(formatter, "invalid {flag}: {value}"),
            Self::InvalidJson {
                flag,
                value,
                message,
            } => {
                write!(formatter, "invalid JSON for {flag}: {value}: {message}")
            }
            Self::InvalidZmqPortRange(value) => {
                write!(formatter, "invalid --disaggregation-zmq-ports: {value}")
            }
        }
    }
}

impl std::error::Error for CliParseError {}

struct ArgParser {
    args: Vec<String>,
    index: usize,
    parsed: PartialServerArgs,
}

impl ArgParser {
    fn new<I, S>(args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut parsed_args = Vec::new();
        for arg in args {
            let arg = arg.into();
            if let Some((flag, value)) = split_equals_arg(&arg) {
                parsed_args.push(flag.to_string());
                parsed_args.push(value.to_string());
            } else {
                parsed_args.push(arg);
            }
        }

        Self {
            args: parsed_args,
            index: 0,
            parsed: PartialServerArgs::default(),
        }
    }

    fn parse(&mut self) -> Result<ServerArgs, CliParseError> {
        self.parse_optional_command()?;

        while self.index < self.args.len() {
            let arg = self.args[self.index].clone();
            self.index += 1;

            match arg.as_str() {
                "--model-path" | "--model" => {
                    self.parsed.model_path = Some(self.take_value("--model-path")?);
                }
                "--host" => {
                    self.parsed.host = self.take_value("--host")?;
                }
                "--log-level" => {
                    self.parsed.log_level = Some(self.take_value("--log-level")?);
                }
                "--port" => {
                    let value = self.take_value("--port")?;
                    self.parsed.port = value
                        .parse::<u16>()
                        .map_err(|_| CliParseError::InvalidPort(value))?;
                }
                "--tp-size" | "--tp" => {
                    self.parsed.tp_size = parse_usize("--tp-size", self.take_value("--tp-size")?)?;
                }
                "--dp-size" | "--dp" => {
                    self.parsed.dp_size = parse_usize("--dp-size", self.take_value("--dp-size")?)?;
                }
                "--kv-cache-dtype" => {
                    self.parsed.kv_cache_dtype = self.take_value("--kv-cache-dtype")?;
                }
                "--kv-cache-num-layers" => {
                    self.parsed.kv_cache_num_layers = Some(parse_usize(
                        "--kv-cache-num-layers",
                        self.take_value("--kv-cache-num-layers")?,
                    )?);
                }
                "--kv-cache-kv-heads" => {
                    self.parsed.kv_cache_kv_heads = Some(parse_usize(
                        "--kv-cache-kv-heads",
                        self.take_value("--kv-cache-kv-heads")?,
                    )?);
                }
                "--kv-cache-head-dim" => {
                    self.parsed.kv_cache_head_dim = Some(parse_usize(
                        "--kv-cache-head-dim",
                        self.take_value("--kv-cache-head-dim")?,
                    )?);
                }
                "--page-size" => {
                    self.parsed.page_size =
                        parse_usize("--page-size", self.take_value("--page-size")?)?;
                }
                "--base-gpu-id" => {
                    self.parsed.base_gpu_id =
                        parse_usize("--base-gpu-id", self.take_value("--base-gpu-id")?)?;
                }
                "--gpu-id-step" => {
                    self.parsed.gpu_id_step =
                        parse_usize("--gpu-id-step", self.take_value("--gpu-id-step")?)?;
                }
                "--nnodes" => {
                    self.parsed.nnodes = parse_usize("--nnodes", self.take_value("--nnodes")?)?;
                }
                "--node-rank" => {
                    self.parsed.node_rank =
                        parse_usize("--node-rank", self.take_value("--node-rank")?)?;
                }
                "--dist-init-addr" => {
                    self.parsed.dist_init_addr = Some(self.take_value("--dist-init-addr")?);
                }
                "--trust-remote-code" => {
                    self.parsed.trust_remote_code = true;
                }
                "--enable-dp-attention" => {
                    self.parsed.enable_dp_attention = true;
                }
                "--enable-dp-lm-head" => {
                    self.parsed.enable_dp_lm_head = true;
                }
                "--disable-cuda-graph" => {
                    self.parsed.disable_cuda_graph = true;
                }
                "--moe-a2a-backend" => {
                    self.parsed.moe_a2a_backend = Some(self.take_value("--moe-a2a-backend")?);
                }
                "--moe-dense-tp-size" => {
                    self.parsed.moe_dense_tp_size = Some(parse_usize(
                        "--moe-dense-tp-size",
                        self.take_value("--moe-dense-tp-size")?,
                    )?);
                }
                "--mem-fraction-static" => {
                    self.parsed.mem_fraction_static = Some(parse_f32(
                        "--mem-fraction-static",
                        self.take_value("--mem-fraction-static")?,
                    )?);
                }
                "--max-running-requests" => {
                    self.parsed.max_running_requests = Some(parse_usize(
                        "--max-running-requests",
                        self.take_value("--max-running-requests")?,
                    )?);
                }
                "--max-prefill-tokens" => {
                    self.parsed.max_prefill_tokens = Some(parse_usize(
                        "--max-prefill-tokens",
                        self.take_value("--max-prefill-tokens")?,
                    )?);
                }
                "--max-total-tokens" => {
                    self.parsed.max_total_tokens = Some(parse_usize(
                        "--max-total-tokens",
                        self.take_value("--max-total-tokens")?,
                    )?);
                }
                "--grpc-mode" => {
                    self.parsed.grpc_mode = true;
                }
                "--served-model-name" => {
                    self.parsed.served_model_name = Some(self.take_value("--served-model-name")?);
                }
                "--tokenizer-path" => {
                    self.parsed.tokenizer_path = Some(self.take_value("--tokenizer-path")?);
                }
                "--disaggregation-mode" => {
                    self.parsed.disaggregation_mode = self.take_value("--disaggregation-mode")?;
                }
                "--disaggregation-transfer-backend" => {
                    self.parsed.disaggregation_transfer_backend =
                        self.take_value("--disaggregation-transfer-backend")?;
                }
                "--disaggregation-bootstrap-port" => {
                    let value = self.take_value("--disaggregation-bootstrap-port")?;
                    self.parsed.disaggregation_bootstrap_port = value
                        .parse::<u16>()
                        .map_err(|_| CliParseError::InvalidPort(value))?;
                }
                "--disaggregation-mooncake-rpc-port" => {
                    let value = self.take_value("--disaggregation-mooncake-rpc-port")?;
                    self.parsed.disaggregation_mooncake_rpc_port = Some(
                        value
                            .parse::<u16>()
                            .map_err(|_| CliParseError::InvalidPort(value))?,
                    );
                }
                "--disaggregation-ib-device" => {
                    self.parsed.disaggregation_ib_device =
                        Some(self.take_value("--disaggregation-ib-device")?);
                }
                "--disaggregation-zmq-ports" => {
                    self.parsed.disaggregation_zmq_ports = Some(parse_zmq_port_range(
                        self.take_value("--disaggregation-zmq-ports")?,
                    )?);
                }
                "--disaggregation-decode-enable-radix-cache" => {
                    self.parsed.disaggregation_decode_enable_radix_cache = true;
                }
                "--disaggregation-decode-enable-offload-kvcache" => {
                    self.parsed.disaggregation_decode_enable_offload_kvcache = true;
                }
                "--num-reserved-decode-tokens" => {
                    self.parsed.num_reserved_decode_tokens = parse_usize(
                        "--num-reserved-decode-tokens",
                        self.take_value("--num-reserved-decode-tokens")?,
                    )?;
                }
                "--disaggregation-decode-polling-interval" => {
                    self.parsed.disaggregation_decode_polling_interval = parse_usize(
                        "--disaggregation-decode-polling-interval",
                        self.take_value("--disaggregation-decode-polling-interval")?,
                    )?;
                }
                "--deepep-config" => {
                    self.parsed.deepep_config = Some(parse_json_value(
                        "--deepep-config",
                        self.take_value("--deepep-config")?,
                    )?);
                }
                "--deepep-mode" => {
                    self.parsed.deepep_mode = Some(self.take_value("--deepep-mode")?);
                }
                "--attention-backend" => {
                    self.parsed.attention_backend = Some(self.take_value("--attention-backend")?);
                }
                "--enable-nsa-prefill-context-parallel" => {
                    self.parsed.enable_nsa_prefill_context_parallel = true;
                }
                "--nsa-prefill-backend" => {
                    self.parsed.nsa_prefill_backend =
                        Some(self.take_value("--nsa-prefill-backend")?);
                }
                "--nsa-prefill-cp-mode" => {
                    self.parsed.nsa_prefill_cp_mode =
                        Some(self.take_value("--nsa-prefill-cp-mode")?);
                }
                "--speculative-algorithm" => {
                    self.parsed.speculative_algorithm =
                        Some(self.take_value("--speculative-algorithm")?);
                }
                "--speculative-eagle-topk" => {
                    self.parsed.speculative_eagle_topk = Some(parse_usize(
                        "--speculative-eagle-topk",
                        self.take_value("--speculative-eagle-topk")?,
                    )?);
                }
                "--speculative-num-draft-tokens" => {
                    self.parsed.speculative_num_draft_tokens = Some(parse_usize(
                        "--speculative-num-draft-tokens",
                        self.take_value("--speculative-num-draft-tokens")?,
                    )?);
                }
                "--speculative-num-steps" => {
                    self.parsed.speculative_num_steps = Some(parse_usize(
                        "--speculative-num-steps",
                        self.take_value("--speculative-num-steps")?,
                    )?);
                }
                "--chunked-prefill-size" => {
                    self.parsed.chunked_prefill_size = Some(parse_usize(
                        "--chunked-prefill-size",
                        self.take_value("--chunked-prefill-size")?,
                    )?);
                }
                "--decode-log-interval" => {
                    self.parsed.decode_log_interval = Some(parse_usize(
                        "--decode-log-interval",
                        self.take_value("--decode-log-interval")?,
                    )?);
                }
                "--disable-overlap-schedule" => {
                    self.parsed.disable_overlap_schedule = true;
                }
                "--model-loader-extra-config" => {
                    self.parsed.model_loader_extra_config = Some(parse_json_value(
                        "--model-loader-extra-config",
                        self.take_value("--model-loader-extra-config")?,
                    )?);
                }
                "--tokenizer-worker-num" => {
                    self.parsed.tokenizer_worker_num = Some(parse_usize(
                        "--tokenizer-worker-num",
                        self.take_value("--tokenizer-worker-num")?,
                    )?);
                }
                "--allow-auto-truncate" => {
                    self.parsed.allow_auto_truncate = true;
                }
                "--collect-tokens-histogram" => {
                    self.parsed.collect_tokens_histogram = true;
                }
                "--enable-cache-report" => {
                    self.parsed.enable_cache_report = true;
                }
                "--enable-metrics" => {
                    self.parsed.enable_metrics = true;
                }
                "--disable-radix-cache" => {
                    self.parsed.disable_radix_cache = true;
                }
                "--tool-call-parser" => {
                    self.parsed.tool_call_parser = Some(self.take_value("--tool-call-parser")?);
                }
                _ => self.preserve_unknown(arg),
            }
        }

        Ok(ServerArgs {
            command: CliCommand::Serve,
            model_path: self
                .parsed
                .model_path
                .take()
                .ok_or(CliParseError::MissingModelPath)?,
            host: self.parsed.host.clone(),
            port: self.parsed.port,
            log_level: self.parsed.log_level.clone(),
            tp_size: self.parsed.tp_size,
            dp_size: self.parsed.dp_size,
            kv_cache_dtype: self.parsed.kv_cache_dtype.clone(),
            kv_cache_num_layers: self.parsed.kv_cache_num_layers,
            kv_cache_kv_heads: self.parsed.kv_cache_kv_heads,
            kv_cache_head_dim: self.parsed.kv_cache_head_dim,
            page_size: self.parsed.page_size,
            base_gpu_id: self.parsed.base_gpu_id,
            gpu_id_step: self.parsed.gpu_id_step,
            nnodes: self.parsed.nnodes,
            node_rank: self.parsed.node_rank,
            dist_init_addr: self.parsed.dist_init_addr.clone(),
            trust_remote_code: self.parsed.trust_remote_code,
            enable_dp_attention: self.parsed.enable_dp_attention,
            enable_dp_lm_head: self.parsed.enable_dp_lm_head,
            disable_cuda_graph: self.parsed.disable_cuda_graph,
            moe_a2a_backend: self.parsed.moe_a2a_backend.clone(),
            moe_dense_tp_size: self.parsed.moe_dense_tp_size,
            mem_fraction_static: self.parsed.mem_fraction_static,
            max_running_requests: self.parsed.max_running_requests,
            max_prefill_tokens: self.parsed.max_prefill_tokens,
            max_total_tokens: self.parsed.max_total_tokens,
            grpc_mode: self.parsed.grpc_mode,
            served_model_name: self.parsed.served_model_name.clone(),
            tokenizer_path: self.parsed.tokenizer_path.clone(),
            disaggregation_mode: self.parsed.disaggregation_mode.clone(),
            disaggregation_transfer_backend: self.parsed.disaggregation_transfer_backend.clone(),
            disaggregation_bootstrap_port: self.parsed.disaggregation_bootstrap_port,
            disaggregation_mooncake_rpc_port: self.parsed.disaggregation_mooncake_rpc_port,
            disaggregation_ib_device: self.parsed.disaggregation_ib_device.clone(),
            disaggregation_zmq_ports: self.parsed.disaggregation_zmq_ports,
            disaggregation_decode_enable_radix_cache: self
                .parsed
                .disaggregation_decode_enable_radix_cache,
            disaggregation_decode_enable_offload_kvcache: self
                .parsed
                .disaggregation_decode_enable_offload_kvcache,
            num_reserved_decode_tokens: self.parsed.num_reserved_decode_tokens,
            disaggregation_decode_polling_interval: self
                .parsed
                .disaggregation_decode_polling_interval,
            deepep_config: self.parsed.deepep_config.clone(),
            deepep_mode: self.parsed.deepep_mode.clone(),
            attention_backend: self.parsed.attention_backend.clone(),
            enable_nsa_prefill_context_parallel: self.parsed.enable_nsa_prefill_context_parallel,
            nsa_prefill_backend: self.parsed.nsa_prefill_backend.clone(),
            nsa_prefill_cp_mode: self.parsed.nsa_prefill_cp_mode.clone(),
            speculative_algorithm: self.parsed.speculative_algorithm.clone(),
            speculative_eagle_topk: self.parsed.speculative_eagle_topk,
            speculative_num_draft_tokens: self.parsed.speculative_num_draft_tokens,
            speculative_num_steps: self.parsed.speculative_num_steps,
            chunked_prefill_size: self.parsed.chunked_prefill_size,
            decode_log_interval: self.parsed.decode_log_interval,
            disable_overlap_schedule: self.parsed.disable_overlap_schedule,
            model_loader_extra_config: self.parsed.model_loader_extra_config.clone(),
            tokenizer_worker_num: self.parsed.tokenizer_worker_num,
            allow_auto_truncate: self.parsed.allow_auto_truncate,
            collect_tokens_histogram: self.parsed.collect_tokens_histogram,
            enable_cache_report: self.parsed.enable_cache_report,
            enable_metrics: self.parsed.enable_metrics,
            disable_radix_cache: self.parsed.disable_radix_cache,
            tool_call_parser: self.parsed.tool_call_parser.clone(),
            extra_args: self.parsed.extra_args.clone(),
        })
    }

    fn parse_optional_command(&mut self) -> Result<(), CliParseError> {
        let Some(first) = self.args.first() else {
            return Ok(());
        };

        if first == "serve" || first == "launch_server" {
            self.index = 1;
            return Ok(());
        }

        if !first.starts_with('-') {
            return Err(CliParseError::InvalidCommand(first.clone()));
        }

        Ok(())
    }

    fn take_value(&mut self, flag: &'static str) -> Result<String, CliParseError> {
        let value = self
            .args
            .get(self.index)
            .cloned()
            .ok_or(CliParseError::MissingValue(flag))?;
        self.index += 1;
        Ok(value)
    }

    fn preserve_unknown(&mut self, arg: String) {
        self.parsed.extra_args.push(arg);

        if self.index < self.args.len() && !self.args[self.index].starts_with('-') {
            self.parsed.extra_args.push(self.args[self.index].clone());
            self.index += 1;
        }
    }
}

#[derive(Clone, Debug)]
struct PartialServerArgs {
    model_path: Option<String>,
    host: String,
    port: u16,
    log_level: Option<String>,
    tp_size: usize,
    dp_size: usize,
    kv_cache_dtype: String,
    kv_cache_num_layers: Option<usize>,
    kv_cache_kv_heads: Option<usize>,
    kv_cache_head_dim: Option<usize>,
    page_size: usize,
    base_gpu_id: usize,
    gpu_id_step: usize,
    nnodes: usize,
    node_rank: usize,
    dist_init_addr: Option<String>,
    trust_remote_code: bool,
    enable_dp_attention: bool,
    enable_dp_lm_head: bool,
    disable_cuda_graph: bool,
    moe_a2a_backend: Option<String>,
    moe_dense_tp_size: Option<usize>,
    mem_fraction_static: Option<f32>,
    max_running_requests: Option<usize>,
    max_prefill_tokens: Option<usize>,
    max_total_tokens: Option<usize>,
    grpc_mode: bool,
    served_model_name: Option<String>,
    tokenizer_path: Option<String>,
    disaggregation_mode: String,
    disaggregation_transfer_backend: String,
    disaggregation_bootstrap_port: u16,
    disaggregation_mooncake_rpc_port: Option<u16>,
    disaggregation_ib_device: Option<String>,
    disaggregation_zmq_ports: Option<ZmqPortRange>,
    disaggregation_decode_enable_radix_cache: bool,
    disaggregation_decode_enable_offload_kvcache: bool,
    num_reserved_decode_tokens: usize,
    disaggregation_decode_polling_interval: usize,
    deepep_config: Option<serde_json::Value>,
    deepep_mode: Option<String>,
    attention_backend: Option<String>,
    enable_nsa_prefill_context_parallel: bool,
    nsa_prefill_backend: Option<String>,
    nsa_prefill_cp_mode: Option<String>,
    speculative_algorithm: Option<String>,
    speculative_eagle_topk: Option<usize>,
    speculative_num_draft_tokens: Option<usize>,
    speculative_num_steps: Option<usize>,
    chunked_prefill_size: Option<usize>,
    decode_log_interval: Option<usize>,
    disable_overlap_schedule: bool,
    model_loader_extra_config: Option<serde_json::Value>,
    tokenizer_worker_num: Option<usize>,
    allow_auto_truncate: bool,
    collect_tokens_histogram: bool,
    enable_cache_report: bool,
    enable_metrics: bool,
    disable_radix_cache: bool,
    tool_call_parser: Option<String>,
    extra_args: Vec<String>,
}

impl Default for PartialServerArgs {
    fn default() -> Self {
        Self {
            model_path: None,
            host: "127.0.0.1".to_string(),
            port: 30000,
            log_level: None,
            tp_size: 1,
            dp_size: 1,
            kv_cache_dtype: "auto".to_string(),
            kv_cache_num_layers: None,
            kv_cache_kv_heads: None,
            kv_cache_head_dim: None,
            page_size: 1,
            base_gpu_id: 0,
            gpu_id_step: 1,
            nnodes: 1,
            node_rank: 0,
            dist_init_addr: None,
            trust_remote_code: false,
            enable_dp_attention: false,
            enable_dp_lm_head: false,
            disable_cuda_graph: false,
            moe_a2a_backend: None,
            moe_dense_tp_size: None,
            mem_fraction_static: None,
            max_running_requests: None,
            max_prefill_tokens: None,
            max_total_tokens: None,
            grpc_mode: false,
            served_model_name: None,
            tokenizer_path: None,
            disaggregation_mode: "null".to_string(),
            disaggregation_transfer_backend: "mooncake".to_string(),
            disaggregation_bootstrap_port: 8998,
            disaggregation_mooncake_rpc_port: None,
            disaggregation_ib_device: None,
            disaggregation_zmq_ports: None,
            disaggregation_decode_enable_radix_cache: false,
            disaggregation_decode_enable_offload_kvcache: false,
            num_reserved_decode_tokens: 512,
            disaggregation_decode_polling_interval: 1,
            deepep_config: None,
            deepep_mode: None,
            attention_backend: None,
            enable_nsa_prefill_context_parallel: false,
            nsa_prefill_backend: None,
            nsa_prefill_cp_mode: None,
            speculative_algorithm: None,
            speculative_eagle_topk: None,
            speculative_num_draft_tokens: None,
            speculative_num_steps: None,
            chunked_prefill_size: None,
            decode_log_interval: None,
            disable_overlap_schedule: false,
            model_loader_extra_config: None,
            tokenizer_worker_num: None,
            allow_auto_truncate: false,
            collect_tokens_histogram: false,
            enable_cache_report: false,
            enable_metrics: false,
            disable_radix_cache: false,
            tool_call_parser: None,
            extra_args: Vec::new(),
        }
    }
}

fn split_equals_arg(arg: &str) -> Option<(&str, &str)> {
    let (flag, value) = arg.split_once('=')?;
    flag.starts_with("--").then_some((flag, value))
}

fn parse_usize(flag: &'static str, value: String) -> Result<usize, CliParseError> {
    value
        .parse::<usize>()
        .map_err(|_| CliParseError::InvalidUsize { flag, value })
}

fn parse_f32(flag: &'static str, value: String) -> Result<f32, CliParseError> {
    value
        .parse::<f32>()
        .map_err(|_| CliParseError::InvalidFloat { flag, value })
}

fn parse_json_value(flag: &'static str, value: String) -> Result<serde_json::Value, CliParseError> {
    serde_json::from_str(&value).map_err(|error| CliParseError::InvalidJson {
        flag,
        value,
        message: error.to_string(),
    })
}

fn parse_zmq_port_range(value: String) -> Result<ZmqPortRange, CliParseError> {
    let Some((start, end)) = value.split_once('-') else {
        let port = value
            .parse::<u16>()
            .map_err(|_| CliParseError::InvalidZmqPortRange(value.clone()))?;
        return Ok(ZmqPortRange {
            start: port,
            end: port,
        });
    };
    let start = start
        .parse::<u16>()
        .map_err(|_| CliParseError::InvalidZmqPortRange(value.clone()))?;
    let end = end
        .parse::<u16>()
        .map_err(|_| CliParseError::InvalidZmqPortRange(value.clone()))?;
    if start > end {
        return Err(CliParseError::InvalidZmqPortRange(value));
    }
    Ok(ZmqPortRange { start, end })
}
