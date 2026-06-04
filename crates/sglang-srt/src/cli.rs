use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CliCommand {
    Serve,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServerArgs {
    pub command: CliCommand,
    pub model_path: String,
    pub host: String,
    pub port: u16,
    pub tp_size: usize,
    pub dp_size: usize,
    pub kv_cache_dtype: String,
    pub page_size: usize,
    pub base_gpu_id: usize,
    pub gpu_id_step: usize,
    pub nnodes: usize,
    pub node_rank: usize,
    pub dist_init_addr: Option<String>,
    pub trust_remote_code: bool,
    pub enable_dp_attention: bool,
    pub moe_a2a_backend: Option<String>,
    pub mem_fraction_static: Option<f32>,
    pub max_running_requests: Option<usize>,
    pub grpc_mode: bool,
    pub served_model_name: Option<String>,
    pub tokenizer_path: Option<String>,
    pub disaggregation_mode: String,
    pub disaggregation_transfer_backend: String,
    pub disaggregation_bootstrap_port: u16,
    pub disaggregation_ib_device: Option<String>,
    pub disaggregation_decode_enable_radix_cache: bool,
    pub disaggregation_decode_enable_offload_kvcache: bool,
    pub num_reserved_decode_tokens: usize,
    pub disaggregation_decode_polling_interval: usize,
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
    InvalidUsize { flag: &'static str, value: String },
    InvalidFloat { flag: &'static str, value: String },
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
        Self {
            args: args.into_iter().map(Into::into).collect(),
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
                "--moe-a2a-backend" => {
                    self.parsed.moe_a2a_backend = Some(self.take_value("--moe-a2a-backend")?);
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
                "--disaggregation-ib-device" => {
                    self.parsed.disaggregation_ib_device =
                        Some(self.take_value("--disaggregation-ib-device")?);
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
            tp_size: self.parsed.tp_size,
            dp_size: self.parsed.dp_size,
            kv_cache_dtype: self.parsed.kv_cache_dtype.clone(),
            page_size: self.parsed.page_size,
            base_gpu_id: self.parsed.base_gpu_id,
            gpu_id_step: self.parsed.gpu_id_step,
            nnodes: self.parsed.nnodes,
            node_rank: self.parsed.node_rank,
            dist_init_addr: self.parsed.dist_init_addr.clone(),
            trust_remote_code: self.parsed.trust_remote_code,
            enable_dp_attention: self.parsed.enable_dp_attention,
            moe_a2a_backend: self.parsed.moe_a2a_backend.clone(),
            mem_fraction_static: self.parsed.mem_fraction_static,
            max_running_requests: self.parsed.max_running_requests,
            grpc_mode: self.parsed.grpc_mode,
            served_model_name: self.parsed.served_model_name.clone(),
            tokenizer_path: self.parsed.tokenizer_path.clone(),
            disaggregation_mode: self.parsed.disaggregation_mode.clone(),
            disaggregation_transfer_backend: self.parsed.disaggregation_transfer_backend.clone(),
            disaggregation_bootstrap_port: self.parsed.disaggregation_bootstrap_port,
            disaggregation_ib_device: self.parsed.disaggregation_ib_device.clone(),
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
    tp_size: usize,
    dp_size: usize,
    kv_cache_dtype: String,
    page_size: usize,
    base_gpu_id: usize,
    gpu_id_step: usize,
    nnodes: usize,
    node_rank: usize,
    dist_init_addr: Option<String>,
    trust_remote_code: bool,
    enable_dp_attention: bool,
    moe_a2a_backend: Option<String>,
    mem_fraction_static: Option<f32>,
    max_running_requests: Option<usize>,
    grpc_mode: bool,
    served_model_name: Option<String>,
    tokenizer_path: Option<String>,
    disaggregation_mode: String,
    disaggregation_transfer_backend: String,
    disaggregation_bootstrap_port: u16,
    disaggregation_ib_device: Option<String>,
    disaggregation_decode_enable_radix_cache: bool,
    disaggregation_decode_enable_offload_kvcache: bool,
    num_reserved_decode_tokens: usize,
    disaggregation_decode_polling_interval: usize,
    extra_args: Vec<String>,
}

impl Default for PartialServerArgs {
    fn default() -> Self {
        Self {
            model_path: None,
            host: "127.0.0.1".to_string(),
            port: 30000,
            tp_size: 1,
            dp_size: 1,
            kv_cache_dtype: "auto".to_string(),
            page_size: 1,
            base_gpu_id: 0,
            gpu_id_step: 1,
            nnodes: 1,
            node_rank: 0,
            dist_init_addr: None,
            trust_remote_code: false,
            enable_dp_attention: false,
            moe_a2a_backend: None,
            mem_fraction_static: None,
            max_running_requests: None,
            grpc_mode: false,
            served_model_name: None,
            tokenizer_path: None,
            disaggregation_mode: "null".to_string(),
            disaggregation_transfer_backend: "mooncake".to_string(),
            disaggregation_bootstrap_port: 8998,
            disaggregation_ib_device: None,
            disaggregation_decode_enable_radix_cache: false,
            disaggregation_decode_enable_offload_kvcache: false,
            num_reserved_decode_tokens: 512,
            disaggregation_decode_polling_interval: 1,
            extra_args: Vec::new(),
        }
    }
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
