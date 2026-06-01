use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CliCommand {
    Serve,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerArgs {
    pub command: CliCommand,
    pub model_path: String,
    pub host: String,
    pub port: u16,
    pub tp_size: usize,
    pub dp_size: usize,
    pub grpc_mode: bool,
    pub served_model_name: Option<String>,
    pub tokenizer_path: Option<String>,
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
}

impl fmt::Display for CliParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCommand(command) => write!(formatter, "invalid command: {command}"),
            Self::MissingModelPath => formatter.write_str("missing --model-path"),
            Self::MissingValue(flag) => write!(formatter, "missing value for {flag}"),
            Self::InvalidPort(value) => write!(formatter, "invalid port: {value}"),
            Self::InvalidUsize { flag, value } => write!(formatter, "invalid {flag}: {value}"),
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
                "--grpc-mode" => {
                    self.parsed.grpc_mode = true;
                }
                "--served-model-name" => {
                    self.parsed.served_model_name = Some(self.take_value("--served-model-name")?);
                }
                "--tokenizer-path" => {
                    self.parsed.tokenizer_path = Some(self.take_value("--tokenizer-path")?);
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
            grpc_mode: self.parsed.grpc_mode,
            served_model_name: self.parsed.served_model_name.clone(),
            tokenizer_path: self.parsed.tokenizer_path.clone(),
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
    grpc_mode: bool,
    served_model_name: Option<String>,
    tokenizer_path: Option<String>,
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
            grpc_mode: false,
            served_model_name: None,
            tokenizer_path: None,
            extra_args: Vec::new(),
        }
    }
}

fn parse_usize(flag: &'static str, value: String) -> Result<usize, CliParseError> {
    value
        .parse::<usize>()
        .map_err(|_| CliParseError::InvalidUsize { flag, value })
}
