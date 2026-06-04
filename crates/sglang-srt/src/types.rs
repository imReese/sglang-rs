#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestId(String);

impl RequestId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for RequestId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SamplingParams {
    pub max_new_tokens: usize,
}

pub const FAKE_BOOTSTRAP_HOST: &str = "2.2.2.2";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DisaggregatedParams {
    pub bootstrap_host: String,
    pub bootstrap_port: u16,
    pub bootstrap_room: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerateRequest {
    pub request_id: RequestId,
    pub prompt: String,
    pub sampling: SamplingParams,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerateOutput {
    pub request_id: RequestId,
    pub text: String,
    pub finished: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenGenerateRequest {
    pub request_id: RequestId,
    pub input_ids: Vec<u32>,
    pub sampling: SamplingParams,
    pub disaggregated_params: Option<DisaggregatedParams>,
    pub data_parallel_rank: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenGenerateOutput {
    pub request_id: RequestId,
    pub output_ids: Vec<u32>,
    pub cached_tokens: usize,
    pub finished: bool,
}
