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
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenGenerateOutput {
    pub request_id: RequestId,
    pub output_ids: Vec<u32>,
    pub finished: bool,
}
