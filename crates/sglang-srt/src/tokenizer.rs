use std::fmt;

#[derive(Debug, Eq, PartialEq)]
pub enum TokenizerError {
    InvalidUtf8,
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUtf8 => formatter.write_str("token ids are not valid UTF-8 bytes"),
        }
    }
}

impl std::error::Error for TokenizerError {}

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError>;
}

#[derive(Clone, Debug, Default)]
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(u32::from).collect()
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        let bytes = token_ids
            .iter()
            .map(|token_id| u8::try_from(*token_id).map_err(|_| TokenizerError::InvalidUtf8))
            .collect::<Result<Vec<_>, _>>()?;

        String::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8)
    }
}
