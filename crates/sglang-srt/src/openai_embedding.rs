use serde_json::{Value, json};

use crate::router::RouterRuntime;
use crate::tokenizer::Tokenizer;
use crate::worker::WorkerExecutor;

const DEFAULT_EMBEDDING_DIMENSIONS: usize = 8;
const MAX_EMBEDDING_DIMENSIONS: usize = 4096;

#[derive(Debug)]
pub(crate) struct OpenAiEmbeddingRequest {
    model: String,
    inputs: Vec<EmbeddingInput>,
    dimensions: usize,
}

#[derive(Debug)]
enum EmbeddingInput {
    Text(String),
    TokenIds(Vec<u32>),
}

#[derive(Debug)]
struct EmbeddingOutput {
    embedding: Vec<f32>,
    prompt_tokens: usize,
    index: usize,
}

pub(crate) fn embeddings_response_json<T, W>(
    runtime: &RouterRuntime<T, W>,
    request: &OpenAiEmbeddingRequest,
) -> Value
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let outputs = request
        .inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let token_ids = match input {
                EmbeddingInput::Text(text) => runtime.tokenize(text).token_ids,
                EmbeddingInput::TokenIds(token_ids) => token_ids.clone(),
            };
            EmbeddingOutput {
                embedding: token_ids_to_embedding(&token_ids, request.dimensions),
                prompt_tokens: token_ids.len(),
                index,
            }
        })
        .collect::<Vec<_>>();

    let prompt_tokens = outputs
        .iter()
        .map(|output| output.prompt_tokens)
        .sum::<usize>();
    json!({
        "object": "list",
        "model": request.model,
        "data": outputs.into_iter().map(|output| json!({
            "object": "embedding",
            "embedding": output.embedding,
            "index": output.index,
        })).collect::<Vec<_>>(),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        }
    })
}

pub(crate) fn parse_embedding_request(
    payload: &Value,
    served_model_name: &str,
) -> Result<OpenAiEmbeddingRequest, String> {
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing model".to_string())?;
    if model != served_model_name {
        return Err(format!(
            "model {model} is not served by this worker ({served_model_name})"
        ));
    }

    let input = payload
        .get("input")
        .ok_or_else(|| "missing input".to_string())?;
    let inputs = parse_embedding_inputs(input)?;
    let dimensions = parse_dimensions(payload)?;
    let encoding_format = payload
        .get("encoding_format")
        .and_then(Value::as_str)
        .unwrap_or("float");
    if encoding_format != "float" {
        return Err("encoding_format must be `float`".to_string());
    }

    Ok(OpenAiEmbeddingRequest {
        model: model.to_string(),
        inputs,
        dimensions,
    })
}

fn parse_embedding_inputs(value: &Value) -> Result<Vec<EmbeddingInput>, String> {
    match value {
        Value::String(text) => {
            validate_non_empty_text(text, "Input")?;
            Ok(vec![EmbeddingInput::Text(text.clone())])
        }
        Value::Array(items) => parse_embedding_input_array(items),
        _ => Err(
            "input must be a string, list of strings, token ids, or token id batches".to_string(),
        ),
    }
}

fn parse_embedding_input_array(items: &[Value]) -> Result<Vec<EmbeddingInput>, String> {
    if items.is_empty() {
        return Err("Input cannot be empty".to_string());
    }

    if items.iter().all(Value::is_string) {
        return items
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let text = value.as_str().expect("checked string");
                validate_non_empty_text(text, "Input")?;
                if text.trim().is_empty() {
                    return Err(format!(
                        "Input at index {index} cannot be empty or whitespace only"
                    ));
                }
                Ok(EmbeddingInput::Text(text.to_string()))
            })
            .collect();
    }

    if items.iter().all(Value::is_u64) {
        return Ok(vec![EmbeddingInput::TokenIds(parse_token_ids(items)?)]);
    }

    if items.iter().all(Value::is_array) {
        return items
            .iter()
            .map(|value| {
                let token_ids = value.as_array().expect("checked array");
                Ok(EmbeddingInput::TokenIds(parse_token_ids(token_ids)?))
            })
            .collect();
    }

    Err("input list must contain only strings, integers, or token id arrays".to_string())
}

fn parse_token_ids(items: &[Value]) -> Result<Vec<u32>, String> {
    if items.is_empty() {
        return Err("Token ID input cannot be empty".to_string());
    }
    items
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let raw = value
                .as_u64()
                .ok_or_else(|| "All items in token ID input must be integers".to_string())?;
            u32::try_from(raw).map_err(|_| format!("Token ID at index {index} exceeds u32 range"))
        })
        .collect()
}

fn parse_dimensions(payload: &Value) -> Result<usize, String> {
    let Some(value) = payload.get("dimensions") else {
        return Ok(DEFAULT_EMBEDDING_DIMENSIONS);
    };
    let Some(raw) = value.as_u64() else {
        return Err("dimensions must be a positive integer".to_string());
    };
    let dimensions =
        usize::try_from(raw).map_err(|_| "dimensions exceeds usize range".to_string())?;
    if dimensions == 0 {
        return Err("dimensions must be at least 1".to_string());
    }
    if dimensions > MAX_EMBEDDING_DIMENSIONS {
        return Err(format!(
            "dimensions must not exceed {MAX_EMBEDDING_DIMENSIONS}"
        ));
    }
    Ok(dimensions)
}

fn validate_non_empty_text(text: &str, field: &'static str) -> Result<(), String> {
    if text.trim().is_empty() {
        return Err(format!("{field} cannot be empty or whitespace only"));
    }
    Ok(())
}

fn token_ids_to_embedding(token_ids: &[u32], dimensions: usize) -> Vec<f32> {
    let mut embedding = vec![0.0_f32; dimensions];
    for (position, token_id) in token_ids.iter().enumerate() {
        let hash = splitmix64(u64::from(*token_id) ^ ((position as u64) << 32));
        let index = (hash as usize) % dimensions;
        let sign = if (hash >> 63) == 0 { 1.0 } else { -1.0 };
        let magnitude = ((hash >> 16) as u32 % 997) as f32 / 997.0 + 0.001;
        embedding[index] += sign * magnitude;
    }

    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm > 0.0 {
        for value in &mut embedding {
            *value /= norm;
        }
    }
    embedding
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
