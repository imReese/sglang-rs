use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::openai_embedding::token_ids_to_embedding;
use crate::router::RouterRuntime;
use crate::tokenizer::Tokenizer;
use crate::worker::WorkerExecutor;

const DEFAULT_CLASS_COUNT: usize = 3;

#[derive(Debug)]
pub(crate) struct OpenAiClassifyRequest {
    model: String,
    inputs: Vec<ClassifyInput>,
}

#[derive(Debug)]
enum ClassifyInput {
    Text(String),
    TokenIds(Vec<u32>),
}

#[derive(Debug)]
struct ClassifyOutput {
    label: String,
    probs: Vec<f32>,
    prompt_tokens: usize,
    index: usize,
}

pub(crate) fn classify_response_json<T, W>(
    runtime: &RouterRuntime<T, W>,
    request: &OpenAiClassifyRequest,
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
                ClassifyInput::Text(text) => runtime.tokenize(text).token_ids,
                ClassifyInput::TokenIds(token_ids) => token_ids.clone(),
            };
            let logits = token_ids_to_embedding(&token_ids, DEFAULT_CLASS_COUNT);
            let probs = softmax(&logits);
            let predicted = probs
                .iter()
                .enumerate()
                .max_by(|(left_index, left), (right_index, right)| {
                    left.total_cmp(right)
                        .then_with(|| right_index.cmp(left_index))
                })
                .map(|(index, _)| index)
                .unwrap_or(0);
            ClassifyOutput {
                label: format!("LABEL_{predicted}"),
                probs,
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
        "id": format!("classify-{}", unix_timestamp_micros()),
        "object": "list",
        "created": unix_timestamp_secs(),
        "model": request.model,
        "data": outputs.into_iter().map(|output| json!({
            "index": output.index,
            "label": output.label,
            "probs": output.probs,
            "num_classes": DEFAULT_CLASS_COUNT,
        })).collect::<Vec<_>>(),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
            "completion_tokens": 0,
            "prompt_tokens_details": Value::Null,
        }
    })
}

pub(crate) fn parse_classify_request(
    payload: &Value,
    served_model_name: &str,
) -> Result<OpenAiClassifyRequest, String> {
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
    Ok(OpenAiClassifyRequest {
        model: model.to_string(),
        inputs: parse_classify_inputs(input)?,
    })
}

fn parse_classify_inputs(value: &Value) -> Result<Vec<ClassifyInput>, String> {
    match value {
        Value::String(text) => {
            validate_non_empty_text(text, "Input")?;
            Ok(vec![ClassifyInput::Text(text.clone())])
        }
        Value::Array(items) => parse_classify_input_array(items),
        _ => Err("input must be a string, list of strings, or token ids".to_string()),
    }
}

fn parse_classify_input_array(items: &[Value]) -> Result<Vec<ClassifyInput>, String> {
    if items.is_empty() {
        return Err("Input cannot be empty".to_string());
    }
    if items.iter().all(Value::is_string) {
        return items
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let text = value.as_str().expect("checked string");
                if text.trim().is_empty() {
                    return Err(format!(
                        "Input at index {index} cannot be empty or whitespace only"
                    ));
                }
                Ok(ClassifyInput::Text(text.to_string()))
            })
            .collect();
    }
    if items.iter().all(Value::is_u64) {
        return Ok(vec![ClassifyInput::TokenIds(parse_token_ids(items)?)]);
    }
    Err("input list must contain only strings or integers".to_string())
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

fn validate_non_empty_text(text: &str, field: &'static str) -> Result<(), String> {
    if text.trim().is_empty() {
        return Err(format!("{field} cannot be empty or whitespace only"));
    }
    Ok(())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = logits
        .iter()
        .map(|logit| (*logit - max).exp())
        .collect::<Vec<_>>();
    let sum = exp.iter().sum::<f32>();
    if sum == 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exp.into_iter().map(|value| value / sum).collect()
}

fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn unix_timestamp_micros() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros()
}
