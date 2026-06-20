use serde_json::{Value, json};

use crate::openai_embedding::token_ids_to_embedding;
use crate::router::RouterRuntime;
use crate::tokenizer::Tokenizer;
use crate::worker::WorkerExecutor;

const SCORE_EMBEDDING_DIMENSIONS: usize = 8;

#[derive(Debug)]
pub(crate) struct OpenAiScoreRequest {
    model: String,
    query: ScoreInput,
    items: Vec<ScoreInput>,
    label_token_ids: Option<Vec<u32>>,
    apply_softmax: bool,
    item_first: bool,
}

#[derive(Debug)]
enum ScoreInput {
    Text(String),
    TokenIds(Vec<u32>),
}

pub(crate) fn parse_score_request(
    payload: &Value,
    served_model_name: &str,
) -> Result<OpenAiScoreRequest, String> {
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing model".to_string())?;
    if model != served_model_name {
        return Err(format!(
            "model {model} is not served by this worker ({served_model_name})"
        ));
    }

    if payload.get("query_embed_overrides").is_some()
        || payload.get("item_embed_overrides").is_some()
        || payload.get("embed_override_token_id").is_some()
    {
        return Err("embedding overrides are not supported by the Rust score worker".to_string());
    }
    if payload
        .get("return_pooled_hidden_states")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err(
            "return_pooled_hidden_states is not supported by the Rust score worker".to_string(),
        );
    }

    let items = parse_score_items(
        payload
            .get("items")
            .ok_or_else(|| "items must be provided".to_string())?,
    )?;
    let query = match payload.get("query") {
        Some(value) => parse_score_query(value)?,
        None => default_query_for_items(&items),
    };

    Ok(OpenAiScoreRequest {
        model: model.to_string(),
        query,
        items,
        label_token_ids: parse_optional_label_token_ids(payload)?,
        apply_softmax: optional_bool(payload, "apply_softmax")?.unwrap_or(false),
        item_first: optional_bool(payload, "item_first")?.unwrap_or(false),
    })
}

pub(crate) fn score_response_json<T, W>(
    runtime: &RouterRuntime<T, W>,
    request: &OpenAiScoreRequest,
) -> Value
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let query_token_ids = score_input_to_token_ids(runtime, &request.query);
    let mut prompt_tokens = 0usize;
    let mut scores = request
        .items
        .iter()
        .map(|item| {
            let item_token_ids = score_input_to_token_ids(runtime, item);
            let prompt_token_ids =
                compose_score_prompt(&query_token_ids, &item_token_ids, request.item_first);
            prompt_tokens += prompt_token_ids.len();
            match &request.label_token_ids {
                Some(label_token_ids) => {
                    let mut row = label_token_ids
                        .iter()
                        .map(|label_token_id| score_label_token(&prompt_token_ids, *label_token_id))
                        .collect::<Vec<_>>();
                    if request.apply_softmax {
                        row = softmax(&row);
                    }
                    row
                }
                None => vec![score_prompt_similarity(&query_token_ids, &item_token_ids)],
            }
        })
        .collect::<Vec<_>>();

    if request.apply_softmax && request.label_token_ids.is_none() {
        for row in &mut scores {
            row[0] = 1.0;
        }
    }

    json!({
        "object": "scoring",
        "model": request.model,
        "scores": scores,
        "pooled_hidden_states": Value::Null,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
            "completion_tokens": 0,
            "prompt_tokens_details": Value::Null,
            "reasoning_tokens": 0,
        }
    })
}

fn parse_score_query(value: &Value) -> Result<ScoreInput, String> {
    match value {
        Value::String(text) => Ok(ScoreInput::Text(text.clone())),
        Value::Array(items) => Ok(ScoreInput::TokenIds(parse_token_ids(items, "query")?)),
        _ => Err("query must be a string or token ID array".to_string()),
    }
}

fn parse_score_items(value: &Value) -> Result<Vec<ScoreInput>, String> {
    match value {
        Value::String(text) => Ok(vec![ScoreInput::Text(text.clone())]),
        Value::Array(items) if items.is_empty() => Ok(Vec::new()),
        Value::Array(items) if items.iter().all(Value::is_string) => items
            .iter()
            .map(|item| {
                Ok(ScoreInput::Text(
                    item.as_str().expect("checked string").to_string(),
                ))
            })
            .collect(),
        Value::Array(items) if items.iter().all(Value::is_array) => items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let token_ids = item.as_array().expect("checked array");
                Ok(ScoreInput::TokenIds(parse_token_ids(
                    token_ids,
                    format!("items[{index}]").as_str(),
                )?))
            })
            .collect(),
        Value::Array(_) => {
            Err("items must be a string, list of strings, or token ID batches".to_string())
        }
        _ => Err("items must be provided".to_string()),
    }
}

fn default_query_for_items(items: &[ScoreInput]) -> ScoreInput {
    if matches!(items.first(), Some(ScoreInput::TokenIds(_))) {
        ScoreInput::TokenIds(Vec::new())
    } else {
        ScoreInput::Text(String::new())
    }
}

fn parse_optional_label_token_ids(payload: &Value) -> Result<Option<Vec<u32>>, String> {
    let Some(value) = payload.get("label_token_ids") else {
        return Ok(None);
    };
    let Some(items) = value.as_array() else {
        return Err("label_token_ids must be an array of token IDs".to_string());
    };
    if items.is_empty() {
        return Err("label_token_ids cannot be empty".to_string());
    }
    parse_token_ids(items, "label_token_ids").map(Some)
}

fn parse_token_ids(items: &[Value], field: &str) -> Result<Vec<u32>, String> {
    items
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let raw = value
                .as_u64()
                .ok_or_else(|| format!("{field} must contain only token ID integers"))?;
            u32::try_from(raw).map_err(|_| format!("{field}[{index}] exceeds u32 range"))
        })
        .collect()
}

fn optional_bool(payload: &Value, field: &'static str) -> Result<Option<bool>, String> {
    let Some(value) = payload.get(field) else {
        return Ok(None);
    };
    value
        .as_bool()
        .ok_or_else(|| format!("{field} must be a boolean"))
        .map(Some)
}

fn score_input_to_token_ids<T, W>(runtime: &RouterRuntime<T, W>, input: &ScoreInput) -> Vec<u32>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    match input {
        ScoreInput::Text(text) => runtime.tokenize(text).token_ids,
        ScoreInput::TokenIds(token_ids) => token_ids.clone(),
    }
}

fn compose_score_prompt(query: &[u32], item: &[u32], item_first: bool) -> Vec<u32> {
    let mut prompt = Vec::with_capacity(query.len() + item.len());
    if item_first {
        prompt.extend_from_slice(item);
        prompt.extend_from_slice(query);
    } else {
        prompt.extend_from_slice(query);
        prompt.extend_from_slice(item);
    }
    prompt
}

fn score_label_token(prompt_token_ids: &[u32], label_token_id: u32) -> f64 {
    let prompt = token_ids_to_embedding(prompt_token_ids, SCORE_EMBEDDING_DIMENSIONS);
    let label = token_ids_to_embedding(&[label_token_id], SCORE_EMBEDDING_DIMENSIONS);
    prompt
        .iter()
        .zip(label.iter())
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum()
}

fn score_prompt_similarity(query_token_ids: &[u32], item_token_ids: &[u32]) -> f64 {
    if query_token_ids.is_empty() || item_token_ids.is_empty() {
        return 0.0;
    }
    let query = token_ids_to_embedding(query_token_ids, SCORE_EMBEDDING_DIMENSIONS);
    let item = token_ids_to_embedding(item_token_ids, SCORE_EMBEDDING_DIMENSIONS);
    query
        .iter()
        .zip(item.iter())
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum()
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp = logits
        .iter()
        .map(|logit| (*logit - max).exp())
        .collect::<Vec<_>>();
    let sum = exp.iter().sum::<f64>();
    if sum == 0.0 {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    exp.into_iter().map(|value| value / sum).collect()
}
