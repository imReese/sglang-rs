use std::collections::BTreeSet;

use serde_json::{Value, json};

use crate::router::RouterRuntime;
use crate::tokenizer::Tokenizer;
use crate::worker::WorkerExecutor;

#[derive(Debug)]
pub(crate) struct OpenAiRerankRequest {
    query: String,
    documents: Vec<String>,
    top_k: Option<usize>,
    return_documents: bool,
}

#[derive(Debug)]
pub(crate) struct OpenAiRerankResult {
    score: f64,
    document: String,
    index: usize,
}

pub(crate) fn parse_rerank_request(
    payload: &Value,
    served_model_name: &str,
) -> Result<OpenAiRerankRequest, String> {
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing model".to_string())?;
    if model != served_model_name {
        return Err(format!(
            "model {model} is not served by this worker ({served_model_name})"
        ));
    }

    let mut query = rerank_content_to_text(
        payload
            .get("query")
            .ok_or_else(|| "missing query".to_string())?,
        "query",
    )?;
    if let Some(instruct) = payload.get("instruct").and_then(Value::as_str)
        && !instruct.trim().is_empty()
    {
        query = format!("{instruct}\n{query}");
    }
    if query.trim().is_empty() {
        return Err("Query cannot be empty or whitespace only".to_string());
    }

    let documents = payload
        .get("documents")
        .and_then(Value::as_array)
        .ok_or_else(|| "documents must be an array".to_string())?;
    if documents.is_empty() {
        return Err("Documents cannot be empty".to_string());
    }

    let mut parsed_documents = Vec::with_capacity(documents.len());
    for document in documents {
        let document = rerank_content_to_text(document, "documents")?;
        if document.trim().is_empty() {
            return Err("Each document cannot be empty or whitespace only".to_string());
        }
        parsed_documents.push(document);
    }

    Ok(OpenAiRerankRequest {
        query,
        documents: parsed_documents,
        top_k: optional_positive_usize(payload, "top_k")?
            .or(optional_positive_usize(payload, "top_n")?),
        return_documents: optional_bool(payload, "return_documents")?.unwrap_or(true),
    })
}

pub(crate) fn score_rerank_documents<T, W>(
    runtime: &RouterRuntime<T, W>,
    request: &OpenAiRerankRequest,
) -> Vec<OpenAiRerankResult>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let query_tokens = token_set(runtime, &request.query);
    let mut results = request
        .documents
        .iter()
        .enumerate()
        .map(|(index, document)| {
            let document_tokens = token_set(runtime, document);
            OpenAiRerankResult {
                score: jaccard_score(&query_tokens, &document_tokens),
                document: document.clone(),
                index,
            }
        })
        .collect::<Vec<_>>();
    results.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.index.cmp(&right.index))
    });
    results
}

pub(crate) fn truncate_rerank_results(
    request: &OpenAiRerankRequest,
    results: &mut Vec<OpenAiRerankResult>,
) {
    if let Some(top_k) = request.top_k {
        results.truncate(top_k.min(results.len()));
    }
}

pub(crate) fn rerank_results_to_json(
    request: &OpenAiRerankRequest,
    results: Vec<OpenAiRerankResult>,
) -> Value {
    Value::Array(
        results
            .into_iter()
            .map(|result| {
                if request.return_documents {
                    json!({
                        "score": result.score,
                        "document": result.document,
                        "index": result.index,
                    })
                } else {
                    json!({
                        "score": result.score,
                        "index": result.index,
                    })
                }
            })
            .collect(),
    )
}

fn rerank_content_to_text(value: &Value, field: &'static str) -> Result<String, String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut texts = Vec::new();
            for part in parts {
                match part {
                    Value::String(text) => texts.push(text.clone()),
                    Value::Object(object) => {
                        let content_type = object.get("type").and_then(Value::as_str);
                        if content_type.is_none_or(|kind| kind == "text")
                            && let Some(text) = object.get("text").and_then(Value::as_str)
                        {
                            texts.push(text.to_string());
                            continue;
                        }
                        return Err(format!(
                            "{field} only supports text content in the Rust rerank worker"
                        ));
                    }
                    _ => {
                        return Err(format!(
                            "{field} must be a string or an array of text content parts"
                        ));
                    }
                }
            }
            if texts.is_empty() {
                return Err(format!(
                    "{field} must include at least one text content part"
                ));
            }
            Ok(texts.join("\n"))
        }
        _ => Err(format!(
            "{field} must be a string or an array of text content parts"
        )),
    }
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

fn optional_positive_usize(payload: &Value, field: &'static str) -> Result<Option<usize>, String> {
    let Some(value) = payload.get(field) else {
        return Ok(None);
    };
    let Some(value) = value.as_u64() else {
        return Err(format!("{field} must be a positive integer"));
    };
    if value == 0 {
        return Err(format!("{field} must be at least 1"));
    }
    usize::try_from(value)
        .map(Some)
        .map_err(|_| format!("{field} exceeds usize range"))
}

fn token_set<T, W>(runtime: &RouterRuntime<T, W>, text: &str) -> BTreeSet<u32>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    runtime.tokenize(text).token_ids.into_iter().collect()
}

fn jaccard_score(query_tokens: &BTreeSet<u32>, document_tokens: &BTreeSet<u32>) -> f64 {
    if query_tokens.is_empty() || document_tokens.is_empty() {
        return 0.0;
    }
    let intersection = query_tokens.intersection(document_tokens).count();
    let union = query_tokens.union(document_tokens).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}
