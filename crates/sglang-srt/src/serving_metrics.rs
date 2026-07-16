use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::router::{RouterGenerateResponse, RouterGenerateResponseBody};

#[derive(Default)]
pub(crate) struct ServingMetrics {
    requests_total: AtomicU64,
    requests_failed_total: AtomicU64,
    requests_in_flight: AtomicU64,
    prompt_tokens_total: AtomicU64,
    generation_tokens_total: AtomicU64,
    cached_tokens_total: AtomicU64,
    time_to_first_token_micros_total: AtomicU64,
    time_to_first_token_count: AtomicU64,
    request_duration_micros_total: AtomicU64,
    request_duration_count: AtomicU64,
}

impl ServingMetrics {
    pub(crate) fn render(&self) -> String {
        let seconds = |micros: u64| micros as f64 / 1_000_000.0;
        format!(
            concat!(
                "# TYPE sglang_requests_total counter\nsglang_requests_total {}\n",
                "# TYPE sglang_requests_failed_total counter\nsglang_requests_failed_total {}\n",
                "# TYPE sglang_requests_in_flight gauge\nsglang_requests_in_flight {}\n",
                "# TYPE sglang_prompt_tokens_total counter\nsglang_prompt_tokens_total {}\n",
                "# TYPE sglang_generation_tokens_total counter\nsglang_generation_tokens_total {}\n",
                "# TYPE sglang_cached_tokens_total counter\nsglang_cached_tokens_total {}\n",
                "# TYPE sglang_time_to_first_token_seconds summary\n",
                "sglang_time_to_first_token_seconds_sum {}\n",
                "sglang_time_to_first_token_seconds_count {}\n",
                "# TYPE sglang_request_duration_seconds summary\n",
                "sglang_request_duration_seconds_sum {}\n",
                "sglang_request_duration_seconds_count {}\n"
            ),
            self.requests_total.load(Ordering::Relaxed),
            self.requests_failed_total.load(Ordering::Relaxed),
            self.requests_in_flight.load(Ordering::Relaxed),
            self.prompt_tokens_total.load(Ordering::Relaxed),
            self.generation_tokens_total.load(Ordering::Relaxed),
            self.cached_tokens_total.load(Ordering::Relaxed),
            seconds(
                self.time_to_first_token_micros_total
                    .load(Ordering::Relaxed)
            ),
            self.time_to_first_token_count.load(Ordering::Relaxed),
            seconds(self.request_duration_micros_total.load(Ordering::Relaxed)),
            self.request_duration_count.load(Ordering::Relaxed),
        )
    }
}

pub(crate) struct RequestObservation {
    metrics: Arc<ServingMetrics>,
    started: Instant,
    first_output_observed: bool,
    finished: bool,
}

impl RequestObservation {
    pub(crate) fn new(metrics: Arc<ServingMetrics>) -> Self {
        metrics.requests_total.fetch_add(1, Ordering::Relaxed);
        metrics.requests_in_flight.fetch_add(1, Ordering::Relaxed);
        Self {
            metrics,
            started: Instant::now(),
            first_output_observed: false,
            finished: false,
        }
    }

    pub(crate) fn observe(&mut self, response: &RouterGenerateResponse) {
        if !self.first_output_observed {
            self.first_output_observed = true;
            self.metrics
                .time_to_first_token_micros_total
                .fetch_add(elapsed_micros(self.started), Ordering::Relaxed);
            self.metrics
                .time_to_first_token_count
                .fetch_add(1, Ordering::Relaxed);
        }
        if let RouterGenerateResponseBody::Complete(complete) = &response.body {
            self.metrics
                .prompt_tokens_total
                .fetch_add(complete.prompt_tokens.max(0) as u64, Ordering::Relaxed);
            self.metrics
                .generation_tokens_total
                .fetch_add(complete.completion_tokens.max(0) as u64, Ordering::Relaxed);
            self.metrics
                .cached_tokens_total
                .fetch_add(complete.cached_tokens.max(0) as u64, Ordering::Relaxed);
        }
    }

    pub(crate) fn finish(&mut self, success: bool) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.metrics
            .requests_in_flight
            .fetch_sub(1, Ordering::Relaxed);
        if !success {
            self.metrics
                .requests_failed_total
                .fetch_add(1, Ordering::Relaxed);
        }
        self.metrics
            .request_duration_micros_total
            .fetch_add(elapsed_micros(self.started), Ordering::Relaxed);
        self.metrics
            .request_duration_count
            .fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for RequestObservation {
    fn drop(&mut self) {
        self.finish(false);
    }
}

pub(crate) fn observe_response(
    observation: &mut Option<RequestObservation>,
    response: &RouterGenerateResponse,
) {
    if let Some(observation) = observation {
        observation.observe(response);
    }
}

pub(crate) fn observe_optional_responses(
    observation: &mut Option<RequestObservation>,
    responses: &[RouterGenerateResponse],
) {
    for response in responses {
        observe_response(observation, response);
    }
}

pub(crate) fn observe_responses(
    observation: &mut RequestObservation,
    responses: &[RouterGenerateResponse],
) {
    for response in responses {
        observation.observe(response);
    }
}

pub(crate) fn finish_observation(observation: &mut Option<RequestObservation>, success: bool) {
    if let Some(observation) = observation {
        observation.finish(success);
    }
}

fn elapsed_micros(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}
