// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, bail, Context, Result};
use clap::{ArgAction, Parser, Subcommand};
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, DiscoveryConfig, LogFormat, ModelConfig,
    ObservabilityConfig, PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal::unix::{signal, Signal, SignalKind};

#[derive(Parser, Debug)]
#[command(
    name = "sgl-router",
    alias = "sglang-router",
    alias = "smg",
    version,
    args_conflicts_with_subcommands = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[command(flatten)]
    router_args: RouterArgs,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Launch the router, matching sgl-model-gateway's primary command shape.
    #[command(visible_alias = "start")]
    Launch {
        #[command(flatten)]
        args: RouterArgs,
    },
}

#[derive(Parser, Debug, Clone)]
struct RouterArgs {
    #[arg(long, env = "SGL_ROUTER_CONFIG")]
    config: Option<PathBuf>,

    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    #[arg(long, default_value_t = 30000)]
    port: u16,

    /// Regular-mode worker URLs, matching sgl-model-gateway's `--worker-urls`.
    #[arg(long, num_args = 0..)]
    worker_urls: Vec<String>,

    /// Load-balancing policy name accepted by sgl-model-gateway.
    #[arg(
        long,
        default_value = "round_robin",
        value_parser = ["random", "round_robin", "cache_aware", "cache_aware_zmq", "power_of_two"]
    )]
    policy: String,

    /// Enable PD (prefill/decode) routing mode.
    #[arg(long, default_value_t = false)]
    pd_disaggregation: bool,

    /// Prefill worker URL, optionally followed by a bootstrap port.
    #[arg(long, num_args = 1..=2, action = ArgAction::Append)]
    prefill: Vec<String>,

    /// Decode worker URL. Can be specified multiple times.
    #[arg(long, action = ArgAction::Append)]
    decode: Vec<String>,

    /// Optional static model id for config-file parity.
    #[arg(long)]
    model_id: Option<String>,

    /// sgl-model-gateway-compatible alias for `--model-id`.
    #[arg(long)]
    served_model_name: Option<String>,

    /// Optional tokenizer path for a statically configured model.
    #[arg(long)]
    tokenizer_path: Option<String>,

    #[arg(long, default_value = "info")]
    log_level: String,
}

impl Cli {
    fn into_config(self) -> Result<Config> {
        match self.command {
            Some(Commands::Launch { args }) => args.into_config(),
            None => self.router_args.into_config(),
        }
    }
}

impl RouterArgs {
    fn into_config(self) -> Result<Config> {
        if let Some(config_path) = self.config {
            return Config::from_path(&config_path)
                .with_context(|| format!("load config from {}", config_path.display()));
        }

        let urls = if self.pd_disaggregation {
            let prefill_urls = extract_url_values(&self.prefill);
            let decode_urls = extract_url_values(&self.decode);
            if prefill_urls.is_empty() || decode_urls.is_empty() {
                bail!(
                    "direct PD launch requires at least one --prefill URL and at least one --decode URL"
                );
            }
            let mut urls = prefill_urls;
            urls.extend(decode_urls);
            urls
        } else {
            extract_url_values(&self.worker_urls)
        };
        let urls = validate_and_normalize_urls(urls)?;
        if urls.is_empty() {
            bail!(
                "direct launch requires --worker-urls, or --pd-disaggregation with --prefill and --decode"
            );
        }

        let models = self.static_models()?;
        Ok(Config {
            server: ServerConfig {
                host: self.host,
                port: self.port,
            },
            observability: ObservabilityConfig {
                log_level: self.log_level,
                log_format: LogFormat::Text,
            },
            models,
            discovery: DiscoveryConfig {
                backend: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig { urls }),
            },
            proxy: ProxyConfig::default(),
            active_load: ActiveLoadConfig::default(),
        })
    }

    fn static_models(&self) -> Result<Vec<ModelConfig>> {
        let model_id = self
            .served_model_name
            .as_ref()
            .or(self.model_id.as_ref())
            .map(|id| id.trim())
            .filter(|id| !id.is_empty());
        let tokenizer_path = self
            .tokenizer_path
            .as_ref()
            .map(|path| path.trim())
            .filter(|path| !path.is_empty());
        match (model_id, tokenizer_path) {
            (None, None) => Ok(Vec::new()),
            (Some(_), None) => bail!(
                "direct launch with --model-id/--served-model-name also requires --tokenizer-path"
            ),
            (None, Some(_)) => bail!(
                "direct launch with --tokenizer-path also requires --model-id or --served-model-name"
            ),
            (Some(id), Some(path)) => Ok(vec![ModelConfig {
                id: id.to_string(),
                tokenizer_path: path.to_string(),
                policy: policy_kind_from_smg_name(&self.policy)?,
                circuit_breaker: None,
                cache_aware: None,
            }]),
        }
    }
}

fn extract_url_values(values: &[String]) -> Vec<String> {
    values
        .iter()
        .filter(|value| {
            let trimmed = value.trim();
            trimmed.starts_with("http://")
                || trimmed.starts_with("https://")
                || trimmed.starts_with("grpc://")
                || trimmed.starts_with("grpcs://")
        })
        .cloned()
        .collect()
}

fn validate_and_normalize_urls(values: Vec<String>) -> Result<Vec<String>> {
    let mut urls = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for raw in values {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parsed = url::Url::parse(trimmed)
            .map_err(|e| anyhow!("worker URL {raw:?} is not a valid URL: {e}"))?;
        match parsed.scheme() {
            "http" | "https" | "grpc" | "grpcs" => {}
            other => bail!(
                "worker URL {raw:?} has unsupported scheme {other:?}; only http, https, grpc, and grpcs are supported"
            ),
        }
        let normalized = parsed.as_str().trim_end_matches('/').to_string();
        if seen.insert(normalized.clone()) {
            urls.push(normalized);
        }
    }
    Ok(urls)
}

fn policy_kind_from_smg_name(name: &str) -> Result<PolicyKind> {
    match name {
        "round_robin" => Ok(PolicyKind::RoundRobin),
        "random" => Ok(PolicyKind::Random),
        "power_of_two" => Ok(PolicyKind::PowerOfTwo),
        "cache_aware" | "cache_aware_zmq" => Ok(PolicyKind::CacheAwareZmq),
        other => bail!("unsupported router policy {other:?}"),
    }
}

/// Install the global tracing subscriber.
///
/// Idempotent: a second call returns `Ok` without panicking. When
/// `try_init` errors, some other code has already installed a subscriber,
/// so the `tracing::debug!` below is delivered through THAT subscriber —
/// no recursive init.
///
/// `format` selects the output shape: `Json` emits one JSON record per
/// line (target for production / k8s log aggregators), `Text` is the
/// human-readable default. The `RUST_LOG` environment variable always
/// wins over `default_level`.
fn init_tracing(default_level: &str, format: LogFormat) -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));
    let install_result = match format {
        LogFormat::Json => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .json()
            .try_init(),
        LogFormat::Text => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .try_init(),
    };
    if let Err(e) = install_result {
        // A second install attempt; the existing subscriber is fine.
        // Surface the attempted default level so an operator can see
        // what we tried.
        tracing::debug!(
            default_level = %default_level,
            ?format,
            error = %e,
            "tracing subscriber already installed; continuing"
        );
    }
    Ok(())
}

/// Install a minimal text-format subscriber BEFORE config parsing so a
/// config-load error has somewhere to surface. The real subscriber
/// (driven by `Config.observability`) is installed after; the second
/// `try_init` is a no-op because a subscriber is already present.
/// The bootstrap subscriber respects `RUST_LOG` so an operator can
/// debug startup with `RUST_LOG=debug` even when the config file is
/// missing or malformed.
fn install_bootstrap_subscriber() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}

/// Install SIGTERM and SIGINT handlers up front so a failure here surfaces
/// before `axum::serve` starts. If installation fails (rare: container
/// without signal capability, seccomp policy), we return an error and the
/// process exits cleanly rather than running deaf to k8s termination.
fn install_signal_handlers() -> Result<(Signal, Signal)> {
    let sigterm = signal(SignalKind::terminate()).context("install SIGTERM handler")?;
    let sigint = signal(SignalKind::interrupt()).context("install SIGINT handler")?;
    Ok((sigterm, sigint))
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    // Bootstrap subscriber so a Config::from_path error has structured
    // output. The configured-format subscriber installs after this and
    // becomes a no-op via try_init's idempotency.
    install_bootstrap_subscriber();
    let cfg = cli.into_config()?;

    init_tracing(&cfg.observability.log_level, cfg.observability.log_format)?;

    tracing::info!(
        "sgl-router {} starting on {}:{}",
        env!("CARGO_PKG_VERSION"),
        cfg.server.host,
        cfg.server.port
    );

    let tokenizers = Arc::new(
        sgl_router::tokenizer::TokenizerRegistry::load_from_config(&cfg)
            .context("load tokenizers")?,
    );

    let registry = Arc::new(sgl_router::workers::WorkerRegistry::default());

    // Build the KV-event index up front so the cache-aware-zmq policy can
    // share its `HashTree` handle + `BlockSizeOracle`. When no model uses
    // `cache_aware_zmq`, the index is still constructed (cheap) but no
    // subscribers are ever added.
    let block_size_oracle = sgl_router::policies::kv_events::BlockSizeOracle::new();
    let kv_index = sgl_router::policies::kv_events::KvEventIndex::new_with_http_and_oracle(
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .expect("default http client builds"),
        Arc::clone(&block_size_oracle),
    );
    let policies = Arc::new(
        sgl_router::policies::factory::build_registry(
            &cfg,
            kv_index.tree(),
            Arc::clone(&tokenizers),
            Arc::clone(&block_size_oracle),
        )
        .context("build policy registry")?,
    );

    // Shared ActiveLoadRegistry + janitor task. The janitor reaps
    // request entries whose lifetime exceeded `stale_request_timeout`,
    // so a leaked guard (proxy task panic, etc.) does not inflate a
    // worker's load forever. The registry is built BEFORE the manager
    // is spawned so the manager can call `forget_worker` on
    // `DiscoveryEvent::Removed`.
    let stale_timeout = std::time::Duration::from_secs(cfg.active_load.stale_request_timeout_secs);
    let active_load = sgl_router::policies::active_load::ActiveLoadRegistry::new(
        Arc::new(sgl_router::policies::active_load::SystemTimeClock),
        stale_timeout,
    );
    // Sweep cadence is 1/10 of the configured timeout, clamped to
    // [1 s, 60 s]. A short timeout (test setting) needs frequent
    // sweeps to fire within the test's window; a long timeout
    // (production) doesn't need sub-minute checks.
    let sweep_interval = std::time::Duration::from_secs(
        (cfg.active_load.stale_request_timeout_secs / 10).clamp(1, 60),
    );
    let janitor_handle =
        sgl_router::policies::active_load::spawn_janitor(Arc::clone(&active_load), sweep_interval);

    // Spawn discovery + manager tasks.
    let (event_rx, discovery_handle) = sgl_router::discovery::spawn_discovery(&cfg)
        .await
        .context("spawn discovery")?;
    let kv_index_opt: Option<Arc<sgl_router::policies::kv_events::KvEventIndex>> =
        Some(Arc::clone(&kv_index));
    let manager_handle = tokio::spawn(sgl_router::workers::manager::run_with_config(
        event_rx,
        registry.clone(),
        Some(Arc::new(cfg.clone())),
        kv_index_opt,
        Some(Arc::clone(&active_load)),
    ));

    let proxy = Arc::new(
        sgl_router::proxy::Proxy::new(std::time::Duration::from_secs(
            cfg.proxy.request_timeout_secs,
        ))
        .context("build proxy client")?,
    );

    let ctx = Arc::new(
        sgl_router::server::app_context::AppContext::with_active_load(
            cfg.clone(),
            tokenizers,
            proxy,
            registry,
            policies,
            active_load,
        ),
    );
    ctx.mark_ready();

    let app = sgl_router::server::app::build_router(ctx.clone());

    let bind = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("bind {bind}"))?;
    tracing::info!("listening on {bind}");

    let (sigterm, sigint) = install_signal_handlers()?;

    let serve = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal(sigterm, sigint));
    let server_result = serve.await.context("axum serve");

    // Best-effort: cancel discovery + manager + janitor on shutdown.
    // The janitor handle's drop signals cancellation; we additionally
    // await `shutdown` so the task joins cleanly before the process
    // exits — useful for tracing tail logs.
    discovery_handle.abort();
    manager_handle.abort();
    janitor_handle.shutdown().await;
    server_result
}

async fn shutdown_signal(mut sigterm: Signal, mut sigint: Signal) {
    tokio::select! {
        _ = sigterm.recv() => tracing::info!("got SIGTERM, shutting down"),
        _ = sigint.recv()  => tracing::info!("got SIGINT, shutting down"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn install_signal_handlers_returns_both() {
        // Pins the contract that handler installation works on a standard
        // tokio runtime. If this fails on a sandboxed runner, the real
        // service would also fail to install — which is the point.
        assert!(install_signal_handlers().is_ok());
    }

    #[test]
    fn init_tracing_is_idempotent() {
        let _ = init_tracing("info", LogFormat::Text);
        let _ = init_tracing("info", LogFormat::Text);
    }

    #[test]
    fn init_tracing_accepts_json_format() {
        // Doesn't matter whether we win or lose the race against another
        // subscriber install — the function must return Ok either way.
        assert!(init_tracing("info", LogFormat::Json).is_ok());
    }

    #[test]
    fn launch_cli_accepts_sgl_model_gateway_pd_args() {
        let cli = Cli::try_parse_from([
            "sgl-router",
            "launch",
            "--host",
            "0.0.0.0",
            "--port",
            "30000",
            "--pd-disaggregation",
            "--prefill",
            "http://127.0.0.1:30001",
            "8200",
            "--decode",
            "http://127.0.0.1:30002",
            "--policy",
            "cache_aware",
        ])
        .expect("SMG-compatible PD launch args should parse");

        let cfg = cli
            .into_config()
            .expect("SMG-compatible PD launch args should build router config");

        assert_eq!(cfg.server.host, "0.0.0.0");
        assert_eq!(cfg.server.port, 30000);
        match cfg.discovery.backend {
            DiscoveryBackend::StaticUrls(static_urls) => {
                assert_eq!(
                    static_urls.urls,
                    vec![
                        "http://127.0.0.1:30001".to_string(),
                        "http://127.0.0.1:30002".to_string(),
                    ]
                );
            }
            other => panic!("expected static_urls discovery, got {other:?}"),
        }
    }

    #[test]
    fn launch_cli_rejects_partial_pd_pools() {
        let cli = Cli::try_parse_from([
            "sgl-router",
            "launch",
            "--pd-disaggregation",
            "--prefill",
            "http://127.0.0.1:30001",
            "8200",
        ])
        .expect("CLI syntax is valid; semantic validation belongs to into_config");

        let err = cli
            .into_config()
            .expect_err("PD direct launch must require both prefill and decode pools");
        assert!(
            err.to_string().contains("--prefill") && err.to_string().contains("--decode"),
            "unexpected error: {err}"
        );
    }
}
