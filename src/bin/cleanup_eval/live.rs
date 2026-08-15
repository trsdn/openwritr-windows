//! Opt-in live-provider mode.
//!
//! Never required by tests or releases, and never runs unless `--live` is
//! passed explicitly. Sends each corpus case's source transcript through the
//! real configured provider (GitHub Copilot or an OpenAI-compatible
//! endpoint), reusing the exact same production prompt resolution, request
//! construction, response parsing, and integrity pipeline that offline mode
//! exercises with simulated responses (via [`crate::eval::evaluate_case`]).
//! Network failures never abort the run; they are tallied as a failure
//! category.
//!
//! Configuration precedence: `--provider`/`--model`/`--base-url` CLI flags,
//! then the user's real `settings.json` (read-only, generic field
//! extraction — this module does not depend on `crate::settings`), then
//! compiled defaults. Credentials are read from the same secure storage the
//! shipped app uses (Windows Credential Manager for OpenAI-compatible keys,
//! `gh auth token` for Copilot) and are never printed or serialized.

use std::collections::BTreeMap;
use std::io;
use std::time::{Duration, Instant};

use crate::cleanup::{self, EndpointScope, EnhanceProvider, PromptTarget, Transcript};
use crate::corpus::CorpusCase;
use crate::report::{LatencySummary, LiveConfig, LiveSummary};

/// Non-secret desired live configuration, resolved from CLI flags / on-disk
/// settings / defaults before any credential lookup happens.
pub struct LiveConfigInput {
    pub provider: EnhanceProvider,
    pub model: String,
    pub base_url: String,
}

impl Default for LiveConfigInput {
    fn default() -> Self {
        LiveConfigInput {
            provider: EnhanceProvider::GithubCopilot,
            model: cleanup::pipeline::DEFAULT_MODEL.to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }
}

/// Read `enhance.provider` / `enhance.model` / `enhance.base_url` directly
/// out of the real `settings.json` as generic JSON, without depending on
/// `crate::settings` (its validation/migration/prompt-override machinery is
/// out of scope for this evaluator). Missing file or fields fall back to
/// `defaults` field-by-field.
pub fn config_from_disk_or_default(defaults: LiveConfigInput) -> LiveConfigInput {
    let path = crate::paths::settings_path();
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return defaults;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return defaults;
    };
    let enhance = value.get("enhance");
    let provider = enhance
        .and_then(|e| e.get("provider"))
        .and_then(|v| v.as_str())
        .and_then(|s| EnhanceProvider::from_settings_str(s).ok())
        .unwrap_or(defaults.provider);
    let model = enhance
        .and_then(|e| e.get("model"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .map(str::to_string)
        .unwrap_or(defaults.model);
    let base_url = enhance
        .and_then(|e| e.get("base_url"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .map(str::to_string)
        .unwrap_or(defaults.base_url);
    LiveConfigInput {
        provider,
        model,
        base_url,
    }
}

#[derive(Debug)]
pub enum LiveSetupError {
    InvalidEndpoint(String),
    MissingCredential,
    CredentialLookupFailed(String),
    GithubCliUnavailable,
    GithubCliNotAuthenticated,
    HttpClientBuildFailed(String),
}

impl std::fmt::Display for LiveSetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiveSetupError::InvalidEndpoint(msg) => write!(f, "invalid endpoint: {msg}"),
            LiveSetupError::MissingCredential => write!(f, "no credential configured"),
            LiveSetupError::CredentialLookupFailed(msg) => {
                write!(f, "credential lookup failed: {msg}")
            }
            LiveSetupError::GithubCliUnavailable => {
                write!(f, "GitHub CLI (`gh`) is not installed or not on PATH")
            }
            LiveSetupError::GithubCliNotAuthenticated => {
                write!(f, "GitHub CLI is not authenticated; run `gh auth login`")
            }
            LiveSetupError::HttpClientBuildFailed(msg) => {
                write!(f, "failed to build HTTP client: {msg}")
            }
        }
    }
}

struct Resolved {
    target: PromptTarget,
    provider: EnhanceProvider,
    token: String,
    endpoint_scope: Option<String>,
    credential_source: &'static str,
}

fn resolve(input: &LiveConfigInput) -> Result<Resolved, LiveSetupError> {
    match input.provider {
        EnhanceProvider::GithubCopilot => {
            let target = PromptTarget::github_copilot(&input.model)
                .map_err(|error| LiveSetupError::InvalidEndpoint(error.to_string()))?;
            let token = gh_auth_token()?;
            Ok(Resolved {
                target,
                provider: EnhanceProvider::GithubCopilot,
                token,
                endpoint_scope: None,
                credential_source: "gh auth token",
            })
        }
        EnhanceProvider::OpenAiCompatible => {
            let scope = EndpointScope::parse(&input.base_url)
                .map_err(|error| LiveSetupError::InvalidEndpoint(error.to_string()))?;
            let target = PromptTarget::openai_compatible(scope.clone(), &input.model)
                .map_err(|error| LiveSetupError::InvalidEndpoint(error.to_string()))?;
            let token = crate::credentials::read_openai_api_key()
                .map_err(|error| LiveSetupError::CredentialLookupFailed(error.to_string()))?
                .filter(|secret| !secret.trim().is_empty())
                .ok_or(LiveSetupError::MissingCredential)?;
            Ok(Resolved {
                target,
                provider: EnhanceProvider::OpenAiCompatible,
                token,
                endpoint_scope: Some(scope.base_url()),
                credential_source: "Windows Credential Manager",
            })
        }
    }
}

/// Shells out to `gh auth token`. Deliberately self-contained (rather than
/// reusing `enhance.rs`'s private helper) so this evaluator does not pull in
/// `crate::settings` — out of scope per this task.
fn gh_auth_token() -> Result<String, LiveSetupError> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let output = std::process::Command::new("gh")
        .args(["auth", "token"])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|error| match error.kind() {
            io::ErrorKind::NotFound => LiveSetupError::GithubCliUnavailable,
            _ => LiveSetupError::CredentialLookupFailed(error.to_string()),
        })?;
    if !output.status.success() {
        return Err(LiveSetupError::GithubCliNotAuthenticated);
    }
    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if token.is_empty() {
        Err(LiveSetupError::GithubCliNotAuthenticated)
    } else {
        Ok(token)
    }
}

/// Run every eligible corpus case's source transcript through the real
/// configured provider. Always returns a summary — individual request
/// failures are tallied, never propagated as a hard error. Only credential
/// or configuration setup failures (fixed before any request is sent) return
/// `Err`.
pub fn run_live(
    cases: &[&CorpusCase],
    input: &LiveConfigInput,
) -> Result<LiveSummary, LiveSetupError> {
    let resolved = resolve(input)?;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|error| LiveSetupError::HttpClientBuildFailed(error.to_string()))?;

    let mut attempted = 0usize;
    let mut succeeded = 0usize;
    let mut failed = 0usize;
    let mut decision_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut failure_categories: BTreeMap<String, usize> = BTreeMap::new();
    let mut latencies: Vec<Duration> = Vec::new();

    let resolved_prompt = cleanup::resolve_prompt(&resolved.target, &cleanup::NoOverrides);

    for case in cases.iter().filter(|case| case.eligible_for_live_probe()) {
        let source = case.resolved_source();
        if source.trim().is_empty() {
            continue;
        }
        attempted += 1;

        let transcript = Transcript::new(&source);
        let body = cleanup::pipeline::build_request(
            &resolved.target,
            &resolved_prompt.system,
            &transcript,
        );
        let url = cleanup::pipeline::chat_completions_url(&resolved.target);

        let started = Instant::now();
        let mut request = client.post(&url).bearer_auth(&resolved.token).json(&body);
        for &(name, value) in resolved.provider.request_headers() {
            request = request.header(name, value);
        }

        match request.send() {
            Ok(response) if response.status().is_success() => {
                latencies.push(started.elapsed());
                match response.json::<serde_json::Value>() {
                    Ok(value) => {
                        succeeded += 1;
                        let outcome = crate::eval::evaluate_case(&source, &value);
                        *decision_counts
                            .entry(outcome.tag().to_string())
                            .or_insert(0) += 1;
                    }
                    Err(_) => {
                        failed += 1;
                        *failure_categories
                            .entry("ResponseUnparseable".to_string())
                            .or_insert(0) += 1;
                    }
                }
            }
            Ok(response) => {
                latencies.push(started.elapsed());
                failed += 1;
                *failure_categories
                    .entry(format!("HttpStatus{}", response.status().as_u16()))
                    .or_insert(0) += 1;
            }
            Err(_) => {
                failed += 1;
                *failure_categories
                    .entry("RequestFailed".to_string())
                    .or_insert(0) += 1;
            }
        }
    }

    let latency_ms = summarize_latency(&latencies);

    Ok(LiveSummary {
        config: LiveConfig {
            provider: resolved.provider.as_str().to_string(),
            model: input.model.clone(),
            endpoint_scope: resolved.endpoint_scope,
            credential_source: resolved.credential_source,
        },
        attempted,
        succeeded,
        failed,
        decision_counts,
        failure_categories,
        latency_ms,
    })
}

fn summarize_latency(latencies: &[Duration]) -> Option<LatencySummary> {
    if latencies.is_empty() {
        return None;
    }
    let millis: Vec<f64> = latencies
        .iter()
        .map(Duration::as_secs_f64)
        .map(|s| s * 1000.0)
        .collect();
    let total: f64 = millis.iter().sum();
    let min = millis.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = millis.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Some(LatencySummary {
        min_ms: min,
        max_ms: max,
        mean_ms: total / millis.len() as f64,
        total_ms: total,
        samples: millis.len(),
    })
}
