//! LLM cleanup pass — GitHub Copilot or any OpenAI-compatible endpoint.
//!
//! Blocking reqwest call so we can run it inline on the transcribe thread. The
//! request/response handling, prompt resolution, normalization, and integrity
//! checks live in the typed [`crate::cleanup`] core; this file keeps the
//! existing `enhance` facade plus the GitHub Copilot readiness probe.

use crate::cleanup::{
    adapter, pipeline, EndpointScope, EnhanceOutcome, EnhanceProvider, FallbackReason,
    PromptTarget, SkipReason, Transcript,
};
use crate::credentials::read_openai_api_key;
use crate::settings::{EnhanceMode, Settings};
use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use std::io;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::warn;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CopilotAuthError {
    #[error("GitHub CLI is not installed or is not available on PATH")]
    CliMissing,
    #[error("GitHub CLI is not authenticated; run `gh auth login`")]
    NotAuthenticated,
    #[error("GitHub CLI token lookup failed: {0}")]
    CommandFailed(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CopilotReadiness {
    Ready,
    CliMissing,
    NotAuthenticated,
    Failed(String),
}

pub fn github_copilot_readiness() -> CopilotReadiness {
    match gh_token() {
        Ok(_) => CopilotReadiness::Ready,
        Err(CopilotAuthError::CliMissing) => CopilotReadiness::CliMissing,
        Err(CopilotAuthError::NotAuthenticated) => CopilotReadiness::NotAuthenticated,
        Err(error) => CopilotReadiness::Failed(error.to_string()),
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedCleanupRequest {
    pub provider: EnhanceProvider,
    pub url: String,
    pub body: serde_json::Value,
    /// The frozen prompt target this request was built from. Used to verify,
    /// just before send, that the current runtime provider/endpoint still
    /// matches so a just-in-time credential is never sent to a stale endpoint.
    pub target: PromptTarget,
}

/// Resolve provider/model/endpoint/prompt and build the exact adapter request
/// from a frozen settings snapshot. Credentials are intentionally absent.
pub(crate) fn prepare_cleanup_request(
    text: &str,
    settings: &Settings,
) -> std::result::Result<PreparedCleanupRequest, FallbackReason> {
    let cfg = &settings.enhance;
    let provider = EnhanceProvider::from_settings_str(&cfg.provider)
        .map_err(|_| FallbackReason::UnknownProvider)?;
    let model_id = pipeline::effective_model_id(&cfg.model);
    let target = match provider {
        EnhanceProvider::GithubCopilot => {
            PromptTarget::github_copilot(&model_id).map_err(|_| FallbackReason::EmptyModelId)?
        }
        EnhanceProvider::OpenAiCompatible => {
            let scope =
                EndpointScope::parse(&cfg.base_url).map_err(|_| FallbackReason::InvalidEndpoint)?;
            PromptTarget::openai_compatible(scope, &model_id)
                .map_err(|_| FallbackReason::EmptyModelId)?
        }
    };
    let resolved = settings.resolve_prompt(&target);
    let transcript = Transcript::new(text);
    Ok(PreparedCleanupRequest {
        provider,
        url: pipeline::chat_completions_url(&target),
        body: pipeline::build_request(&target, &resolved.system, &transcript),
        target,
    })
}

/// Whether two targets share the same credential binding: the same provider
/// and, for OpenAI-compatible providers, the same canonical endpoint scope.
///
/// The model/deployment ID is deliberately ignored — the API key is bound to
/// the endpoint, not the model, and the request body already carries the frozen
/// model. Credentials are never part of the comparison, so replacing the key on
/// an unchanged endpoint stays allowed (preserving just-in-time credentials).
pub(crate) fn credential_target_unchanged(frozen: &PromptTarget, current: &PromptTarget) -> bool {
    frozen.provider() == current.provider() && frozen.endpoint() == current.endpoint()
}

/// Decide whether a just-in-time credential may be sent to the job's `frozen`
/// target given the *current* runtime target.
///
/// GitHub Copilot's endpoint is fixed, so those jobs are never gated. For
/// OpenAI-compatible jobs the current provider and canonical endpoint scope
/// must still match the frozen ones; a mismatch — or an unavailable current
/// target (missing/invalid settings) — returns
/// [`FallbackReason::CredentialTargetChanged`] so no request is sent.
pub(crate) fn credential_binding_outcome(
    frozen: &PromptTarget,
    current: Option<&PromptTarget>,
) -> std::result::Result<(), FallbackReason> {
    if frozen.provider() != EnhanceProvider::OpenAiCompatible {
        return Ok(());
    }
    match current {
        Some(current) if credential_target_unchanged(frozen, current) => Ok(()),
        _ => Err(FallbackReason::CredentialTargetChanged),
    }
}

/// Load the current runtime provider/endpoint target, ignoring anything that
/// cannot be resolved into a valid target. Reads only the non-secret settings
/// document; it never touches the credential store.
fn current_credential_target() -> Option<PromptTarget> {
    Settings::load().ok().and_then(|s| s.prompt_target().ok())
}

/// Run cleanup using the immutable non-secret settings snapshot supplied by
/// the recording job. Provider credentials are deliberately resolved here,
/// immediately before execution, so credential changes do not require a new
/// recording while provider/model/endpoint/prompt changes do.
///
/// `before_validate` is called after the provider response has been parsed and
/// immediately before the integrity validator. Returning `false` tombstones
/// the attempt without producing an outcome.
pub fn enhance_outcome(
    text: &str,
    settings: &Settings,
    before_validate: impl FnOnce() -> bool,
) -> Option<EnhanceOutcome> {
    let cfg = &settings.enhance;
    if cfg.mode == EnhanceMode::Never {
        return Some(EnhanceOutcome::skipped(
            text,
            SkipReason::EnhancementDisabled,
        ));
    }
    if !text.chars().any(char::is_alphanumeric) {
        return Some(EnhanceOutcome::skipped(text, SkipReason::EmptyTranscript));
    }

    let prepared = match prepare_cleanup_request(text, settings) {
        Ok(prepared) => prepared,
        Err(reason) => {
            warn!(?reason, "cleanup request configuration is invalid");
            return Some(EnhanceOutcome::raw_fallback(text, reason));
        }
    };
    let provider = prepared.provider;

    // Secrets are the sole just-in-time setting: read them only when the job
    // actually reaches provider execution.
    let token = match provider {
        EnhanceProvider::GithubCopilot => match gh_token() {
            Ok(token) => token,
            Err(error) => {
                warn!(error = %error, "GitHub Copilot credential lookup failed");
                return Some(EnhanceOutcome::raw_fallback(
                    text,
                    FallbackReason::MissingCredential,
                ));
            }
        },
        EnhanceProvider::OpenAiCompatible => match read_openai_api_key() {
            Ok(Some(secret)) if !secret.trim().is_empty() => secret,
            Ok(_) => {
                warn!("OpenAI-compatible credential is not configured");
                return Some(EnhanceOutcome::raw_fallback(
                    text,
                    FallbackReason::MissingCredential,
                ));
            }
            Err(error) => {
                warn!(error = %error, "OpenAI-compatible credential lookup failed");
                return Some(EnhanceOutcome::raw_fallback(
                    text,
                    FallbackReason::MissingCredential,
                ));
            }
        },
    };

    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            warn!(error = %error, "cleanup HTTP client creation failed");
            return Some(EnhanceOutcome::raw_fallback(
                text,
                FallbackReason::RequestFailed,
            ));
        }
    };
    let mut request = client
        .post(&prepared.url)
        .bearer_auth(&token)
        .json(&prepared.body);
    for &(name, value) in provider.request_headers() {
        request = request.header(name, value);
    }

    // Race hardening: as late as possible — after the just-in-time credential
    // read and immediately before send — confirm the current runtime provider
    // and canonical endpoint still match this job's frozen target. Otherwise a
    // key that now belongs to a different endpoint could be sent to the stale
    // one. GitHub Copilot's fixed endpoint short-circuits inside the check.
    if let Err(reason) =
        credential_binding_outcome(&prepared.target, current_credential_target().as_ref())
    {
        warn!(
            ?reason,
            "cleanup target changed before send; no request was made"
        );
        return Some(EnhanceOutcome::raw_fallback(text, reason));
    }

    let response = match request.send() {
        Ok(response) => response,
        Err(error) => {
            warn!(error = %error, "cleanup provider request failed");
            return Some(EnhanceOutcome::raw_fallback(
                text,
                FallbackReason::RequestFailed,
            ));
        }
    };
    if !response.status().is_success() {
        warn!(status = %response.status(), "cleanup provider returned an error");
        return Some(EnhanceOutcome::raw_fallback(
            text,
            FallbackReason::RequestFailed,
        ));
    }
    let value: serde_json::Value = match response.json() {
        Ok(value) => value,
        Err(error) => {
            warn!(error = %error, "cleanup provider response was not valid JSON");
            return Some(EnhanceOutcome::raw_fallback(
                text,
                FallbackReason::ResponseUnparseable,
            ));
        }
    };
    let content = match adapter::parse_chat_response(&value) {
        Ok(content) => content,
        Err(error) => {
            warn!(error = %error, "cleanup provider response shape was unsupported");
            return Some(EnhanceOutcome::raw_fallback(
                text,
                FallbackReason::ResponseUnparseable,
            ));
        }
    };

    before_validate().then(|| pipeline::finalize(text, &content))
}

/// Compatibility facade retained for callers that still expect provider and
/// credential failures as `Err`. New worker code consumes [`EnhanceOutcome`]
/// directly.
pub fn enhance(text: &str, settings: &Settings) -> Result<String> {
    let outcome = enhance_outcome(text, settings, || true)
        .ok_or_else(|| anyhow!("enhancement was cancelled"))?;
    match outcome {
        EnhanceOutcome::RawFallback { reason, .. }
            if matches!(
                reason,
                FallbackReason::UnknownProvider
                    | FallbackReason::MissingCredential
                    | FallbackReason::CredentialTargetChanged
                    | FallbackReason::InvalidEndpoint
                    | FallbackReason::EmptyModelId
                    | FallbackReason::RequestFailed
                    | FallbackReason::ResponseUnparseable
            ) =>
        {
            Err(anyhow!(fallback_diagnostic(&reason)))
        }
        outcome => Ok(outcome.into_text()),
    }
}

fn fallback_diagnostic(reason: &FallbackReason) -> &'static str {
    match reason {
        FallbackReason::UnknownProvider => "unknown enhancement provider",
        FallbackReason::MissingCredential => "enhancement credential is not configured",
        FallbackReason::CredentialTargetChanged => {
            "enhancement provider or endpoint changed before send"
        }
        FallbackReason::InvalidEndpoint => "invalid OpenAI-compatible endpoint",
        FallbackReason::EmptyModelId => "enhancement model ID is empty",
        FallbackReason::RequestFailed => "enhancement provider request failed",
        FallbackReason::ResponseUnparseable => "enhancement provider response was invalid",
        FallbackReason::EmptyCandidate => "enhancement returned an empty candidate",
        FallbackReason::IntegrityRejected(_) => "enhancement failed integrity validation",
        FallbackReason::ValidatorError => "enhancement validator failed",
    }
}

fn gh_token() -> std::result::Result<String, CopilotAuthError> {
    // Cache the token for 10 minutes so we don't spawn `gh` on every call.
    static CACHE: Mutex<Option<(String, Instant)>> = Mutex::new(None);
    {
        let g = CACHE.lock();
        if let Some((tok, t)) = g.as_ref() {
            if t.elapsed() < Duration::from_secs(600) {
                return Ok(tok.clone());
            }
        }
    }
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let out = std::process::Command::new("gh")
        .args(["auth", "token"])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|error| match error.kind() {
            io::ErrorKind::NotFound => CopilotAuthError::CliMissing,
            _ => CopilotAuthError::CommandFailed(error.to_string()),
        })?;
    if !out.status.success() {
        warn!(status = ?out.status.code(), "`gh auth token` failed");
        return parse_gh_token_output(false, &out.stdout);
    }
    let s = parse_gh_token_output(true, &out.stdout)?;
    *CACHE.lock() = Some((s.clone(), Instant::now()));
    Ok(s)
}

fn parse_gh_token_output(
    success: bool,
    stdout: &[u8],
) -> std::result::Result<String, CopilotAuthError> {
    if !success {
        return Err(CopilotAuthError::NotAuthenticated);
    }
    let token = String::from_utf8_lossy(stdout).trim().to_string();
    if token.is_empty() {
        Err(CopilotAuthError::NotAuthenticated)
    } else {
        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        credential_binding_outcome, credential_target_unchanged, parse_gh_token_output,
        CopilotAuthError,
    };
    use crate::cleanup::{EndpointScope, FallbackReason, PromptTarget};

    fn openai_target(base_url: &str, model: &str) -> PromptTarget {
        let scope = EndpointScope::parse(base_url).unwrap();
        PromptTarget::openai_compatible(scope, model).unwrap()
    }

    #[test]
    fn unchanged_target_allows_the_send() {
        let frozen = openai_target("https://api.openai.com/v1", "gpt-5-mini");
        let current = openai_target("https://api.openai.com/v1", "gpt-5-mini");
        assert!(credential_target_unchanged(&frozen, &current));
        assert_eq!(credential_binding_outcome(&frozen, Some(&current)), Ok(()));
    }

    #[test]
    fn provider_change_blocks_the_send() {
        let frozen = openai_target("https://api.openai.com/v1", "gpt-5-mini");
        let current = PromptTarget::github_copilot("gpt-5-mini").unwrap();
        assert!(!credential_target_unchanged(&frozen, &current));
        assert_eq!(
            credential_binding_outcome(&frozen, Some(&current)),
            Err(FallbackReason::CredentialTargetChanged)
        );
    }

    #[test]
    fn endpoint_host_path_or_port_change_blocks_the_send() {
        let frozen = openai_target("https://api.openai.com/v1", "gpt-5-mini");
        for changed in [
            "https://other.example.com/v1",   // host
            "https://api.openai.com/v2",      // path
            "https://api.openai.com:8443/v1", // port
        ] {
            let current = openai_target(changed, "gpt-5-mini");
            assert!(
                !credential_target_unchanged(&frozen, &current),
                "expected {changed} to differ from the frozen endpoint"
            );
            assert_eq!(
                credential_binding_outcome(&frozen, Some(&current)),
                Err(FallbackReason::CredentialTargetChanged),
                "expected {changed} to block the send"
            );
        }
    }

    #[test]
    fn equivalent_canonical_endpoint_allows_the_send() {
        let frozen = openai_target("https://api.openai.com/v1", "gpt-5-mini");
        // Trailing slash, uppercase host, and explicit default port all
        // canonicalize to the frozen endpoint scope.
        let current = openai_target("HTTPS://API.OpenAI.com:443/v1/", "gpt-5-mini");
        assert!(credential_target_unchanged(&frozen, &current));
        assert_eq!(credential_binding_outcome(&frozen, Some(&current)), Ok(()));
    }

    #[test]
    fn missing_current_settings_blocks_the_send() {
        let frozen = openai_target("https://api.openai.com/v1", "gpt-5-mini");
        assert_eq!(
            credential_binding_outcome(&frozen, None),
            Err(FallbackReason::CredentialTargetChanged)
        );
    }

    #[test]
    fn replacing_the_credential_on_the_same_endpoint_stays_allowed() {
        // The credential value is never part of the binding, so a job frozen on
        // an endpoint keeps sending after the single stored key is replaced, as
        // long as the current provider/endpoint is unchanged. A model change on
        // the same endpoint is likewise allowed.
        let frozen = openai_target("https://api.openai.com/v1", "gpt-5-mini");
        let current = openai_target("https://api.openai.com/v1", "claude-haiku-4.5");
        assert!(credential_target_unchanged(&frozen, &current));
        assert_eq!(credential_binding_outcome(&frozen, Some(&current)), Ok(()));
    }

    #[test]
    fn copilot_target_is_never_gated() {
        let frozen = PromptTarget::github_copilot("gpt-5-mini").unwrap();
        // Even with no resolvable current target, Copilot's fixed endpoint is
        // always allowed to proceed.
        assert_eq!(credential_binding_outcome(&frozen, None), Ok(()));
    }

    #[test]
    fn classifies_github_cli_token_results_without_exposing_tokens() {
        assert_eq!(
            parse_gh_token_output(false, b"").unwrap_err(),
            CopilotAuthError::NotAuthenticated
        );
        assert_eq!(
            parse_gh_token_output(true, b" \r\n").unwrap_err(),
            CopilotAuthError::NotAuthenticated
        );
        assert_eq!(
            parse_gh_token_output(true, b"secret-token\r\n").unwrap(),
            "secret-token"
        );
    }
}
