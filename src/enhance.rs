//! LLM cleanup pass — GitHub Copilot or any OpenAI-compatible endpoint.
//!
//! Blocking reqwest call so we can run it inline on the transcribe thread.

use crate::credentials::read_openai_api_key;
use crate::settings::{EnhanceMode, Settings};
use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use serde_json::json;
use std::io;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::warn;

const SYSTEM: &str = "You are a transcription cleanup assistant. Fix \
punctuation, casing, filler words ('um', 'uh', 'like'), and obvious \
recognition errors in the user message. Preserve the original meaning, \
language, and tone. Return ONLY the cleaned text — no preamble, no \
quotes, no commentary.";

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

pub fn enhance(text: &str, settings: &Settings) -> Result<String> {
    let cfg = &settings.enhance;
    if cfg.mode == EnhanceMode::Never || text.trim().is_empty() {
        return Ok(text.to_string());
    }
    let (url, token) = match cfg.provider.as_str() {
        "github_copilot" => {
            let token = gh_token()?;
            (
                "https://api.githubcopilot.com/chat/completions".to_string(),
                token,
            )
        }
        "openai_compatible" => {
            let token = read_openai_api_key()?
                .filter(|secret| !secret.trim().is_empty())
                .ok_or_else(|| anyhow!("OpenAI-compatible API key is not configured"))?;
            let base = cfg.base_url.trim_end_matches('/');
            (format!("{base}/chat/completions"), token)
        }
        other => return Err(anyhow!("unknown provider {other}")),
    };
    let model = if cfg.model.trim().is_empty() {
        "claude-haiku-4.5".to_string()
    } else {
        cfg.model.clone()
    };

    let body = json!({
        "model": model,
        "temperature": 0.1,
        "messages": [
            { "role": "system", "content": SYSTEM },
            { "role": "user", "content": text }
        ]
    });
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()?;
    let mut req = client.post(&url).bearer_auth(&token).json(&body);
    if cfg.provider == "github_copilot" {
        req = req.header("Copilot-Integration-Id", "vscode-chat").header(
            "Editor-Version",
            concat!("OpenWritr/", env!("CARGO_PKG_VERSION")),
        );
    }
    let resp = req.send()?;
    if !resp.status().is_success() {
        return Err(anyhow!("enhance http {}", resp.status()));
    }
    let v: serde_json::Value = resp.json()?;
    let content = v
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .ok_or_else(|| anyhow!("missing choices[0].message.content"))?;
    Ok(content.trim().to_string())
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
    use super::{parse_gh_token_output, CopilotAuthError};

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
