//! LLM cleanup pass — GitHub Copilot or any OpenAI-compatible endpoint.
//!
//! Blocking reqwest call so we can run it inline on the transcribe thread.

use crate::settings::{Enhance, Settings};
use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use serde_json::json;
use std::time::{Duration, Instant};
use tracing::warn;

const SYSTEM: &str = "You are a transcription cleanup assistant. Fix \
punctuation, casing, filler words ('um', 'uh', 'like'), and obvious \
recognition errors in the user message. Preserve the original meaning, \
language, and tone. Return ONLY the cleaned text — no preamble, no \
quotes, no commentary.";

pub fn enhance(text: &str, settings: &Settings) -> Result<String> {
    let cfg = &settings.enhance;
    if cfg.provider == "off" || text.trim().is_empty() {
        return Ok(text.to_string());
    }
    let (url, token) = match cfg.provider.as_str() {
        "github_copilot" => {
            let token = gh_token().ok_or_else(|| anyhow!("`gh auth token` empty"))?;
            ("https://api.githubcopilot.com/chat/completions".to_string(), token)
        }
        "openai_compatible" => {
            if cfg.api_key.trim().is_empty() {
                return Err(anyhow!("OpenAI api_key empty"));
            }
            let base = cfg.base_url.trim_end_matches('/');
            (format!("{base}/chat/completions"), cfg.api_key.clone())
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
        req = req
            .header("Copilot-Integration-Id", "vscode-chat")
            .header("Editor-Version", "OpenWritr/0.2");
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

fn gh_token() -> Option<String> {
    // Cache the token for 10 minutes so we don't spawn `gh` on every call.
    static CACHE: Mutex<Option<(String, Instant)>> = Mutex::new(None);
    {
        let g = CACHE.lock();
        if let Some((tok, t)) = g.as_ref() {
            if t.elapsed() < Duration::from_secs(600) {
                return Some(tok.clone());
            }
        }
    }
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let out = std::process::Command::new("gh")
        .args(["auth", "token"])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .ok()?;
    if !out.status.success() {
        warn!("`gh auth token` failed: {}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { return None; }
    *CACHE.lock() = Some((s.clone(), Instant::now()));
    Some(s)
}
