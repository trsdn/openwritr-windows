// Grammar / cleanup enhancement provider.
//
// Mirrors the macOS GrammarEnhancer.swift. The provider is selectable via
// settings; credentials live in Windows Credential Manager via the `keyring`
// crate (service name = "OpenWritr").
//
// Supported providers:
//   * GitHub Copilot — POST https://api.githubcopilot.com/chat/completions
//     with `Authorization: Bearer <token>` and `Copilot-Integration-Id: vscode-chat`.
//   * Any OpenAI-compatible endpoint (`base_url` + `api_key`).

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::warn;

const SERVICE: &str = "OpenWritr";
const SYSTEM_PROMPT: &str = "You are a transcription cleanup assistant. Fix punctuation, casing, \
filler words ('um', 'uh', 'like'), and obvious recognition errors in the user message. \
Preserve the original meaning, language, and tone. Return ONLY the cleaned text — no \
preamble, no quotes, no commentary.";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Provider {
    Copilot,
    OpenAi { base_url: String, model: String },
}

impl Default for Provider {
    fn default() -> Self {
        Provider::Copilot
    }
}

pub async fn enhance(text: &str) -> Result<String> {
    if text.trim().is_empty() {
        return Ok(text.to_string());
    }
    let provider = load_provider();
    match enhance_with(&provider, text).await {
        Ok(s) if !s.trim().is_empty() => Ok(s),
        Ok(_) => Ok(text.to_string()),
        Err(e) => {
            warn!(error = %e, "enhance failed, returning raw transcript");
            Ok(text.to_string())
        }
    }
}

async fn enhance_with(provider: &Provider, text: &str) -> Result<String> {
    let (url, model, token) = match provider {
        Provider::Copilot => (
            "https://api.githubcopilot.com/chat/completions".to_string(),
            "gpt-4o-mini".to_string(),
            secret("copilot_token")?,
        ),
        Provider::OpenAi { base_url, model } => (
            format!("{}/chat/completions", base_url.trim_end_matches('/')),
            model.clone(),
            secret("openai_api_key")?,
        ),
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()?;
    let body = json!({
        "model": model,
        "temperature": 0.1,
        "messages": [
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user",   "content": text }
        ]
    });

    let mut req = client.post(&url).bearer_auth(&token).json(&body);
    if matches!(provider, Provider::Copilot) {
        req = req
            .header("Copilot-Integration-Id", "vscode-chat")
            .header("Editor-Version", "OpenWritr/0.1");
    }

    let resp = req.send().await.context("enhance http")?;
    if !resp.status().is_success() {
        return Err(anyhow!("enhance http {}", resp.status()));
    }
    let payload: ChatResp = resp.json().await?;
    payload
        .choices
        .into_iter()
        .next()
        .and_then(|c| c.message.content)
        .ok_or_else(|| anyhow!("empty completion"))
}

fn load_provider() -> Provider {
    // Provider selection itself is stored in keyring under "provider_json".
    if let Ok(s) = secret("provider_json") {
        if let Ok(p) = serde_json::from_str::<Provider>(&s) {
            return p;
        }
    }
    Provider::default()
}

fn secret(key: &str) -> Result<String> {
    let entry = keyring::Entry::new(SERVICE, key)?;
    entry.get_password().context("read secret from credential manager")
}

#[derive(Deserialize)]
struct ChatResp {
    choices: Vec<ChatChoice>,
}
#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}
#[derive(Deserialize)]
struct ChatMessage {
    content: Option<String>,
}
