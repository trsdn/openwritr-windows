//! Typed enhancement provider.
//!
//! Replaces the stringly `provider: String` handling with a closed enum whose
//! serialization values are frozen so settings written by older builds keep
//! round-tripping. `from_settings_str` is the compatibility adapter for the
//! existing `Settings.enhance.provider` string.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Fixed base URL for the GitHub Copilot chat completions endpoint.
pub const COPILOT_CHAT_COMPLETIONS_URL: &str = "https://api.githubcopilot.com/chat/completions";

/// The LLM provider used for the transcription cleanup pass.
///
/// The wire values are pinned with explicit `#[serde(rename = ...)]` rather
/// than `rename_all` so they can never drift (serde's `snake_case` would
/// otherwise render `OpenAiCompatible` as `open_ai_compatible`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnhanceProvider {
    /// GitHub Copilot chat completions (token from `gh auth token`).
    #[serde(rename = "github_copilot")]
    GithubCopilot,
    /// Any OpenAI-compatible `/chat/completions` endpoint (bearer API key).
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible,
}

impl EnhanceProvider {
    /// Stable, snake_case wire value. Must never change: it is persisted in
    /// `settings.json`.
    pub const fn as_str(self) -> &'static str {
        match self {
            EnhanceProvider::GithubCopilot => "github_copilot",
            EnhanceProvider::OpenAiCompatible => "openai_compatible",
        }
    }

    /// Compatibility adapter for the existing `Settings.enhance.provider`
    /// string. Unknown values are rejected rather than silently coerced.
    pub fn from_settings_str(raw: &str) -> Result<Self, UnknownProvider> {
        match raw {
            "github_copilot" => Ok(EnhanceProvider::GithubCopilot),
            "openai_compatible" => Ok(EnhanceProvider::OpenAiCompatible),
            other => Err(UnknownProvider(other.to_string())),
        }
    }

    /// Every provider, in stable order — handy for building UI pickers.
    pub const fn all() -> [EnhanceProvider; 2] {
        [
            EnhanceProvider::GithubCopilot,
            EnhanceProvider::OpenAiCompatible,
        ]
    }

    /// Whether this provider needs a caller-supplied endpoint scope. Copilot's
    /// endpoint is fixed; OpenAI-compatible providers require a base URL.
    pub const fn requires_endpoint_scope(self) -> bool {
        matches!(self, EnhanceProvider::OpenAiCompatible)
    }

    /// Provider-specific request headers. Copilot requires an integration id
    /// and editor version; OpenAI-compatible providers need none. The values
    /// are static and contain no secrets.
    pub fn request_headers(self) -> &'static [(&'static str, &'static str)] {
        match self {
            EnhanceProvider::GithubCopilot => &[
                ("Copilot-Integration-Id", "vscode-chat"),
                (
                    "Editor-Version",
                    concat!("OpenWritr/", env!("CARGO_PKG_VERSION")),
                ),
            ],
            EnhanceProvider::OpenAiCompatible => &[],
        }
    }
}

/// The provider string in settings did not match a known provider.
///
/// Carries only the offending provider identifier (never transcript or secret
/// data) so it is safe to surface in logs and error messages.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("unknown provider `{0}`")]
pub struct UnknownProvider(pub String);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_values_are_frozen() {
        assert_eq!(EnhanceProvider::GithubCopilot.as_str(), "github_copilot");
        assert_eq!(
            EnhanceProvider::OpenAiCompatible.as_str(),
            "openai_compatible"
        );
    }

    #[test]
    fn serialization_uses_stable_snake_case() {
        assert_eq!(
            serde_json::to_string(&EnhanceProvider::GithubCopilot).unwrap(),
            "\"github_copilot\""
        );
        assert_eq!(
            serde_json::to_string(&EnhanceProvider::OpenAiCompatible).unwrap(),
            "\"openai_compatible\""
        );
    }

    #[test]
    fn deserialization_round_trips() {
        for provider in EnhanceProvider::all() {
            let json = serde_json::to_string(&provider).unwrap();
            let back: EnhanceProvider = serde_json::from_str(&json).unwrap();
            assert_eq!(back, provider);
        }
    }

    #[test]
    fn from_settings_str_matches_wire_values() {
        assert_eq!(
            EnhanceProvider::from_settings_str("github_copilot").unwrap(),
            EnhanceProvider::GithubCopilot
        );
        assert_eq!(
            EnhanceProvider::from_settings_str("openai_compatible").unwrap(),
            EnhanceProvider::OpenAiCompatible
        );
    }

    #[test]
    fn from_settings_str_rejects_unknown_without_coercion() {
        let err = EnhanceProvider::from_settings_str("off").unwrap_err();
        assert_eq!(err, UnknownProvider("off".to_string()));
        assert!(err.to_string().contains("off"));
    }

    #[test]
    fn only_openai_requires_endpoint_scope() {
        assert!(!EnhanceProvider::GithubCopilot.requires_endpoint_scope());
        assert!(EnhanceProvider::OpenAiCompatible.requires_endpoint_scope());
    }

    #[test]
    fn copilot_headers_are_present_and_openai_has_none() {
        assert_eq!(EnhanceProvider::GithubCopilot.request_headers().len(), 2);
        assert!(EnhanceProvider::OpenAiCompatible
            .request_headers()
            .is_empty());
    }
}
