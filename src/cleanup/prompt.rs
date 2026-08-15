//! Prompt target identity, transcript payload, and prompt resolution.
//!
//! A [`PromptTarget`] is the stable identity of "which prompt applies here":
//! the provider, the endpoint scope (for OpenAI-compatible providers), and the
//! exact trimmed model/deployment ID. It is hashable so callers can key a
//! per-target override table on it.
//!
//! The [`Transcript`] is a *distinct, untrusted* user payload. It is kept
//! structurally separate from the system prompt at every layer so untrusted
//! recognized speech can never be promoted into instructions, and its `Debug`
//! impl redacts the content so it is never logged.
//!
//! Prompt resolution layers, highest precedence first:
//!   1. caller-supplied custom override (per target),
//!   2. compiled bundled model default,
//!   3. compiled bundled provider default,
//!   4. compiled bundled global default.

use std::fmt;

use thiserror::Error;

use super::endpoint::EndpointScope;
use super::provider::EnhanceProvider;

/// The compiled global default system prompt.
pub const GLOBAL_DEFAULT_SYSTEM: &str = "You are a transcription cleanup assistant. Fix \
punctuation, casing, filler words ('um', 'uh', 'like'), and obvious \
recognition errors in the user message. Preserve the original meaning, \
language, and tone. Return ONLY the cleaned text — no preamble, no \
quotes, no commentary. If the user message has no meaningful content, \
reply with exactly [[EMPTY]].";

/// A distinct, untrusted transcript payload.
///
/// The content is only ever exposed via [`Transcript::as_untrusted_str`], whose
/// name is a reminder to callers that it must be treated as data, never as
/// instructions. `Debug` redacts the content.
#[derive(Clone, Copy)]
pub struct Transcript<'a> {
    text: &'a str,
}

impl<'a> Transcript<'a> {
    pub fn new(text: &'a str) -> Self {
        Transcript { text }
    }

    /// The raw transcript. The name is deliberate: this is untrusted user data
    /// and must only ever be placed in a user-role message payload.
    pub fn as_untrusted_str(&self) -> &'a str {
        self.text
    }

    pub fn is_effectively_empty(&self) -> bool {
        self.text.trim().is_empty()
    }
}

impl fmt::Debug for Transcript<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never log transcript content — expose only a length hint.
        f.debug_struct("Transcript")
            .field("chars", &self.text.chars().count())
            .finish_non_exhaustive()
    }
}

/// The stable identity of a prompt target.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum PromptTarget {
    GithubCopilot {
        model_id: String,
    },
    OpenAiCompatible {
        endpoint: EndpointScope,
        model_id: String,
    },
}

impl PromptTarget {
    /// Build a Copilot target from a model ID. The ID is trimmed; empty IDs are
    /// rejected.
    pub fn github_copilot(model_id: &str) -> Result<Self, PromptTargetError> {
        Ok(PromptTarget::GithubCopilot {
            model_id: normalize_model_id(model_id)?,
        })
    }

    /// Build an OpenAI-compatible target from an endpoint scope and model ID.
    /// The ID is trimmed; empty IDs are rejected.
    pub fn openai_compatible(
        endpoint: EndpointScope,
        model_id: &str,
    ) -> Result<Self, PromptTargetError> {
        Ok(PromptTarget::OpenAiCompatible {
            endpoint,
            model_id: normalize_model_id(model_id)?,
        })
    }

    pub fn provider(&self) -> EnhanceProvider {
        match self {
            PromptTarget::GithubCopilot { .. } => EnhanceProvider::GithubCopilot,
            PromptTarget::OpenAiCompatible { .. } => EnhanceProvider::OpenAiCompatible,
        }
    }

    /// The exact trimmed model/deployment ID.
    pub fn model_id(&self) -> &str {
        match self {
            PromptTarget::GithubCopilot { model_id } => model_id,
            PromptTarget::OpenAiCompatible { model_id, .. } => model_id,
        }
    }

    /// The endpoint scope, present only for OpenAI-compatible targets.
    pub fn endpoint(&self) -> Option<&EndpointScope> {
        match self {
            PromptTarget::GithubCopilot { .. } => None,
            PromptTarget::OpenAiCompatible { endpoint, .. } => Some(endpoint),
        }
    }
}

impl fmt::Debug for PromptTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PromptTarget")
            .field("provider", &self.provider().as_str())
            .field("endpoint", &self.endpoint().map(EndpointScope::base_url))
            .field("model_id", &self.model_id())
            .finish()
    }
}

fn normalize_model_id(model_id: &str) -> Result<String, PromptTargetError> {
    let trimmed = model_id.trim();
    if trimmed.is_empty() {
        return Err(PromptTargetError::EmptyModelId);
    }
    Ok(trimmed.to_string())
}

/// Why a prompt target could not be constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum PromptTargetError {
    #[error("model/deployment ID must not be empty")]
    EmptyModelId,
}

/// Which layer supplied the resolved system prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptSource {
    CustomOverride,
    ModelDefault,
    ProviderDefault,
    GlobalDefault,
}

/// The outcome of prompt resolution: the system prompt and where it came from.
#[derive(Clone, PartialEq, Eq)]
pub struct ResolvedPrompt {
    pub system: String,
    pub source: PromptSource,
}

impl fmt::Debug for ResolvedPrompt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Prompts may contain caller-supplied overrides; redact the text.
        f.debug_struct("ResolvedPrompt")
            .field("source", &self.source)
            .field("chars", &self.system.chars().count())
            .finish_non_exhaustive()
    }
}

/// A caller-supplied lookup for per-target custom system prompts.
pub trait PromptOverrides {
    /// Return a custom system prompt for the target, or `None` to fall through
    /// to the compiled defaults.
    fn lookup(&self, target: &PromptTarget) -> Option<String>;
}

/// A [`PromptOverrides`] implementation that never overrides anything.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoOverrides;

impl PromptOverrides for NoOverrides {
    fn lookup(&self, _target: &PromptTarget) -> Option<String> {
        None
    }
}

/// Compiled bundled default for a specific model, if any. Currently sparse;
/// the mechanism exists so future model-specific prompts are a one-line change.
fn bundled_model_default(_model_id: &str) -> Option<&'static str> {
    None
}

/// Compiled bundled default for a provider, if any.
fn bundled_provider_default(_provider: EnhanceProvider) -> Option<&'static str> {
    None
}

/// Resolve the system prompt for a target using the caller's overrides layered
/// over the compiled bundled defaults.
pub fn resolve_prompt(target: &PromptTarget, overrides: &dyn PromptOverrides) -> ResolvedPrompt {
    resolve_layers(
        overrides.lookup(target),
        bundled_model_default(target.model_id()).map(str::to_string),
        bundled_provider_default(target.provider()).map(str::to_string),
        GLOBAL_DEFAULT_SYSTEM.to_string(),
    )
}

/// Pure layering logic, split out so every precedence branch is directly
/// testable without constructing targets.
fn resolve_layers(
    custom: Option<String>,
    model_default: Option<String>,
    provider_default: Option<String>,
    global_default: String,
) -> ResolvedPrompt {
    if let Some(custom) = custom {
        let trimmed = custom.trim();
        if !trimmed.is_empty() {
            return ResolvedPrompt {
                system: trimmed.to_string(),
                source: PromptSource::CustomOverride,
            };
        }
    }
    if let Some(model_default) = model_default {
        return ResolvedPrompt {
            system: model_default,
            source: PromptSource::ModelDefault,
        };
    }
    if let Some(provider_default) = provider_default {
        return ResolvedPrompt {
            system: provider_default,
            source: PromptSource::ProviderDefault,
        };
    }
    ResolvedPrompt {
        system: global_default,
        source: PromptSource::GlobalDefault,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedOverride(Option<&'static str>);
    impl PromptOverrides for FixedOverride {
        fn lookup(&self, _target: &PromptTarget) -> Option<String> {
            self.0.map(str::to_string)
        }
    }

    fn scope() -> EndpointScope {
        EndpointScope::parse("https://api.openai.com/v1").unwrap()
    }

    #[test]
    fn model_id_is_trimmed_and_empty_is_rejected() {
        let target = PromptTarget::github_copilot("  gpt-5-mini  ").unwrap();
        assert_eq!(target.model_id(), "gpt-5-mini");
        assert_eq!(
            PromptTarget::github_copilot("   ").unwrap_err(),
            PromptTargetError::EmptyModelId
        );
        assert_eq!(
            PromptTarget::openai_compatible(scope(), "").unwrap_err(),
            PromptTargetError::EmptyModelId
        );
    }

    #[test]
    fn identity_includes_provider_endpoint_and_model() {
        use std::collections::HashSet;
        let a = PromptTarget::openai_compatible(scope(), "gpt-5-mini").unwrap();
        let b = PromptTarget::openai_compatible(scope(), "gpt-5-mini").unwrap();
        let c = PromptTarget::openai_compatible(scope(), "claude-haiku-4.5").unwrap();
        let d = PromptTarget::github_copilot("gpt-5-mini").unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);

        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
        assert!(!set.contains(&c));
    }

    #[test]
    fn accessors_expose_the_identity_parts() {
        let target = PromptTarget::openai_compatible(scope(), "gpt-5-mini").unwrap();
        assert_eq!(target.provider(), EnhanceProvider::OpenAiCompatible);
        assert_eq!(target.model_id(), "gpt-5-mini");
        assert_eq!(
            target.endpoint().unwrap().base_url(),
            "https://api.openai.com/v1"
        );

        let copilot = PromptTarget::github_copilot("gpt-5-mini").unwrap();
        assert_eq!(copilot.provider(), EnhanceProvider::GithubCopilot);
        assert!(copilot.endpoint().is_none());
    }

    #[test]
    fn resolution_prefers_custom_override_when_non_empty() {
        let target = PromptTarget::github_copilot("gpt-5-mini").unwrap();
        let resolved = resolve_prompt(&target, &FixedOverride(Some("  custom prompt  ")));
        assert_eq!(resolved.source, PromptSource::CustomOverride);
        assert_eq!(resolved.system, "custom prompt");
    }

    #[test]
    fn blank_override_falls_through_to_global_default() {
        let target = PromptTarget::github_copilot("gpt-5-mini").unwrap();
        let resolved = resolve_prompt(&target, &FixedOverride(Some("   ")));
        assert_eq!(resolved.source, PromptSource::GlobalDefault);
        assert_eq!(resolved.system, GLOBAL_DEFAULT_SYSTEM);
    }

    #[test]
    fn no_overrides_yields_global_default() {
        let target = PromptTarget::github_copilot("gpt-5-mini").unwrap();
        let resolved = resolve_prompt(&target, &NoOverrides);
        assert_eq!(resolved.source, PromptSource::GlobalDefault);
        assert_eq!(resolved.system, GLOBAL_DEFAULT_SYSTEM);
    }

    #[test]
    fn layering_precedence_is_deterministic() {
        // custom > model > provider > global
        assert_eq!(
            resolve_layers(
                Some("c".into()),
                Some("m".into()),
                Some("p".into()),
                "g".into()
            )
            .source,
            PromptSource::CustomOverride
        );
        assert_eq!(
            resolve_layers(None, Some("m".into()), Some("p".into()), "g".into()).source,
            PromptSource::ModelDefault
        );
        assert_eq!(
            resolve_layers(None, None, Some("p".into()), "g".into()).source,
            PromptSource::ProviderDefault
        );
        assert_eq!(
            resolve_layers(None, None, None, "g".into()).source,
            PromptSource::GlobalDefault
        );
    }

    #[test]
    fn transcript_debug_redacts_content() {
        let transcript = Transcript::new("secret speech content");
        let rendered = format!("{transcript:?}");
        assert!(!rendered.contains("secret speech content"));
        assert!(rendered.contains("chars"));
    }

    #[test]
    fn resolved_prompt_debug_redacts_content() {
        let resolved = ResolvedPrompt {
            system: "sensitive override".into(),
            source: PromptSource::CustomOverride,
        };
        let rendered = format!("{resolved:?}");
        assert!(!rendered.contains("sensitive override"));
        assert!(rendered.contains("CustomOverride"));
    }
}
