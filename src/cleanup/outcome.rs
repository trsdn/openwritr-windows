//! Typed enhancement outcomes for worker integration.
//!
//! The worker never needs to inspect strings to decide what happened: every
//! terminal state is a typed variant carrying the text to use plus a reason.
//!
//! `Debug` is implemented by hand for [`EnhanceOutcome`] so it reports only the
//! character count of the text, never the text itself — outcomes are safe to
//! log.

use std::fmt;

use super::integrity::ViolationKind;

/// Why the enhancement pass was skipped entirely (no request was made).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkipReason {
    /// Enhancement mode is disabled.
    EnhancementDisabled,
    /// The transcript had no meaningful content to enhance.
    EmptyTranscript,
}

/// Why the caller fell back to the raw transcript instead of using the model
/// output. Carries only content-free structural detail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FallbackReason {
    /// The configured provider string was not recognized.
    UnknownProvider,
    /// The provider credential was missing or empty.
    MissingCredential,
    /// The provider/endpoint changed between recording and execution, so the
    /// freshly read credential could belong to a different endpoint than the
    /// one this job froze. No request was sent to avoid leaking the current
    /// key to a stale endpoint.
    CredentialTargetChanged,
    /// The OpenAI-compatible endpoint URL was invalid.
    InvalidEndpoint,
    /// The configured model/deployment ID was empty.
    EmptyModelId,
    /// The HTTP request failed or returned a non-success status.
    RequestFailed,
    /// The response could not be parsed into candidate text.
    ResponseUnparseable,
    /// The model produced empty output for a meaningful transcript.
    EmptyCandidate,
    /// The integrity validator rejected the candidate.
    IntegrityRejected(Vec<ViolationKind>),
    /// The integrity validator failed internally.
    ValidatorError,
}

/// The terminal result of an enhancement attempt.
///
/// Every variant carries the `text` the caller should ultimately use, so the
/// worker can always paste something safe.
#[derive(Clone, PartialEq, Eq)]
pub enum EnhanceOutcome {
    /// The candidate was accepted. `warnings` are non-fatal integrity notes.
    Enhanced {
        text: String,
        warnings: Vec<ViolationKind>,
    },
    /// No request was made; `text` is the untouched transcript.
    Skipped { text: String, reason: SkipReason },
    /// The raw transcript is used instead of the model output.
    RawFallback {
        text: String,
        reason: FallbackReason,
    },
}

impl EnhanceOutcome {
    pub fn enhanced(text: impl Into<String>, warnings: Vec<ViolationKind>) -> Self {
        EnhanceOutcome::Enhanced {
            text: text.into(),
            warnings,
        }
    }

    pub fn skipped(text: impl Into<String>, reason: SkipReason) -> Self {
        EnhanceOutcome::Skipped {
            text: text.into(),
            reason,
        }
    }

    pub fn raw_fallback(text: impl Into<String>, reason: FallbackReason) -> Self {
        EnhanceOutcome::RawFallback {
            text: text.into(),
            reason,
        }
    }

    /// The text the caller should use, regardless of variant.
    pub fn text(&self) -> &str {
        match self {
            EnhanceOutcome::Enhanced { text, .. }
            | EnhanceOutcome::Skipped { text, .. }
            | EnhanceOutcome::RawFallback { text, .. } => text,
        }
    }

    /// Consume the outcome, returning the text the caller should use.
    pub fn into_text(self) -> String {
        match self {
            EnhanceOutcome::Enhanced { text, .. }
            | EnhanceOutcome::Skipped { text, .. }
            | EnhanceOutcome::RawFallback { text, .. } => text,
        }
    }

    /// Whether the model output was actually accepted.
    pub fn is_enhanced(&self) -> bool {
        matches!(self, EnhanceOutcome::Enhanced { .. })
    }
}

impl fmt::Debug for EnhanceOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnhanceOutcome::Enhanced { text, warnings } => f
                .debug_struct("Enhanced")
                .field("chars", &text.chars().count())
                .field("warnings", warnings)
                .finish(),
            EnhanceOutcome::Skipped { text, reason } => f
                .debug_struct("Skipped")
                .field("chars", &text.chars().count())
                .field("reason", reason)
                .finish(),
            EnhanceOutcome::RawFallback { text, reason } => f
                .debug_struct("RawFallback")
                .field("chars", &text.chars().count())
                .field("reason", reason)
                .finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_accessors_agree_across_variants() {
        let enhanced = EnhanceOutcome::enhanced("clean", vec![]);
        assert_eq!(enhanced.text(), "clean");
        assert!(enhanced.is_enhanced());
        assert_eq!(enhanced.into_text(), "clean");

        let skipped = EnhanceOutcome::skipped("raw", SkipReason::EnhancementDisabled);
        assert_eq!(skipped.text(), "raw");
        assert!(!skipped.is_enhanced());

        let fallback = EnhanceOutcome::raw_fallback("raw", FallbackReason::RequestFailed);
        assert_eq!(fallback.text(), "raw");
        assert!(!fallback.is_enhanced());
    }

    #[test]
    fn debug_never_leaks_text() {
        let outcome = EnhanceOutcome::enhanced("super secret transcript", vec![]);
        let rendered = format!("{outcome:?}");
        assert!(!rendered.contains("super secret transcript"));
        assert!(rendered.contains("chars"));

        let fallback = EnhanceOutcome::raw_fallback(
            "another secret",
            FallbackReason::IntegrityRejected(vec![ViolationKind::NegationChanged]),
        );
        let rendered = format!("{fallback:?}");
        assert!(!rendered.contains("another secret"));
        assert!(rendered.contains("NegationChanged"));
    }

    #[test]
    fn outcomes_compare_by_value() {
        assert_eq!(
            EnhanceOutcome::raw_fallback("x", FallbackReason::EmptyCandidate),
            EnhanceOutcome::raw_fallback("x", FallbackReason::EmptyCandidate)
        );
        assert_ne!(
            EnhanceOutcome::raw_fallback("x", FallbackReason::EmptyCandidate),
            EnhanceOutcome::raw_fallback("x", FallbackReason::RequestFailed)
        );
    }
}
