//! Offline evaluation engine.
//!
//! Every decision here is delegated to the production `cleanup` core:
//! [`cleanup::adapter::parse_chat_response`] parses the simulated provider
//! response exactly as the live HTTP path does, and [`cleanup::pipeline::finalize`]
//! (normalization + the integrity validator) decides accept vs. raw-fallback.
//! This module only compares that real decision against the fixture's
//! expectation and computes content-free, aggregate health signals.

use std::collections::BTreeMap;

use crate::cleanup;
use crate::corpus::{CorpusCase, ExpectedKind};

/// The categorized, content-free outcome of running one case through the
/// production pipeline. `final_text` is kept only transiently to compute
/// non-secret scores (length, trailing punctuation, filler/repetition
/// presence) — it is never retained on the result or serialized.
#[derive(Debug)]
pub enum CaseOutcome {
    Enhanced {
        text: String,
        warnings: Vec<String>,
    },
    RawFallback {
        text: String,
        reason: String,
        error_kinds: Vec<String>,
    },
    ParseError {
        error: String,
    },
}

impl CaseOutcome {
    pub fn tag(&self) -> &'static str {
        match self {
            CaseOutcome::Enhanced { .. } => "enhanced",
            CaseOutcome::RawFallback { .. } => "raw_fallback",
            CaseOutcome::ParseError { .. } => "parse_error",
        }
    }
}

/// Run one case through the real production pipeline: parse -> normalize ->
/// validate. No logic is duplicated here; this is a thin, typed wrapper.
pub fn evaluate_case(source: &str, response_value: &serde_json::Value) -> CaseOutcome {
    let raw_content = match cleanup::adapter::parse_chat_response(response_value) {
        Ok(content) => content,
        Err(error) => {
            return CaseOutcome::ParseError {
                error: format!("{error:?}"),
            }
        }
    };

    match cleanup::pipeline::finalize(source, &raw_content) {
        cleanup::EnhanceOutcome::Enhanced { text, warnings } => CaseOutcome::Enhanced {
            text,
            warnings: warnings.iter().map(|kind| format!("{kind:?}")).collect(),
        },
        cleanup::EnhanceOutcome::RawFallback { text, reason } => {
            let (reason_tag, error_kinds) = fallback_reason_parts(&reason);
            CaseOutcome::RawFallback {
                text,
                reason: reason_tag,
                error_kinds,
            }
        }
        // `finalize` never produces `Skipped` — that variant is only used by
        // higher-level worker/settings integration this evaluator does not
        // touch (out of scope). Kept exhaustive so a future outcome variant
        // fails to compile here instead of being silently mishandled.
        cleanup::EnhanceOutcome::Skipped { text, .. } => CaseOutcome::RawFallback {
            text,
            reason: "Skipped".to_string(),
            error_kinds: vec![],
        },
    }
}

fn fallback_reason_parts(reason: &cleanup::FallbackReason) -> (String, Vec<String>) {
    use cleanup::FallbackReason;
    match reason {
        FallbackReason::UnknownProvider => ("UnknownProvider".to_string(), vec![]),
        FallbackReason::MissingCredential => ("MissingCredential".to_string(), vec![]),
        FallbackReason::CredentialTargetChanged => ("CredentialTargetChanged".to_string(), vec![]),
        FallbackReason::InvalidEndpoint => ("InvalidEndpoint".to_string(), vec![]),
        FallbackReason::EmptyModelId => ("EmptyModelId".to_string(), vec![]),
        FallbackReason::RequestFailed => ("RequestFailed".to_string(), vec![]),
        FallbackReason::ResponseUnparseable => ("ResponseUnparseable".to_string(), vec![]),
        FallbackReason::EmptyCandidate => ("EmptyCandidate".to_string(), vec![]),
        FallbackReason::ValidatorError => ("ValidatorError".to_string(), vec![]),
        FallbackReason::IntegrityRejected(kinds) => (
            "IntegrityRejected".to_string(),
            kinds.iter().map(|kind| format!("{kind:?}")).collect(),
        ),
    }
}

/// The result of comparing one case's real outcome to its fixture
/// expectation. Never carries source/candidate text — only ids, categories,
/// counts, and structural (content-free) tags.
pub struct CaseResult {
    pub id: String,
    pub category: String,
    pub tags: Vec<String>,
    pub tricky: bool,
    pub outcome_tag: &'static str,
    pub warning_kinds: Vec<String>,
    pub error_kinds: Vec<String>,
    pub fallback_reason: Option<String>,
    pub parse_error: Option<String>,
    pub passed: bool,
    pub mismatch: Option<String>,
    pub final_text_chars: usize,
    /// The actual user-visible final text for this case — whatever
    /// `EnhanceOutcome::text()` would be in production (the enhanced
    /// candidate, or the raw transcript on any fallback). Used only to
    /// compute the aggregate punctuation/filler/repetition scores, then
    /// discarded; never retained on the result or serialized.
    scored_text: Option<String>,
}

/// Evaluate a single case and compare against its fixture expectation.
pub fn run_case(case: &CorpusCase) -> CaseResult {
    let source = case.resolved_source();
    let response_value = case.provider_response.to_response_value();
    let outcome = evaluate_case(&source, &response_value);

    let mut warning_kinds = Vec::new();
    let mut error_kinds = Vec::new();
    let mut fallback_reason = None;
    let mut parse_error = None;
    let outcome_tag = outcome.tag();

    // Every arm below sets both `final_text_chars` and `scored_text` — the
    // actual user-visible final text is never omitted, whether the outcome
    // was accepted, fell back to the raw transcript, or (in this
    // evaluator-only variant) failed to parse at all.
    let (final_text_chars, scored_text, mismatch) = match &outcome {
        CaseOutcome::Enhanced { text, warnings } => {
            warning_kinds = warnings.clone();
            (
                text.chars().count(),
                Some(text.clone()),
                check_enhanced(case, text, warnings),
            )
        }
        CaseOutcome::RawFallback {
            text,
            reason,
            error_kinds: kinds,
        } => {
            error_kinds = kinds.clone();
            fallback_reason = Some(reason.clone());
            // A raw fallback's `text` is exactly what the user ends up with
            // (the untouched transcript), so it must be scored too — never
            // omitted — to reflect the real user-visible outcome.
            (
                text.chars().count(),
                Some(text.clone()),
                check_raw_fallback(case, &source, text, reason, kinds),
            )
        }
        CaseOutcome::ParseError { error } => {
            // In production a parse failure maps to
            // `RawFallback(ResponseUnparseable)`, whose user-visible text is
            // the original source transcript — score it the same way so
            // this evaluator-only outcome variant doesn't silently drop out
            // of the aggregate scores.
            parse_error = Some(error.clone());
            (
                source.chars().count(),
                Some(source.clone()),
                check_parse_error(case, error),
            )
        }
    };

    CaseResult {
        id: case.id.clone(),
        category: case.category.clone(),
        tags: case.tags.clone(),
        tricky: case.is_tricky(),
        outcome_tag,
        warning_kinds,
        error_kinds,
        fallback_reason,
        parse_error,
        passed: mismatch.is_none(),
        mismatch,
        final_text_chars,
        scored_text,
    }
}

fn check_enhanced(case: &CorpusCase, text: &str, warnings: &[String]) -> Option<String> {
    let expect = &case.expect;
    if expect.outcome != ExpectedKind::Enhanced {
        return Some(format!(
            "expected outcome `{:?}`, got `enhanced`",
            expect.outcome
        ));
    }
    if let Some(expected_text) = &expect.text {
        if expected_text != text {
            return Some("enhanced text did not match the expected text (lengths differ or content differs — redacted)".to_string());
        }
    }
    if expect.warning_kinds != warnings {
        return Some(format!(
            "expected warning kinds {:?}, got {:?}",
            expect.warning_kinds, warnings
        ));
    }
    None
}

fn check_raw_fallback(
    case: &CorpusCase,
    source: &str,
    text: &str,
    reason: &str,
    error_kinds: &[String],
) -> Option<String> {
    let expect = &case.expect;
    if expect.outcome != ExpectedKind::RawFallback {
        return Some(format!(
            "expected outcome `{:?}`, got `raw_fallback`({reason})",
            expect.outcome
        ));
    }
    if let Some(expected_reason) = &expect.reason {
        if expected_reason != reason {
            return Some(format!(
                "expected fallback reason `{expected_reason}`, got `{reason}`"
            ));
        }
    }
    if expect.error_kinds != error_kinds {
        return Some(format!(
            "expected error kinds {:?}, got {:?}",
            expect.error_kinds, error_kinds
        ));
    }
    if expect.text_equals_source && text != source {
        return Some("raw fallback text did not equal the source (redacted)".to_string());
    }
    if let Some(expected_text) = &expect.text {
        if expected_text != text {
            return Some(
                "raw fallback text did not match the expected text (redacted)".to_string(),
            );
        }
    }
    None
}

fn check_parse_error(case: &CorpusCase, error: &str) -> Option<String> {
    let expect = &case.expect;
    if expect.outcome != ExpectedKind::ParseError {
        return Some(format!(
            "expected outcome `{:?}`, got `parse_error`({error})",
            expect.outcome
        ));
    }
    match &expect.parse_error {
        Some(expected) if expected == error => None,
        Some(expected) => Some(format!("expected parse error `{expected}`, got `{error}`")),
        None => Some("fixture is missing `expect.parse_error`".to_string()),
    }
}

/// Aggregate, content-free health scores over every case's actual
/// user-visible final text — the enhanced candidate when accepted, or the
/// raw transcript on any fallback (raw-fallback or parse-error) — never
/// omitting fallback cases, since those are exactly what a real user would
/// end up seeing. Reuses the same private-but-`pub(crate)` helper predicates
/// the integrity validator itself uses, rather than re-detecting
/// punctuation, fillers, or repetition with new logic.
#[derive(Debug, Clone, Copy)]
pub struct ScoreSample {
    pub punctuation_ok: bool,
    pub filler_ok: bool,
    pub repetition_ok: bool,
}

pub fn score_results(results: &[CaseResult]) -> Vec<ScoreSample> {
    results
        .iter()
        .filter_map(|result| result.scored_text.as_deref())
        .map(|text| ScoreSample {
            punctuation_ok: cleanup::integrity::terminal_punctuation(text).is_some(),
            filler_ok: !cleanup::integrity::contains_filler(text),
            repetition_ok: !cleanup::integrity::has_adjacent_repetition(text),
        })
        .collect()
}

/// Tally integrity violation kinds (from either warnings or fallback error
/// kinds) across a set of results, keyed by the `ViolationKind` debug label.
pub fn tally_violation_kinds<'a>(
    results: impl IntoIterator<Item = &'a CaseResult>,
) -> (BTreeMap<String, usize>, BTreeMap<String, usize>) {
    let mut errors = BTreeMap::new();
    let mut warnings = BTreeMap::new();
    for result in results {
        for kind in &result.error_kinds {
            *errors.entry(kind.clone()).or_insert(0) += 1;
        }
        for kind in &result.warning_kinds {
            *warnings.entry(kind.clone()).or_insert(0) += 1;
        }
    }
    (errors, warnings)
}
