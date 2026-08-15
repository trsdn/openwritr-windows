//! Conservative integrity validator.
//!
//! A pure comparison of a source transcript against a candidate cleanup. It
//! never performs I/O and never retains transcript or candidate content in its
//! output — violations carry only a [`ViolationKind`] and a [`Severity`], so
//! reports are always safe to log.
//!
//! Design bias: prefer *false negatives over false positives*. When a check
//! fires it means the caller should fall back to the raw transcript, which is
//! always safe, so the checks only fire when reasonably confident.
//!
//! Error-level violations (candidate should be rejected):
//!   * a factual explicit-digit value changed or disappeared,
//!   * a negation was added, removed, or changed,
//!   * a protected token (URL/email/inline code/acronym) was dropped,
//!   * confident German/English language drift,
//!   * a meaningful source collapsed to empty.
//!
//! Warning-level violations (candidate kept, but noted):
//!   * an introduced digit that most likely came from spelling out a number
//!     (warning-only *initially* — deliberately not an error yet),
//!   * terminal punctuation changed,
//!   * a filler word survived,
//!   * adjacent word repetition remained.
//!
//! The validator can also fail *internally* (see [`ValidatorError`]); the
//! caller represents that as a raw fallback.

use std::collections::BTreeMap;

use thiserror::Error;

/// Upper bound on input size. Beyond this the validator refuses to run (a real,
/// representable internal failure) rather than doing pathological work; the
/// caller maps the failure to a raw fallback.
pub const MAX_INPUT_CHARS: usize = 100_000;

/// Identifies the validator's observable check set and thresholds. Bump this
/// whenever a check is added/removed/re-tuned in a way that could change a
/// decision on existing input, so consumers pinned to a version (e.g. the
/// opt-in `cleanup_eval` evaluator's fixture corpus) can detect drift instead
/// of silently comparing against a moved target.
pub const VALIDATOR_VERSION: &str = "integrity-v1";

/// Severity of an integrity violation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// The candidate should be rejected in favor of the raw transcript.
    Error,
    /// The candidate is kept but the issue is worth surfacing.
    Warning,
}

/// A category of integrity violation. Deliberately coarse and content-free so
/// reports never leak transcript or candidate text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViolationKind {
    /// An explicit-digit factual value changed or disappeared.
    FactualDigitChanged,
    /// A negation was added, removed, or changed.
    NegationChanged,
    /// A protected token (URL/email/inline code/acronym) was dropped.
    ProtectedTokenDropped,
    /// Confident German/English language drift.
    LanguageDrift,
    /// A meaningful source became empty.
    SourceBecameEmpty,
    /// A digit appeared in the candidate that was not in the source (most
    /// likely from spelling out a number). Warning-only initially.
    IntroducedDigit,
    /// Terminal punctuation changed.
    TerminalPunctuationChanged,
    /// A filler word survived into the candidate.
    RetainedFiller,
    /// Adjacent word repetition remained in the candidate.
    AdjacentRepetition,
}

/// A single integrity violation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Violation {
    pub kind: ViolationKind,
    pub severity: Severity,
}

impl Violation {
    fn error(kind: ViolationKind) -> Self {
        Violation {
            kind,
            severity: Severity::Error,
        }
    }

    fn warning(kind: ViolationKind) -> Self {
        Violation {
            kind,
            severity: Severity::Warning,
        }
    }
}

/// The result of validating a candidate against a source.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct IntegrityReport {
    violations: Vec<Violation>,
}

impl IntegrityReport {
    pub fn violations(&self) -> &[Violation] {
        &self.violations
    }

    pub fn is_rejected(&self) -> bool {
        self.violations
            .iter()
            .any(|violation| violation.severity == Severity::Error)
    }

    pub fn error_kinds(&self) -> Vec<ViolationKind> {
        self.kinds(Severity::Error)
    }

    pub fn warning_kinds(&self) -> Vec<ViolationKind> {
        self.kinds(Severity::Warning)
    }

    fn kinds(&self, severity: Severity) -> Vec<ViolationKind> {
        self.violations
            .iter()
            .filter(|violation| violation.severity == severity)
            .map(|violation| violation.kind)
            .collect()
    }
}

/// An internal validator failure. The caller represents this as a raw fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum ValidatorError {
    #[error("integrity validator input exceeds the safe size limit")]
    InputTooLarge,
}

/// Validate a candidate cleanup against its source transcript.
pub fn validate(source: &str, candidate: &str) -> Result<IntegrityReport, ValidatorError> {
    if source.len() > MAX_INPUT_CHARS || candidate.len() > MAX_INPUT_CHARS {
        return Err(ValidatorError::InputTooLarge);
    }

    let mut violations = Vec::new();

    // --- Error-level checks -------------------------------------------------
    if is_meaningful(source) && !is_meaningful(candidate) {
        violations.push(Violation::error(ViolationKind::SourceBecameEmpty));
    }

    check_numeric_integrity(source, candidate, &mut violations);

    if negation_count(source) != negation_count(candidate) {
        violations.push(Violation::error(ViolationKind::NegationChanged));
    }

    if protected_token_dropped(source, candidate) {
        violations.push(Violation::error(ViolationKind::ProtectedTokenDropped));
    }

    if let (Some(source_lang), Some(candidate_lang)) =
        (confident_language(source), confident_language(candidate))
    {
        if source_lang != candidate_lang {
            violations.push(Violation::error(ViolationKind::LanguageDrift));
        }
    }

    // --- Warning-level checks ----------------------------------------------
    if terminal_punctuation(source) != terminal_punctuation(candidate) {
        violations.push(Violation::warning(
            ViolationKind::TerminalPunctuationChanged,
        ));
    }

    if contains_filler(candidate) {
        violations.push(Violation::warning(ViolationKind::RetainedFiller));
    }

    if has_adjacent_repetition(candidate) {
        violations.push(Violation::warning(ViolationKind::AdjacentRepetition));
    }

    Ok(IntegrityReport { violations })
}

/// Whether the text carries any meaningful content: at least one word token
/// that is not one of the small, unambiguous set of hesitation filler words
/// (see [`FILLERS`] — `um`, `uh`, `äh`, ...).
///
/// This is deliberately conservative in both directions:
/// * Text made up *only* of those hesitation fillers (plus punctuation/
///   whitespace) is treated as having no meaningful content, matching what
///   the cleanup prompt actually asks a model to signal as empty. That keeps
///   a genuinely filler-only source (DE or EN) from being misclassified as
///   "meaningful", which matters for two callers: the [[EMPTY]] sentinel
///   gate in [`super::pipeline::finalize`] correctly *accepts* the sentinel
///   for such a source, and the [`ViolationKind::SourceBecameEmpty`] check
///   below correctly does *not* fire for it.
/// * Any other word (ordinary discourse words like "so"/"well"/"like"
///   included, since those are far too common in real speech to safely
///   treat as filler here) still counts as meaningful. So a source with any
///   real content stays meaningful, meaning a malicious or accidental
///   `[[EMPTY]]` sentinel for it is still rejected in favor of the raw
///   transcript, and a candidate that actually erases real content still
///   trips `SourceBecameEmpty`.
pub(crate) fn is_meaningful(text: &str) -> bool {
    let lower = text.to_lowercase();
    word_tokens(&lower)
        .into_iter()
        .any(|word| !FILLERS.contains(&word))
}

// --------------------------------------------------------------------------
// Numeric integrity
// --------------------------------------------------------------------------

fn check_numeric_integrity(source: &str, candidate: &str, out: &mut Vec<Violation>) {
    let source_counts = multiset(numeric_tokens(source));
    let candidate_counts = multiset(numeric_tokens(candidate));

    let mut factual_changed = false;
    let mut introduced = false;

    for (token, source_count) in &source_counts {
        let candidate_count = candidate_counts.get(token).copied().unwrap_or(0);
        if candidate_count < *source_count {
            factual_changed = true;
        } else if candidate_count > *source_count {
            introduced = true;
        }
    }
    for (token, candidate_count) in &candidate_counts {
        if !source_counts.contains_key(token) && *candidate_count > 0 {
            introduced = true;
        }
    }

    if factual_changed {
        out.push(Violation::error(ViolationKind::FactualDigitChanged));
    }
    if introduced {
        out.push(Violation::warning(ViolationKind::IntroducedDigit));
    }
}

/// Extract normalized explicit-digit numeric tokens. Thousands separators
/// (commas) are removed and case is normalized; a decimal point between digits
/// is preserved. Trailing/leading separators are stripped.
fn numeric_tokens(text: &str) -> Vec<String> {
    let lower = text.to_ascii_lowercase();
    let mut tokens = Vec::new();
    let mut run = String::new();
    for ch in lower.chars() {
        if ch.is_ascii_digit() || ch == ',' || ch == '.' {
            run.push(ch);
        } else {
            if let Some(token) = normalize_number_run(&run) {
                tokens.push(token);
            }
            run.clear();
        }
    }
    if let Some(token) = normalize_number_run(&run) {
        tokens.push(token);
    }
    tokens
}

fn normalize_number_run(run: &str) -> Option<String> {
    if !run.chars().any(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let without_commas: String = run.chars().filter(|&ch| ch != ',').collect();
    let trimmed = without_commas.trim_matches('.');
    if trimmed.is_empty() || !trimmed.chars().any(|ch| ch.is_ascii_digit()) {
        return None;
    }
    Some(trimmed.to_string())
}

fn multiset(tokens: Vec<String>) -> BTreeMap<String, usize> {
    let mut map = BTreeMap::new();
    for token in tokens {
        *map.entry(token).or_insert(0) += 1;
    }
    map
}

// --------------------------------------------------------------------------
// Negation
// --------------------------------------------------------------------------

const NEGATION_WORDS: &[&str] = &[
    // English
    "not",
    "no",
    "never",
    "none",
    "nobody",
    "nothing",
    "nowhere",
    "neither",
    "nor",
    "without",
    "cannot",
    "cant",
    "dont",
    "doesnt",
    "didnt",
    "wont",
    "isnt",
    "arent",
    "wasnt",
    "werent",
    "havent",
    "hasnt",
    "hadnt",
    "wouldnt",
    "couldnt",
    "shouldnt", // German
    "nicht",
    "kein",
    "keine",
    "keinen",
    "keiner",
    "keinem",
    "keines",
    "nie",
    "niemals",
    "niemand",
    "nichts",
    "weder",
    "ohne",
    "nirgends",
    "nirgendwo",
];

fn negation_count(text: &str) -> usize {
    let lower = text.to_ascii_lowercase();
    // Contractions like "don't"/"isn't" — count the "n't" marker directly so we
    // are robust to the apostrophe splitting words during tokenization.
    let mut count = lower.matches("n't").count();
    for word in word_tokens(&lower) {
        if NEGATION_WORDS.contains(&word) {
            count += 1;
        }
    }
    count
}

// --------------------------------------------------------------------------
// Protected tokens
// --------------------------------------------------------------------------

fn protected_token_dropped(source: &str, candidate: &str) -> bool {
    for token in protected_tokens(source) {
        if !candidate.contains(&token) {
            return true;
        }
    }
    false
}

fn protected_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();

    for raw in text.split_whitespace() {
        let trimmed = trim_edges(raw);
        if trimmed.is_empty() {
            continue;
        }
        if is_url(trimmed) || is_email(trimmed) {
            tokens.push(trimmed.to_string());
        }
    }

    tokens.extend(inline_code_spans(text));
    tokens.extend(acronyms(text));

    tokens
}

fn trim_edges(token: &str) -> &str {
    token.trim_matches(|ch: char| {
        matches!(
            ch,
            '.' | ','
                | ';'
                | ':'
                | '!'
                | '?'
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '"'
                | '\''
                | '<'
                | '>'
        )
    })
}

fn is_url(token: &str) -> bool {
    token.starts_with("http://") || token.starts_with("https://") || token.starts_with("www.")
}

fn is_email(token: &str) -> bool {
    match token.split_once('@') {
        Some((local, domain)) => {
            !local.is_empty()
                && !domain.is_empty()
                && !domain.starts_with('.')
                && domain.contains('.')
                && !domain.ends_with('.')
        }
        None => false,
    }
}

fn inline_code_spans(text: &str) -> Vec<String> {
    let mut spans = Vec::new();
    for (index, part) in text.split('`').enumerate() {
        if index % 2 == 1 && !part.trim().is_empty() {
            spans.push(part.to_string());
        }
    }
    spans
}

const ACRONYM_STOPLIST: &[&str] = &["OK", "TV", "PM", "AM"];

fn acronyms(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut run = String::new();
    for ch in text.chars() {
        if ch.is_ascii_uppercase() || (!run.is_empty() && ch.is_ascii_digit()) {
            run.push(ch);
        } else {
            push_acronym(&mut out, &run);
            run.clear();
        }
    }
    push_acronym(&mut out, &run);
    out
}

fn push_acronym(out: &mut Vec<String>, run: &str) {
    let letters = run.chars().filter(|ch| ch.is_ascii_alphabetic()).count();
    if letters >= 2 && !ACRONYM_STOPLIST.contains(&run) {
        out.push(run.to_string());
    }
}

// --------------------------------------------------------------------------
// Language drift
// --------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Language {
    English,
    German,
}

const ENGLISH_STOPWORDS: &[&str] = &[
    "the", "and", "is", "are", "was", "you", "your", "this", "that", "with", "have", "has", "for",
    "but", "they", "what", "when", "will", "would", "there", "their", "about", "from", "which",
    "been", "being", "does", "of", "to", "it", "he", "she", "we", "on", "at", "as", "an", "be",
];

const GERMAN_STOPWORDS: &[&str] = &[
    "der", "die", "das", "und", "ist", "sind", "ich", "nicht", "ein", "eine", "einen", "zu", "den",
    "dem", "mit", "auf", "für", "dass", "es", "sie", "wir", "aber", "oder", "auch", "noch",
    "schon", "sehr", "haben", "hat", "war", "waren", "werden", "wird", "nach", "über", "ohne",
    "durch", "weil", "wenn", "wie", "mehr", "ja", "nein",
];

fn language_scores(text: &str) -> (usize, usize) {
    let lower = text.to_ascii_lowercase();
    let mut english = 0;
    let mut german = 0;
    for word in word_tokens(&lower) {
        if ENGLISH_STOPWORDS.contains(&word) {
            english += 1;
        }
        if GERMAN_STOPWORDS.contains(&word) {
            german += 1;
        }
    }
    (english, german)
}

fn confident_language(text: &str) -> Option<Language> {
    let (english, german) = language_scores(text);
    if english + german < 3 {
        return None;
    }
    if english >= 3 && english >= german * 2 {
        Some(Language::English)
    } else if german >= 3 && german >= english * 2 {
        Some(Language::German)
    } else {
        None
    }
}

// --------------------------------------------------------------------------
// Warning-level checks
// --------------------------------------------------------------------------

/// Exposed at `pub(crate)` (rather than private) solely so the opt-in
/// `cleanup_eval` binary (`src/bin/cleanup_eval/`) can compute non-secret
/// punctuation/filler/repetition health scores over the synthetic evaluation
/// corpus without duplicating this logic. Not part of the crate's public API.
pub(crate) fn terminal_punctuation(text: &str) -> Option<char> {
    text.trim_end()
        .chars()
        .last()
        .filter(|ch| matches!(ch, '.' | '!' | '?' | '…'))
}

const FILLERS: &[&str] = &[
    "um", "uh", "uhh", "uhm", "erm", "hmm", "hm", "äh", "ähm", "ähh", "öh", "öhm",
];

/// See the `terminal_punctuation` doc comment above: `pub(crate)` only for the
/// opt-in evaluator's scoring, not part of the public API.
pub(crate) fn contains_filler(text: &str) -> bool {
    let lower = text.to_lowercase();
    word_tokens(&lower)
        .into_iter()
        .any(|word| FILLERS.contains(&word))
}

/// See the `terminal_punctuation` doc comment above: `pub(crate)` only for the
/// opt-in evaluator's scoring, not part of the public API.
pub(crate) fn has_adjacent_repetition(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let words: Vec<&str> = word_tokens(&lower);
    words
        .windows(2)
        .any(|pair| pair[0] == pair[1] && pair[0].chars().any(|ch| ch.is_alphabetic()))
}

// --------------------------------------------------------------------------
// Shared tokenization
// --------------------------------------------------------------------------

/// Split into alphabetic/numeric word tokens, dropping punctuation. Operates on
/// already-lowercased text at call sites that care about case.
fn word_tokens(text: &str) -> Vec<&str> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(report: &IntegrityReport) -> Vec<ViolationKind> {
        report.violations().iter().map(|v| v.kind).collect()
    }

    #[test]
    fn clean_pass_produces_no_violations() {
        let report = validate(
            "um so i think we should uh ship it tomorrow",
            "So I think we should ship it tomorrow.",
        )
        .unwrap();
        // Only a benign terminal-punctuation warning (a period was added).
        assert!(!report.is_rejected());
        assert_eq!(report.error_kinds(), Vec::<ViolationKind>::new());
    }

    #[test]
    fn changed_factual_digit_is_an_error() {
        let report = validate("transfer 42 dollars", "Transfer 24 dollars.").unwrap();
        assert!(report.is_rejected());
        assert!(report
            .error_kinds()
            .contains(&ViolationKind::FactualDigitChanged));
    }

    #[test]
    fn dropped_number_is_an_error() {
        let report = validate("we need 3 servers", "We need servers.").unwrap();
        assert!(report
            .error_kinds()
            .contains(&ViolationKind::FactualDigitChanged));
    }

    #[test]
    fn thousands_separator_and_case_normalization_avoid_false_positives() {
        let report = validate("it cost 1,000 USD", "It cost 1000 USD.").unwrap();
        assert!(
            !report
                .error_kinds()
                .contains(&ViolationKind::FactualDigitChanged),
            "1,000 and 1000 must be treated as equal"
        );
    }

    #[test]
    fn trailing_period_does_not_create_a_phantom_number_change() {
        let report = validate("I have 5", "I have 5.").unwrap();
        assert!(!report
            .error_kinds()
            .contains(&ViolationKind::FactualDigitChanged));
    }

    #[test]
    fn introduced_digit_is_warning_only_initially() {
        let report = validate("give me five apples", "Give me 5 apples.").unwrap();
        assert!(!report.is_rejected(), "introduced digits must not reject");
        assert!(report
            .warning_kinds()
            .contains(&ViolationKind::IntroducedDigit));
    }

    #[test]
    fn removed_negation_is_an_error() {
        let report = validate("i do not agree", "I agree.").unwrap();
        assert!(report
            .error_kinds()
            .contains(&ViolationKind::NegationChanged));
    }

    #[test]
    fn added_negation_is_an_error() {
        let report = validate("i agree", "I do not agree.").unwrap();
        assert!(report
            .error_kinds()
            .contains(&ViolationKind::NegationChanged));
    }

    #[test]
    fn contraction_preserves_negation_count() {
        let report = validate("i do not know", "I don't know.").unwrap();
        assert!(!report
            .error_kinds()
            .contains(&ViolationKind::NegationChanged));
    }

    #[test]
    fn dropped_url_is_an_error() {
        let report = validate(
            "see https://example.com/docs for details",
            "See for details.",
        )
        .unwrap();
        assert!(report
            .error_kinds()
            .contains(&ViolationKind::ProtectedTokenDropped));
    }

    #[test]
    fn preserved_url_with_trailing_punctuation_is_ok() {
        let report = validate(
            "see https://example.com/docs.",
            "See https://example.com/docs.",
        )
        .unwrap();
        assert!(!report
            .error_kinds()
            .contains(&ViolationKind::ProtectedTokenDropped));
    }

    #[test]
    fn dropped_email_is_an_error() {
        let report = validate("mail me at a@b.com please", "Mail me please.").unwrap();
        assert!(report
            .error_kinds()
            .contains(&ViolationKind::ProtectedTokenDropped));
    }

    #[test]
    fn dropped_inline_code_is_an_error() {
        let report = validate("run `cargo test` now", "Run now.").unwrap();
        assert!(report
            .error_kinds()
            .contains(&ViolationKind::ProtectedTokenDropped));
    }

    #[test]
    fn dropped_acronym_is_an_error() {
        let report = validate("the API is down", "The is down.").unwrap();
        assert!(report
            .error_kinds()
            .contains(&ViolationKind::ProtectedTokenDropped));
    }

    #[test]
    fn preserved_acronym_is_ok() {
        let report = validate("the API is down", "The API is down.").unwrap();
        assert!(!report
            .error_kinds()
            .contains(&ViolationKind::ProtectedTokenDropped));
    }

    #[test]
    fn confident_language_drift_is_an_error() {
        let report = validate(
            "ich glaube das ist nicht richtig und wir sollten warten",
            "I believe this is not right and we should wait.",
        )
        .unwrap();
        assert!(report.error_kinds().contains(&ViolationKind::LanguageDrift));
    }

    #[test]
    fn short_ambiguous_text_does_not_trigger_language_drift() {
        let report = validate("ok", "OK.").unwrap();
        assert!(!report.error_kinds().contains(&ViolationKind::LanguageDrift));
    }

    #[test]
    fn meaningful_source_becoming_empty_is_an_error() {
        let report = validate("this matters a lot", "   ").unwrap();
        assert!(report
            .error_kinds()
            .contains(&ViolationKind::SourceBecameEmpty));
    }

    #[test]
    fn filler_only_sources_are_not_meaningful_in_english_and_german() {
        assert!(!is_meaningful("um uh erm"));
        assert!(!is_meaningful("ähm äh hm"));
        assert!(!is_meaningful("Ähm Äh Öhm"));
        // Whitespace/punctuation-only text remains non-meaningful too.
        assert!(!is_meaningful("   ..."));
    }

    #[test]
    fn any_real_word_mixed_with_fillers_is_still_meaningful() {
        assert!(is_meaningful("um please call me back"));
        assert!(is_meaningful("ähm bitte ruf mich zurück"));
    }

    #[test]
    fn filler_only_source_collapsing_to_empty_is_not_a_source_became_empty_error() {
        let report = validate("um uh erm", "").unwrap();
        assert!(!report
            .error_kinds()
            .contains(&ViolationKind::SourceBecameEmpty));

        let report = validate("ähm äh hm", "").unwrap();
        assert!(!report
            .error_kinds()
            .contains(&ViolationKind::SourceBecameEmpty));

        let report = validate("Ähm Äh Öhm", "").unwrap();
        assert!(!report
            .error_kinds()
            .contains(&ViolationKind::SourceBecameEmpty));
    }

    #[test]
    fn terminal_punctuation_change_is_warning_only() {
        let report = validate("are you sure", "Are you sure?").unwrap();
        assert!(!report.is_rejected());
        assert!(report
            .warning_kinds()
            .contains(&ViolationKind::TerminalPunctuationChanged));
    }

    #[test]
    fn retained_filler_is_warning_only() {
        let report = validate("um hello there", "Um, hello there.").unwrap();
        assert!(!report.is_rejected());
        assert!(report
            .warning_kinds()
            .contains(&ViolationKind::RetainedFiller));
    }

    #[test]
    fn retained_capitalized_german_filler_is_warning_only() {
        let report = validate("please proceed.", "Ähm please proceed.").unwrap();
        assert!(!report.is_rejected());
        assert!(report
            .warning_kinds()
            .contains(&ViolationKind::RetainedFiller));
    }

    #[test]
    fn adjacent_repetition_is_warning_only() {
        let report = validate("the the plan", "The the plan.").unwrap();
        assert!(!report.is_rejected());
        assert!(report
            .warning_kinds()
            .contains(&ViolationKind::AdjacentRepetition));
    }

    #[test]
    fn oversized_input_is_a_representable_internal_failure() {
        let big = "a".repeat(MAX_INPUT_CHARS + 1);
        assert_eq!(
            validate(&big, "x").unwrap_err(),
            ValidatorError::InputTooLarge
        );
        assert_eq!(
            validate("x", &big).unwrap_err(),
            ValidatorError::InputTooLarge
        );
    }

    #[test]
    fn report_helpers_partition_by_severity() {
        let report = validate("i do not agree", "I agree!").unwrap();
        assert!(report.is_rejected());
        assert!(report
            .error_kinds()
            .contains(&ViolationKind::NegationChanged));
        // Adding "!" where there was no terminal punctuation is a warning.
        assert!(report
            .warning_kinds()
            .contains(&ViolationKind::TerminalPunctuationChanged));
    }

    #[test]
    fn violations_never_embed_source_or_candidate_text() {
        let report = validate("secret 42 value", "Secret 24 value.").unwrap();
        let rendered = format!("{report:?}");
        assert!(!rendered.contains("secret"));
        assert!(!rendered.contains("42"));
        assert!(!rendered.contains("24"));
    }

    #[test]
    fn kinds_helper_matches_expected() {
        let report = IntegrityReport {
            violations: vec![
                Violation::error(ViolationKind::NegationChanged),
                Violation::warning(ViolationKind::RetainedFiller),
            ],
        };
        assert_eq!(kinds(&report).len(), 2);
        assert_eq!(report.error_kinds(), vec![ViolationKind::NegationChanged]);
        assert_eq!(report.warning_kinds(), vec![ViolationKind::RetainedFiller]);
    }
}
