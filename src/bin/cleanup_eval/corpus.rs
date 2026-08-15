//! Versioned synthetic evaluation corpus: types + loader.
//!
//! The corpus is a committed, privacy-safe JSON fixture
//! (`fixtures/cleanup-eval/v1/corpus.json`) of fully synthetic DE/EN
//! source/candidate pairs with expected validator decisions. It is embedded
//! into the binary via `include_str!` so offline evaluation is deterministic
//! and needs no filesystem access at test time.

use serde::Deserialize;
use serde_json::Value;

/// Corpus format version this evaluator was written against. Bump alongside
/// `fixtures/cleanup-eval/<vN>/corpus.json` when the fixture schema or
/// expectations change in an incompatible way.
pub const EXPECTED_CORPUS_VERSION: &str = "cleanup-eval-corpus-v1";

/// The committed default corpus, embedded at compile time so offline
/// evaluation (including `cargo test`) never depends on the working
/// directory or filesystem layout at run time.
pub const EMBEDDED_CORPUS_JSON: &str =
    include_str!("../../../fixtures/cleanup-eval/v1/corpus.json");

#[derive(Debug, Deserialize)]
pub struct CorpusFile {
    pub version: String,
    #[allow(dead_code)]
    pub description: String,
    pub cases: Vec<CorpusCase>,
}

#[derive(Debug, Deserialize)]
pub struct CorpusCase {
    pub id: String,
    pub category: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[allow(dead_code)]
    pub lang: String,
    #[allow(dead_code)]
    pub notes: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub source_repeat: Option<SourceRepeat>,
    pub provider_response: ProviderResponseSpec,
    pub expect: ExpectedOutcome,
}

impl CorpusCase {
    /// Resolve this case's source transcript text. Exactly one of `source` /
    /// `source_repeat` is present — enforced by [`load_str`] at parse time.
    pub fn resolved_source(&self) -> String {
        if let Some(source) = &self.source {
            return source.clone();
        }
        if let Some(repeat) = &self.source_repeat {
            return repeat.char.repeat(repeat.count);
        }
        String::new()
    }

    /// True if this case is a "known tricky case" — either filed directly
    /// under the dedicated `tricky` category, or tagged `tricky` while
    /// living in whichever category it most naturally belongs to (e.g. a
    /// tricky date-reformatting case stays under
    /// `digits_dates_times_versions` but is still tagged so it is
    /// discoverable and counted as tricky).
    pub fn is_tricky(&self) -> bool {
        self.category == "tricky" || self.tags.iter().any(|tag| tag == "tricky")
    }

    /// Whether this case is a good candidate for the opt-in live-provider
    /// probe: it must have literal, human-scale source text (not a generated
    /// stress payload, which exists only to exercise the local size guard).
    pub fn eligible_for_live_probe(&self) -> bool {
        self.source.is_some()
    }
}

/// A generated-source case: repeats `char` `count` times. Keeps the
/// committed corpus small for stress fixtures (e.g. the oversized-input
/// case) instead of embedding a huge literal string.
#[derive(Debug, Deserialize)]
pub struct SourceRepeat {
    pub char: String,
    pub count: usize,
}

/// The simulated provider response for a case, in exactly the shapes
/// `cleanup::adapter::parse_chat_response` accepts (or an arbitrary `raw`
/// value for exercising its error paths).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ProviderResponseSpec {
    Content { content: String },
    ContentParts { content_parts: Vec<String> },
    Raw { raw: Value },
}

impl ProviderResponseSpec {
    /// Build the full chat-completions-shaped response `Value` exactly as
    /// `cleanup::adapter::parse_chat_response` would receive it over the
    /// wire, so offline evaluation exercises the real parsing code path.
    pub fn to_response_value(&self) -> Value {
        match self {
            ProviderResponseSpec::Content { content } => serde_json::json!({
                "choices": [ { "message": { "content": content } } ]
            }),
            ProviderResponseSpec::ContentParts { content_parts } => serde_json::json!({
                "choices": [ { "message": { "content": content_parts } } ]
            }),
            ProviderResponseSpec::Raw { raw } => raw.clone(),
        }
    }
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedKind {
    Enhanced,
    RawFallback,
    ParseError,
}

#[derive(Debug, Deserialize)]
pub struct ExpectedOutcome {
    pub outcome: ExpectedKind,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub text_equals_source: bool,
    #[serde(default)]
    pub warning_kinds: Vec<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub error_kinds: Vec<String>,
    #[serde(default)]
    pub parse_error: Option<String>,
}

#[derive(Debug)]
pub struct LoadError(pub String);

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "corpus load error: {}", self.0)
    }
}

impl std::error::Error for LoadError {}

/// Load and structurally validate the embedded default corpus.
pub fn load_embedded() -> Result<CorpusFile, LoadError> {
    load_str(EMBEDDED_CORPUS_JSON)
}

/// Parse and structurally validate a corpus JSON document.
pub fn load_str(raw: &str) -> Result<CorpusFile, LoadError> {
    let file: CorpusFile =
        serde_json::from_str(raw).map_err(|error| LoadError(error.to_string()))?;
    for case in &file.cases {
        if case.source.is_some() == case.source_repeat.is_some() {
            return Err(LoadError(format!(
                "case `{}` must set exactly one of `source` / `source_repeat`",
                case.id
            )));
        }
    }
    if file.cases.is_empty() {
        return Err(LoadError("corpus has no cases".to_string()));
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_corpus_parses_and_matches_the_expected_version() {
        let file = load_embedded().expect("embedded corpus must parse");
        assert_eq!(file.version, EXPECTED_CORPUS_VERSION);
        assert!(!file.cases.is_empty());
    }

    #[test]
    fn rejects_a_case_with_both_source_kinds() {
        let raw = r#"{
            "version": "cleanup-eval-corpus-v1",
            "description": "d",
            "cases": [ {
                "id": "bad",
                "category": "tricky",
                "lang": "en",
                "notes": "n",
                "source": "hi",
                "source_repeat": { "char": "a", "count": 1 },
                "provider_response": { "content": "Hi." },
                "expect": { "outcome": "enhanced", "text": "Hi." }
            } ]
        }"#;
        assert!(load_str(raw).is_err());
    }

    #[test]
    fn rejects_a_case_with_neither_source_kind() {
        let raw = r#"{
            "version": "cleanup-eval-corpus-v1",
            "description": "d",
            "cases": [ {
                "id": "bad",
                "category": "tricky",
                "lang": "en",
                "notes": "n",
                "provider_response": { "content": "Hi." },
                "expect": { "outcome": "enhanced", "text": "Hi." }
            } ]
        }"#;
        assert!(load_str(raw).is_err());
    }

    #[test]
    fn source_repeat_generates_the_expected_length() {
        let case = CorpusCase {
            id: "gen".into(),
            category: "tricky".into(),
            tags: vec![],
            lang: "en".into(),
            notes: "n".into(),
            source: None,
            source_repeat: Some(SourceRepeat {
                char: "a".into(),
                count: 5,
            }),
            provider_response: ProviderResponseSpec::Content {
                content: "x".into(),
            },
            expect: ExpectedOutcome {
                outcome: ExpectedKind::Enhanced,
                text: None,
                text_equals_source: false,
                warning_kinds: vec![],
                reason: None,
                error_kinds: vec![],
                parse_error: None,
            },
        };
        assert_eq!(case.resolved_source(), "aaaaa");
    }
}
