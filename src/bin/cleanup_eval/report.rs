//! The evaluator's report: complete, non-secret run configuration plus
//! aggregate, content-free results. Every field here is safe to print or
//! commit to CI logs — no transcript, prompt, candidate, or credential text
//! ever reaches this struct.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::eval::{score_results, tally_violation_kinds, CaseResult, ScoreSample};

#[derive(Serialize, Clone)]
pub struct PromptProfile {
    /// Which resolution layer supplied the prompt (e.g. `"GlobalDefault"`),
    /// from `cleanup::PromptSource`'s debug label.
    pub layer: String,
    /// SHA-256 of the resolved system prompt text — a stable fingerprint for
    /// detecting prompt drift without ever printing the prompt itself.
    pub sha256: String,
    pub chars: usize,
}

#[derive(Serialize)]
pub struct ScoreSummary {
    /// Fraction in `[0, 1]` of sampled `Enhanced` outcomes with the property.
    pub score: f64,
    pub sampled: usize,
}

#[derive(Serialize, Default)]
pub struct CategoryBreakdown {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
}

#[derive(Serialize)]
pub struct LatencySummary {
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub total_ms: f64,
    pub samples: usize,
}

#[derive(Serialize)]
pub struct RunConfig {
    /// `"offline"` or `"offline+live"`.
    pub mode: &'static str,
    pub corpus_version: String,
    pub corpus_case_count: usize,
    pub validator_version: &'static str,
    pub prompt_profile: PromptProfile,
    pub category_filter: Option<String>,
    pub app_version: &'static str,
    pub target_arch: &'static str,
    pub target_os: &'static str,
    pub started_at_unix_secs: u64,
    /// Always `true`: transcript/prompt/candidate content is never included
    /// in this report, regardless of mode.
    pub redacted: bool,
}

#[derive(Serialize)]
pub struct LiveConfig {
    pub provider: String,
    pub model: String,
    /// Canonical `EndpointScope::base_url()` — never includes credentials
    /// (the parser rejects endpoint URLs with embedded credentials).
    pub endpoint_scope: Option<String>,
    pub credential_source: &'static str,
}

#[derive(Serialize)]
pub struct LiveSummary {
    pub config: LiveConfig,
    pub attempted: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub decision_counts: BTreeMap<String, usize>,
    pub failure_categories: BTreeMap<String, usize>,
    pub latency_ms: Option<LatencySummary>,
}

#[derive(Serialize)]
pub struct FailingCase {
    pub id: String,
    pub category: String,
    pub tags: Vec<String>,
    /// Redacted, structural mismatch reason (e.g. "expected warning kinds
    /// [...], got [...]") — never source/candidate text.
    pub reason: Option<String>,
}

#[derive(Serialize)]
pub struct Report {
    pub run_config: RunConfig,
    pub decision_counts: BTreeMap<String, usize>,
    pub passed: usize,
    pub failed: usize,
    pub failing_cases: Vec<FailingCase>,
    pub tricky_case_count: usize,
    pub failure_categories: BTreeMap<String, usize>,
    pub integrity_error_kind_counts: BTreeMap<String, usize>,
    pub integrity_warning_kind_counts: BTreeMap<String, usize>,
    pub category_breakdown: BTreeMap<String, CategoryBreakdown>,
    pub scores: BTreeMap<String, ScoreSummary>,
    /// Mean length (in chars) of each case's final text — a coarse, content-
    /// free size signal, never the text itself.
    pub avg_final_text_chars: f64,
    pub live: Option<LiveSummary>,
}

pub fn build_report(
    run_config: RunConfig,
    results: &[CaseResult],
    live: Option<LiveSummary>,
) -> Report {
    let mut decision_counts = BTreeMap::new();
    let mut failure_categories = BTreeMap::new();
    let mut category_breakdown: BTreeMap<String, CategoryBreakdown> = BTreeMap::new();
    let mut failing_cases = Vec::new();
    let mut tricky_case_count = 0usize;
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut total_final_text_chars = 0usize;

    for result in results {
        *decision_counts
            .entry(result.outcome_tag.to_string())
            .or_insert(0) += 1;
        if result.tricky {
            tricky_case_count += 1;
        }
        total_final_text_chars += result.final_text_chars;
        let entry = category_breakdown
            .entry(result.category.clone())
            .or_default();
        entry.total += 1;
        if result.passed {
            passed += 1;
            entry.passed += 1;
        } else {
            failed += 1;
            entry.failed += 1;
            failing_cases.push(FailingCase {
                id: result.id.clone(),
                category: result.category.clone(),
                tags: result.tags.clone(),
                reason: result.mismatch.clone(),
            });
            if let Some(reason) = &result.fallback_reason {
                *failure_categories.entry(reason.clone()).or_insert(0) += 1;
            }
            if let Some(error) = &result.parse_error {
                *failure_categories.entry(error.clone()).or_insert(0) += 1;
            }
        }
    }

    let (integrity_error_kind_counts, integrity_warning_kind_counts) =
        tally_violation_kinds(results);

    let samples = score_results(results);
    let scores = build_scores(&samples);
    let avg_final_text_chars = if results.is_empty() {
        0.0
    } else {
        total_final_text_chars as f64 / results.len() as f64
    };

    Report {
        run_config,
        decision_counts,
        passed,
        failed,
        failing_cases,
        tricky_case_count,
        failure_categories,
        integrity_error_kind_counts,
        integrity_warning_kind_counts,
        category_breakdown,
        scores,
        avg_final_text_chars,
        live,
    }
}

fn build_scores(samples: &[ScoreSample]) -> BTreeMap<String, ScoreSummary> {
    let sampled = samples.len();
    let ratio = |count: usize| -> f64 {
        if sampled == 0 {
            1.0
        } else {
            count as f64 / sampled as f64
        }
    };
    let punctuation = samples.iter().filter(|s| s.punctuation_ok).count();
    let filler = samples.iter().filter(|s| s.filler_ok).count();
    let repetition = samples.iter().filter(|s| s.repetition_ok).count();

    let mut scores = BTreeMap::new();
    scores.insert(
        "punctuation".to_string(),
        ScoreSummary {
            score: ratio(punctuation),
            sampled,
        },
    );
    scores.insert(
        "filler".to_string(),
        ScoreSummary {
            score: ratio(filler),
            sampled,
        },
    );
    scores.insert(
        "repetition".to_string(),
        ScoreSummary {
            score: ratio(repetition),
            sampled,
        },
    );
    scores
}
