//! `cleanup_eval` — focused, opt-in evaluator for the `cleanup` core.
//!
//! Deterministic **offline** mode (default, and the one tests/CI rely on)
//! replays a committed, versioned, fully synthetic DE/EN fixture corpus
//! through the real production `cleanup` pipeline — prompt resolution,
//! request construction, response parsing, normalization, and the integrity
//! validator — and checks the resulting decision against each fixture's
//! expectation. See `fixtures/cleanup-eval/v1/corpus.json` and
//! `README.md`'s "cleanup_eval" section for corpus format and usage.
//!
//! Optional **live** mode (`--live`, never required by tests or releases)
//! additionally sends the same corpus through a real configured provider to
//! sanity-check latency and decision distribution. Never prints secrets,
//! transcripts, prompts, or raw candidate text.
//!
//! This binary shares the production `cleanup` core with `openwritr.exe` by
//! including the same module tree (`src/cleanup/`) directly — there is no
//! library target, so `#[path]` includes are the standard way multiple
//! binaries in this package share source without duplicating logic. See
//! `src/bin/package.rs` for the existing precedent of a second, independent
//! binary target in this package.

#[path = "../../cleanup/mod.rs"]
mod cleanup;
// `credentials`, `paths`, and `single_instance` are mounted whole (see the
// module-level doc comment above) so live mode can reuse
// `credentials::read_openai_api_key()` and `paths::settings_path()` without
// duplicating them; most of their other items exist only for `openwritr`'s
// own use and are legitimately unused here, mirroring `cleanup::mod`'s own
// crate-wide `#![allow(dead_code, unused_imports)]` for the same reason.
#[path = "../../credentials.rs"]
#[allow(dead_code)]
mod credentials;
#[path = "../../paths.rs"]
#[allow(dead_code)]
mod paths;
#[path = "../../single_instance.rs"]
#[allow(dead_code)]
mod single_instance;

mod corpus;
mod eval;
mod live;
mod report;

use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use cleanup::EnhanceProvider;
use corpus::CorpusCase;
use report::{build_report, PromptProfile, Report, RunConfig};

struct Cli {
    live: bool,
    json: bool,
    category: Option<String>,
    corpus_path: Option<String>,
    provider: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    help: bool,
}

fn parse_cli(args: impl Iterator<Item = String>) -> Result<Cli, String> {
    let mut cli = Cli {
        live: false,
        json: false,
        category: None,
        corpus_path: None,
        provider: None,
        model: None,
        base_url: None,
        help: false,
    };
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--live" => cli.live = true,
            "--json" => cli.json = true,
            "-h" | "--help" => cli.help = true,
            "--category" => cli.category = Some(args.next().ok_or("--category requires a value")?),
            "--corpus" => cli.corpus_path = Some(args.next().ok_or("--corpus requires a value")?),
            "--provider" => cli.provider = Some(args.next().ok_or("--provider requires a value")?),
            "--model" => cli.model = Some(args.next().ok_or("--model requires a value")?),
            "--base-url" => cli.base_url = Some(args.next().ok_or("--base-url requires a value")?),
            other => return Err(format!("unknown argument `{other}` (see --help)")),
        }
    }
    Ok(cli)
}

const HELP: &str = "\
cleanup_eval — opt-in evaluator for the cleanup pipeline (prompt resolution,
request construction, response parsing, normalization, integrity validator).

USAGE:
    cleanup_eval [OPTIONS]

OFFLINE MODE (default; deterministic, no network, safe for CI):
    --category <name>   only evaluate fixture cases in this category
    --corpus <path>      evaluate an external corpus JSON file instead of the
                          embedded committed one (not used by tests)
    --json               print the full report as JSON instead of text

LIVE MODE (opt-in; never required by tests or releases):
    --live               additionally probe the real configured provider
    --provider <name>    github_copilot | openai_compatible (default: read
                         from settings.json, else github_copilot)
    --model <id>         model/deployment id (default: from settings.json,
                         else the compiled default)
    --base-url <url>     OpenAI-compatible base URL (default: from
                         settings.json, else https://api.openai.com/v1)

    -h, --help           print this help
";

fn main() -> ExitCode {
    let cli = match parse_cli(std::env::args().skip(1)) {
        Ok(cli) => cli,
        Err(error) => {
            eprintln!("error: {error}\n\n{HELP}");
            return ExitCode::FAILURE;
        }
    };
    if cli.help {
        println!("{HELP}");
        return ExitCode::SUCCESS;
    }

    let corpus_file = match &cli.corpus_path {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(contents) => match corpus::load_str(&contents) {
                Ok(file) => {
                    if file.version != corpus::EXPECTED_CORPUS_VERSION {
                        eprintln!(
                            "warning: corpus `{path}` has version `{}`, evaluator was written against `{}`",
                            file.version,
                            corpus::EXPECTED_CORPUS_VERSION
                        );
                    }
                    file
                }
                Err(error) => {
                    eprintln!("error: {error}");
                    return ExitCode::FAILURE;
                }
            },
            Err(error) => {
                eprintln!("error: failed to read corpus file {path}: {error}");
                return ExitCode::FAILURE;
            }
        },
        None => match corpus::load_embedded() {
            Ok(file) => file,
            Err(error) => {
                eprintln!("error: {error}");
                return ExitCode::FAILURE;
            }
        },
    };

    let cases: Vec<&CorpusCase> = corpus_file
        .cases
        .iter()
        .filter(|case| {
            cli.category
                .as_deref()
                .map(|category| case.category == category)
                .unwrap_or(true)
        })
        .collect();
    if cases.is_empty() {
        eprintln!("error: no cases matched (category filter too narrow?)");
        return ExitCode::FAILURE;
    }

    let results: Vec<eval::CaseResult> = cases.iter().map(|case| eval::run_case(case)).collect();
    let all_passed = results.iter().all(|result| result.passed);

    let live_summary = if cli.live {
        let input = resolve_live_input(&cli);
        match live::run_live(&cases, &input) {
            Ok(summary) => Some(summary),
            Err(error) => {
                eprintln!("live mode setup failed: {error}");
                None
            }
        }
    } else {
        None
    };

    let run_config = RunConfig {
        mode: if cli.live { "offline+live" } else { "offline" },
        corpus_version: corpus_file.version.clone(),
        corpus_case_count: cases.len(),
        validator_version: cleanup::VALIDATOR_VERSION,
        prompt_profile: prompt_profile(),
        category_filter: cli.category.clone(),
        app_version: env!("CARGO_PKG_VERSION"),
        target_arch: std::env::consts::ARCH,
        target_os: std::env::consts::OS,
        started_at_unix_secs: unix_now(),
        redacted: true,
    };

    let report = build_report(run_config, &results, live_summary);

    if cli.json {
        match serde_json::to_string_pretty(&report) {
            Ok(json) => println!("{json}"),
            Err(error) => eprintln!("error: failed to serialize report: {error}"),
        }
    } else {
        print_text_report(&report);
    }

    if all_passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Resolve live-mode config: CLI flags override on-disk `settings.json`
/// fields, which override compiled defaults. Never touches credentials —
/// that happens later in `live::resolve`.
fn resolve_live_input(cli: &Cli) -> live::LiveConfigInput {
    let mut resolved = live::config_from_disk_or_default(live::LiveConfigInput::default());
    if let Some(provider) = cli
        .provider
        .as_deref()
        .and_then(|s| EnhanceProvider::from_settings_str(s).ok())
    {
        resolved.provider = provider;
    }
    if let Some(model) = &cli.model {
        if !model.trim().is_empty() {
            resolved.model = model.clone();
        }
    }
    if let Some(base_url) = &cli.base_url {
        if !base_url.trim().is_empty() {
            resolved.base_url = base_url.clone();
        }
    }
    resolved
}

/// Resolve the compiled default prompt profile once for reporting: which
/// resolution layer supplied it, plus a content-free SHA-256 fingerprint so
/// prompt drift is detectable without ever printing the prompt text.
fn prompt_profile() -> PromptProfile {
    let target = cleanup::PromptTarget::github_copilot(cleanup::pipeline::DEFAULT_MODEL)
        .expect("compiled default model id is non-empty");
    let resolved = cleanup::resolve_prompt(&target, &cleanup::NoOverrides);
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(resolved.system.as_bytes());
    let sha256 = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    PromptProfile {
        layer: format!("{:?}", resolved.source),
        sha256,
        chars: resolved.system.chars().count(),
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn print_text_report(report: &Report) {
    let run = &report.run_config;
    println!("cleanup_eval report ({})", run.mode);
    println!("  app version:        {}", run.app_version);
    println!(
        "  target:              {}-{}",
        run.target_arch, run.target_os
    );
    println!(
        "  corpus:              {} ({} cases{})",
        run.corpus_version,
        run.corpus_case_count,
        run.category_filter
            .as_deref()
            .map(|c| format!(", category={c}"))
            .unwrap_or_default()
    );
    println!("  validator version:   {}", run.validator_version);
    println!(
        "  prompt profile:      {} sha256={} chars={}",
        run.prompt_profile.layer, run.prompt_profile.sha256, run.prompt_profile.chars
    );
    println!("  redacted:            {}", run.redacted);
    println!();
    println!(
        "  passed: {}  failed: {}  tricky: {}",
        report.passed, report.failed, report.tricky_case_count
    );
    println!("  decisions: {:?}", report.decision_counts);
    if !report.failing_cases.is_empty() {
        println!("  failing cases:");
        for case in &report.failing_cases {
            println!(
                "    {} (category={}, tags={:?}): {}",
                case.id,
                case.category,
                case.tags,
                case.reason.as_deref().unwrap_or("(no reason recorded)")
            );
        }
    }
    if !report.failure_categories.is_empty() {
        println!("  failure categories: {:?}", report.failure_categories);
    }
    println!(
        "  integrity error kinds:   {:?}",
        report.integrity_error_kind_counts
    );
    println!(
        "  integrity warning kinds: {:?}",
        report.integrity_warning_kind_counts
    );
    println!("  category breakdown:");
    for (category, breakdown) in &report.category_breakdown {
        println!(
            "    {category}: total={} passed={} failed={}",
            breakdown.total, breakdown.passed, breakdown.failed
        );
    }
    println!("  scores:");
    for (name, score) in &report.scores {
        println!("    {name}: {:.3} (sampled={})", score.score, score.sampled);
    }
    println!("  avg final text chars: {:.1}", report.avg_final_text_chars);
    if let Some(live) = &report.live {
        println!();
        println!("  live mode:");
        println!(
            "    provider={} model={} endpoint={} credential_source={}",
            live.config.provider,
            live.config.model,
            live.config
                .endpoint_scope
                .as_deref()
                .unwrap_or("(none — fixed Copilot endpoint)"),
            live.config.credential_source
        );
        println!(
            "    attempted={} succeeded={} failed={}",
            live.attempted, live.succeeded, live.failed
        );
        println!("    decisions: {:?}", live.decision_counts);
        if !live.failure_categories.is_empty() {
            println!("    failure categories: {:?}", live.failure_categories);
        }
        if let Some(latency) = &live.latency_ms {
            println!(
                "    latency ms: min={:.1} max={:.1} mean={:.1} samples={}",
                latency.min_ms, latency.max_ms, latency.mean_ms, latency.samples
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const REQUIRED_CATEGORIES: &[&str] = &[
        "punctuation",
        "filler",
        "repetition",
        "negation",
        "digits_dates_times_versions",
        "commands_code_acronyms",
        "urls_emails",
        "language_preservation",
        "filler_only_empty",
        "tricky",
    ];

    fn embedded_results() -> Vec<eval::CaseResult> {
        let file = corpus::load_embedded().expect("embedded corpus must parse");
        file.cases.iter().map(eval::run_case).collect()
    }

    #[test]
    fn corpus_version_is_pinned() {
        let file = corpus::load_embedded().unwrap();
        assert_eq!(file.version, corpus::EXPECTED_CORPUS_VERSION);
    }

    #[test]
    fn every_required_category_is_covered() {
        let file = corpus::load_embedded().unwrap();
        let present: HashSet<&str> = file
            .cases
            .iter()
            .map(|case| case.category.as_str())
            .collect();
        for required in REQUIRED_CATEGORIES {
            assert!(
                present.contains(required),
                "corpus is missing required category `{required}`"
            );
        }
    }

    #[test]
    fn every_fixture_case_matches_its_expected_decision() {
        let results = embedded_results();
        let failing: Vec<&str> = results
            .iter()
            .filter(|result| !result.passed)
            .map(|result| result.id.as_str())
            .collect();
        assert!(
            failing.is_empty(),
            "cases failed to match their expected decision: {failing:?}"
        );
    }

    #[test]
    fn at_least_one_tricky_case_is_present_and_documented() {
        let file = corpus::load_embedded().unwrap();
        let tricky_count = file.cases.iter().filter(|case| case.is_tricky()).count();
        assert!(
            tricky_count >= 3,
            "expected several documented tricky cases"
        );
    }

    #[test]
    fn serialized_report_never_contains_fixture_source_or_candidate_text() {
        let file = corpus::load_embedded().unwrap();
        let results = embedded_results();
        let run_config = RunConfig {
            mode: "offline",
            corpus_version: file.version.clone(),
            corpus_case_count: results.len(),
            validator_version: cleanup::VALIDATOR_VERSION,
            prompt_profile: prompt_profile(),
            category_filter: None,
            app_version: env!("CARGO_PKG_VERSION"),
            target_arch: std::env::consts::ARCH,
            target_os: std::env::consts::OS,
            started_at_unix_secs: 0,
            redacted: true,
        };
        let report = build_report(run_config, &results, None);
        let json = serde_json::to_string(&report).unwrap();

        // No forbidden field names.
        for forbidden in [
            "\"source\"",
            "\"candidate\"",
            "\"prompt\"",
            "\"token\"",
            "\"api_key\"",
            "\"secret\"",
        ] {
            assert!(
                !json.contains(forbidden),
                "serialized report must never contain a `{forbidden}` field"
            );
        }

        // No literal fixture text leaked into the report.
        for case in &file.cases {
            if let Some(source) = &case.source {
                if source.chars().count() > 3 {
                    assert!(
                        !json.contains(source.as_str()),
                        "serialized report leaked source text for case `{}`",
                        case.id
                    );
                }
            }
            if let corpus::ProviderResponseSpec::Content { content } = &case.provider_response {
                if content.chars().count() > 3 && content != "[[EMPTY]]" {
                    assert!(
                        !json.contains(content.as_str()),
                        "serialized report leaked candidate text for case `{}`",
                        case.id
                    );
                }
            }
        }
    }

    #[test]
    fn harness_delegates_to_the_real_pipeline_rather_than_reimplementing_it() {
        // Spot-check: directly call the production modules the same way
        // `eval::evaluate_case` does, and assert the harness produced the
        // identical decision. If the harness ever drifted into a parallel
        // reimplementation, this would catch it.
        let source = "transfer 42 dollars";
        let response = serde_json::json!({
            "choices": [ { "message": { "content": "Transfer 24 dollars." } } ]
        });
        let direct_content = cleanup::adapter::parse_chat_response(&response).unwrap();
        let direct_outcome = cleanup::pipeline::finalize(source, &direct_content);

        let harness_outcome = eval::evaluate_case(source, &response);
        match (direct_outcome, harness_outcome) {
            (
                cleanup::EnhanceOutcome::RawFallback {
                    text: direct_text,
                    reason: cleanup::FallbackReason::IntegrityRejected(direct_kinds),
                },
                eval::CaseOutcome::RawFallback {
                    text: harness_text,
                    error_kinds: harness_kinds,
                    ..
                },
            ) => {
                assert_eq!(direct_text, harness_text);
                let direct_kind_labels: Vec<String> = direct_kinds
                    .iter()
                    .map(|kind| format!("{kind:?}"))
                    .collect();
                assert_eq!(direct_kind_labels, harness_kinds);
            }
            other => panic!("expected matching IntegrityRejected outcomes, got {other:?}"),
        }
    }

    #[test]
    fn cli_parses_known_flags_and_rejects_unknown_ones() {
        let cli = parse_cli(
            [
                "--live",
                "--json",
                "--category",
                "digits_dates_times_versions",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert!(cli.live);
        assert!(cli.json);
        assert_eq!(cli.category.as_deref(), Some("digits_dates_times_versions"));

        assert!(parse_cli(["--nope".to_string()].into_iter()).is_err());
    }
}
