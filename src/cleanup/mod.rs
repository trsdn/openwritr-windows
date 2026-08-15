//! Typed enhancement/cleanup core.
//!
//! This module replaces the ad-hoc, stringly enhancement handling with typed,
//! composable primitives:
//!
//! * [`provider`] — the typed [`EnhanceProvider`] with frozen wire values,
//! * [`endpoint`] — the strict canonical OpenAI [`EndpointScope`] parser,
//! * [`catalog`] — advisory model metadata and capability hints,
//! * [`prompt`] — prompt target identity, the untrusted [`Transcript`], and
//!   layered prompt resolution,
//! * [`adapter`] — request construction and response parsing boundaries,
//! * [`normalize`] — output normalization (incl. the `[[EMPTY]]` sentinel),
//! * [`integrity`] — the conservative, pure integrity validator,
//! * [`outcome`] — typed outcomes/fallbacks/skips for worker integration,
//! * [`pipeline`] — the network-free assembly that ties it all together.
//!
//! The `enhance` facade builds on these while keeping its existing public API,
//! so the crate compiles before the settings/worker integration lands.
//!
//! Several items are used only by tests or by the pending integration task, so
//! `dead_code`/`unused_imports` are allowed module-wide: the re-exports below
//! form the stable public surface the settings/worker integration will consume.

#![allow(dead_code, unused_imports)]

pub mod adapter;
pub mod catalog;
pub mod endpoint;
pub mod integrity;
pub mod normalize;
pub mod outcome;
pub mod pipeline;
pub mod prompt;
pub mod provider;

pub use adapter::{ChatRequestOptions, ResponseParseError};
pub use catalog::{CatalogModel, ModelCapabilities};
pub use endpoint::{EndpointScope, EndpointScopeError};
pub use integrity::{
    validate as validate_integrity, IntegrityReport, Severity, ValidatorError, Violation,
    ViolationKind, VALIDATOR_VERSION,
};
pub use normalize::{is_empty_sentinel, normalize_output, EMPTY_SENTINEL};
pub use outcome::{EnhanceOutcome, FallbackReason, SkipReason};
pub use pipeline::{
    build_request, chat_completions_url, effective_model_id, finalize, DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
};
pub use prompt::{
    resolve_prompt, NoOverrides, PromptOverrides, PromptSource, PromptTarget, PromptTargetError,
    ResolvedPrompt, Transcript, GLOBAL_DEFAULT_SYSTEM,
};
pub use provider::{EnhanceProvider, UnknownProvider, COPILOT_CHAT_COMPLETIONS_URL};
