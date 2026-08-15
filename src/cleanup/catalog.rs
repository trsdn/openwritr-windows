//! Advisory model catalog.
//!
//! The catalog is a set of *hints*: display names and capability metadata for a
//! handful of well-known models. Arbitrary model IDs remain fully supported —
//! anything not in the catalog falls back to [`ModelCapabilities::DEFAULT`].
//!
//! No pricing is encoded here. Pricing changes frequently and is provider- and
//! contract-specific; hardcoding it would be stale on arrival.

/// What request parameters a model is known to accept.
///
/// This lets catalog additions declare, per model, whether they accept a custom
/// sampling temperature or a `system` role message. Adapters use this so we
/// never blindly send parameters a model rejects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelCapabilities {
    /// Model accepts a caller-chosen sampling temperature.
    pub accepts_temperature: bool,
    /// Model accepts a dedicated `system` role message.
    pub accepts_system_message: bool,
}

impl ModelCapabilities {
    /// Conservative default for unknown/arbitrary model IDs: assume the model
    /// accepts the standard chat parameters. This preserves the historical
    /// behavior of always sending `temperature` + a `system` message.
    pub const DEFAULT: ModelCapabilities = ModelCapabilities {
        accepts_temperature: true,
        accepts_system_message: true,
    };

    /// Reasoning-style models that only run at their fixed default temperature
    /// but still accept a system message.
    const FIXED_TEMPERATURE: ModelCapabilities = ModelCapabilities {
        accepts_temperature: false,
        accepts_system_message: true,
    };
}

/// A catalog entry: an advisory display name and capability hints for a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CatalogModel {
    /// Canonical model ID as sent on the wire.
    pub id: &'static str,
    /// Human-friendly label for pickers.
    pub display_name: &'static str,
    /// Known request capabilities.
    pub capabilities: ModelCapabilities,
}

const CATALOG: &[CatalogModel] = &[
    CatalogModel {
        id: "gpt-5.6-luna",
        display_name: "GPT-5.6 Luna",
        capabilities: ModelCapabilities::FIXED_TEMPERATURE,
    },
    CatalogModel {
        id: "gemini-3.7-flash",
        display_name: "Gemini 3.7 Flash",
        capabilities: ModelCapabilities::DEFAULT,
    },
    CatalogModel {
        id: "mai-code-1.1-flash",
        display_name: "MAI Code 1.1 Flash",
        capabilities: ModelCapabilities::DEFAULT,
    },
    CatalogModel {
        id: "gpt-5-mini",
        display_name: "GPT-5 Mini",
        capabilities: ModelCapabilities::FIXED_TEMPERATURE,
    },
    CatalogModel {
        id: "claude-haiku-4.5",
        display_name: "Claude Haiku 4.5",
        capabilities: ModelCapabilities::DEFAULT,
    },
];

/// Every catalog entry, in stable display order.
pub fn all() -> &'static [CatalogModel] {
    CATALOG
}

/// Look up a catalog entry by exact model ID. The caller is expected to trim
/// the ID first; matching is exact.
pub fn lookup(model_id: &str) -> Option<&'static CatalogModel> {
    CATALOG.iter().find(|model| model.id == model_id)
}

/// Capabilities for a model ID, falling back to [`ModelCapabilities::DEFAULT`]
/// for anything not in the catalog.
pub fn capabilities_for(model_id: &str) -> ModelCapabilities {
    lookup(model_id)
        .map(|model| model.capabilities)
        .unwrap_or(ModelCapabilities::DEFAULT)
}

/// Advisory display name for a model ID, or `None` for arbitrary IDs.
pub fn display_name_for(model_id: &str) -> Option<&'static str> {
    lookup(model_id).map(|model| model.display_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_contains_the_expected_models() {
        let ids: Vec<&str> = all().iter().map(|model| model.id).collect();
        assert_eq!(
            ids,
            vec![
                "gpt-5.6-luna",
                "gemini-3.7-flash",
                "mai-code-1.1-flash",
                "gpt-5-mini",
                "claude-haiku-4.5",
            ]
        );
    }

    #[test]
    fn display_names_are_advisory_and_present() {
        assert_eq!(display_name_for("gpt-5.6-luna"), Some("GPT-5.6 Luna"));
        assert_eq!(
            display_name_for("gemini-3.7-flash"),
            Some("Gemini 3.7 Flash")
        );
        assert_eq!(
            display_name_for("mai-code-1.1-flash"),
            Some("MAI Code 1.1 Flash")
        );
        assert_eq!(display_name_for("gpt-5-mini"), Some("GPT-5 Mini"));
        assert_eq!(
            display_name_for("claude-haiku-4.5"),
            Some("Claude Haiku 4.5")
        );
    }

    #[test]
    fn arbitrary_model_ids_are_supported_with_default_capabilities() {
        assert!(lookup("some-future-model-2099").is_none());
        assert_eq!(
            capabilities_for("some-future-model-2099"),
            ModelCapabilities::DEFAULT
        );
        assert!(display_name_for("some-future-model-2099").is_none());
    }

    #[test]
    fn capabilities_are_not_uniform_across_the_catalog() {
        // The whole point of capability metadata: catalog additions must not
        // assume every model accepts the same temperature/system parameters.
        assert!(!capabilities_for("gpt-5-mini").accepts_temperature);
        assert!(!capabilities_for("gpt-5.6-luna").accepts_temperature);
        assert!(capabilities_for("claude-haiku-4.5").accepts_temperature);
        assert!(capabilities_for("gemini-3.7-flash").accepts_temperature);
    }

    #[test]
    fn lookup_is_exact_and_trims_are_the_callers_job() {
        assert!(lookup(" gpt-5-mini ").is_none());
        assert!(lookup("gpt-5-mini").is_some());
    }
}
