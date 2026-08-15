//! Output normalization for enhancement responses.
//!
//! Normalization trims surrounding whitespace and recognizes the `[[EMPTY]]`
//! sentinel — the value a model is asked to return when the transcript has no
//! meaningful content — collapsing it to an empty string.

/// Sentinel a model returns to signal "there was nothing to clean up".
pub const EMPTY_SENTINEL: &str = "[[EMPTY]]";

/// Normalize a raw model response into candidate output text.
///
/// * Surrounding whitespace is trimmed.
/// * The `[[EMPTY]]` sentinel (after trimming) becomes an empty string.
pub fn normalize_output(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed == EMPTY_SENTINEL {
        return String::new();
    }
    trimmed.to_string()
}

/// Whether the raw model response is (after trimming) the empty sentinel.
pub fn is_empty_sentinel(raw: &str) -> bool {
    raw.trim() == EMPTY_SENTINEL
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_surrounding_whitespace() {
        assert_eq!(normalize_output("  hello world  \n"), "hello world");
    }

    #[test]
    fn empty_sentinel_becomes_empty_string() {
        assert_eq!(normalize_output("[[EMPTY]]"), "");
        assert_eq!(normalize_output("  [[EMPTY]]\n"), "");
        assert!(is_empty_sentinel("  [[EMPTY]]  "));
    }

    #[test]
    fn sentinel_must_be_exact() {
        // Extra content around the sentinel means it is not a sentinel.
        assert_eq!(normalize_output("[[EMPTY]] and more"), "[[EMPTY]] and more");
        assert!(!is_empty_sentinel("[[EMPTY]] and more"));
        assert_eq!(normalize_output("[[empty]]"), "[[empty]]");
    }

    #[test]
    fn whitespace_only_stays_empty() {
        assert_eq!(normalize_output("   \n\t "), "");
    }
}
