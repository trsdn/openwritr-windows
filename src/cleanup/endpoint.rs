//! Strict, canonical OpenAI-compatible endpoint scope parser.
//!
//! There is exactly one way to turn a user-supplied base URL into an
//! [`EndpointScope`]. Canonicalization: lowercase scheme and host, drop the
//! scheme's default port, strip a trailing slash from the path. Credentials,
//! fragments, and query strings are rejected rather than silently dropped so
//! two settings that look different can never collapse into the same identity
//! by accident.
//!
//! The scope is hashable and serializable (as its canonical string) so it can
//! be used as part of a prompt-target key.

use std::fmt;

use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

/// A canonicalized OpenAI-compatible endpoint base URL.
///
/// Construct via [`EndpointScope::parse`]; the fields are always canonical.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct EndpointScope {
    scheme: String,
    host: String,
    /// `None` when the port equals the scheme default (80/443).
    port: Option<u16>,
    /// Normalized path with any trailing slash removed. Empty for a bare host.
    path: String,
}

impl EndpointScope {
    /// Parse and canonicalize a base URL into an endpoint scope.
    pub fn parse(raw: &str) -> Result<Self, EndpointScopeError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(EndpointScopeError::Empty);
        }
        let url = reqwest::Url::parse(trimmed).map_err(|_| EndpointScopeError::Malformed)?;

        let scheme = url.scheme().to_ascii_lowercase();
        if scheme != "http" && scheme != "https" {
            return Err(EndpointScopeError::UnsupportedScheme(scheme));
        }

        if !url.username().is_empty() || url.password().is_some() {
            return Err(EndpointScopeError::CredentialsPresent);
        }
        if url.fragment().is_some() {
            return Err(EndpointScopeError::FragmentPresent);
        }
        if url.query().is_some() {
            return Err(EndpointScopeError::QueryPresent);
        }

        let host = url
            .host_str()
            .ok_or(EndpointScopeError::MissingHost)?
            .to_ascii_lowercase();
        if host.is_empty() {
            return Err(EndpointScopeError::MissingHost);
        }

        let default_port = if scheme == "https" { 443 } else { 80 };
        let port = match url.port() {
            Some(port) if port == default_port => None,
            other => other,
        };

        let path = url.path().trim_end_matches('/').to_string();

        Ok(EndpointScope {
            scheme,
            host,
            port,
            path,
        })
    }

    pub fn scheme(&self) -> &str {
        &self.scheme
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn port(&self) -> Option<u16> {
        self.port
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    /// The canonical base URL string (no trailing slash).
    pub fn base_url(&self) -> String {
        let mut out = format!("{}://{}", self.scheme, self.host);
        if let Some(port) = self.port {
            out.push(':');
            out.push_str(&port.to_string());
        }
        out.push_str(&self.path);
        out
    }

    /// Join a path suffix (e.g. `"/chat/completions"`) onto the canonical base.
    pub fn join(&self, suffix: &str) -> String {
        let mut out = self.base_url();
        if !suffix.is_empty() && !suffix.starts_with('/') {
            out.push('/');
        }
        out.push_str(suffix);
        out
    }
}

impl fmt::Debug for EndpointScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("EndpointScope")
            .field(&self.base_url())
            .finish()
    }
}

impl fmt::Display for EndpointScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.base_url())
    }
}

impl Serialize for EndpointScope {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.base_url())
    }
}

impl<'de> Deserialize<'de> for EndpointScope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        EndpointScope::parse(&raw).map_err(D::Error::custom)
    }
}

/// Why a base URL could not be canonicalized. Contains only structural detail
/// about the URL, never transcript or secret data.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum EndpointScopeError {
    #[error("endpoint URL is empty")]
    Empty,
    #[error("endpoint URL is malformed")]
    Malformed,
    #[error("endpoint scheme `{0}` is not http or https")]
    UnsupportedScheme(String),
    #[error("endpoint URL must not embed credentials")]
    CredentialsPresent,
    #[error("endpoint URL must not contain a fragment")]
    FragmentPresent,
    #[error("endpoint URL must not contain a query string")]
    QueryPresent,
    #[error("endpoint URL is missing a host")]
    MissingHost,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden canonicalization table: raw input -> canonical base URL.
    #[test]
    fn canonicalizes_to_golden_base_urls() {
        let cases = [
            ("https://api.openai.com/v1", "https://api.openai.com/v1"),
            ("https://api.openai.com/v1/", "https://api.openai.com/v1"),
            ("HTTPS://API.OpenAI.COM/v1", "https://api.openai.com/v1"),
            ("https://api.openai.com", "https://api.openai.com"),
            ("https://api.openai.com/", "https://api.openai.com"),
            ("https://api.openai.com:443/v1", "https://api.openai.com/v1"),
            ("http://localhost:80/v1", "http://localhost/v1"),
            ("http://localhost:11434/v1", "http://localhost:11434/v1"),
            (
                "  https://api.openai.com/v1/  ",
                "https://api.openai.com/v1",
            ),
            (
                "https://Example.COM:8443/OpenAI/V1/",
                "https://example.com:8443/OpenAI/V1",
            ),
            ("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
        ];
        for (raw, expected) in cases {
            let scope = EndpointScope::parse(raw)
                .unwrap_or_else(|error| panic!("{raw:?} should parse but got {error:?}"));
            assert_eq!(scope.base_url(), expected, "input {raw:?}");
        }
    }

    #[test]
    fn path_case_is_preserved_but_scheme_and_host_are_lowercased() {
        let scope = EndpointScope::parse("HTTPS://API.Example.com/OpenAI/V1").unwrap();
        assert_eq!(scope.scheme(), "https");
        assert_eq!(scope.host(), "api.example.com");
        assert_eq!(scope.path(), "/OpenAI/V1");
        assert_eq!(scope.port(), None);
    }

    #[test]
    fn default_ports_are_dropped_and_custom_ports_kept() {
        assert_eq!(
            EndpointScope::parse("https://h.example:443/v1")
                .unwrap()
                .port(),
            None
        );
        assert_eq!(
            EndpointScope::parse("http://h.example:80/v1")
                .unwrap()
                .port(),
            None
        );
        assert_eq!(
            EndpointScope::parse("https://h.example:8443/v1")
                .unwrap()
                .port(),
            Some(8443)
        );
    }

    #[test]
    fn rejects_credentials_fragments_and_queries() {
        assert_eq!(
            EndpointScope::parse("https://user:pass@api.example.com/v1").unwrap_err(),
            EndpointScopeError::CredentialsPresent
        );
        assert_eq!(
            EndpointScope::parse("https://user@api.example.com/v1").unwrap_err(),
            EndpointScopeError::CredentialsPresent
        );
        assert_eq!(
            EndpointScope::parse("https://api.example.com/v1#frag").unwrap_err(),
            EndpointScopeError::FragmentPresent
        );
        assert_eq!(
            EndpointScope::parse("https://api.example.com/v1?key=1").unwrap_err(),
            EndpointScopeError::QueryPresent
        );
    }

    #[test]
    fn rejects_non_http_schemes_and_empty_input() {
        assert_eq!(
            EndpointScope::parse("ftp://api.example.com/v1").unwrap_err(),
            EndpointScopeError::UnsupportedScheme("ftp".to_string())
        );
        assert_eq!(
            EndpointScope::parse("   ").unwrap_err(),
            EndpointScopeError::Empty
        );
        assert_eq!(
            EndpointScope::parse("not a url").unwrap_err(),
            EndpointScopeError::Malformed
        );
    }

    #[test]
    fn equal_inputs_hash_and_compare_equal_after_canonicalization() {
        use std::collections::HashSet;
        let a = EndpointScope::parse("https://API.OpenAI.com/v1/").unwrap();
        let b = EndpointScope::parse("https://api.openai.com:443/v1").unwrap();
        assert_eq!(a, b);
        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
    }

    #[test]
    fn join_appends_a_single_separator() {
        let scope = EndpointScope::parse("https://api.openai.com/v1/").unwrap();
        assert_eq!(
            scope.join("/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            scope.join("chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn serializes_as_canonical_string_and_round_trips() {
        let scope = EndpointScope::parse("HTTPS://API.OpenAI.com/v1/").unwrap();
        let json = serde_json::to_string(&scope).unwrap();
        assert_eq!(json, "\"https://api.openai.com/v1\"");
        let back: EndpointScope = serde_json::from_str(&json).unwrap();
        assert_eq!(back, scope);
    }

    #[test]
    fn deserializing_non_canonical_string_canonicalizes() {
        let back: EndpointScope = serde_json::from_str("\"https://API.OpenAI.com/v1/\"").unwrap();
        assert_eq!(back.base_url(), "https://api.openai.com/v1");
    }
}
