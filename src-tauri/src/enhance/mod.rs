// Grammar / cleanup enhancement provider.
//
// Ports the macOS GrammarEnhancer.swift. Provider is configurable:
//   - GitHub Copilot (default, uses the user's gh CLI token from keyring)
//   - Any OpenAI-compatible endpoint (base_url + api_key from Credential Manager)
//
// For now a stub that returns the input unchanged so the rest of the pipeline
// can be exercised end-to-end before we wire in HTTP.

use anyhow::Result;

pub async fn enhance(text: &str) -> Result<String> {
    Ok(text.to_string())
}
