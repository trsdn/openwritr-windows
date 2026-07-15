//! Pinned multilingual Whisper tokenizer and decoder prompts.

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use tokenizers::Tokenizer;

pub const TOKEN_EOT: i32 = 50_257;
pub const TOKEN_SOT: i32 = 50_258;
pub const LANGUAGE_TOKEN_START: i32 = 50_259;
pub const LANGUAGE_TOKEN_END: i32 = 50_358;
pub const TOKEN_TRANSLATE: i32 = 50_359;
pub const TOKEN_TRANSCRIBE: i32 = 50_360;
pub const TOKEN_NO_TIMESTAMPS: i32 = 50_364;

const SPECIAL_TOKENS: &[(&str, i32)] = &[
    ("<|endoftext|>", TOKEN_EOT),
    ("<|startoftranscript|>", TOKEN_SOT),
    ("<|translate|>", TOKEN_TRANSLATE),
    ("<|transcribe|>", TOKEN_TRANSCRIBE),
    ("<|notimestamps|>", TOKEN_NO_TIMESTAMPS),
];

#[derive(Deserialize)]
struct GenerationConfig {
    bos_token_id: u32,
    decoder_start_token_id: u32,
    eos_token_id: u32,
    forced_decoder_ids: Vec<(u32, Option<u32>)>,
    is_multilingual: bool,
    lang_to_id: HashMap<String, u32>,
    no_timestamps_token_id: u32,
    pad_token_id: u32,
    task_to_id: HashMap<String, u32>,
}

pub struct WhisperTokenizer {
    tokenizer: Tokenizer,
    language_codes: HashMap<i32, String>,
}

impl WhisperTokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| anyhow!("load {}: {error}", tokenizer_path.display()))?;
        let config_path = model_dir.join("generation_config.json");
        let config: GenerationConfig = serde_json::from_slice(
            &std::fs::read(&config_path)
                .with_context(|| format!("read {}", config_path.display()))?,
        )
        .with_context(|| format!("parse {}", config_path.display()))?;
        validate_pinned_config(&tokenizer, &config)?;

        let language_codes = config
            .lang_to_id
            .into_iter()
            .map(|(token, id)| {
                let code = token
                    .strip_prefix("<|")
                    .and_then(|value| value.strip_suffix("|>"))
                    .ok_or_else(|| anyhow!("invalid Whisper language token {token}"))?;
                Ok((id as i32, code.to_string()))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Self {
            tokenizer,
            language_codes,
        })
    }

    pub fn language_code(&self, token: i32) -> Option<&str> {
        self.language_codes.get(&token).map(String::as_str)
    }

    pub fn decode(&self, tokens: &[i32]) -> Result<String> {
        let tokens = tokens
            .iter()
            .map(|token| {
                u32::try_from(*token)
                    .map_err(|_| anyhow!("Whisper produced a negative token ID {token}"))
            })
            .collect::<Result<Vec<_>>>()?;
        self.tokenizer
            .decode(&tokens, true)
            .map(|text| text.trim().to_string())
            .map_err(|error| anyhow!("decode Whisper tokens: {error}"))
    }
}

fn validate_pinned_config(tokenizer: &Tokenizer, config: &GenerationConfig) -> Result<()> {
    require_id("bos_token_id", config.bos_token_id, TOKEN_EOT)?;
    require_id(
        "decoder_start_token_id",
        config.decoder_start_token_id,
        TOKEN_SOT,
    )?;
    require_id("eos_token_id", config.eos_token_id, TOKEN_EOT)?;
    require_id("pad_token_id", config.pad_token_id, TOKEN_EOT)?;
    require_id(
        "no_timestamps_token_id",
        config.no_timestamps_token_id,
        TOKEN_NO_TIMESTAMPS,
    )?;
    if !config.is_multilingual {
        bail!("Whisper generation config is not multilingual");
    }
    if config.task_to_id.get("transcribe").copied() != Some(TOKEN_TRANSCRIBE as u32)
        || config.task_to_id.get("translate").copied() != Some(TOKEN_TRANSLATE as u32)
    {
        bail!("Whisper generation config has unexpected task token IDs");
    }
    if config.forced_decoder_ids.as_slice() != [(1, None), (2, Some(TOKEN_TRANSCRIBE as u32))] {
        bail!("Whisper generation config has unexpected forced decoder IDs");
    }

    for (token, expected) in SPECIAL_TOKENS {
        let actual = tokenizer
            .token_to_id(token)
            .ok_or_else(|| anyhow!("Whisper tokenizer is missing {token}"))?;
        require_id(token, actual, *expected)?;
    }
    if tokenizer.token_to_id("<|en|>") != Some(LANGUAGE_TOKEN_START as u32)
        || tokenizer.token_to_id("<|yue|>") != Some(LANGUAGE_TOKEN_END as u32)
    {
        bail!("Whisper tokenizer language token range is not pinned");
    }

    let language_ids = config
        .lang_to_id
        .iter()
        .map(|(token, id)| {
            if tokenizer.token_to_id(token) != Some(*id) {
                bail!("Whisper tokenizer/config disagree on language token {token}");
            }
            i32::try_from(*id).map_err(|_| anyhow!("language token {token} exceeds i32"))
        })
        .collect::<Result<HashSet<_>>>()?;
    let expected_language_ids = (LANGUAGE_TOKEN_START..=LANGUAGE_TOKEN_END).collect::<HashSet<_>>();
    if language_ids != expected_language_ids {
        bail!("Whisper language token IDs are not the expected contiguous range");
    }
    Ok(())
}

fn require_id(name: &str, actual: u32, expected: i32) -> Result<()> {
    if actual != expected as u32 {
        bail!("Whisper {name} mismatch: expected {expected}, got {actual}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_manager::{CancellationToken, ModelManager};

    #[test]
    fn resolves_only_pinned_language_tokens() {
        let tokenizer = WhisperTokenizer {
            tokenizer: Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
            language_codes: [
                (LANGUAGE_TOKEN_START, "en".to_string()),
                (LANGUAGE_TOKEN_END, "yue".to_string()),
            ]
            .into_iter()
            .collect(),
        };

        assert_eq!(tokenizer.language_code(LANGUAGE_TOKEN_START), Some("en"));
        assert_eq!(tokenizer.language_code(TOKEN_TRANSCRIBE), None);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "loads the real pinned Whisper tokenizer assets"]
    fn decodes_reference_vectors_with_the_pinned_tokenizer() {
        let models = ModelManager::new().unwrap();
        let model_dir = models
            .ensure("whisper_npu", &CancellationToken::default(), |_| {})
            .unwrap();
        let tokenizer = WhisperTokenizer::load(&model_dir).unwrap();

        assert_eq!(
            tokenizer
                .decode(&[
                    TOKEN_SOT,
                    LANGUAGE_TOKEN_START,
                    TOKEN_TRANSCRIBE,
                    TOKEN_NO_TIMESTAMPS,
                    2_425,
                    11,
                    1_002,
                    0,
                    TOKEN_EOT,
                ])
                .unwrap(),
            "Hello, world!"
        );
        assert_eq!(
            tokenizer.decode(&[38_908, 2_536, 10_390, 13]).unwrap(),
            "Grüß dich."
        );
        assert_eq!(tokenizer.decode(&[10_930, 2_131, 1_543]).unwrap(), "你好。");
        assert_eq!(
            tokenizer
                .decode(&[3_714, 2_288, 5_016, 3_555, 995, 20_666, 3_615, 45_340])
                .unwrap(),
            "مرحبا بالعالم"
        );
        assert_eq!(tokenizer.language_code(LANGUAGE_TOKEN_END), Some("yue"));
    }
}
