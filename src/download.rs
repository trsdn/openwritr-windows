//! Hugging Face model download with simple progress logging.
//!
//! Uses `hf-hub` blocking client. Models live under %LOCALAPPDATA%/OpenWritr/models/<repo>
//! with the same filenames as on the Hub.

use crate::paths::models_dir;
use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use std::path::PathBuf;
use tracing::{info, warn};

pub struct ModelSpec {
    pub repo: &'static str,         // e.g. "istupakov/parakeet-tdt-0.6b-v3-onnx"
    pub local_dir: &'static str,    // e.g. "parakeet-tdt-0.6b-v3-onnx"
    pub files: &'static [&'static str],
}

pub fn ensure(spec: &ModelSpec) -> Result<PathBuf> {
    let target = models_dir().join(spec.local_dir);
    std::fs::create_dir_all(&target)?;

    // Skip the download if every file is already present and non-empty.
    let all_present = spec.files.iter().all(|f| {
        target
            .join(f)
            .metadata()
            .map(|m| m.len() > 0)
            .unwrap_or(false)
    });
    if all_present {
        info!(repo = spec.repo, "model already cached");
        return Ok(target);
    }

    let cache_dir = models_dir().join(".hf-cache");
    std::fs::create_dir_all(&cache_dir)?;
    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir)
        .build()
        .context("hf-hub api init")?;
    let repo = api.model(spec.repo.to_string());

    for &name in spec.files {
        let dst = target.join(name);
        if dst.exists() && dst.metadata().map(|m| m.len() > 0).unwrap_or(false) {
            continue;
        }
        info!(file = name, "downloading from {}", spec.repo);
        let cached = repo
            .get(name)
            .with_context(|| format!("download {}", name))?;
        // Copy into our flat per-model directory so the runtime doesn't need
        // to chase hf-hub's snapshot folder layout.
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::copy(&cached, &dst)
            .with_context(|| format!("copy {} -> {}", cached.display(), dst.display()))?;
        info!(file = name, bytes = std::fs::metadata(&dst)?.len(), "downloaded");
    }
    Ok(target)
}

pub fn ensure_parakeet_cpu_int8() -> Result<PathBuf> {
    ensure(&ModelSpec {
        repo: "istupakov/parakeet-tdt-0.6b-v3-onnx",
        local_dir: "parakeet-tdt-0.6b-v3-onnx",
        files: &[
            "encoder-model.int8.onnx",
            "decoder_joint-model.int8.onnx",
            "nemo128.onnx",
            "vocab.txt",
            "config.json",
        ],
    })
}

#[allow(dead_code)]
pub fn ensure_parakeet_npu_int8() -> Result<PathBuf> {
    ensure(&ModelSpec {
        repo: "trsdn/parakeet-tdt-0.6b-v3-htp-int8",
        local_dir: "parakeet-tdt-0.6b-v3-htp-int8",
        files: &[
            "encoder-model.onnx",
            "encoder-model.onnx.data",
            "decoder_joint-model.onnx",
            "nemo128.onnx",
            "vocab.txt",
            "config.json",
        ],
    })
}

#[allow(dead_code)]
pub fn ensure_whisper_npu() -> Result<PathBuf> {
    warn!("Whisper NPU model download not yet implemented (~1.6 GB from Qualcomm AI Hub)");
    Err(anyhow::anyhow!("whisper NPU not yet implemented in native build"))
}
