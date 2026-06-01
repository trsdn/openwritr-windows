// ONNX Runtime ASR engine.
//
// EP probe order: QNN (Snapdragon NPU) -> DirectML (GPU) -> CPU.
// Model: NVIDIA Parakeet TDT 0.6B v3, exported via scripts/export_parakeet_onnx.py.
// We expect three artifacts under models/parakeet-tdt-0.6b-v3/:
//   encoder.onnx, decoder.onnx (predictor), joint.onnx
// Plus a SentencePiece tokenizer (tokenizer.model) and a config.json that
// captures the TDT durations (typically [0,1,2,3,4]).

use anyhow::{anyhow, Context, Result};
use std::path::PathBuf;
use tracing::{info, warn};

pub struct Engine {
    active_ep: String,
    // Held for later: ort sessions for encoder/decoder/joint, tokenizer, config.
    // Wired up once scripts/export_parakeet_onnx.py produces validated artifacts.
}

impl Engine {
    pub async fn load() -> Result<Self> {
        let model_dir = model_dir();
        if !model_dir.exists() {
            // First-launch: we don't auto-download yet — surface a clear message.
            return Err(anyhow!(
                "Parakeet model not found at {}. Run scripts/export_parakeet_onnx.py and place ONNX files there.",
                model_dir.display()
            ));
        }

        let ep = probe_ep()?;
        info!(ep = %ep, "selected execution provider");

        Ok(Self { active_ep: ep })
    }

    pub fn active_ep(&self) -> &str {
        &self.active_ep
    }

    pub async fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        // Resample to 16 kHz mono, mel-spectrogram, encoder forward,
        // TDT greedy decode with durations. Implemented incrementally.
        warn!(
            samples = samples.len(),
            sample_rate, "transcribe() called — full pipeline not yet wired"
        );
        let _ = (samples, sample_rate);
        Ok(String::new())
    }
}

fn model_dir() -> PathBuf {
    // %LOCALAPPDATA%\OpenWritr\models\parakeet-tdt-0.6b-v3
    let base = std::env::var("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    base.join("OpenWritr").join("models").join("parakeet-tdt-0.6b-v3")
}

fn probe_ep() -> Result<String> {
    // The real probe will instantiate small ort sessions and catch failures.
    // For now we report what the build was compiled with and let runtime pick.
    if cfg!(feature = "qnn") {
        return Ok("QNN (Hexagon NPU)".into());
    }
    if cfg!(feature = "directml") {
        return Ok("DirectML".into());
    }
    Ok("CPU".into())
}

// Keep ndarray + ort symbols referenced so future commits compile cleanly.
#[allow(dead_code)]
fn _link_check() {
    let _ = ndarray::Array1::<f32>::zeros(0);
    let _ = ort::execution_providers::CPUExecutionProvider::default();
}

// Pulled into scope for the link check.
use ort as _;
use ndarray as _;
