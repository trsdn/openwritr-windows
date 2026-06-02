//! ASR engine abstraction.
//!
//! For v0.2 native we ship Parakeet TDT 0.6B v3 (INT8 CPU) only. NPU + Whisper
//! follow in subsequent commits. The engine signature is intentionally minimal:
//! one async `transcribe(samples, sr) -> String` call.

use anyhow::Result;

mod parakeet;
mod resample;
mod tokenizer;
mod tdt;
mod whisper_npu;

pub use parakeet::ParakeetEngine;

pub trait Engine: Send + Sync {
    fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<String>;
    fn label(&self) -> &'static str;
}

pub fn load(name: &str) -> Result<Box<dyn Engine>> {
    match name {
        "parakeet_npu" => match ParakeetEngine::load_npu() {
            Ok(e) => Ok(Box::new(e)),
            Err(e) => {
                tracing::warn!(error = %e, "Parakeet NPU load failed — using CPU");
                Ok(Box::new(ParakeetEngine::load_cpu()?))
            }
        },
        "whisper_npu" => match whisper_npu::WhisperNpuEngine::load() {
            Ok(e) => Ok(Box::new(e)),
            Err(e) => {
                tracing::warn!(error = %e, "Whisper NPU unavailable — using Parakeet CPU");
                Ok(Box::new(ParakeetEngine::load_cpu()?))
            }
        },
        _ => Ok(Box::new(ParakeetEngine::load_cpu()?)),
    }
}
