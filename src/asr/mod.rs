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

pub use parakeet::ParakeetCpu;

pub trait Engine: Send + Sync {
    fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<String>;
    fn label(&self) -> &'static str;
}

pub fn load(name: &str) -> Result<Box<dyn Engine>> {
    match name {
        "parakeet_cpu" | "parakeet_npu" | "whisper_npu" => Ok(Box::new(ParakeetCpu::load()?)),
        other => {
            tracing::warn!("unknown engine '{other}', using parakeet_cpu");
            Ok(Box::new(ParakeetCpu::load()?))
        }
    }
}
