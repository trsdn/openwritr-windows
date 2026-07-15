//! ASR engine abstraction.
//!
//! For v0.2 native we ship Parakeet TDT 0.6B v3 (INT8 CPU) only. NPU + Whisper
//! follow in subsequent commits. The engine signature is intentionally minimal:
//! one async `transcribe(samples, sr) -> String` call.

use anyhow::Result;
use std::path::PathBuf;

mod ort_helpers;
mod parakeet;
mod qnn_ffi;
mod resample;
mod tdt;
mod tokenizer;
mod whisper_decoder;
mod whisper_mel;
mod whisper_npu;
mod whisper_tokenizer;

pub use parakeet::ParakeetEngine;

pub trait Engine: Send {
    fn transcribe(&mut self, samples: &[f32], sample_rate: u32) -> Result<String>;
    fn label(&self) -> &'static str;
}

pub fn verify_runtime_installation() -> Result<PathBuf> {
    qnn_ffi::initialize_ort_runtime()
}

pub fn whisper_hardware_status() -> Result<String> {
    whisper_npu::hardware_status()
}

pub fn load_from_dir(name: &str, model_dir: PathBuf) -> Result<Box<dyn Engine>> {
    match name {
        "parakeet_cpu" => Ok(Box::new(ParakeetEngine::load_cpu_from(model_dir)?)),
        "parakeet_npu" => Ok(Box::new(ParakeetEngine::load_npu_from(model_dir)?)),
        "whisper_npu" => Ok(Box::new(whisper_npu::WhisperNpuEngine::load_from(
            model_dir,
        )?)),
        other => anyhow::bail!("unknown transcription engine {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::load_from_dir;
    use crate::model_manager::{CancellationToken, ModelManager};

    #[test]
    #[ignore = "loads the real pinned ONNX Runtime and Parakeet CPU sessions"]
    fn loads_real_parakeet_cpu_without_fallback() {
        let models = ModelManager::new().unwrap();
        let cpu = models
            .ensure("parakeet_cpu", &CancellationToken::default(), |_| {})
            .unwrap();
        let cpu_engine = load_from_dir("parakeet_cpu", cpu).unwrap();
        assert!(cpu_engine.label().contains("CPU"));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "loads and reloads the real pinned Parakeet NPU sessions"]
    fn reloads_real_parakeet_npu_without_fallback() {
        let models = ModelManager::new().unwrap();
        let npu = models
            .ensure("parakeet_npu", &CancellationToken::default(), |_| {})
            .unwrap();
        for _ in 0..2 {
            let mut npu_engine = load_from_dir("parakeet_npu", npu.clone()).unwrap();
            assert!(npu_engine.label().contains("NPU"));
            npu_engine.transcribe(&vec![0.0; 16_000], 16_000).unwrap();
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "loads the real pinned Whisper encoder and decoder QNN sessions"]
    fn loads_real_whisper_npu_sessions() {
        let models = ModelManager::new().unwrap();
        let whisper = models
            .ensure("whisper_npu", &CancellationToken::default(), |_| {})
            .unwrap();
        let engine = load_from_dir("whisper_npu", whisper).unwrap();
        assert!(engine.label().contains("Whisper"));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "runs the real Whisper encoder and autoregressive QNN decoder"]
    fn transcribes_real_whisper_npu_silence() {
        let models = ModelManager::new().unwrap();
        let whisper = models
            .ensure("whisper_npu", &CancellationToken::default(), |_| {})
            .unwrap();
        let mut engine = load_from_dir("whisper_npu", whisper).unwrap();
        engine.transcribe(&vec![0.0; 16_000], 16_000).unwrap();
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "runs two real Whisper chunks while reusing one detected language"]
    fn transcribes_real_whisper_npu_two_chunks() {
        let models = ModelManager::new().unwrap();
        let whisper = models
            .ensure("whisper_npu", &CancellationToken::default(), |_| {})
            .unwrap();
        let mut engine = load_from_dir("whisper_npu", whisper).unwrap();
        engine.transcribe(&vec![0.0; 480_001], 16_000).unwrap();
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "reloads Whisper and Parakeet through the shared QNN provider"]
    fn reloads_whisper_and_parakeet_npu_sessions() {
        let models = ModelManager::new().unwrap();
        let whisper = models
            .ensure("whisper_npu", &CancellationToken::default(), |_| {})
            .unwrap();
        let parakeet = models
            .ensure("parakeet_npu", &CancellationToken::default(), |_| {})
            .unwrap();

        for _ in 0..2 {
            let whisper_engine = load_from_dir("whisper_npu", whisper.clone()).unwrap();
            assert!(whisper_engine.label().contains("Whisper"));
            drop(whisper_engine);

            let parakeet_engine = load_from_dir("parakeet_npu", parakeet.clone()).unwrap();
            assert!(parakeet_engine.label().contains("Parakeet"));
            drop(parakeet_engine);
        }
    }
}
