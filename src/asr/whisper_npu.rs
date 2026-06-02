//! Whisper Large v3 Turbo on Hexagon NPU — stub.
//!
//! The full implementation (mel features, encoder/decoder with KV cache,
//! greedy decoding) lives in `python/whisper_npu.py`. Porting it to
//! native Rust + ort 2.0-rc.12 + QNN is a larger project than v0.2.
//! For now this engine returns a clear error so the tray app can fall
//! back to Parakeet when the user picks Whisper.

use anyhow::Result;
use tracing::warn;

use crate::asr::Engine;
use crate::paths::models_dir;

pub struct WhisperNpuEngine;

impl WhisperNpuEngine {
    pub fn load() -> Result<Self> {
        let dir = models_dir().join("whisper-large-v3-turbo-qnn");
        if !dir.join("encoder.onnx").exists() {
            anyhow::bail!(
                "Whisper Turbo NPU model not found at {}. \
                 Run `python\\fetch_whisper.py` from the v0.1 Python build \
                 to download it (1.6 GB), or pick a Parakeet engine instead.",
                dir.display()
            );
        }
        warn!(
            "Whisper Turbo NPU native runtime not yet implemented in v0.2. \
             Use the Python v0.1 app for Whisper-on-NPU, or pick a Parakeet \
             engine (which now runs natively on NPU)."
        );
        anyhow::bail!("whisper_npu native runtime pending")
    }
}

impl Engine for WhisperNpuEngine {
    fn label(&self) -> &'static str { "Whisper Large v3 Turbo (NPU)" }

    fn transcribe(&self, _samples: &[f32], _sample_rate: u32) -> Result<String> {
        anyhow::bail!("whisper_npu native runtime not yet available")
    }
}
