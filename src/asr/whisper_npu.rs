//! Whisper Large v3 Turbo on Qualcomm Hexagon NPU — native Rust port of
//! the python/whisper_npu.py implementation.
//!
//! Same Qualcomm AI Hub pre-compiled QNN ONNX context binaries we already
//! download for the Python build. We can NOT use those on NPU yet from
//! Rust because the `ort` 2.0-rc.10 we are pinned to does not expose
//! `RegisterExecutionProviderLibrary`. This module therefore loads the
//! encoder/decoder on the CPU EP. Once we move to a newer `ort` that
//! exposes the EP-library API, switching this to NPU is a one-line change.

use anyhow::{anyhow, Context, Result};
use ndarray::{Array2, Array3, Array4, ArrayD, Ix3};
use ort::execution_providers::CPUExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use parking_lot::Mutex;
use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::Instant;
use tracing::{info, warn};

use crate::asr::Engine;
use crate::paths::models_dir;

// Whisper Large v3 Turbo constants (match Python whisper_npu.py).
const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const N_MELS: usize = 128;
const N_FRAMES: usize = 3000;
const N_DECODER_LAYERS: usize = 4;
const N_HEADS: usize = 20;
const HEAD_DIM: usize = 64;
const SELF_CACHE_LEN: usize = 199;
const CROSS_CACHE_LEN: usize = 1500;
const MEAN_DECODE_LEN: usize = 200;
const MAX_DECODE_STEPS: usize = 200;
const TOK_SOT: i32 = 50258;
const TOK_EOT: i32 = 50257;

static ORT_INIT: Once = Once::new();

fn init_ort_once() -> Result<()> {
    let mut err = None;
    ORT_INIT.call_once(|| {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let local = dir.join("onnxruntime.dll");
                if local.exists() {
                    std::env::set_var("ORT_DYLIB_PATH", &local);
                }
            }
        }
        if let Err(e) = ort::init()
            .with_name("openwritr")
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .commit()
        {
            err = Some(e);
        }
    });
    if let Some(e) = err { return Err(e.into()); }
    Ok(())
}

pub struct WhisperNpuEngine {
    _dir: PathBuf,
    _encoder: Mutex<Session>,
    _decoder: Mutex<Session>,
}

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
            "Whisper Turbo NPU: native ort 2.0-rc.10 cannot load QNN EP \
             libraries dynamically. Refusing to load to avoid silent CPU \
             fallback on a 1.6 GB model. Use the Python v0.1 app for now."
        );
        anyhow::bail!("whisper_npu native runtime pending ort EP library API")
    }
}

impl Engine for WhisperNpuEngine {
    fn label(&self) -> &'static str { "Whisper Large v3 Turbo (NPU)" }

    fn transcribe(&self, _samples: &[f32], _sample_rate: u32) -> Result<String> {
        anyhow::bail!("whisper_npu native runtime not yet available")
    }
}
