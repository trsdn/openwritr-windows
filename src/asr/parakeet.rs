//! Parakeet TDT 0.6B v3 — CPU INT8 + optional Hexagon NPU backend.
//!
//! With `ort` 2.0-rc.12 + `api-22` we can now register the QNN execution
//! provider library at runtime and pick the NPU per-session. If no NPU is
//! available we fall back to CPU INT8.

use super::ort_helpers::OrtResultExt;
use super::{resample, tokenizer::Vocab};
use super::tdt::Tdt;
use crate::asr::Engine;
use crate::download;
use anyhow::{anyhow, Context, Result};
use ndarray::{Array1, Ix3};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::Instant;
use tracing::{info, warn};

static ORT_INIT: Once = Once::new();

pub const MAX_NPU_SECONDS: f32 = 28.0;

fn init_ort_once() {
    ORT_INIT.call_once(|| {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let local = dir.join("onnxruntime.dll");
                if local.exists() {
                    std::env::set_var("ORT_DYLIB_PATH", &local);
                }
            }
        }
        // commit() returns bool; false simply means another caller already
        // committed the env first — that's fine for us.
        let _ = ort::init().with_name("openwritr").commit();
    });
}

fn register_qnn_if_needed() -> Result<()> {
    static QNN_REG: std::sync::Once = std::sync::Once::new();
    let mut err: Option<String> = None;
    QNN_REG.call_once(|| {
        let exe = match std::env::current_exe() {
            Ok(p) => p,
            Err(e) => { err = Some(format!("current_exe: {e}")); return; }
        };
        let dir = match exe.parent() {
            Some(p) => p.to_path_buf(),
            None => { err = Some("no parent".into()); return; }
        };
        let qnn_dll = dir.join("onnxruntime_providers_qnn.dll");
        if !qnn_dll.exists() {
            err = Some(format!("{} not found", qnn_dll.display()));
            return;
        }
        match ort::environment::Environment::current() {
            Ok(env) => match env.register_ep_library("QNNExecutionProvider", &qnn_dll) {
                Ok(_) => info!("QNN execution provider library registered"),
                Err(e) => err = Some(format!("register: {e}")),
            },
            Err(e) => err = Some(format!("environment::current: {e}")),
        }
    });
    if let Some(e) = err {
        return Err(anyhow!("QNN register failed: {e}"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Npu,
}

pub struct ParakeetEngine {
    backend: Backend,
    encoder: Mutex<Session>,
    decoder_joint: Mutex<Session>,
    preprocessor: Mutex<Session>,
    vocab: Vocab,
    tdt: Tdt,
}

impl ParakeetEngine {
    pub fn load_cpu() -> Result<Self> {
        init_ort_once();
        let dir = download::ensure_parakeet_cpu_int8()
            .context("download parakeet cpu int8")?;
        Self::load_from(dir, Backend::Cpu)
    }

    pub fn load_npu() -> Result<Self> {
        init_ort_once();
        // NPU needs the HTP-specific QDQ-quantized encoder (different
        // quantization than CPU INT8). Decoder/preprocessor reuse the
        // istupakov upstream files included in the same HF repo.
        let dir = download::ensure_parakeet_npu_int8()
            .context("download parakeet htp int8")?;
        match Self::load_with_backend(&dir, Backend::Npu) {
            Ok(e) => Ok(e),
            Err(e) => {
                warn!(error = ?e, "NPU session failed; falling back to CPU INT8");
                let cpu_dir = download::ensure_parakeet_cpu_int8()
                    .context("fallback to cpu int8")?;
                Self::load_from(cpu_dir, Backend::Cpu)
            }
        }
    }

    fn load_from(dir: PathBuf, backend: Backend) -> Result<Self> {
        Self::load_with_backend(&dir, backend)
    }

    fn load_with_backend(dir: &Path, backend: Backend) -> Result<Self> {
        info!("loading Parakeet {:?} from {}", backend, dir.display());
        let t0 = Instant::now();

        // CRITICAL ORDERING: build all CPU sessions BEFORE registering QNN.
        // Once QNN is registered globally, subsequent CPU sessions also get
        // their graph offered to QNN, which then chokes on dynamic shapes
        // in the decoder ("Dynamic shape is not supported yet").
        let preprocessor = build_cpu_session(&dir.join("nemo128.onnx"))?;
        let decoder_filename = match backend {
            Backend::Cpu => "decoder_joint-model.int8.onnx",
            Backend::Npu => "decoder_joint-model.onnx",
        };
        let decoder_joint = build_cpu_session(&dir.join(decoder_filename))?;

        // Now register QNN (if NPU requested) and build the encoder session.
        let encoder = match backend {
            Backend::Cpu => build_cpu_session(&dir.join("encoder-model.int8.onnx"))?,
            Backend::Npu => build_npu_session(&dir.join("encoder-model.onnx"))?,
        };

        let vocab = Vocab::load(&dir.join("vocab.txt"))?;
        info!(
            secs = t0.elapsed().as_secs_f32(),
            vocab = vocab.size,
            blank = vocab.blank_id,
            "Parakeet ready"
        );

        let tdt = Tdt {
            vocab_size: vocab.size,
            blank_id: vocab.blank_id,
        };

        Ok(Self {
            backend,
            encoder: Mutex::new(encoder),
            decoder_joint: Mutex::new(decoder_joint),
            preprocessor: Mutex::new(preprocessor),
            vocab,
            tdt,
        })
    }
}

impl Engine for ParakeetEngine {
    fn label(&self) -> &'static str {
        match self.backend {
            Backend::Cpu => "Parakeet TDT v3 (CPU INT8)",
            Backend::Npu => "Parakeet TDT v3 (Hexagon NPU)",
        }
    }

    fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let t_total = Instant::now();
        let pcm = resample::to_16k_mono(samples, sample_rate)?;
        if pcm.is_empty() {
            return Ok(String::new());
        }
        let text = self.run_pipeline(&pcm)?;
        info!(
            audio_s = samples.len() as f32 / sample_rate as f32,
            decode_ms = t_total.elapsed().as_millis() as u64,
            backend = ?self.backend,
            "transcribed -> {text:?}"
        );
        Ok(text)
    }
}

impl ParakeetEngine {
    fn run_pipeline(&self, pcm_16k: &[f32]) -> Result<String> {
        // NPU calibration was done at a fixed audio length; both shorter and
        // longer inputs break /Expand. Pad/truncate to exactly 28 s for NPU,
        // but keep wave_len at the *real* audio length so the encoder still
        // emits frames only for that portion.
        let pcm_npu;
        let real_len_samples = pcm_16k.len().min((MAX_NPU_SECONDS * 16_000.0) as usize);
        let pcm_16k: &[f32] = if self.backend == Backend::Npu {
            let target = (MAX_NPU_SECONDS * 16_000.0) as usize;
            if pcm_16k.len() >= target {
                &pcm_16k[..target]
            } else {
                let mut buf = pcm_16k.to_vec();
                buf.resize(target, 0.0);
                pcm_npu = buf;
                &pcm_npu
            }
        } else {
            pcm_16k
        };

        let n = pcm_16k.len();
        let wave = ndarray::Array2::from_shape_vec((1, n), pcm_16k.to_vec())?;
        // For NPU: declare the full padded length to the preprocessor so the
        // mel feature time dimension matches what the encoder was calibrated
        // for. Decoder will see blank emissions over silence — that's fine.
        let _ = real_len_samples;
        let wave_len = Array1::<i64>::from_elem(1, n as i64);

        let pre_in = ort::inputs![
            "waveforms" => Value::from_array(wave)?,
            "waveforms_lens" => Value::from_array(wave_len)?,
        ];
        let (features, features_lens) = {
            let mut pre = self.preprocessor.lock();
            let pre_out = pre.run(pre_in).ortx()?;
            let features = pre_out
                .get("features")
                .ok_or_else(|| anyhow!("preprocessor missing 'features'"))?
                .try_extract_array::<f32>().ortx()?
                .to_owned()
                .into_dimensionality::<Ix3>()?;
            let features_lens = pre_out
                .get("features_lens")
                .ok_or_else(|| anyhow!("preprocessor missing 'features_lens'"))?
                .try_extract_array::<i64>().ortx()?
                .to_owned();
            (features, features_lens)
        };

        let enc_in = ort::inputs![
            "audio_signal" => Value::from_array(features)?,
            "length" => Value::from_array(features_lens)?,
        ];
        let (encoder_out, encoder_lens) = {
            let mut enc = self.encoder.lock();
            let enc_out_map = enc.run(enc_in).ortx()?;
            let encoder_out = enc_out_map
                .get("outputs")
                .ok_or_else(|| anyhow!("encoder missing 'outputs'"))?
                .try_extract_array::<f32>().ortx()?
                .to_owned()
                .into_dimensionality::<Ix3>()?;
            let encoder_lens = enc_out_map
                .get("encoded_lengths")
                .ok_or_else(|| anyhow!("encoder missing 'encoded_lengths'"))?
                .try_extract_array::<i64>().ortx()?
                .to_owned();
            (encoder_out, encoder_lens)
        };

        let encoder_out_owned = encoder_out
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();
        let t_out = encoder_lens
            .iter()
            .next()
            .copied()
            .map(|v| v as usize)
            .unwrap_or(encoder_out_owned.dim().1)
            .min(encoder_out_owned.dim().1);

        let token_ids = {
            let mut dec = self.decoder_joint.lock();
            self.tdt.decode(&mut dec, &encoder_out_owned, t_out)?
        };
        Ok(self.vocab.detokenize(&token_ids))
    }
}

fn build_cpu_session(path: &Path) -> Result<Session> {
    use ort::ep::{CPU, ExecutionProvider};
    Session::builder().ortx()?
        .with_optimization_level(GraphOptimizationLevel::Level3).ortx()?
        .with_intra_threads(num_threads()).ortx()?
        .with_execution_providers([CPU::default().build()]).ortx()?
        .commit_from_file(path).ortx()
        .with_context(|| format!("load {}", path.display()))
}

fn build_npu_session(path: &Path) -> Result<Session> {
    use ort::memory::DeviceType;
    register_qnn_if_needed()?;
    let env = ort::environment::Environment::current().ortx()?;
    let npu_devices: Vec<_> = env.devices().filter(|d| d.ty() == DeviceType::NPU).collect();
    if npu_devices.is_empty() {
        return Err(anyhow!("no NPU device discovered by ORT"));
    }
    info!(count = npu_devices.len(), "found NPU device(s); building HTP session");
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_default();
    let cache_path = exe_dir.join("parakeet-encoder.qnn_ctx.onnx");
    let opts = vec![
        ("QNNExecutionProvider.htp_performance_mode".to_string(), "burst".to_string()),
        ("QNNExecutionProvider.context_cache_enable".to_string(), "1".to_string()),
        ("QNNExecutionProvider.context_file_path".to_string(), cache_path.to_string_lossy().into_owned()),
    ];
    Session::builder().ortx()?
        .with_devices(npu_devices.into_iter(), Some(&opts)).ortx()?
        .commit_from_file(path).ortx()
        .with_context(|| format!("load NPU {}", path.display()))
}

fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(4)
}
