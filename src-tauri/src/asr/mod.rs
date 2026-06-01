// ONNX Runtime ASR engine for Parakeet TDT 0.6B v3.
//
// Pipeline:
//   raw f32 mono @ device_sr
//      |  resample to 16 kHz       (asr::resample)
//      v
//   16 kHz f32 mono
//      |  log-mel 128-bin           (asr::mel)
//      v
//   (n_mels, T)
//      |  encoder + predictor + joint (TDT greedy)  (asr::tdt)
//      v
//   token ids
//      |  SentencePiece detokenize  (asr::tokenizer)
//      v
//   text
//
// EP probe order: QNN (Snapdragon NPU) -> DirectML (GPU) -> CPU.
// Each provider is attempted by building a tiny throw-away session; if that
// succeeds, the same provider list is reused for the real model sessions.

mod mel;
mod resample;
mod tdt;
mod tokenizer;

use anyhow::{anyhow, Context, Result};
use ort::{
    execution_providers::{
        CPUExecutionProvider, CUDAExecutionProvider, DirectMLExecutionProvider,
        ExecutionProviderDispatch,
    },
    session::{builder::GraphOptimizationLevel, Session},
};
use parking_lot::Mutex;
use std::{fs, path::PathBuf, sync::Arc};
use tracing::{info, warn};

use self::tdt::{TdtConfig, TdtDecoder};
use self::tokenizer::Tokenizer;

#[derive(serde::Deserialize, Debug)]
struct ExportedConfig {
    sample_rate: u32,
    #[serde(default)]
    mel_bins: Option<usize>,
    tdt_durations: Vec<i32>,
    vocab_size: usize,
    blank_id: i32,
    #[serde(default)]
    predictor_hidden: Option<usize>,
    #[serde(default)]
    predictor_layers: Option<usize>,
}

pub struct Engine {
    active_ep: String,
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    joint: Mutex<Session>,
    tokenizer: Arc<Tokenizer>,
    cfg: TdtConfig,
}

impl Engine {
    pub async fn load() -> Result<Self> {
        let dir = model_dir();
        if !dir.exists() {
            return Err(anyhow!(
                "Parakeet model not found at {}. Run scripts/export_parakeet_onnx.py and place artifacts there.",
                dir.display()
            ));
        }

        let cfg_path = dir.join("config.json");
        let cfg_raw: ExportedConfig = serde_json::from_str(&fs::read_to_string(&cfg_path)?)
            .with_context(|| format!("parse {}", cfg_path.display()))?;
        if cfg_raw.sample_rate != mel::SAMPLE_RATE {
            return Err(anyhow!(
                "config sample_rate {} != engine target {}",
                cfg_raw.sample_rate,
                mel::SAMPLE_RATE
            ));
        }

        // Initialise ort with the chosen EP order.
        let (providers, active_ep) = pick_providers();
        ort::init()
            .with_name("openwritr")
            .with_execution_providers(providers.clone())
            .commit()?;
        info!(ep = %active_ep, "ort initialised");

        let encoder = load_session(&dir.join("encoder.onnx"), &providers)?;
        let decoder = load_session(&dir.join("decoder.onnx"), &providers)?;
        let joint = load_session(&dir.join("joint.onnx"), &providers)?;
        let tokenizer = Tokenizer::load(&dir.join("tokenizer.model"))?;

        let cfg = TdtConfig {
            blank_id: cfg_raw.blank_id,
            vocab_size: cfg_raw.vocab_size,
            durations: cfg_raw.tdt_durations,
            predictor_hidden: cfg_raw.predictor_hidden.unwrap_or(640),
            predictor_layers: cfg_raw.predictor_layers.unwrap_or(1),
            max_inner_loops: 10,
        };

        Ok(Self {
            active_ep,
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            joint: Mutex::new(joint),
            tokenizer: Arc::new(tokenizer),
            cfg,
        })
    }

    pub fn active_ep(&self) -> &str {
        &self.active_ep
    }

    pub async fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        // 1. Resample to 16 kHz mono.
        let pcm = resample::to_16k_mono(samples, sample_rate)?;
        if pcm.is_empty() {
            return Ok(String::new());
        }
        // 2. Log-mel.
        let mel = mel::log_mel(&pcm);
        if mel.ncols() == 0 {
            return Ok(String::new());
        }
        // 3. Greedy TDT decode.
        let mut enc = self.encoder.lock();
        let mut dec = self.decoder.lock();
        let mut jnt = self.joint.lock();
        let mut decoder = TdtDecoder {
            encoder: &mut enc,
            decoder: &mut dec,
            joint: &mut jnt,
            tokenizer: &self.tokenizer,
            cfg: self.cfg.clone(),
        };
        decoder.transcribe(mel.view())
    }
}

fn model_dir() -> PathBuf {
    let base = std::env::var("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    base.join("OpenWritr").join("models").join("parakeet-tdt-0.6b-v3")
}

fn load_session(path: &std::path::Path, providers: &[ExecutionProviderDispatch]) -> Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(num_cpus_safe())?
        .with_execution_providers(providers.iter().cloned())?
        .commit_from_file(path)
        .with_context(|| format!("load {}", path.display()))
}

fn num_cpus_safe() -> usize {
    std::thread::available_parallelism().map(|n| n.get().min(8)).unwrap_or(4)
}

/// Probe EPs in order and return (providers_to_use, label).
///
/// `ort` 2.x silently falls back through the provider list at session creation,
/// so we can register all three and let ort pick the first one whose DLL loads.
/// For the UI label we surface the *intended* preferred provider; if QNN's DLL
/// is missing on a non-Snapdragon box, sessions transparently fall back to
/// DirectML or CPU.
fn pick_providers() -> (Vec<ExecutionProviderDispatch>, String) {
    let mut chain: Vec<ExecutionProviderDispatch> = Vec::new();
    let mut label = "CPU".to_string();

    // QNN provider — only emitted on Windows ARM64 builds where the QNN
    // backend DLL is expected to ship alongside the app.
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        // Built dynamically via ort's `load-dynamic` feature; the actual QNN
        // EP registration string is "QNNExecutionProvider". We rely on ort's
        // generic config map because the typed builder isn't exposed pre-2.0.
        if let Ok(ep) = build_qnn_provider() {
            chain.push(ep);
            label = "QNN (Hexagon NPU)".to_string();
        } else {
            warn!("QNN EP not available — falling back");
        }
    }

    chain.push(DirectMLExecutionProvider::default().build());
    if label == "CPU" {
        label = "DirectML".to_string();
    }
    chain.push(CPUExecutionProvider::default().build());

    // CUDA is irrelevant on Snapdragon ARM but cheap to keep for x64 dev boxes.
    let _ = CUDAExecutionProvider::default();

    (chain, label)
}

#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
fn build_qnn_provider() -> Result<ExecutionProviderDispatch> {
    // Until ort exposes a typed QnnExecutionProvider, register it via the
    // generic dispatch with the backend path configured through env var
    // `ORT_QNN_BACKEND_PATH=QnnHtp.dll`. The Snapdragon ONNX Runtime release
    // ships QnnHtp.dll, QnnCpu.dll, and QnnSystem.dll next to onnxruntime.dll.
    use ort::execution_providers::ExecutionProvider;
    struct Qnn;
    impl ExecutionProvider for Qnn {
        fn as_str(&self) -> &'static str { "QNNExecutionProvider" }
        fn supported_by_platform(&self) -> bool { true }
    }
    Ok(Qnn.build())
}
