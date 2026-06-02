//! Parakeet TDT 0.6B v3 CPU INT8 — Rust port of the onnx-asr nemo adapter.
//!
//! Pipeline per utterance:
//!   1. resample input audio to 16 kHz mono (rubato)
//!   2. run nemo128.onnx ONNX preprocessor -> (B, n_mels, T) features
//!   3. run encoder-model.int8.onnx -> (B, D, T') + encoded_lengths
//!      then transpose to (B, T', D) like onnx-asr does
//!   4. greedy TDT decode (asr::tdt) over the time axis
//!   5. detokenize via vocab.txt
//!
//! All three onnxruntime sessions are reused across calls. CPU EP only —
//! NPU comes in a follow-up commit.

use super::{resample, tokenizer::Vocab};
use super::tdt::Tdt;
use crate::asr::Engine;
use crate::download;
use anyhow::{anyhow, Context, Result};
use ndarray::{Array1, Ix3};
use ort::execution_providers::CPUExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Once;
use std::time::Instant;
use tracing::info;

static ORT_INIT: Once = Once::new();

fn init_ort_once() -> Result<()> {
    let mut err = None;
    ORT_INIT.call_once(|| {
        // load-dynamic build — try to find onnxruntime.dll next to the exe
        // first, then fall back to whatever is on PATH.
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
    if let Some(e) = err {
        return Err(e.into());
    }
    Ok(())
}

pub struct ParakeetCpu {
    model_dir: PathBuf,
    encoder: Mutex<Session>,
    decoder_joint: Mutex<Session>,
    preprocessor: Mutex<Session>,
    vocab: Vocab,
    tdt: Tdt,
}

impl ParakeetCpu {
    pub fn load() -> Result<Self> {
        init_ort_once()?;
        let dir = download::ensure_parakeet_cpu_int8()
            .context("download parakeet cpu int8")?;
        info!("loading Parakeet CPU INT8 from {}", dir.display());
        let t0 = Instant::now();

        let make = |name: &str| -> Result<Session> {
            let path = dir.join(name);
            Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(num_threads())?
                .commit_from_file(&path)
                .with_context(|| format!("load {}", path.display()))
        };

        let preprocessor = make("nemo128.onnx")?;
        let encoder = make("encoder-model.int8.onnx")?;
        let decoder_joint = make("decoder_joint-model.int8.onnx")?;
        let vocab = Vocab::load(&dir.join("vocab.txt"))?;
        info!(
            secs = t0.elapsed().as_secs_f32(),
            vocab = vocab.size,
            blank = vocab.blank_id,
            "Parakeet CPU ready"
        );

        let tdt = Tdt {
            vocab_size: vocab.size - 1,  // exclude blank from token logits range
            blank_id: vocab.blank_id,
        };

        Ok(Self {
            model_dir: dir,
            encoder: Mutex::new(encoder),
            decoder_joint: Mutex::new(decoder_joint),
            preprocessor: Mutex::new(preprocessor),
            vocab,
            tdt,
        })
    }
}

impl Engine for ParakeetCpu {
    fn label(&self) -> &'static str { "Parakeet TDT v3 (CPU INT8)" }

    fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let t_total = Instant::now();
        let pcm = resample::to_16k_mono(samples, sample_rate)?;
        if pcm.is_empty() {
            return Ok(String::new());
        }

        // 1) Preprocessor: waveforms (1, N) + waveforms_lens (1) -> features (1, n_mels, T), lens.
        let n = pcm.len();
        let wave = ndarray::Array2::from_shape_vec((1, n), pcm)?;
        let wave_len = Array1::<i64>::from_elem(1, n as i64);

        let pre_in = ort::inputs![
            "waveforms" => Value::from_array(wave.into_dyn())?,
            "waveforms_lens" => Value::from_array(wave_len.into_dyn())?,
        ];
        let (features, features_lens) = {
            let mut pre = self.preprocessor.lock();
            let pre_out = pre.run(pre_in)?;
            let features = pre_out
                .get("features")
                .ok_or_else(|| anyhow!("preprocessor missing 'features'"))?
                .try_extract_array::<f32>()?
                .to_owned()
                .into_owned()
                .into_dimensionality::<Ix3>()?;
            let features_lens = pre_out
                .get("features_lens")
                .ok_or_else(|| anyhow!("preprocessor missing 'features_lens'"))?
                .try_extract_array::<i64>()?
                .to_owned()
                .into_owned();
            (features, features_lens)
        };

        // 2) Encoder.
        let enc_in = ort::inputs![
            "audio_signal" => Value::from_array(features.into_dyn())?,
            "length" => Value::from_array(features_lens.into_dyn())?,
        ];
        let (encoder_out, encoder_lens) = {
            let mut enc = self.encoder.lock();
            let enc_out_map = enc.run(enc_in)?;
            let encoder_out = enc_out_map
                .get("outputs")
                .ok_or_else(|| anyhow!("encoder missing 'outputs'"))?
                .try_extract_array::<f32>()?
                .to_owned()
                .into_owned()
                .into_dimensionality::<Ix3>()?;
            let encoder_lens = enc_out_map
                .get("encoded_lengths")
                .ok_or_else(|| anyhow!("encoder missing 'encoded_lengths'"))?
                .try_extract_array::<i64>()?
                .to_owned()
                .into_owned();
            (encoder_out, encoder_lens)
        };

        // istupakov returns (B, D, T'); we transpose to (B, T', D) so TDT
        // can index time on axis 1.
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
        let text = self.vocab.detokenize(&token_ids);

        info!(
            audio_s = samples.len() as f32 / sample_rate as f32,
            decode_ms = t_total.elapsed().as_millis() as u64,
            tokens = token_ids.len(),
            "transcribed -> {text:?}"
        );
        let _ = self.model_dir;
        Ok(text)
    }
}

fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(4)
}
