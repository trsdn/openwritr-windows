//! Parakeet TDT 0.6B v3 — CPU INT8 + optional Hexagon NPU backend.
//!
//! With `ort` 2.0-rc.12 + `api-24` we can now register the QNN execution
//! provider library at runtime and pick the NPU per-session. If no NPU is
//! available the selected engine fails visibly; fallback is never implicit.

use super::ort_helpers::OrtResultExt;
use super::qnn_ffi::{
    acquire_qnn_provider, enumerate_qnn_npu_devices, initialize_ort_runtime, QnnSession,
    SessionContract, TensorElementType, TensorInput, TensorSpec,
};
use super::tdt::Tdt;
use super::{resample, tokenizer::Vocab};
use crate::asr::Engine;
use anyhow::{anyhow, Context, Result};
use ndarray::{Array1, Array2, Array3, Axis, Ix3};
use ort::session::{
    builder::{AutoDevicePolicy, GraphOptimizationLevel},
    Session,
};
use ort::value::Value;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::info;

// The AI-Hub-compiled QNN context binary was calibrated and compiled at a
// fixed 8-second audio window. Inputs are padded to exactly this length;
// utterances longer than this are truncated. Trade-off vs. a longer window:
// the encoder cost is dominated by the static window, not the real audio
// length, so a shorter window means faster steady-state inference at the
// cost of capping the max push-to-talk duration.
pub const MAX_NPU_SECONDS: f32 = 8.0;

fn init_ort_once() -> Result<()> {
    initialize_ort_runtime()?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Npu,
}

enum Encoder {
    Cpu(Session),
    Npu(QnnSession),
}

pub struct ParakeetEngine {
    backend: Backend,
    encoder: Encoder,
    decoder_joint: Session,
    preprocessor: Session,
    vocab: Vocab,
    tdt: Tdt,
}

impl ParakeetEngine {
    pub fn load_cpu_from(dir: PathBuf) -> Result<Self> {
        init_ort_once()?;
        Self::load_from(dir, Backend::Cpu)
    }

    pub fn load_npu_from(dir: PathBuf) -> Result<Self> {
        init_ort_once()?;
        Self::load_with_backend(&dir, Backend::Npu)
    }

    fn load_from(dir: PathBuf, backend: Backend) -> Result<Self> {
        Self::load_with_backend(&dir, backend)
    }

    fn load_with_backend(dir: &Path, backend: Backend) -> Result<Self> {
        info!("loading Parakeet {:?} from {}", backend, dir.display());
        let t0 = Instant::now();

        // Build CPU sessions before registering QNN on the first NPU load.
        // build_cpu_session also pins automatic device selection to CPU so
        // later engine reloads remain CPU-only after QNN is process-global.
        let preprocessor = build_cpu_session(&dir.join("nemo128.onnx"))?;
        // CPU INT8 decoder for both backends — only the encoder differs (HTP
        // for NPU, CPU INT8 for CPU). The decoder runs on the CPU EP either way
        // because its dynamic-shape TDT loop doesn't map to HTP.
        let decoder_joint = build_cpu_session(&dir.join("decoder_joint-model.int8.onnx"))?;

        // Now register QNN (if NPU requested) and build the encoder session.
        let encoder = match backend {
            Backend::Cpu => Encoder::Cpu(build_cpu_session(&dir.join("encoder-model.int8.onnx"))?),
            Backend::Npu => Encoder::Npu(build_npu_ffi(&dir.join("encoder-model.onnx"))?),
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
            encoder,
            decoder_joint,
            preprocessor,
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

    fn transcribe(&mut self, samples: &[f32], sample_rate: u32) -> Result<String> {
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
            chars = text.chars().count(),
            "transcribed"
        );
        Ok(text)
    }
}

impl ParakeetEngine {
    /// Preprocess + encode one PCM buffer. For NPU the length must equal
    /// the compiled window (= MAX_NPU_SECONDS); caller pads. CPU is dynamic
    /// shape, any length works. Returns encoder_out [1, 1024, T] and valid
    /// T_out (after clipping to the encoder's reported `encoded_length`).
    fn preprocess_and_encode(&mut self, pcm: &[f32]) -> Result<(Array3<f32>, usize)> {
        let n = pcm.len();
        let wave = ndarray::Array2::from_shape_vec((1, n), pcm.to_vec())?;
        let wave_len = Array1::<i64>::from_elem(1, n as i64);
        let pre_in = ort::inputs![
            "waveforms" => Value::from_array(wave)?,
            "waveforms_lens" => Value::from_array(wave_len)?,
        ];
        let (features, features_lens) = {
            let pre_out = self.preprocessor.run(pre_in).ortx()?;
            let features = pre_out
                .get("features")
                .ok_or_else(|| anyhow!("preprocessor missing 'features'"))?
                .try_extract_array::<f32>()
                .ortx()?
                .to_owned()
                .into_dimensionality::<Ix3>()?;
            let features_lens = pre_out
                .get("features_lens")
                .ok_or_else(|| anyhow!("preprocessor missing 'features_lens'"))?
                .try_extract_array::<i64>()
                .ortx()?
                .to_owned();
            (features, features_lens)
        };

        let (encoder_out, encoder_lens) = match &mut self.encoder {
            Encoder::Cpu(encoder) => {
                let enc_in = ort::inputs![
                    "audio_signal" => Value::from_array(features)?,
                    "length" => Value::from_array(features_lens)?,
                ];
                let enc_out_map = encoder.run(enc_in).ortx()?;
                let encoder_out = enc_out_map
                    .get("outputs")
                    .ok_or_else(|| anyhow!("encoder missing 'outputs'"))?
                    .try_extract_array::<f32>()
                    .ortx()?
                    .to_owned()
                    .into_dimensionality::<Ix3>()?;
                let encoder_lens = enc_out_map
                    .get("encoded_lengths")
                    .ok_or_else(|| anyhow!("encoder missing 'encoded_lengths'"))?
                    .try_extract_array::<i64>()
                    .ortx()?
                    .to_owned();
                (encoder_out, encoder_lens)
            }
            Encoder::Npu(npu) => {
                let features = features.as_standard_layout();
                let dimensions = [
                    features.dim().0 as i64,
                    features.dim().1 as i64,
                    features.dim().2 as i64,
                ];
                let feature_values = features
                    .as_slice()
                    .ok_or_else(|| anyhow!("NPU features are not contiguous"))?;
                let length_i32 = i32::try_from(features_lens.iter().next().copied().unwrap_or(0))
                    .context("NPU feature length does not fit i32")?;
                let length_values = [length_i32];
                let length_dimensions = [1];
                let mut outputs = npu.run(&[
                    TensorInput::f32("audio_signal", &dimensions, feature_values),
                    TensorInput::i32("length", &length_dimensions, &length_values),
                ])?;
                let (output_dimensions, output_values) = outputs.remove(0).into_f32()?;
                if output_dimensions.len() != 3 {
                    return Err(anyhow!(
                        "NPU encoder output rank mismatch: {:?}",
                        output_dimensions
                    ));
                }
                let output_shape = (
                    usize::try_from(output_dimensions[0])?,
                    usize::try_from(output_dimensions[1])?,
                    usize::try_from(output_dimensions[2])?,
                );
                let out_arr = Array3::from_shape_vec(output_shape, output_values)?;
                let (_, encoded_lengths) = outputs.remove(0).into_i32()?;
                let encoded_len_i32 = encoded_lengths
                    .first()
                    .copied()
                    .ok_or_else(|| anyhow!("NPU encoded length output is empty"))?;
                let encoder_lens = Array1::from_elem(1, encoded_len_i32 as i64).into_dyn();
                (out_arr, encoder_lens)
            }
        };

        let valid_t = encoder_lens
            .iter()
            .next()
            .copied()
            .map(|v| v as usize)
            .unwrap_or(encoder_out.dim().2)
            .min(encoder_out.dim().2);
        Ok((encoder_out, valid_t))
    }

    /// Take a single-pass encoder output [1, 1024, T] and run the TDT decoder
    /// over the first `valid_t` frames.
    fn decode_features(&mut self, encoder_out: Array3<f32>, valid_t: usize) -> Result<String> {
        let permuted = encoder_out
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();
        let t_out = valid_t.min(permuted.dim().1);
        let token_ids = self.tdt.decode(&mut self.decoder_joint, &permuted, t_out)?;
        Ok(self.vocab.detokenize(&token_ids))
    }

    fn run_pipeline(&mut self, pcm_16k: &[f32]) -> Result<String> {
        // NPU window the binary was compiled for (samples + mel-frame stride).
        let chunk_samples = (MAX_NPU_SECONDS * 16_000.0) as usize;
        // 1 s of overlap between successive NPU chunks, in audio samples and
        // in encoder-output frames (encoder downsamples by ~8 → 12 frames/s).
        let stride_samples = ((MAX_NPU_SECONDS - 1.0) * 16_000.0) as usize;
        const OVERLAP_FRAMES: usize = 12;

        let is_npu = matches!(&self.encoder, Encoder::Npu(_));
        let n_real = pcm_16k.len();

        // CPU mode: encoder is dynamic-shape, no padding or chunking.
        if !is_npu {
            let (encoder_out, valid_t) = self.preprocess_and_encode(pcm_16k)?;
            return self.decode_features(encoder_out, valid_t);
        }

        // NPU mode, audio fits in one window: pad with silence to the
        // compiled length and run once.
        if n_real <= chunk_samples {
            let mut padded = pcm_16k.to_vec();
            padded.resize(chunk_samples, 0.0);
            let (encoder_out, valid_t) = self.preprocess_and_encode(&padded)?;
            return self.decode_features(encoder_out, valid_t);
        }

        // NPU mode, audio longer than the window: slide 8-s windows with 1 s
        // overlap, stitch the encoder feature streams at the seam (drop the
        // overlapping 12 leading frames of each non-first chunk), then run
        // the TDT decoder once over the concatenated features.
        let mut stitched: Vec<Array2<f32>> = Vec::new();
        let mut chunk_start = 0usize;
        let mut chunk_idx = 0usize;
        loop {
            let chunk_end_real = (chunk_start + chunk_samples).min(n_real);
            let mut chunk_pcm = pcm_16k[chunk_start..chunk_end_real].to_vec();
            chunk_pcm.resize(chunk_samples, 0.0);

            let (encoder_out, valid_t) = self.preprocess_and_encode(&chunk_pcm)?;
            // [1, 1024, T] → [T, 1024]
            let permuted = encoder_out
                .permuted_axes([0, 2, 1])
                .as_standard_layout()
                .to_owned();
            let chunk_frames: Array2<f32> = permuted.slice(ndarray::s![0, .., ..]).to_owned();

            let start_t = if chunk_idx == 0 { 0 } else { OVERLAP_FRAMES };
            let end_t = valid_t;
            if start_t < end_t {
                stitched.push(
                    chunk_frames
                        .slice(ndarray::s![start_t..end_t, ..])
                        .to_owned(),
                );
            }

            if chunk_end_real == n_real {
                break;
            }
            chunk_start += stride_samples;
            chunk_idx += 1;
        }
        info!(chunks = chunk_idx + 1, "NPU chunked encode");

        if stitched.is_empty() {
            return Ok(String::new());
        }
        let views: Vec<_> = stitched.iter().map(|a| a.view()).collect();
        let final_2d = ndarray::concatenate(Axis(0), &views)
            .map_err(|e| anyhow!("concat encoder features: {e}"))?;
        let final_features: Array3<f32> = final_2d.insert_axis(Axis(0));
        let t_out = final_features.dim().1;
        let token_ids = self
            .tdt
            .decode(&mut self.decoder_joint, &final_features, t_out)?;
        Ok(self.vocab.detokenize(&token_ids))
    }
}

fn build_cpu_session(path: &Path) -> Result<Session> {
    use ort::ep::CPU;
    Session::builder()
        .ortx()?
        .with_auto_device(AutoDevicePolicy::PreferCPU)
        .ortx()?
        .with_no_environment_execution_providers()
        .ortx()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .ortx()?
        .with_intra_threads(num_threads())
        .ortx()?
        .with_execution_providers([CPU::default().build()])
        .ortx()?
        .commit_from_file(path)
        .ortx()
        .with_context(|| format!("load {}", path.display()))
}

fn build_npu_ffi(path: &Path) -> Result<QnnSession> {
    use ort::AsPointer;
    let runtime_dir = initialize_ort_runtime()?;
    let provider_lease = acquire_qnn_provider(&runtime_dir.join("onnxruntime_providers_qnn.dll"))?;
    let env = ort::environment::Environment::current().ortx()?;
    let env_ptr = env.ptr();
    let npu_devs = enumerate_qnn_npu_devices(env_ptr).context("enumerate QNN NPU devices")?;
    if npu_devs.is_empty() {
        return Err(anyhow!("no QNN NPU device discovered by ORT"));
    }
    info!(
        count = npu_devs.len(),
        "found QNN NPU device(s); building HTP session via FFI"
    );
    let contract = SessionContract::new(
        vec![
            TensorSpec::new("audio_signal", TensorElementType::F32, vec![1, 128, 801]),
            TensorSpec::new("length", TensorElementType::I32, vec![1]),
        ],
        vec![
            TensorSpec::new("output_0", TensorElementType::F32, vec![1, 1024, -1]),
            TensorSpec::new("output_1", TensorElementType::I32, vec![1]),
        ],
    )?;
    QnnSession::load(env_ptr, &npu_devs, path, contract, provider_lease)
        .with_context(|| format!("load NPU {}", path.display()))
}

fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(4)
}
