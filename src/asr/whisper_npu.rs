//! Whisper Large v3 Turbo on the Qualcomm Hexagon NPU.

use super::qnn_ffi::{
    acquire_qnn_provider, enumerate_qnn_npu_devices, initialize_ort_runtime, QnnSession,
};
use super::whisper_decoder::{
    decode_chunk, decoder_contract, encoder_contract, run_encoder, QnnDecoderBackend,
};
use super::whisper_mel::{log_mel_30s, SAMPLE_RATE};
use super::whisper_tokenizer::WhisperTokenizer;
use super::{resample, Engine};
use anyhow::{anyhow, bail, Context, Result};
use ort::AsPointer;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::info;

const CHUNK_SECONDS: usize = 30;
const CHUNK_SAMPLES: usize = SAMPLE_RATE as usize * CHUNK_SECONDS;

pub struct WhisperNpuEngine {
    encoder: QnnSession,
    decoder: QnnSession,
    tokenizer: WhisperTokenizer,
}

impl WhisperNpuEngine {
    pub fn load_from(model_dir: PathBuf) -> Result<Self> {
        ensure_supported_hardware()?;

        let tokenizer = WhisperTokenizer::load(&model_dir)
            .with_context(|| format!("load Whisper tokenizer from {}", model_dir.display()))?;
        let runtime_dir = initialize_ort_runtime()?;
        let qnn_library = runtime_dir.join("onnxruntime_providers_qnn.dll");
        let encoder_lease = acquire_qnn_provider(&qnn_library)?;
        let decoder_lease = acquire_qnn_provider(&qnn_library)?;
        let environment = ort::environment::Environment::current()
            .map_err(|error| anyhow!("Environment::current: {error}"))?;
        let devices =
            enumerate_qnn_npu_devices(environment.ptr()).context("enumerate QNN NPU devices")?;
        if devices.is_empty() {
            bail!("no QNN NPU device discovered by ONNX Runtime");
        }

        let started = Instant::now();
        let encoder_session = load_session(
            environment.ptr(),
            &devices,
            &model_dir.join("encoder.onnx"),
            encoder_contract()?,
            encoder_lease,
        )?;
        info!(
            elapsed_ms = started.elapsed().as_millis() as u64,
            "Whisper NPU encoder loaded"
        );

        let started = Instant::now();
        let decoder_session = load_session(
            environment.ptr(),
            &devices,
            &model_dir.join("decoder.onnx"),
            decoder_contract()?,
            decoder_lease,
        )?;
        info!(
            elapsed_ms = started.elapsed().as_millis() as u64,
            "Whisper NPU decoder loaded"
        );

        Ok(Self {
            encoder: encoder_session,
            decoder: decoder_session,
            tokenizer,
        })
    }
}

impl Engine for WhisperNpuEngine {
    fn label(&self) -> &'static str {
        "Whisper Large v3 Turbo (NPU)"
    }

    fn transcribe(&mut self, samples: &[f32], sample_rate: u32) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        validate_sample_rate(sample_rate)?;
        let audio = resample::to_16k_mono(samples, sample_rate)
            .with_context(|| format!("resample {sample_rate} Hz audio to 16 kHz"))?;
        let chunk_count = audio.len().div_ceil(CHUNK_SAMPLES);
        let started = Instant::now();
        let mut language_token = None;
        let mut transcript_tokens = Vec::new();

        for (index, chunk) in audio.chunks(CHUNK_SAMPLES).enumerate() {
            let features = log_mel_30s(chunk)
                .with_context(|| format!("prepare Whisper chunk {}", index + 1))?;

            let encoder_started = Instant::now();
            let cross_cache = run_encoder(&mut self.encoder, &features)
                .with_context(|| format!("encode Whisper chunk {}", index + 1))?;
            let encoder_ms = encoder_started.elapsed().as_millis() as u64;

            let decoder_started = Instant::now();
            let mut decoder = QnnDecoderBackend::new(&mut self.decoder);
            let decoded = decode_chunk(&mut decoder, &cross_cache, language_token)
                .with_context(|| format!("decode Whisper chunk {}", index + 1))?;
            let decoder_ms = decoder_started.elapsed().as_millis() as u64;

            if language_token.is_none() {
                let language = self
                    .tokenizer
                    .language_code(decoded.language_token)
                    .ok_or_else(|| {
                        anyhow!(
                            "Whisper returned unsupported language token {}",
                            decoded.language_token
                        )
                    })?;
                info!(
                    language,
                    token = decoded.language_token,
                    "Whisper recording language detected"
                );
                language_token = Some(decoded.language_token);
            }

            transcript_tokens.extend(decoded.tokens);
            info!(
                chunk = index + 1,
                chunks = chunk_count,
                encoder_ms,
                decoder_ms,
                decode_steps = decoded.steps,
                "Whisper NPU chunk complete"
            );
        }

        info!(
            chunks = chunk_count,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "Whisper NPU recording complete"
        );
        self.tokenizer.decode(&transcript_tokens)
    }
}

fn load_session(
    environment: *const ort::sys::OrtEnv,
    devices: &[*const ort::sys::OrtEpDevice],
    path: &Path,
    contract: super::qnn_ffi::SessionContract,
    provider_lease: super::qnn_ffi::QnnProviderLease,
) -> Result<QnnSession> {
    QnnSession::load(environment, devices, path, contract, provider_lease)
        .with_context(|| format!("load Whisper NPU session {}", path.display()))
}

fn ensure_supported_hardware() -> Result<()> {
    #[cfg(not(target_arch = "aarch64"))]
    {
        bail!("Whisper NPU requires the ARM64 build on Snapdragon X Elite");
    }
    #[cfg(all(target_arch = "aarch64", not(windows)))]
    {
        bail!("Whisper NPU requires Windows on Snapdragon X Elite");
    }
    #[cfg(all(target_arch = "aarch64", windows))]
    {
        let processor = processor_name()?;
        if !is_snapdragon_x_elite(&processor) {
            bail!("Whisper NPU requires Snapdragon X Elite; detected processor: {processor}");
        }
        info!(processor, "supported Whisper NPU hardware detected");
        Ok(())
    }
}

pub(super) fn hardware_status() -> Result<String> {
    #[cfg(not(target_arch = "aarch64"))]
    {
        return Ok("unavailable: Whisper NPU requires ARM64 Snapdragon X Elite".to_string());
    }
    #[cfg(all(target_arch = "aarch64", not(windows)))]
    {
        return Ok("unavailable: Whisper NPU requires Windows".to_string());
    }
    #[cfg(all(target_arch = "aarch64", windows))]
    {
        let processor = processor_name()?;
        let state = if is_snapdragon_x_elite(&processor) {
            "supported"
        } else {
            "unsupported"
        };
        Ok(format!("{state}: {processor}"))
    }
}

#[cfg(any(target_arch = "aarch64", test))]
fn is_snapdragon_x_elite(processor: &str) -> bool {
    let processor = processor.to_ascii_lowercase();
    processor.contains("snapdragon") && (processor.contains("x elite") || processor.contains("x1e"))
}

#[cfg(all(target_arch = "aarch64", windows))]
fn processor_name() -> Result<String> {
    use windows::core::w;
    use windows::Win32::System::Registry::{RegGetValueW, HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ};

    let mut bytes = 0_u32;
    unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            w!("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
            w!("ProcessorNameString"),
            RRF_RT_REG_SZ,
            None,
            None,
            Some(&mut bytes),
        )
        .ok()
        .context("read Snapdragon processor name size")?;
    }
    if bytes < 2 {
        bail!("Windows returned an empty processor name");
    }

    let mut buffer = vec![0_u16; (bytes as usize).div_ceil(2)];
    unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            w!("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
            w!("ProcessorNameString"),
            RRF_RT_REG_SZ,
            None,
            Some(buffer.as_mut_ptr().cast()),
            Some(&mut bytes),
        )
        .ok()
        .context("read Snapdragon processor name")?;
    }
    let length = buffer
        .iter()
        .position(|value| *value == 0)
        .unwrap_or(buffer.len());
    String::from_utf16(&buffer[..length]).context("decode Snapdragon processor name")
}

fn validate_sample_rate(sample_rate: u32) -> Result<()> {
    if sample_rate == 0 {
        bail!("cannot transcribe audio with a zero sample rate");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_sample_rate_before_resampling() {
        assert_eq!(
            validate_sample_rate(0).unwrap_err().to_string(),
            "cannot transcribe audio with a zero sample rate"
        );
    }

    #[test]
    fn chunk_size_is_exactly_thirty_seconds() {
        assert_eq!(CHUNK_SAMPLES, 480_000);
        assert_eq!(480_001usize.div_ceil(CHUNK_SAMPLES), 2);
    }

    #[test]
    fn recognizes_only_snapdragon_x_elite_processor_names() {
        assert!(is_snapdragon_x_elite(
            "Snapdragon(R) X 12-core X1E80100 @ 3.40 GHz"
        ));
        assert!(is_snapdragon_x_elite("Qualcomm Snapdragon X Elite"));
        assert!(!is_snapdragon_x_elite("Snapdragon X Plus X1P64100"));
        assert!(!is_snapdragon_x_elite("Intel Core Ultra"));
    }
}
