use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parking_lot::Mutex;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc,
};
use tracing::{info, warn};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureInfo {
    pub device_name: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub max_samples: usize,
}

#[derive(Debug)]
pub struct Recording {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub device_name: String,
    pub reached_limit: bool,
    pub stream_error: Option<String>,
}

/// Push-to-talk capture.
///
/// The WASAPI capture stream is built fresh on every `start()` and dropped on
/// `stop()`, rather than held open for the process lifetime. Holding it open
/// caused a "mic stops working after idle" bug: after ~30 min Windows puts the
/// Qualcomm Aqstic array into a low-power state (or the default device changes)
/// and silently invalidates the stream — sometimes without firing the error
/// callback at all. The data callback then never runs again, recordings come
/// back empty, and only an app restart fixed it. Rebuilding per recording also
/// means we always capture from the *current* default input device, so
/// switching mics mid-session just works.
pub struct Recorder {
    pub recording: Arc<AtomicBool>,
    pub last_rms_x10000: Arc<AtomicU32>,
    samples: Arc<Mutex<Vec<f32>>>,
    active: Mutex<Option<CaptureInfo>>,
    limit_reached: Arc<AtomicBool>,
    stream_error: Arc<Mutex<Option<String>>>,
    stream: Mutex<Option<cpal::Stream>>,
}

impl Recorder {
    pub fn new() -> Result<Self> {
        // Probe only for diagnostics. Every recording uses metadata from the
        // newly opened stream, so switching devices cannot leave a stale rate.
        match Self::default_device_format() {
            Ok((_, sr, ch, name)) => {
                info!(device = %name, sample_rate = sr, channels = ch, "audio device");
            }
            Err(e) => {
                warn!(error = %e, "no input device at startup; will retry on first record");
            }
        }

        Ok(Self {
            recording: Arc::new(AtomicBool::new(false)),
            last_rms_x10000: Arc::new(AtomicU32::new(0)),
            samples: Arc::new(Mutex::new(Vec::new())),
            active: Mutex::new(None),
            limit_reached: Arc::new(AtomicBool::new(false)),
            stream_error: Arc::new(Mutex::new(None)),
            stream: Mutex::new(None),
        })
    }

    fn default_device_format() -> Result<(cpal::Device, u32, u16, String)> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device"))?;
        let config = device.default_input_config()?;
        let name = device.name().unwrap_or_default();
        Ok((device, config.sample_rate().0, config.channels(), name))
    }

    /// Build + play a fresh capture stream against the current default device.
    fn build_stream(&self, max_record_seconds: f32) -> Result<(cpal::Stream, CaptureInfo)> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device"))?;
        let config = device.default_input_config()?;
        let device_name = device.name().unwrap_or_else(|_| "Default input".into());
        let sample_rate = config.sample_rate().0;
        let channels = config.channels();
        let max_samples = max_samples_for(sample_rate, max_record_seconds)?;
        let stream_config: cpal::StreamConfig = config.clone().into();

        let samples_cb = self.samples.clone();
        let recording_cb = self.recording.clone();
        let level_cb = self.last_rms_x10000.clone();
        let limit_cb = self.limit_reached.clone();
        let error_cb = self.stream_error.clone();
        let recording_error_cb = recording_cb.clone();
        let ch = channels as usize;
        let on_error = move |err: cpal::StreamError| {
            let message = err.to_string();
            warn!(error = %message, "cpal stream error");
            *error_cb.lock() = Some(message);
            recording_error_cb.store(false, Ordering::Relaxed);
        };

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &stream_config,
                move |data: &[f32], _| {
                    append_interleaved(
                        data,
                        ch,
                        max_samples,
                        &samples_cb,
                        &recording_cb,
                        &level_cb,
                        &limit_cb,
                        |sample| sample,
                    );
                },
                on_error,
                None,
            )?,
            cpal::SampleFormat::I16 => {
                let samples_cb = self.samples.clone();
                let recording_cb = self.recording.clone();
                let level_cb = self.last_rms_x10000.clone();
                let limit_cb = self.limit_reached.clone();
                let error_cb = self.stream_error.clone();
                let recording_error_cb = recording_cb.clone();
                device.build_input_stream(
                    &stream_config,
                    move |data: &[i16], _| {
                        append_interleaved(
                            data,
                            ch,
                            max_samples,
                            &samples_cb,
                            &recording_cb,
                            &level_cb,
                            &limit_cb,
                            i16_to_f32,
                        );
                    },
                    move |err| {
                        let message = err.to_string();
                        warn!(error = %message, "cpal stream error");
                        *error_cb.lock() = Some(message);
                        recording_error_cb.store(false, Ordering::Relaxed);
                    },
                    None,
                )?
            }
            cpal::SampleFormat::U16 => {
                let samples_cb = self.samples.clone();
                let recording_cb = self.recording.clone();
                let level_cb = self.last_rms_x10000.clone();
                let limit_cb = self.limit_reached.clone();
                let error_cb = self.stream_error.clone();
                let recording_error_cb = recording_cb.clone();
                device.build_input_stream(
                    &stream_config,
                    move |data: &[u16], _| {
                        append_interleaved(
                            data,
                            ch,
                            max_samples,
                            &samples_cb,
                            &recording_cb,
                            &level_cb,
                            &limit_cb,
                            u16_to_f32,
                        );
                    },
                    move |err| {
                        let message = err.to_string();
                        warn!(error = %message, "cpal stream error");
                        *error_cb.lock() = Some(message);
                        recording_error_cb.store(false, Ordering::Relaxed);
                    },
                    None,
                )?
            }
            other => return Err(anyhow!("unsupported sample format: {other:?}")),
        };
        Ok((
            stream,
            CaptureInfo {
                device_name,
                sample_rate,
                channels,
                max_samples,
            },
        ))
    }

    pub fn start(&self, max_record_seconds: f32) -> Result<CaptureInfo> {
        if self.recording.load(Ordering::Relaxed) || self.active.lock().is_some() {
            bail!("capture stream is already active");
        }
        self.samples.lock().clear();
        self.last_rms_x10000.store(0, Ordering::Relaxed);
        self.limit_reached.store(false, Ordering::Relaxed);
        self.stream_error.lock().take();
        // Drop any previous stream before building a new one.
        *self.stream.lock() = None;
        let (stream, info) = self.build_stream(max_record_seconds)?;
        stream.play().context("start capture stream")?;
        *self.active.lock() = Some(info.clone());
        *self.stream.lock() = Some(stream);
        self.recording.store(true, Ordering::Relaxed);
        Ok(info)
    }

    pub fn stop(&self) -> Result<Recording> {
        self.recording.store(false, Ordering::Relaxed);
        self.last_rms_x10000.store(0, Ordering::Relaxed);
        // Drop the stream so the device is released between recordings.
        *self.stream.lock() = None;
        let info = self
            .active
            .lock()
            .take()
            .ok_or_else(|| anyhow!("capture stream is not active"))?;
        let samples = std::mem::take(&mut *self.samples.lock());
        Ok(Recording {
            samples,
            sample_rate: info.sample_rate,
            channels: info.channels,
            device_name: info.device_name,
            reached_limit: self.limit_reached.swap(false, Ordering::Relaxed),
            stream_error: self.stream_error.lock().take(),
        })
    }

    pub fn limit_reached(&self) -> bool {
        self.limit_reached.load(Ordering::Relaxed)
    }

    pub fn stream_failed(&self) -> bool {
        self.stream_error.lock().is_some()
    }
}

fn max_samples_for(sample_rate: u32, max_record_seconds: f32) -> Result<usize> {
    if sample_rate == 0 {
        bail!("input sample rate is zero");
    }
    if !max_record_seconds.is_finite() || max_record_seconds <= 0.0 {
        bail!("max_record_seconds must be a positive finite number");
    }
    let samples = (sample_rate as f64 * max_record_seconds as f64).ceil();
    if samples > usize::MAX as f64 {
        bail!("recording sample limit is too large");
    }
    Ok((samples as usize).max(1))
}

fn i16_to_f32(sample: i16) -> f32 {
    sample as f32 / 32_768.0
}

fn u16_to_f32(sample: u16) -> f32 {
    (sample as f32 - 32_768.0) / 32_768.0
}

fn append_interleaved<T: Copy>(
    data: &[T],
    channels: usize,
    max_samples: usize,
    samples: &Mutex<Vec<f32>>,
    recording: &AtomicBool,
    level_x10000: &AtomicU32,
    limit_reached: &AtomicBool,
    convert: fn(T) -> f32,
) {
    if !recording.load(Ordering::Relaxed) {
        return;
    }

    let channels = channels.max(1);
    let frames = data.len() / channels;
    let mut out = samples.lock();
    if !recording.load(Ordering::Relaxed) {
        return;
    }
    let frames_to_take = frames.min(max_samples.saturating_sub(out.len()));
    let mut sumsq = 0.0f32;

    for frame in data.chunks_exact(channels).take(frames_to_take) {
        let mono = frame.iter().copied().map(convert).sum::<f32>() / channels as f32;
        out.push(mono);
        sumsq += mono * mono;
    }

    if frames_to_take > 0 {
        let rms = (sumsq / frames_to_take as f32).sqrt();
        let level_gain = if channels == 1 {
            1.0
        } else {
            (channels as f32).sqrt()
        };
        let scaled = ((rms * level_gain).min(1.0) * 10_000.0) as u32;
        level_x10000.store(scaled, Ordering::Relaxed);
    }

    if out.len() >= max_samples {
        limit_reached.store(true, Ordering::Relaxed);
        recording.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture<T: Copy>(
        data: &[T],
        channels: usize,
        max_samples: usize,
        convert: fn(T) -> f32,
    ) -> (Vec<f32>, bool, bool) {
        let samples = Mutex::new(Vec::new());
        let recording = AtomicBool::new(true);
        let level = AtomicU32::new(0);
        let limit = AtomicBool::new(false);
        append_interleaved(
            data,
            channels,
            max_samples,
            &samples,
            &recording,
            &level,
            &limit,
            convert,
        );
        let captured = samples.lock().clone();
        (
            captured,
            recording.load(Ordering::Relaxed),
            limit.load(Ordering::Relaxed),
        )
    }

    #[test]
    fn downmixes_interleaved_f32_to_mono() {
        let (samples, recording, limit) = capture(&[1.0, -1.0, 0.5, 0.25], 2, 10, |sample| sample);
        assert_eq!(samples, vec![0.0, 0.375]);
        assert!(recording);
        assert!(!limit);
    }

    #[test]
    fn converts_common_integer_formats() {
        let (signed, _, _) = capture(&[i16::MIN, 0, i16::MAX], 1, 10, i16_to_f32);
        assert_eq!(signed[0], -1.0);
        assert_eq!(signed[1], 0.0);
        assert!((signed[2] - 0.999_969_5).abs() < 0.000_001);

        let (unsigned, _, _) = capture(&[u16::MIN, 32_768, u16::MAX], 1, 10, u16_to_f32);
        assert_eq!(unsigned[0], -1.0);
        assert_eq!(unsigned[1], 0.0);
        assert!((unsigned[2] - 0.999_969_5).abs() < 0.000_001);
    }

    #[test]
    fn enforces_the_hard_sample_limit() {
        let samples = Mutex::new(Vec::new());
        let recording = AtomicBool::new(true);
        let level = AtomicU32::new(0);
        let limit = AtomicBool::new(false);
        append_interleaved(
            &[0.1, 0.2, 0.3, 0.4],
            1,
            2,
            &samples,
            &recording,
            &level,
            &limit,
            |sample| sample,
        );
        append_interleaved(
            &[0.5, 0.6],
            1,
            2,
            &samples,
            &recording,
            &level,
            &limit,
            |sample| sample,
        );

        assert_eq!(*samples.lock(), vec![0.1, 0.2]);
        assert!(!recording.load(Ordering::Relaxed));
        assert!(limit.load(Ordering::Relaxed));
    }

    #[test]
    fn derives_limit_from_the_active_sample_rate() {
        assert_eq!(max_samples_for(48_000, 1.5).unwrap(), 72_000);
        assert!(max_samples_for(48_000, 0.0).is_err());
        assert!(max_samples_for(48_000, f32::NAN).is_err());
    }

    #[test]
    fn stop_returns_the_active_stream_metadata() {
        let recorder = Recorder {
            recording: Arc::new(AtomicBool::new(true)),
            last_rms_x10000: Arc::new(AtomicU32::new(123)),
            samples: Arc::new(Mutex::new(vec![0.1, 0.2])),
            active: Mutex::new(Some(CaptureInfo {
                device_name: "Switched microphone".into(),
                sample_rate: 48_000,
                channels: 2,
                max_samples: 96_000,
            })),
            limit_reached: Arc::new(AtomicBool::new(false)),
            stream_error: Arc::new(Mutex::new(None)),
            stream: Mutex::new(None),
        };

        let recording = recorder.stop().unwrap();
        assert_eq!(recording.sample_rate, 48_000);
        assert_eq!(recording.channels, 2);
        assert_eq!(recording.device_name, "Switched microphone");
        assert_eq!(recording.samples, vec![0.1, 0.2]);
    }
}
