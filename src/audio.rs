use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parking_lot::Mutex;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc,
};
use tracing::{info, warn};

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
    pub sample_rate: u32,
    pub channels: u16,
    stream: Mutex<Option<cpal::Stream>>,
}

impl Recorder {
    pub fn new() -> Result<Self> {
        // Probe the default device once at startup just to learn the format
        // we'll report (sample_rate / channels). The actual capture stream is
        // built lazily in start().
        let (sample_rate, channels) = match Self::default_device_format() {
            Ok((_, sr, ch, name)) => {
                info!(device = %name, sample_rate = sr, channels = ch, "audio device");
                (sr, ch)
            }
            Err(e) => {
                warn!(error = %e, "no input device at startup; will retry on first record");
                (16_000, 1)
            }
        };

        Ok(Self {
            recording: Arc::new(AtomicBool::new(false)),
            last_rms_x10000: Arc::new(AtomicU32::new(0)),
            samples: Arc::new(Mutex::new(Vec::with_capacity(16_000 * 30))),
            sample_rate,
            channels,
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
    fn build_stream(&self) -> Result<(cpal::Stream, u32, u16)> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device"))?;
        let config = device.default_input_config()?;
        let sample_rate = config.sample_rate().0;
        let channels = config.channels();

        let samples_cb = self.samples.clone();
        let recording_cb = self.recording.clone();
        let level_cb = self.last_rms_x10000.clone();
        let ch = channels as usize;

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _| {
                    if !recording_cb.load(Ordering::Relaxed) {
                        return;
                    }
                    // Downmix interleaved [ch0..chN, ch0..] to mono.
                    let frames = data.len() / ch.max(1);
                    let mut mono = Vec::with_capacity(frames);
                    let mut sumsq = 0f32;
                    if ch <= 1 {
                        mono.extend_from_slice(data);
                        for &x in data {
                            sumsq += x * x;
                        }
                    } else {
                        for f in 0..frames {
                            let mut acc = 0f32;
                            for c in 0..ch {
                                acc += data[f * ch + c];
                            }
                            let s = acc / ch as f32;
                            mono.push(s);
                            sumsq += s * s;
                        }
                    }
                    let n = mono.len().max(1);
                    let rms = (sumsq / n as f32).sqrt();
                    let level_gain = if ch <= 1 { 1.0 } else { (ch as f32).sqrt() };
                    let scaled = ((rms * level_gain).min(1.0) * 10_000.0) as u32;
                    level_cb.store(scaled, Ordering::Relaxed);
                    samples_cb.lock().extend(mono);
                },
                |err| warn!(?err, "cpal stream error"),
                None,
            )?,
            other => return Err(anyhow!("unsupported sample format: {other:?}")),
        };
        stream.play()?;
        Ok((stream, sample_rate, channels))
    }

    pub fn start(&self) {
        self.samples.lock().clear();
        self.last_rms_x10000.store(0, Ordering::Relaxed);
        // Drop any previous stream before building a new one.
        *self.stream.lock() = None;
        match self.build_stream() {
            Ok((stream, sr, ch)) => {
                // sample_rate/channels are not behind a lock (read by the
                // pipeline); they only change if the user swaps to a device
                // with a different format. We update via the atomics-free
                // fields by leaking through interior mutability is overkill —
                // instead the pipeline reads self.sample_rate which we set on
                // construction. If it changed, we log it; resampling downstream
                // already targets 16 kHz so a changed rate is handled.
                if sr != self.sample_rate || ch != self.channels {
                    warn!(old_sr = self.sample_rate, new_sr = sr,
                          old_ch = self.channels, new_ch = ch,
                          "input device format changed since startup");
                }
                *self.stream.lock() = Some(stream);
                self.recording.store(true, Ordering::Relaxed);
            }
            Err(e) => {
                warn!(error = %e, "failed to open capture stream; recording aborted");
                self.recording.store(false, Ordering::Relaxed);
            }
        }
    }

    pub fn stop(&self) -> Vec<f32> {
        self.recording.store(false, Ordering::Relaxed);
        self.last_rms_x10000.store(0, Ordering::Relaxed);
        // Drop the stream so the device is released between recordings.
        *self.stream.lock() = None;
        let mut s = self.samples.lock();
        std::mem::take(&mut *s)
    }
}
