use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parking_lot::Mutex;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc,
};
use tracing::{info, warn};

pub struct Recorder {
    pub recording: Arc<AtomicBool>,
    pub last_rms_x10000: Arc<AtomicU32>,
    samples: Arc<Mutex<Vec<f32>>>,
    pub sample_rate: u32,
    pub channels: u16,
    _stream: cpal::Stream,
}

impl Recorder {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device"))?;
        let config = device.default_input_config()?;
        let sample_rate = config.sample_rate().0;
        let channels = config.channels();
        info!(device = %device.name().unwrap_or_default(), sample_rate, channels, "audio device");

        let samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::with_capacity(16_000 * 30)));
        let recording = Arc::new(AtomicBool::new(false));
        let level = Arc::new(AtomicU32::new(0));

        let samples_cb = samples.clone();
        let recording_cb = recording.clone();
        let level_cb = level.clone();

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _| {
                    if !recording_cb.load(Ordering::Relaxed) { return; }
                    // Downmix to mono on-the-fly: average across channels.
                    // `data` is interleaved [ch0, ch1, ..., chN-1, ch0, ...].
                    let ch = channels as usize;
                    let frames = data.len() / ch.max(1);
                    let mut mono = Vec::with_capacity(frames);
                    let mut sumsq = 0f32;
                    if ch <= 1 {
                        mono.extend_from_slice(data);
                        for &x in data { sumsq += x * x; }
                    } else {
                        for f in 0..frames {
                            let mut acc = 0f32;
                            for c in 0..ch { acc += data[f * ch + c]; }
                            let s = acc / ch as f32;
                            mono.push(s);
                            sumsq += s * s;
                        }
                    }
                    let n = mono.len().max(1);
                    let rms = (sumsq / n as f32).sqrt();
                    // Multi-channel downmix attenuates each channel; scale the
                    // level report up so the overlay meter reflects real loudness.
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

        Ok(Self { recording, last_rms_x10000: level, samples, sample_rate, channels, _stream: stream })
    }

    pub fn start(&self) {
        self.samples.lock().clear();
        self.last_rms_x10000.store(0, Ordering::Relaxed);
        self.recording.store(true, Ordering::Relaxed);
    }

    pub fn stop(&self) -> Vec<f32> {
        self.recording.store(false, Ordering::Relaxed);
        self.last_rms_x10000.store(0, Ordering::Relaxed);
        let mut s = self.samples.lock();
        std::mem::take(&mut *s)
    }
}
