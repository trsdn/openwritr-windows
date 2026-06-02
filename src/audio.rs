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
        info!(device = %device.name().unwrap_or_default(), sample_rate, "audio device");

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
                    let mut sumsq = 0f32;
                    for &x in data { sumsq += x * x; }
                    let rms = (sumsq / data.len().max(1) as f32).sqrt();
                    let scaled = (rms.min(1.0) * 10_000.0) as u32;
                    level_cb.store(scaled, Ordering::Relaxed);
                    samples_cb.lock().extend_from_slice(data);
                },
                |err| warn!(?err, "cpal stream error"),
                None,
            )?,
            other => return Err(anyhow!("unsupported sample format: {other:?}")),
        };
        stream.play()?;

        Ok(Self { recording, last_rms_x10000: level, samples, sample_rate, _stream: stream })
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
