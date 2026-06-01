// WASAPI shared-mode capture at 16 kHz mono via cpal.
// Pushes f32 samples into a lock-free SPSC ring buffer that the ASR task drains.
//
// Notes:
// - Windows default audio is rarely 16 kHz; we capture at the device's native
//   sample rate and resample on the consumer side (planned: rubato in asr/).
// - The capture thread is owned by cpal and is realtime-sensitive; do NOT
//   allocate, log, or block inside `data_callback`.

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{traits::*, HeapRb};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tracing::{info, warn};

pub struct Capture {
    pub recording: Arc<AtomicBool>,
    pub consumer: ringbuf::HeapCons<f32>,
    pub sample_rate: u32,
    _stream: cpal::Stream,
}

pub struct CaptureHandle {
    pub recording: Arc<AtomicBool>,
    pub take_samples: Box<dyn FnMut() -> Vec<f32> + Send + 'static>,
    pub sample_rate: u32,
}

impl Capture {
    pub fn spawn() -> Result<CaptureHandle> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device"))?;
        let config = device.default_input_config()?;
        let sample_rate = config.sample_rate().0;
        info!(device = %device.name().unwrap_or_default(), sample_rate, "audio device");

        // 30 s of headroom at the native sample rate.
        let rb = HeapRb::<f32>::new((sample_rate as usize) * 30);
        let (mut prod, cons) = rb.split();

        let recording = Arc::new(AtomicBool::new(false));
        let rec_clone = recording.clone();

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _| {
                    if rec_clone.load(Ordering::Relaxed) {
                        let _ = prod.push_slice(data);
                    }
                },
                |err| warn!(?err, "cpal stream error"),
                None,
            )?,
            other => return Err(anyhow!("unsupported sample format: {other:?}")),
        };
        stream.play()?;

        // We move the stream into the handle to keep it alive; consumer is
        // owned via a closure so callers don't need to know about ringbuf types.
        let consumer = std::sync::Mutex::new(cons);
        let take_samples = Box::new(move || {
            let mut c = consumer.lock().unwrap();
            let mut out = Vec::with_capacity(c.occupied_len());
            while let Some(s) = c.try_pop() {
                out.push(s);
            }
            out
        });

        Ok(CaptureHandle {
            recording,
            take_samples,
            sample_rate,
        })
    }
}
