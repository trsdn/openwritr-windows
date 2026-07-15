//! Whisper Large v3 128-bin log-mel frontend.

use anyhow::{bail, Result};
use half::f16;
use once_cell::sync::Lazy;
use realfft::RealFftPlanner;

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
pub const N_FRAMES: usize = 3_000;
pub const N_SAMPLES: usize = (N_FRAMES - 1) * HOP_LENGTH + N_FFT;

static HANN_WINDOW: Lazy<Vec<f32>> = Lazy::new(|| {
    (0..N_FFT)
        .map(|index| {
            let phase = 2.0 * std::f64::consts::PI * index as f64 / N_FFT as f64;
            (0.5 - 0.5 * phase.cos()) as f32
        })
        .collect()
});

static MEL_FILTERS: Lazy<Vec<f32>> = Lazy::new(build_mel_filters);

pub struct LogMel {
    values: Vec<u16>,
}

impl LogMel {
    pub fn f16_bits(&self) -> &[u16] {
        &self.values
    }

    pub fn dimensions(&self) -> [i64; 3] {
        [1, N_MELS as i64, N_FRAMES as i64]
    }
}

pub fn log_mel_30s(audio: &[f32]) -> Result<LogMel> {
    if audio.iter().any(|sample| !sample.is_finite()) {
        bail!("Whisper audio contains a non-finite sample");
    }

    let mut fixed = vec![0.0_f32; N_SAMPLES];
    let copied = audio.len().min(N_SAMPLES);
    fixed[..copied].copy_from_slice(&audio[..copied]);
    let centered = reflect_pad(&fixed, N_FFT / 2);

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut frame = vec![0.0_f32; N_FFT];
    let mut spectrum = fft.make_output_vec();
    let mut scratch = fft.make_scratch_vec();
    let mut power = vec![0.0_f32; N_FFT / 2 + 1];
    let mut mel = vec![0.0_f32; N_MELS * N_FRAMES];

    for frame_index in 0..N_FRAMES {
        let start = frame_index * HOP_LENGTH;
        for index in 0..N_FFT {
            frame[index] = centered[start + index] * HANN_WINDOW[index];
        }
        fft.process_with_scratch(&mut frame, &mut spectrum, &mut scratch)
            .map_err(|error| anyhow::anyhow!("Whisper STFT failed: {error}"))?;
        for (index, value) in spectrum.iter().enumerate() {
            power[index] = value.re.mul_add(value.re, value.im * value.im);
        }
        for mel_index in 0..N_MELS {
            let filter = &MEL_FILTERS[mel_index * power.len()..(mel_index + 1) * power.len()];
            let energy = filter
                .iter()
                .zip(&power)
                .fold(0.0_f32, |sum, (weight, value)| weight.mul_add(*value, sum));
            mel[mel_index * N_FRAMES + frame_index] = energy.max(1.0e-10);
        }
    }

    let mut maximum = f32::NEG_INFINITY;
    for value in &mut mel {
        *value = value.log10();
        maximum = maximum.max(*value);
    }
    let floor = maximum - 8.0;
    let values = mel
        .into_iter()
        .map(|value| f16::from_f32((value.max(floor) + 4.0) / 4.0).to_bits())
        .collect();
    Ok(LogMel { values })
}

fn reflect_pad(audio: &[f32], padding: usize) -> Vec<f32> {
    debug_assert!(audio.len() > padding);
    let mut padded = vec![0.0_f32; audio.len() + padding * 2];
    padded[padding..padding + audio.len()].copy_from_slice(audio);
    for index in 0..padding {
        padded[index] = audio[padding - index];
        padded[padding + audio.len() + index] = audio[audio.len() - index - 2];
    }
    padded
}

fn build_mel_filters() -> Vec<f32> {
    let fft_bins = N_FFT / 2 + 1;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(SAMPLE_RATE as f64 / 2.0);
    let mel_points = (0..N_MELS + 2)
        .map(|index| {
            let ratio = index as f64 / (N_MELS + 1) as f64;
            mel_to_hz(mel_min + (mel_max - mel_min) * ratio)
        })
        .collect::<Vec<_>>();
    let mut filters = vec![0.0_f32; N_MELS * fft_bins];

    for mel_index in 0..N_MELS {
        let left = mel_points[mel_index];
        let center = mel_points[mel_index + 1];
        let right = mel_points[mel_index + 2];
        let normalization = 2.0 / (right - left);
        for fft_index in 0..fft_bins {
            let frequency = fft_index as f64 * (SAMPLE_RATE as f64 / 2.0) / (fft_bins - 1) as f64;
            let weight = if frequency < left || frequency > right {
                0.0
            } else if frequency <= center {
                (frequency - left) / (center - left)
            } else {
                (right - frequency) / (right - center)
            };
            filters[mel_index * fft_bins + fft_index] = (weight * normalization) as f32;
        }
    }
    filters
}

fn hz_to_mel(frequency: f64) -> f64 {
    const FREQ_STEP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1_000.0;
    let min_log_mel = MIN_LOG_HZ / FREQ_STEP;
    let log_step = 6.4_f64.ln() / 27.0;
    if frequency >= MIN_LOG_HZ {
        min_log_mel + (frequency / MIN_LOG_HZ).ln() / log_step
    } else {
        frequency / FREQ_STEP
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    const FREQ_STEP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1_000.0;
    let min_log_mel = MIN_LOG_HZ / FREQ_STEP;
    let log_step = 6.4_f64.ln() / 27.0;
    if mel >= min_log_mel {
        MIN_LOG_HZ * (log_step * (mel - min_log_mel)).exp()
    } else {
        FREQ_STEP * mel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_has_the_expected_shape_and_floor() {
        let mel = log_mel_30s(&[]).unwrap();
        assert_eq!(mel.dimensions(), [1, 128, 3_000]);
        assert_eq!(mel.f16_bits().len(), N_MELS * N_FRAMES);
        assert!(mel
            .f16_bits()
            .iter()
            .all(|value| f16::from_bits(*value).to_f32() == -1.5));
    }

    #[test]
    fn reflect_padding_matches_numpy_semantics() {
        assert_eq!(
            reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2),
            vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
    }

    #[test]
    fn rejects_non_finite_audio() {
        assert!(log_mel_30s(&[f32::NAN]).is_err());
    }

    #[test]
    fn matches_the_historical_scipy_golden_vector() {
        let audio = (0..SAMPLE_RATE * 2)
            .map(|index| (((index * 73) % 257) as i32 - 128) as f32 / 512.0)
            .collect::<Vec<_>>();
        let mel = log_mel_30s(&audio).unwrap();
        let expected = [
            ((0, 0), 0x36d3),
            ((1, 0), 0x3831),
            ((10, 0), 0x38a9),
            ((32, 0), 0x378d),
            ((64, 0), 0x381a),
            ((100, 0), 0x3954),
            ((127, 0), 0x37f7),
            ((10, 1), 0x359c),
            ((32, 10), 0x3788),
            ((64, 100), 0x3527),
            ((100, 199), 0x3697),
            ((127, 200), 0x35c0),
            ((0, 500), 0xbb71),
            ((64, 500), 0xbb71),
            ((127, 2_999), 0xbb71),
        ];
        for ((mel_index, frame_index), expected_bits) in expected {
            assert_eq!(
                mel.f16_bits()[mel_index * N_FRAMES + frame_index],
                expected_bits,
                "golden mismatch at mel {mel_index}, frame {frame_index}"
            );
        }
    }
}
