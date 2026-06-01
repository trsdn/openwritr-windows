// Log-Mel filterbank features for NVIDIA Parakeet (NeMo `AudioToMelSpectrogramPreprocessor`).
//
// Reference config (parakeet-tdt-0.6b-v3):
//   sample_rate           16_000
//   n_fft                 512
//   win_length            400   (25 ms)
//   hop_length            160   (10 ms)
//   window                hann
//   n_mels                128
//   fmin                  0
//   fmax                  8_000  (Nyquist)
//   preemph               0.97
//   features              log (natural log with `log_zero_guard_value = 1e-8`)
//   mag_power             2.0  (power spectrum)
//   normalize             per-feature (mean-var per mel bin across time)
//
// Output shape: (n_mels=128, T) — same layout NeMo's encoder expects after
// the preprocessor module.

use ndarray::Array2;
use once_cell::sync::Lazy;

pub const SAMPLE_RATE: u32 = 16_000;
pub const N_FFT: usize = 512;
pub const WIN_LENGTH: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
pub const FMIN: f32 = 0.0;
pub const FMAX: f32 = 8_000.0;
pub const PREEMPH: f32 = 0.97;
pub const LOG_EPS: f32 = 1e-8;

static HANN: Lazy<Vec<f32>> = Lazy::new(|| {
    // Periodic Hann window of length WIN_LENGTH (matches torch.hann_window default).
    (0..WIN_LENGTH)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / WIN_LENGTH as f32).cos())
        .collect()
});

static MEL_FB: Lazy<Vec<Vec<f32>>> = Lazy::new(|| build_mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE, FMIN, FMAX));

/// Compute log-mel spectrogram. Input is 16 kHz mono f32 in [-1, 1].
/// Returns `Array2<f32>` shaped (n_mels, n_frames).
pub fn log_mel(audio: &[f32]) -> Array2<f32> {
    if audio.len() < WIN_LENGTH {
        return Array2::zeros((N_MELS, 0));
    }

    // Pre-emphasis: y[n] = x[n] - preemph * x[n-1]
    let mut pre = Vec::with_capacity(audio.len());
    pre.push(audio[0]);
    for i in 1..audio.len() {
        pre.push(audio[i] - PREEMPH * audio[i - 1]);
    }

    // Center-padding to match torch.stft(center=True): reflect-pad N_FFT/2 on both sides.
    let pad = N_FFT / 2;
    let padded = reflect_pad(&pre, pad);

    let n_frames = if padded.len() < N_FFT {
        0
    } else {
        1 + (padded.len() - N_FFT) / HOP_LENGTH
    };

    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut buf = vec![0f32; N_FFT];
    let mut spec = fft.make_output_vec();

    let mut out = Array2::<f32>::zeros((N_MELS, n_frames));
    let win_pad_l = (N_FFT - WIN_LENGTH) / 2;

    for t in 0..n_frames {
        let start = t * HOP_LENGTH;
        // Zero-init then write windowed slice centered inside N_FFT.
        buf.iter_mut().for_each(|x| *x = 0.0);
        for j in 0..WIN_LENGTH {
            let s = padded.get(start + j).copied().unwrap_or(0.0);
            buf[win_pad_l + j] = s * HANN[j];
        }
        fft.process(&mut buf, &mut spec).expect("fft");

        // Power spectrum: |X[k]|^2
        let mut power = [0f32; N_FFT / 2 + 1];
        for (k, c) in spec.iter().enumerate() {
            power[k] = c.re * c.re + c.im * c.im;
        }

        // Mel projection.
        for m in 0..N_MELS {
            let fb = &MEL_FB[m];
            let mut acc = 0f32;
            for k in 0..fb.len() {
                acc += fb[k] * power[k];
            }
            out[[m, t]] = (acc + LOG_EPS).ln();
        }
    }

    // Per-feature mean/var normalize (NeMo's `per_feature` norm).
    per_feature_normalize(&mut out);
    out
}

fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    // left: x[pad], x[pad-1], ..., x[1]
    for i in 0..pad {
        let src = if pad >= i { pad - i } else { 0 };
        out.push(x[src.min(n - 1)]);
    }
    out.extend_from_slice(x);
    for i in 0..pad {
        // right: x[n-2], x[n-3], ..., x[n-pad-1]
        let src = n.saturating_sub(2 + i);
        out.push(x[src]);
    }
    out
}

fn per_feature_normalize(spec: &mut Array2<f32>) {
    let (n_mels, n_frames) = spec.dim();
    if n_frames == 0 {
        return;
    }
    for m in 0..n_mels {
        let row = spec.row(m);
        let mean = row.sum() / n_frames as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n_frames as f32;
        let std = (var + 1e-5).sqrt();
        for t in 0..n_frames {
            spec[[m, t]] = (spec[[m, t]] - mean) / std;
        }
    }
}

fn hz_to_mel(f: f32) -> f32 {
    // NeMo uses the librosa "slaney" mel scale.
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + ((f / min_log_hz).ln() / logstep)
    } else {
        (f - f_min) / f_sp
    }
}

fn mel_to_hz(m: f32) -> f32 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_min + f_sp * m
    }
}

fn build_mel_filterbank(n_mels: usize, n_fft: usize, sr: u32, fmin: f32, fmax: f32) -> Vec<Vec<f32>> {
    let n_bins = n_fft / 2 + 1;
    // FFT bin center frequencies in Hz.
    let fft_freqs: Vec<f32> = (0..n_bins).map(|k| k as f32 * sr as f32 / n_fft as f32).collect();
    // Mel-spaced edge frequencies.
    let m_min = hz_to_mel(fmin);
    let m_max = hz_to_mel(fmax);
    let mel_pts: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_to_hz(m_min + (m_max - m_min) * i as f32 / (n_mels as f32 + 1.0)))
        .collect();
    // Slaney-norm triangular filters.
    let mut fb = vec![vec![0f32; n_bins]; n_mels];
    for m in 0..n_mels {
        let f_left = mel_pts[m];
        let f_center = mel_pts[m + 1];
        let f_right = mel_pts[m + 2];
        let enorm = 2.0 / (f_right - f_left);
        for k in 0..n_bins {
            let f = fft_freqs[k];
            let weight = if f < f_left || f > f_right {
                0.0
            } else if f <= f_center {
                (f - f_left) / (f_center - f_left)
            } else {
                (f_right - f) / (f_right - f_center)
            };
            fb[m][k] = weight * enorm;
        }
    }
    fb
}
