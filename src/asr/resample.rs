//! Any-sample-rate f32 mono → 16 kHz mono via rubato sinc resampling.

use anyhow::Result;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

pub const TARGET_SR: u32 = 16_000;

pub fn to_16k_mono(input: &[f32], input_sr: u32) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    if input_sr == TARGET_SR {
        return Ok(input.to_vec());
    }
    let ratio = TARGET_SR as f64 / input_sr as f64;
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };
    let chunk = input.len();
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk, 1)?;
    let waves_in = vec![input.to_vec()];
    let waves_out = resampler.process(&waves_in, None)?;
    Ok(waves_out.into_iter().next().unwrap_or_default())
}
