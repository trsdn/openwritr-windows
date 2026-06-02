//! TDT greedy decoder for Parakeet (matches istupakov/onnx-asr loop).
//!
//! Decoder/joint is a single fused ONNX graph that consumes one encoder
//! frame at a time plus the previous LSTM self-state, and produces:
//!   * `outputs` — logits over vocab + duration head; output[..vocab] are
//!                 token logits (including blank), output[vocab..] are
//!                 duration logits. We argmax both and use the duration to
//!                 advance `t` (the encoder time index).
//!   * `output_states_1/2` — new LSTM hidden + cell state.
//!
//! For pure RNN-T (no TDT) the step would always be 1. With TDT, blank +
//! step>0 is the common case which is what gives the model its real-time
//! lead over plain RNN-T.

use super::ort_helpers::OrtResultExt;
use anyhow::{anyhow, Result};
use ndarray::{Array2, Array3, ArrayD, Axis, Ix3};
use ort::session::Session;
use ort::value::{Outlet, Value, ValueType};

pub const MAX_TOKENS_PER_STEP: usize = 10;

pub struct DecoderState {
    state1: Array3<f32>,
    state2: Array3<f32>,
}

pub struct Tdt {
    pub vocab_size: usize,
    pub blank_id: i32,
}

impl Tdt {
    pub fn init_state(&self, decoder: &Session) -> Result<DecoderState> {
        // We read the input shapes from the decoder's `input_states_*` inputs
        // to honour whatever LSTM size the upstream export used.
        let inputs = decoder.inputs();
        let s1 = find_shape(inputs, "input_states_1")?;
        let s2 = find_shape(inputs, "input_states_2")?;
        Ok(DecoderState {
            state1: Array3::<f32>::zeros((s1.0, 1, s1.2)),
            state2: Array3::<f32>::zeros((s2.0, 1, s2.2)),
        })
    }

    pub fn decode(
        &self,
        decoder: &mut Session,
        encoder_out: &ndarray::Array3<f32>,    // (1, T, D) after transpose
        encoder_out_len: usize,
    ) -> Result<Vec<i32>> {
        let mut state = self.init_state(decoder)?;
        let mut tokens: Vec<i32> = Vec::new();
        let mut t = 0usize;
        let mut emitted_this_step = 0usize;

        while t < encoder_out_len {
            // (1, D, 1) — single encoder frame
            let enc_t = encoder_out
                .index_axis(Axis(1), t)        // (1, D)
                .to_owned();
            let enc_t = enc_t.insert_axis(Axis(2));   // (1, D, 1)

            let prev_token = *tokens.last().unwrap_or(&self.blank_id);
            let targets = Array2::<i32>::from_elem((1, 1), prev_token);
            let target_length = ndarray::Array1::<i32>::from_elem(1, 1);

            let inputs = ort::inputs![
                "encoder_outputs" => Value::from_array(enc_t)?,
                "targets" => Value::from_array(targets)?,
                "target_length" => Value::from_array(target_length)?,
                "input_states_1" => Value::from_array(state.state1.clone())?,
                "input_states_2" => Value::from_array(state.state2.clone())?,
            ];
            let outs = decoder.run(inputs).ortx()?;
            let logits = outs
                .get("outputs")
                .ok_or_else(|| anyhow!("decoder missing 'outputs'"))?
                .try_extract_array::<f32>().ortx()?
                .to_owned();
            let s1 = outs
                .get("output_states_1")
                .ok_or_else(|| anyhow!("decoder missing output_states_1"))?
                .try_extract_array::<f32>().ortx()?
                .to_owned()
                .into_dimensionality::<Ix3>()?;
            let s2 = outs
                .get("output_states_2")
                .ok_or_else(|| anyhow!("decoder missing output_states_2"))?
                .try_extract_array::<f32>().ortx()?
                .to_owned()
                .into_dimensionality::<Ix3>()?;

            let flat: Vec<f32> = logits.iter().copied().collect();
            if flat.len() < self.vocab_size {
                return Err(anyhow!(
                    "joint logits {} smaller than vocab {}",
                    flat.len(),
                    self.vocab_size
                ));
            }
            let token_logits = &flat[..self.vocab_size];
            let duration_logits = &flat[self.vocab_size..];

            let token = argmax(token_logits) as i32;
            let step = if duration_logits.is_empty() {
                1usize
            } else {
                argmax(duration_logits)
            };

            if token != self.blank_id {
                tokens.push(token);
                state = DecoderState { state1: s1, state2: s2 };
                emitted_this_step += 1;
            }

            if step > 0 {
                t += step;
                emitted_this_step = 0;
            } else if token == self.blank_id || emitted_this_step >= MAX_TOKENS_PER_STEP {
                t += 1;
                emitted_this_step = 0;
            }
        }
        Ok(tokens)
    }
}

fn argmax(xs: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn find_shape(inputs: &[Outlet], name: &str) -> Result<(usize, usize, usize)> {
    let input = inputs
        .iter()
        .find(|i| i.name() == name)
        .ok_or_else(|| anyhow!("decoder input '{name}' missing"))?;
    let dims = match input.dtype() {
        ValueType::Tensor { shape, .. } => shape,
        _ => anyhow::bail!("input '{name}' is not a tensor"),
    };
    if dims.len() < 3 {
        anyhow::bail!("input '{name}' rank {} != 3", dims.len());
    }
    let d0 = (dims[0].max(1)) as usize;
    let d1 = (dims[1].max(1)) as usize;
    let d2 = (dims[2].max(1)) as usize;
    Ok((d0, d1, d2))
}

#[allow(dead_code)]
fn _phantom() -> ArrayD<f32> { ArrayD::<f32>::zeros(ndarray::IxDyn(&[])) }
