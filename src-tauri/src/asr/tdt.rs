// Token-and-Duration Transducer (TDT) greedy decoder for Parakeet TDT v3.
//
// Algorithm (Xu et al. 2023 "TDT — Efficient Sequence Transduction by
// Jointly Predicting Tokens and Durations"):
//
//   t = 0                              # encoder time index
//   predictor_state = predictor.zero_state()
//   last_token = blank
//   emitted = []
//   max_inner = 10                     # safety cap per frame
//   while t < T:
//       inner = 0
//       loop:
//           pred_out, predictor_state' = predictor(last_token, predictor_state)
//           joint_out = joint(encoder[t], pred_out)
//           token_logits  = joint_out[..vocab_size]
//           duration_logits = joint_out[vocab_size..]
//           token    = argmax(token_logits)
//           duration = TDT_DURATIONS[argmax(duration_logits)]
//           if token != blank:
//               emitted.push(token)
//               last_token = token
//               predictor_state = predictor_state'
//           t += duration
//           inner += 1
//           if duration > 0 or inner >= max_inner: break
//
// `TDT_DURATIONS` is read from config.json (default [0,1,2,3,4]).
//
// Encoder/Joint/Predictor I/O contracts (from NeMo export):
//   encoder:    audio_signal (1, n_mels, T_in) -> outputs (1, D_enc, T_out), encoded_lengths
//   decoder:    targets (1, U), target_length (1) -> outputs (1, D_dec, U), state_h, state_c
//               (the streaming form takes single-token input + prior state)
//   joint:      encoder_outputs (1, D_enc, 1), decoder_outputs (1, D_dec, 1)
//               -> logits (1, 1, 1, vocab + n_durations)

use crate::asr::tokenizer::Tokenizer;
use anyhow::{anyhow, Result};
use ndarray::{s, Array2, Array3, ArrayView2, Axis, IxDyn};
use ort::{session::Session, value::Value};
use tracing::trace;

#[derive(Clone, Debug)]
pub struct TdtConfig {
    pub blank_id: i32,
    pub vocab_size: usize,
    pub durations: Vec<i32>,        // typically [0, 1, 2, 3, 4]
    pub predictor_hidden: usize,    // LSTM hidden size (read from config)
    pub predictor_layers: usize,    // LSTM layer count
    pub max_inner_loops: usize,     // safety guard, default 10
}

impl Default for TdtConfig {
    fn default() -> Self {
        Self {
            blank_id: 1024,
            vocab_size: 1024,
            durations: vec![0, 1, 2, 3, 4],
            predictor_hidden: 640,
            predictor_layers: 1,
            max_inner_loops: 10,
        }
    }
}

pub struct TdtDecoder<'a> {
    pub encoder: &'a mut Session,
    pub decoder: &'a mut Session,
    pub joint: &'a mut Session,
    pub tokenizer: &'a Tokenizer,
    pub cfg: TdtConfig,
}

impl<'a> TdtDecoder<'a> {
    /// Run end-to-end: mel -> encoder -> TDT greedy -> detokenize.
    pub fn transcribe(&mut self, mel: ArrayView2<f32>) -> Result<String> {
        let (n_mels, t_in) = mel.dim();
        if t_in == 0 {
            return Ok(String::new());
        }
        let audio_signal = mel
            .to_owned()
            .insert_axis(Axis(0)); // (1, n_mels, T_in)
        let length = ndarray::Array1::from_elem(1, t_in as i64);

        // ---- Encoder forward ----
        let enc_inputs = ort::inputs![
            "audio_signal" => Value::from_array(audio_signal.into_dyn())?,
            "length"       => Value::from_array(length.into_dyn())?,
        ]?;
        let enc_out = self.encoder.run(enc_inputs)?;
        let encoded = enc_out
            .get("outputs")
            .or_else(|| enc_out.get("encoded"))
            .ok_or_else(|| anyhow!("encoder output 'outputs' missing"))?
            .try_extract_array::<f32>()?
            .into_owned();
        let _ = n_mels;
        // shape (1, D_enc, T_out)
        let enc = encoded
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| anyhow!("encoder shape: {e}"))?;
        let (_, d_enc, t_out) = enc.dim();
        trace!(d_enc, t_out, "encoded");

        // ---- Greedy TDT loop ----
        let mut emitted: Vec<i32> = Vec::new();
        let mut t = 0usize;
        let mut last_token = self.cfg.blank_id;
        let mut state_h = Array3::<f32>::zeros((self.cfg.predictor_layers, 1, self.cfg.predictor_hidden));
        let mut state_c = Array3::<f32>::zeros((self.cfg.predictor_layers, 1, self.cfg.predictor_hidden));

        while t < t_out {
            // Run predictor once per inner-loop iteration. NeMo's streaming export
            // exposes the predictor with single-token input and explicit LSTM state.
            let mut advanced = false;
            for _inner in 0..self.cfg.max_inner_loops {
                let target = ndarray::Array2::from_elem((1, 1), last_token);
                let target_len = ndarray::Array1::from_elem(1, 1i32);
                let dec_inputs = ort::inputs![
                    "targets"       => Value::from_array(target.into_dyn())?,
                    "target_length" => Value::from_array(target_len.into_dyn())?,
                    "states.1"      => Value::from_array(state_h.clone().into_dyn())?,
                    "states.2"      => Value::from_array(state_c.clone().into_dyn())?,
                ]?;
                let dec_out = self.decoder.run(dec_inputs)?;
                let pred = dec_out
                    .get("outputs")
                    .ok_or_else(|| anyhow!("decoder output 'outputs' missing"))?
                    .try_extract_array::<f32>()?
                    .into_owned()
                    .into_dimensionality::<ndarray::Ix3>()
                    .map_err(|e| anyhow!("decoder shape: {e}"))?;
                let next_h = dec_out
                    .get("states.1")
                    .ok_or_else(|| anyhow!("decoder state_h missing"))?
                    .try_extract_array::<f32>()?
                    .into_owned()
                    .into_dimensionality::<ndarray::Ix3>()
                    .map_err(|e| anyhow!("state_h shape: {e}"))?;
                let next_c = dec_out
                    .get("states.2")
                    .ok_or_else(|| anyhow!("decoder state_c missing"))?
                    .try_extract_array::<f32>()?
                    .into_owned()
                    .into_dimensionality::<ndarray::Ix3>()
                    .map_err(|e| anyhow!("state_c shape: {e}"))?;

                // Joint: (1, D_enc, 1) + (1, D_dec, 1) -> (1, 1, 1, vocab + |durations|)
                let enc_t = enc.slice(s![0..1, .., t..t + 1]).to_owned();
                let dec_u = pred.slice(s![0..1, .., 0..1]).to_owned();
                let joint_inputs = ort::inputs![
                    "encoder_outputs" => Value::from_array(enc_t.into_dyn())?,
                    "decoder_outputs" => Value::from_array(dec_u.into_dyn())?,
                ]?;
                let joint_out = self.joint.run(joint_inputs)?;
                let logits = joint_out
                    .get("outputs")
                    .ok_or_else(|| anyhow!("joint output 'outputs' missing"))?
                    .try_extract_array::<f32>()?
                    .into_owned();
                let flat = logits.iter().copied().collect::<Vec<_>>();
                let total = flat.len();
                let n_dur = self.cfg.durations.len();
                if total < self.cfg.vocab_size + n_dur {
                    return Err(anyhow!(
                        "joint logits too small: {} < vocab {} + durations {}",
                        total, self.cfg.vocab_size, n_dur
                    ));
                }
                let token_logits = &flat[..self.cfg.vocab_size + 1]; // includes blank
                let duration_logits = &flat[self.cfg.vocab_size + 1..self.cfg.vocab_size + 1 + n_dur];

                let token = argmax_i32(token_logits) as i32;
                let dur_idx = argmax_i32(duration_logits);
                let duration = self.cfg.durations[dur_idx] as usize;

                if token != self.cfg.blank_id {
                    emitted.push(token);
                    last_token = token;
                    state_h = next_h;
                    state_c = next_c;
                }
                t += duration;
                if duration > 0 {
                    advanced = true;
                    break;
                }
            }
            if !advanced {
                // Forced advance to prevent infinite loops if duration head is degenerate.
                t += 1;
            }
        }

        self.tokenizer.decode(&emitted)
    }
}

fn argmax_i32(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

// Keep IxDyn re-exported so the module compiles cleanly even if unused above.
#[allow(dead_code)]
fn _phantom() -> IxDyn { IxDyn(&[]) }
#[allow(dead_code)]
fn _array2_check() -> Array2<f32> { Array2::zeros((0, 0)) }
