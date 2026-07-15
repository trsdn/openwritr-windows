//! Whisper QNN tensor contracts and autoregressive KV-cache decoding.

use super::qnn_ffi::{
    QnnSession, SessionContract, TensorDataOwned, TensorElementType, TensorInput, TensorOutput,
    TensorSpec,
};
use super::whisper_mel::{LogMel, N_FRAMES, N_MELS};
use super::whisper_tokenizer::{
    LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START, TOKEN_EOT, TOKEN_NO_TIMESTAMPS, TOKEN_SOT,
    TOKEN_TRANSCRIBE,
};
use anyhow::{anyhow, bail, Result};
use half::f16;

pub const DECODER_LAYERS: usize = 4;
pub const ATTENTION_HEADS: usize = 20;
pub const HEAD_DIMENSION: usize = 64;
pub const SELF_CACHE_LENGTH: usize = 199;
pub const CROSS_CACHE_LENGTH: usize = 1_500;
pub const DECODE_WINDOW: usize = 200;
pub const VOCABULARY_SIZE: usize = 51_866;

const SELF_KEY_SHAPE: [i64; 4] = [
    ATTENTION_HEADS as i64,
    1,
    HEAD_DIMENSION as i64,
    SELF_CACHE_LENGTH as i64,
];
const SELF_VALUE_SHAPE: [i64; 4] = [
    ATTENTION_HEADS as i64,
    1,
    SELF_CACHE_LENGTH as i64,
    HEAD_DIMENSION as i64,
];
const CROSS_KEY_SHAPE: [i64; 4] = [
    ATTENTION_HEADS as i64,
    1,
    HEAD_DIMENSION as i64,
    CROSS_CACHE_LENGTH as i64,
];
const CROSS_VALUE_SHAPE: [i64; 4] = [
    ATTENTION_HEADS as i64,
    1,
    CROSS_CACHE_LENGTH as i64,
    HEAD_DIMENSION as i64,
];

pub struct F16Tensor {
    pub name: String,
    pub dimensions: Vec<i64>,
    pub values: Vec<u16>,
}

pub struct DecodedChunk {
    pub tokens: Vec<i32>,
    pub language_token: i32,
    pub steps: usize,
}

pub fn encoder_contract() -> Result<SessionContract> {
    let mut outputs = Vec::with_capacity(DECODER_LAYERS * 2);
    for layer in 0..DECODER_LAYERS {
        outputs.push(TensorSpec::new(
            format!("k_cache_cross_{layer}"),
            TensorElementType::F16,
            CROSS_KEY_SHAPE.to_vec(),
        ));
        outputs.push(TensorSpec::new(
            format!("v_cache_cross_{layer}"),
            TensorElementType::F16,
            CROSS_VALUE_SHAPE.to_vec(),
        ));
    }
    SessionContract::new(
        vec![TensorSpec::new(
            "input_features",
            TensorElementType::F16,
            vec![1, N_MELS as i64, N_FRAMES as i64],
        )],
        outputs,
    )
}

pub fn decoder_contract() -> Result<SessionContract> {
    let mut inputs = vec![
        TensorSpec::new("input_ids", TensorElementType::I32, vec![1, 1]),
        TensorSpec::new(
            "attention_mask",
            TensorElementType::F16,
            vec![1, 1, 1, DECODE_WINDOW as i64],
        ),
    ];
    for layer in 0..DECODER_LAYERS {
        inputs.push(TensorSpec::new(
            format!("k_cache_self_{layer}_in"),
            TensorElementType::F16,
            SELF_KEY_SHAPE.to_vec(),
        ));
        inputs.push(TensorSpec::new(
            format!("v_cache_self_{layer}_in"),
            TensorElementType::F16,
            SELF_VALUE_SHAPE.to_vec(),
        ));
    }
    for layer in 0..DECODER_LAYERS {
        inputs.push(TensorSpec::new(
            format!("k_cache_cross_{layer}"),
            TensorElementType::F16,
            CROSS_KEY_SHAPE.to_vec(),
        ));
        inputs.push(TensorSpec::new(
            format!("v_cache_cross_{layer}"),
            TensorElementType::F16,
            CROSS_VALUE_SHAPE.to_vec(),
        ));
    }
    inputs.push(TensorSpec::new(
        "position_ids",
        TensorElementType::I32,
        vec![1],
    ));

    let mut outputs = vec![TensorSpec::new(
        "logits",
        TensorElementType::F16,
        vec![1, VOCABULARY_SIZE as i64, 1, 1],
    )];
    for layer in 0..DECODER_LAYERS {
        outputs.push(TensorSpec::new(
            format!("k_cache_self_{layer}_out"),
            TensorElementType::F16,
            SELF_KEY_SHAPE.to_vec(),
        ));
        outputs.push(TensorSpec::new(
            format!("v_cache_self_{layer}_out"),
            TensorElementType::F16,
            SELF_VALUE_SHAPE.to_vec(),
        ));
    }
    SessionContract::new(inputs, outputs)
}

pub fn run_encoder(session: &mut QnnSession, mel: &LogMel) -> Result<Vec<F16Tensor>> {
    let dimensions = mel.dimensions();
    session
        .run(&[TensorInput::f16(
            "input_features",
            &dimensions,
            mel.f16_bits(),
        )])?
        .into_iter()
        .map(output_to_f16_tensor)
        .collect()
}

pub trait DecoderBackend {
    fn run_step(
        &mut self,
        token: i32,
        position: i32,
        attention_mask: &[u16],
        self_cache: &mut Vec<F16Tensor>,
        cross_cache: &[F16Tensor],
        selection: TokenSelection,
    ) -> Result<i32>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenSelection {
    All,
    Language,
}

pub struct QnnDecoderBackend<'a> {
    session: &'a mut QnnSession,
}

impl<'a> QnnDecoderBackend<'a> {
    pub fn new(session: &'a mut QnnSession) -> Self {
        Self { session }
    }
}

impl DecoderBackend for QnnDecoderBackend<'_> {
    fn run_step(
        &mut self,
        token: i32,
        position: i32,
        attention_mask: &[u16],
        self_cache: &mut Vec<F16Tensor>,
        cross_cache: &[F16Tensor],
        selection: TokenSelection,
    ) -> Result<i32> {
        let token_values = [token];
        let token_shape = [1_i64, 1];
        let position_values = [position];
        let position_shape = [1_i64];
        let attention_shape = [1_i64, 1, 1, DECODE_WINDOW as i64];
        let mut inputs = Vec::with_capacity(3 + self_cache.len() + cross_cache.len());
        inputs.push(TensorInput::i32("input_ids", &token_shape, &token_values));
        inputs.push(TensorInput::f16(
            "attention_mask",
            &attention_shape,
            attention_mask,
        ));
        for tensor in self_cache.iter().chain(cross_cache) {
            inputs.push(TensorInput::f16(
                &tensor.name,
                &tensor.dimensions,
                &tensor.values,
            ));
        }
        inputs.push(TensorInput::i32(
            "position_ids",
            &position_shape,
            &position_values,
        ));

        let outputs = self.session.run(&inputs)?;
        let mut logits = None;
        let mut next_cache = Vec::with_capacity(DECODER_LAYERS * 2);
        for output in outputs {
            if output.name == "logits" {
                let tensor = output_to_f16_tensor(output)?;
                logits = Some(tensor.values);
            } else {
                let mut tensor = output_to_f16_tensor(output)?;
                tensor.name = tensor
                    .name
                    .strip_suffix("_out")
                    .map(|prefix| format!("{prefix}_in"))
                    .ok_or_else(|| anyhow!("unexpected decoder cache output {}", tensor.name))?;
                next_cache.push(tensor);
            }
        }
        if next_cache.len() != DECODER_LAYERS * 2 {
            bail!(
                "decoder returned {} self-cache tensors, expected {}",
                next_cache.len(),
                DECODER_LAYERS * 2
            );
        }
        *self_cache = next_cache;
        argmax_f16(
            &logits.ok_or_else(|| anyhow!("decoder did not return logits"))?,
            selection,
        )
    }
}

pub fn decode_chunk(
    backend: &mut impl DecoderBackend,
    cross_cache: &[F16Tensor],
    language_token: Option<i32>,
) -> Result<DecodedChunk> {
    if let Some(language_token) = language_token {
        require_language_token(language_token)?;
    }
    let mut self_cache = initial_self_cache();
    let mut attention_mask = vec![f16::MIN.to_bits(); DECODE_WINDOW];
    let mut forced = language_token
        .map(|language| vec![language, TOKEN_TRANSCRIBE, TOKEN_NO_TIMESTAMPS])
        .unwrap_or_default()
        .into_iter();
    let mut current_token = TOKEN_SOT;
    let mut detected_language = language_token;
    let mut emitted = Vec::new();
    let mut steps = 0;

    for position in 0..SELF_CACHE_LENGTH {
        attention_mask[DECODE_WINDOW - position - 1] = f16::ZERO.to_bits();
        let selection = if detected_language.is_none() {
            TokenSelection::Language
        } else {
            TokenSelection::All
        };
        let predicted = backend.run_step(
            current_token,
            position as i32,
            &attention_mask,
            &mut self_cache,
            cross_cache,
            selection,
        )?;
        steps += 1;

        if detected_language.is_none() {
            require_language_token(predicted)?;
            detected_language = Some(predicted);
            current_token = predicted;
            forced = vec![TOKEN_TRANSCRIBE, TOKEN_NO_TIMESTAMPS].into_iter();
            continue;
        }
        if let Some(forced_token) = forced.next() {
            current_token = forced_token;
            continue;
        }
        if predicted == TOKEN_EOT {
            break;
        }
        emitted.push(predicted);
        current_token = predicted;
    }

    Ok(DecodedChunk {
        tokens: emitted,
        language_token: detected_language.expect("language is set before decoding text"),
        steps,
    })
}

fn initial_self_cache() -> Vec<F16Tensor> {
    let mut tensors = Vec::with_capacity(DECODER_LAYERS * 2);
    for layer in 0..DECODER_LAYERS {
        tensors.push(F16Tensor {
            name: format!("k_cache_self_{layer}_in"),
            dimensions: SELF_KEY_SHAPE.to_vec(),
            values: vec![f16::ZERO.to_bits(); element_count(&SELF_KEY_SHAPE)],
        });
        tensors.push(F16Tensor {
            name: format!("v_cache_self_{layer}_in"),
            dimensions: SELF_VALUE_SHAPE.to_vec(),
            values: vec![f16::ZERO.to_bits(); element_count(&SELF_VALUE_SHAPE)],
        });
    }
    tensors
}

fn output_to_f16_tensor(output: TensorOutput) -> Result<F16Tensor> {
    let TensorOutput {
        name,
        dimensions,
        data,
    } = output;
    let TensorDataOwned::F16(values) = data else {
        bail!("QNN output {name} is not float16");
    };
    Ok(F16Tensor {
        name,
        dimensions,
        values,
    })
}

fn argmax_f16(values: &[u16], selection: TokenSelection) -> Result<i32> {
    if values.len() != VOCABULARY_SIZE {
        bail!(
            "decoder logits contain {} values, expected {VOCABULARY_SIZE}",
            values.len()
        );
    }
    let mut best_index = None;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        let value = f16::from_bits(*value).to_f32();
        if !value.is_finite() {
            bail!("decoder logits contain a non-finite value at index {index}");
        }
        if selection == TokenSelection::Language
            && !(LANGUAGE_TOKEN_START as usize..=LANGUAGE_TOKEN_END as usize).contains(&index)
        {
            continue;
        }
        if value > best_value {
            best_value = value;
            best_index = Some(index);
        }
    }
    i32::try_from(best_index.expect("validated non-empty logits"))
        .map_err(|_| anyhow!("decoder token index exceeds i32"))
}

fn require_language_token(token: i32) -> Result<()> {
    if !(LANGUAGE_TOKEN_START..=LANGUAGE_TOKEN_END).contains(&token) {
        bail!("Whisper language detection returned unexpected token {token}");
    }
    Ok(())
}

fn element_count(shape: &[i64]) -> usize {
    shape.iter().map(|dimension| *dimension as usize).product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct FakeDecoder {
        predictions: VecDeque<i32>,
        inputs: Vec<i32>,
        selections: Vec<TokenSelection>,
    }

    impl FakeDecoder {
        fn new(predictions: impl IntoIterator<Item = i32>) -> Self {
            Self {
                predictions: predictions.into_iter().collect(),
                inputs: Vec::new(),
                selections: Vec::new(),
            }
        }
    }

    impl DecoderBackend for FakeDecoder {
        fn run_step(
            &mut self,
            token: i32,
            position: i32,
            attention_mask: &[u16],
            self_cache: &mut Vec<F16Tensor>,
            _cross_cache: &[F16Tensor],
            selection: TokenSelection,
        ) -> Result<i32> {
            assert_eq!(self_cache.len(), DECODER_LAYERS * 2);
            assert_eq!(attention_mask.len(), DECODE_WINDOW);
            assert_eq!(
                attention_mask
                    .iter()
                    .filter(|value| **value == f16::ZERO.to_bits())
                    .count(),
                position as usize + 1
            );
            self.inputs.push(token);
            self.selections.push(selection);
            self.predictions
                .pop_front()
                .ok_or_else(|| anyhow!("fake decoder ran out of predictions"))
        }
    }

    #[test]
    fn detects_language_once_then_forces_transcription_prompt() {
        let mut decoder =
            FakeDecoder::new([LANGUAGE_TOKEN_START, 123, 456, 1_111, 2_222, TOKEN_EOT]);
        let decoded = decode_chunk(&mut decoder, &[], None).unwrap();

        assert_eq!(decoded.language_token, LANGUAGE_TOKEN_START);
        assert_eq!(decoded.tokens, [1_111, 2_222]);
        assert_eq!(decoded.steps, 6);
        assert_eq!(
            decoder.inputs,
            [
                TOKEN_SOT,
                LANGUAGE_TOKEN_START,
                TOKEN_TRANSCRIBE,
                TOKEN_NO_TIMESTAMPS,
                1_111,
                2_222
            ]
        );
        assert_eq!(decoder.selections[0], TokenSelection::Language);
        assert!(decoder.selections[1..]
            .iter()
            .all(|selection| *selection == TokenSelection::All));
    }

    #[test]
    fn reuses_the_recording_language_on_later_chunks() {
        let mut decoder = FakeDecoder::new([10, 20, 30, 9_999, TOKEN_EOT]);
        let decoded = decode_chunk(&mut decoder, &[], Some(LANGUAGE_TOKEN_END)).unwrap();

        assert_eq!(decoded.language_token, LANGUAGE_TOKEN_END);
        assert_eq!(decoded.tokens, [9_999]);
        assert_eq!(
            decoder.inputs,
            [
                TOKEN_SOT,
                LANGUAGE_TOKEN_END,
                TOKEN_TRANSCRIBE,
                TOKEN_NO_TIMESTAMPS,
                9_999
            ]
        );
    }

    #[test]
    fn rejects_a_non_language_detection_result() {
        let mut decoder = FakeDecoder::new([TOKEN_TRANSCRIBE]);
        assert!(decode_chunk(&mut decoder, &[], None).is_err());
    }

    #[test]
    fn argmax_rejects_bad_shapes_and_non_finite_logits() {
        assert!(argmax_f16(&[], TokenSelection::All).is_err());
        let mut logits = vec![f16::ZERO.to_bits(); VOCABULARY_SIZE];
        logits[42] = f16::from_f32(2.0).to_bits();
        assert_eq!(argmax_f16(&logits, TokenSelection::All).unwrap(), 42);
        logits[17] = f16::NAN.to_bits();
        assert!(argmax_f16(&logits, TokenSelection::All).is_err());
    }

    #[test]
    fn language_selection_ignores_a_larger_non_language_logit() {
        let mut logits = vec![f16::from_f32(-10.0).to_bits(); VOCABULARY_SIZE];
        logits[TOKEN_TRANSCRIBE as usize] = f16::from_f32(10.0).to_bits();
        logits[LANGUAGE_TOKEN_START as usize] = f16::from_f32(2.0).to_bits();
        logits[LANGUAGE_TOKEN_START as usize + 1] = f16::from_f32(3.0).to_bits();

        assert_eq!(
            argmax_f16(&logits, TokenSelection::Language).unwrap(),
            LANGUAGE_TOKEN_START + 1
        );
        assert_eq!(
            argmax_f16(&logits, TokenSelection::All).unwrap(),
            TOKEN_TRANSCRIBE
        );
    }
}
