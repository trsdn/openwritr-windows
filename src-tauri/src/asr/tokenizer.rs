// SentencePiece tokenizer wrapper. Parakeet TDT v3 ships a 1024-vocab
// unigram model with blank_id = vocab_size (last index).

use anyhow::{Context, Result};
use sentencepiece::SentencePieceProcessor;
use std::path::Path;

pub struct Tokenizer {
    sp: SentencePieceProcessor,
}

impl Tokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let sp = SentencePieceProcessor::open(path).with_context(|| format!("load tokenizer {}", path.display()))?;
        Ok(Self { sp })
    }

    pub fn decode(&self, ids: &[i32]) -> Result<String> {
        let pieces: Vec<String> = ids
            .iter()
            .filter_map(|&id| if id < 0 { None } else { self.sp.decode_piece_ids(&[id as u32]).ok() })
            .collect();
        Ok(pieces.concat())
    }

    pub fn vocab_size(&self) -> usize {
        self.sp.len()
    }
}
