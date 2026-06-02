//! Custom vocab loader for the istupakov Parakeet release.
//!
//! The `vocab.txt` is a UTF-8 file with one entry per line: `<token> <id>`.
//! `▁` (U+2581) is the SentencePiece word-boundary marker, which we expand
//! back to a space at decode time. The last entry (`<blk>`) is the transducer
//! blank symbol.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

pub struct Vocab {
    table: HashMap<i32, String>,
    pub blank_id: i32,
    pub size: usize,
}

impl Vocab {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read vocab from {}", path.display()))?;
        let mut table = HashMap::new();
        let mut blank_id = -1i32;
        for line in text.lines() {
            let line = line.trim_end_matches('\n');
            if line.is_empty() {
                continue;
            }
            // Split on the LAST space so tokens that contain spaces survive.
            let Some(idx) = line.rfind(' ') else { continue; };
            let (tok, id_str) = (&line[..idx], &line[idx + 1..]);
            let id: i32 = id_str.parse().with_context(|| format!("vocab id parse: {line}"))?;
            let token = tok.replace('\u{2581}', " ");
            if tok == "<blk>" {
                blank_id = id;
            }
            table.insert(id, token);
        }
        let size = table.len();
        if blank_id < 0 {
            anyhow::bail!("vocab missing <blk> token");
        }
        Ok(Self { table, blank_id, size })
    }

    pub fn detokenize(&self, ids: &[i32]) -> String {
        let mut out = String::with_capacity(ids.len() * 4);
        for &id in ids {
            if let Some(tok) = self.table.get(&id) {
                out.push_str(tok);
            }
        }
        // Trim the leading space from the first word-boundary marker.
        out.trim_start().to_string()
    }
}
