//! Helpers to make `ort` 2.0-rc.12 errors play nicely with `anyhow::Error`.
//!
//! `ort::Error<R>` carries a "recoverable" payload (e.g. SessionBuilder).
//! Those payloads contain raw `NonNull<...>` pointers that are not Send/Sync,
//! so the auto-conversion via `?` into `anyhow::Error` (which requires
//! Send + Sync) fails. We provide a small extension trait that drops the
//! payload and converts to a simple string-based anyhow error.

use anyhow::anyhow;

pub trait OrtResultExt<T> {
    fn ortx(self) -> anyhow::Result<T>;
}

impl<T, R> OrtResultExt<T> for std::result::Result<T, ort::Error<R>> {
    fn ortx(self) -> anyhow::Result<T> {
        self.map_err(|e| anyhow!("ort: {e}"))
    }
}
