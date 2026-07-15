use crate::{asr, model_manager::ModelManager, paths};
use anyhow::{Context, Result};
use serde_json::json;
use std::io::Write;

pub fn run() -> Result<()> {
    let runtime_dir = asr::verify_runtime_installation().context("verify native runtime")?;
    ModelManager::verify_embedded_manifest().context("verify model manifest")?;
    let data_dir = verify_writable_data_directory()?;
    let hardware = asr::whisper_hardware_status().context("inspect Whisper NPU hardware")?;

    let report = json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "architecture": std::env::consts::ARCH,
        "runtime_directory": runtime_dir,
        "model_manifest": "ok",
        "data_directory": data_dir,
        "whisper_npu_hardware": hardware,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn verify_writable_data_directory() -> Result<std::path::PathBuf> {
    let directory = paths::data_dir();
    std::fs::create_dir_all(&directory)
        .with_context(|| format!("create data directory {}", directory.display()))?;
    let probe = directory.join(format!(".self-check-{}.tmp", std::process::id()));
    let result: Result<()> = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&probe)
            .with_context(|| format!("create writable-data probe {}", probe.display()))?;
        file.write_all(b"openwritr self-check\n")?;
        file.sync_all()?;
        Ok(())
    })();
    let remove_result = std::fs::remove_file(&probe)
        .with_context(|| format!("remove writable-data probe {}", probe.display()));
    result?;
    remove_result?;
    Ok(directory)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writable_data_probe_is_removed() {
        let directory = verify_writable_data_directory().unwrap();
        let prefix = format!(".self-check-{}", std::process::id());
        assert!(!std::fs::read_dir(directory)
            .unwrap()
            .filter_map(Result::ok)
            .any(|entry| entry.file_name().to_string_lossy().starts_with(&prefix)));
    }
}
