use crate::{paths, settings::Settings};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};
use zip::write::SimpleFileOptions;

const MAX_LOG_FILES: usize = 7;
const MAX_LOG_BYTES_PER_FILE: u64 = 5 * 1024 * 1024;
const MAX_DIAGNOSTIC_BUNDLES: usize = 5;
const CREATE_NO_WINDOW: u32 = 0x0800_0000;
const DETACHED_PROCESS: u32 = 0x0000_0008;

pub fn prune_logs() -> Result<()> {
    prune_matching(&paths::log_dir(), MAX_LOG_FILES, is_log_file)
}

pub fn open_logs_dir() -> Result<()> {
    let directory = paths::log_dir();
    fs::create_dir_all(&directory).with_context(|| format!("create {}", directory.display()))?;
    open_explorer(&directory, false)
}

pub fn reveal(path: &Path) -> Result<()> {
    open_explorer(path, true)
}

pub fn export_bundle(settings: &Settings) -> Result<PathBuf> {
    let destination = paths::diagnostics_dir();
    fs::create_dir_all(&destination)
        .with_context(|| format!("create {}", destination.display()))?;
    prune_matching(
        &destination,
        MAX_DIAGNOSTIC_BUNDLES.saturating_sub(1),
        is_diagnostic_bundle,
    )?;
    let runtime_dir = std::env::current_exe()
        .context("resolve current executable")?
        .parent()
        .context("executable has no parent directory")?
        .to_path_buf();
    export_bundle_from(
        settings,
        &paths::log_dir(),
        &paths::models_dir(),
        &runtime_dir,
        &destination,
    )
}

fn export_bundle_from(
    settings: &Settings,
    log_dir: &Path,
    models_dir: &Path,
    runtime_dir: &Path,
    destination_dir: &Path,
) -> Result<PathBuf> {
    fs::create_dir_all(destination_dir)
        .with_context(|| format!("create {}", destination_dir.display()))?;
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?;
    let timestamp = elapsed.as_secs();
    let destination = destination_dir.join(format!(
        "openwritr-diagnostics-{}-{}.zip",
        elapsed.as_millis(),
        std::process::id()
    ));
    let temporary = destination.with_extension("zip.tmp");
    temporary.unlink_if_exists()?;

    let result = (|| -> Result<()> {
        let output =
            File::create(&temporary).with_context(|| format!("create {}", temporary.display()))?;
        let mut zip = zip::ZipWriter::new(output);
        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

        let mut redacted_settings = serde_json::to_value(settings)?;
        redact_secrets(&mut redacted_settings);
        write_json(
            &mut zip,
            "settings.redacted.json",
            &redacted_settings,
            options,
        )?;

        let logs = recent_files(log_dir, is_log_file)?;
        let included_logs: Vec<_> = logs.iter().take(MAX_LOG_FILES).collect();
        let metadata = json!({
            "app_version": env!("CARGO_PKG_VERSION"),
            "process_id": std::process::id(),
            "architecture": std::env::consts::ARCH,
            "operating_system": std::env::consts::OS,
            "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "exported_at_unix_seconds": timestamp,
            "log_files_included": included_logs.len(),
        });
        write_json(&mut zip, "metadata.json", &metadata, options)?;
        write_json(
            &mut zip,
            "model-status.json",
            &crate::model_manager::diagnostic_status(models_dir)?,
            options,
        )?;

        let runtime_status = collect_runtime_status(runtime_dir)?;
        write_json(&mut zip, "runtime-status.json", &runtime_status, options)?;
        let runtime_receipt = runtime_dir.join("runtime-versions.json");
        if runtime_receipt.is_file() {
            let bytes = fs::read(&runtime_receipt)
                .with_context(|| format!("read {}", runtime_receipt.display()))?;
            write_bytes(&mut zip, "runtime-versions.json", &bytes, options)?;
        }

        for log_path in included_logs {
            let bytes = read_sanitized_log_tail(log_path)?;
            let name = log_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("openwritr.log");
            write_bytes(&mut zip, &format!("logs/{name}"), &bytes, options)?;
        }

        zip.finish().context("finish diagnostics ZIP")?;
        Ok(())
    })();

    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    fs::rename(&temporary, &destination)
        .with_context(|| format!("promote {}", destination.display()))?;
    Ok(destination)
}

fn write_json<W: Write + Seek>(
    zip: &mut zip::ZipWriter<W>,
    name: &str,
    value: &Value,
    options: SimpleFileOptions,
) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    write_bytes(zip, name, &bytes, options)
}

fn write_bytes<W: Write + Seek>(
    zip: &mut zip::ZipWriter<W>,
    name: &str,
    bytes: &[u8],
    options: SimpleFileOptions,
) -> Result<()> {
    zip.start_file(name, options)
        .with_context(|| format!("start ZIP entry {name}"))?;
    zip.write_all(bytes)
        .with_context(|| format!("write ZIP entry {name}"))?;
    Ok(())
}

fn collect_runtime_status(runtime_dir: &Path) -> Result<Value> {
    let receipt_path = runtime_dir.join("runtime-versions.json");
    if receipt_path.is_file() {
        let receipt: Value = serde_json::from_slice(
            &fs::read(&receipt_path).with_context(|| format!("read {}", receipt_path.display()))?,
        )
        .context("parse runtime-versions.json")?;
        let files = receipt
            .get("files")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let statuses: Vec<_> = files
            .iter()
            .filter_map(|file| {
                let path = file.get("path")?.as_str()?;
                if path.starts_with("third-party-licenses/") {
                    return None;
                }
                let expected_bytes = file.get("bytes").and_then(Value::as_u64);
                let runtime_path = runtime_dir.join(path);
                let metadata = fs::metadata(&runtime_path).ok();
                Some(json!({
                    "path": path,
                    "present": metadata.is_some(),
                    "bytes": metadata.as_ref().map(|value| value.len()),
                    "expected_bytes": expected_bytes,
                    "size_matches": metadata.as_ref().map(|value| Some(value.len()) == expected_bytes),
                    "file_version": file_version(&runtime_path),
                }))
            })
            .collect();
        return Ok(json!({
            "receipt_present": true,
            "architecture": receipt.get("architecture"),
            "packages": receipt.get("packages"),
            "qnn": receipt.get("qnn"),
            "files": statuses,
        }));
    }

    let files: Vec<_> = known_runtime_files()
        .iter()
        .map(|name| {
            let runtime_path = runtime_dir.join(name);
            let metadata = fs::metadata(&runtime_path).ok();
            json!({
                "path": name,
                "present": metadata.is_some(),
                "bytes": metadata.map(|value| value.len()),
                "file_version": file_version(&runtime_path),
            })
        })
        .collect();
    Ok(json!({ "receipt_present": false, "files": files }))
}

fn known_runtime_files() -> &'static [&'static str] {
    if cfg!(target_arch = "aarch64") {
        &[
            "onnxruntime.dll",
            "onnxruntime_providers_qnn.dll",
            "QnnHtp.dll",
            "QnnHtpPrepare.dll",
            "QnnHtpNetRunExtensions.dll",
            "QnnHtpV73Stub.dll",
            "libQnnHtpV73Skel.so",
            "libqnnhtpv73.cat",
            "QnnSystem.dll",
        ]
    } else {
        &["onnxruntime.dll"]
    }
}

fn file_version(path: &Path) -> Option<String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileVersionInfoSizeW, GetFileVersionInfoW, VerQueryValueW, VS_FIXEDFILEINFO,
    };

    if !path.is_file() {
        return None;
    }

    let path_wide = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let mut ignored = 0_u32;
    let size = unsafe { GetFileVersionInfoSizeW(path_wide.as_ptr(), &mut ignored) };
    if size == 0 {
        return None;
    }

    let mut data = vec![0_u8; size as usize];
    if unsafe { GetFileVersionInfoW(path_wide.as_ptr(), 0, size, data.as_mut_ptr().cast()) } == 0 {
        return None;
    }

    let root = ['\\' as u16, 0];
    let mut value = std::ptr::null_mut();
    let mut value_len = 0_u32;
    if unsafe {
        VerQueryValueW(
            data.as_ptr().cast(),
            root.as_ptr(),
            &mut value,
            &mut value_len,
        )
    } == 0
        || value.is_null()
        || value_len < std::mem::size_of::<VS_FIXEDFILEINFO>() as u32
    {
        return None;
    }

    let info = unsafe { &*value.cast::<VS_FIXEDFILEINFO>() };
    if info.dwSignature != 0xFEEF04BD {
        return None;
    }

    Some(format!(
        "{}.{}.{}.{}",
        info.dwFileVersionMS >> 16,
        info.dwFileVersionMS & 0xffff,
        info.dwFileVersionLS >> 16,
        info.dwFileVersionLS & 0xffff
    ))
}

fn redact_secrets(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                let normalized = key.to_ascii_lowercase();
                if ["api_key", "token", "secret", "password", "authorization"]
                    .iter()
                    .any(|sensitive| normalized.contains(sensitive))
                {
                    *value = Value::String("<redacted>".into());
                } else {
                    redact_secrets(value);
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                redact_secrets(value);
            }
        }
        _ => {}
    }
}

fn read_sanitized_log_tail(path: &Path) -> Result<Vec<u8>> {
    read_sanitized_log_tail_with_limit(path, MAX_LOG_BYTES_PER_FILE)
}

fn read_sanitized_log_tail_with_limit(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let len = file.metadata()?.len();
    let truncated = len > max_bytes;
    if truncated {
        file.seek(SeekFrom::End(-(max_bytes as i64)))?;
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    if truncated {
        if let Some(first_newline) = bytes.iter().position(|byte| *byte == b'\n') {
            bytes.drain(..=first_newline);
        } else {
            bytes.clear();
        }
    }
    let mut sanitized = sanitize_log(&bytes);
    if truncated {
        let mut prefixed = format!("[truncated to the last {max_bytes} bytes]\n").into_bytes();
        prefixed.append(&mut sanitized);
        return Ok(prefixed);
    }
    Ok(sanitized)
}

fn sanitize_log(bytes: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(bytes);
    let mut sanitized = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        if line.contains("transcribed ->") {
            sanitized.push_str("[redacted legacy transcript log line]\n");
        } else {
            sanitized.push_str(line);
        }
    }
    sanitized.into_bytes()
}

fn prune_matching(directory: &Path, max_files: usize, predicate: fn(&Path) -> bool) -> Result<()> {
    let files = recent_files(directory, predicate)?;
    for path in files.into_iter().skip(max_files) {
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| format!("remove {}", path.display()));
            }
        }
    }
    Ok(())
}

fn recent_files(directory: &Path, predicate: fn(&Path) -> bool) -> Result<Vec<PathBuf>> {
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut files: Vec<_> = fs::read_dir(directory)
        .with_context(|| format!("read {}", directory.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && predicate(path))
        .collect();
    files.sort_by(|left, right| {
        let left_modified = left
            .metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap_or(UNIX_EPOCH);
        let right_modified = right
            .metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap_or(UNIX_EPOCH);
        right_modified
            .cmp(&left_modified)
            .then_with(|| right.file_name().cmp(&left.file_name()))
    });
    Ok(files)
}

fn is_log_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with("openwritr") && name.ends_with(".log"))
        .unwrap_or(false)
}

fn is_diagnostic_bundle(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with("openwritr-diagnostics-") && name.ends_with(".zip"))
        .unwrap_or(false)
}

fn open_explorer(path: &Path, select: bool) -> Result<()> {
    let mut command = Command::new("explorer.exe");
    if select {
        command.arg(format!("/select,{}", path.display()));
    } else {
        command.arg(path);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(DETACHED_PROCESS | CREATE_NO_WINDOW)
        .spawn()
        .with_context(|| format!("open Explorer for {}", path.display()))?;
    Ok(())
}

trait UnlinkIfExists {
    fn unlink_if_exists(&self) -> Result<()>;
}

impl UnlinkIfExists for Path {
    fn unlink_if_exists(&self) -> Result<()> {
        match fs::remove_file(self) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error).with_context(|| format!("remove {}", self.display())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zip::ZipArchive;

    #[test]
    fn redacts_nested_credentials() {
        let mut value = json!({
            "enhance": { "api_key": "secret-value" },
            "nested": [{ "access_token": "token-value" }],
            "safe": "visible"
        });
        redact_secrets(&mut value);
        let serialized = serde_json::to_string(&value).unwrap();
        assert!(!serialized.contains("secret-value"));
        assert!(!serialized.contains("token-value"));
        assert!(serialized.contains("visible"));
    }

    #[test]
    fn scrubs_legacy_transcript_lines() {
        let sanitized = sanitize_log(
            b"safe line\n2026 INFO transcribed -> \"private words\"\nanother safe line\n",
        );
        let text = String::from_utf8(sanitized).unwrap();
        assert!(!text.contains("private words"));
        assert!(text.contains("safe line"));
        assert!(text.contains("redacted legacy transcript"));
    }

    #[test]
    fn bounded_log_tail_drops_a_partial_private_line() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("openwritr.log");
        fs::write(
            &path,
            b"INFO transcribed -> \"private words that must not escape\"\nINFO recent safe line\n",
        )
        .unwrap();

        let sanitized = read_sanitized_log_tail_with_limit(&path, 45).unwrap();
        let text = String::from_utf8(sanitized).unwrap();

        assert!(!text.contains("private words"));
        assert!(text.contains("recent safe line"));
    }

    #[test]
    fn prunes_logs_to_the_retention_limit() {
        let temp = tempfile::tempdir().unwrap();
        for index in 0..10 {
            fs::write(
                temp.path().join(format!("openwritr-{index:02}.log")),
                b"log",
            )
            .unwrap();
        }
        prune_matching(temp.path(), 7, is_log_file).unwrap();
        assert_eq!(recent_files(temp.path(), is_log_file).unwrap().len(), 7);
    }

    #[test]
    fn export_is_redacted_and_contains_status_files() {
        let temp = tempfile::tempdir().unwrap();
        let logs = temp.path().join("logs");
        let models = temp.path().join("models");
        let runtime = temp.path().join("runtime");
        let output = temp.path().join("diagnostics");
        fs::create_dir_all(&logs).unwrap();
        fs::create_dir_all(models.join("parakeet")).unwrap();
        fs::create_dir_all(&runtime).unwrap();
        fs::write(
            logs.join("openwritr-test.log"),
            b"safe\ntranscribed -> \"private transcript\"\n",
        )
        .unwrap();
        fs::write(models.join("parakeet").join("encoder.onnx"), b"model").unwrap();
        fs::write(
            runtime.join("runtime-versions.json"),
            br#"{"architecture":"arm64","packages":[],"qnn":null,"files":[]}"#,
        )
        .unwrap();

        let settings = Settings::default();
        let bundle = export_bundle_from(&settings, &logs, &models, &runtime, &output).unwrap();
        let mut archive = ZipArchive::new(File::open(bundle).unwrap()).unwrap();

        let mut settings_json = String::new();
        archive
            .by_name("settings.redacted.json")
            .unwrap()
            .read_to_string(&mut settings_json)
            .unwrap();
        assert!(!settings_json.contains("\"api_key\""));

        let mut log = String::new();
        archive
            .by_name("logs/openwritr-test.log")
            .unwrap()
            .read_to_string(&mut log)
            .unwrap();
        assert!(!log.contains("private transcript"));
        assert!(archive.by_name("metadata.json").is_ok());
        assert!(archive.by_name("model-status.json").is_ok());
        assert!(archive.by_name("runtime-status.json").is_ok());
        assert!(archive.by_name("runtime-versions.json").is_ok());
    }
}
