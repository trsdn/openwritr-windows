use crate::paths;
use anyhow::{anyhow, Context};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;
use tracing::{info, warn};

const MANIFEST_JSON: &str = include_str!("../models-manifest.json");
const DOWNLOAD_BUFFER_BYTES: usize = 1024 * 1024;
const STAGING_MARGIN_BYTES: u64 = 128 * 1024 * 1024;

macro_rules! bail {
    ($($argument:tt)*) => {
        return Err(ModelError::Failed(anyhow!($($argument)*)))
    };
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelState {
    Missing,
    Downloading {
        downloaded_bytes: u64,
        total_bytes: u64,
    },
    Verifying,
    Ready,
    Failed {
        message: String,
    },
    Cancelled,
}

impl ModelState {
    pub fn status_text(&self, model_id: &str) -> String {
        match self {
            Self::Missing => format!("{model_id}: preparing model download"),
            Self::Downloading {
                downloaded_bytes,
                total_bytes,
            } => {
                let percent = if *total_bytes == 0 {
                    0
                } else {
                    downloaded_bytes.saturating_mul(100) / total_bytes
                };
                format!("{model_id}: downloading {percent}%")
            }
            Self::Verifying => format!("{model_id}: verifying model files"),
            Self::Ready => format!("{model_id}: model ready"),
            Self::Failed { message } => format!("{model_id}: {message}"),
            Self::Cancelled => format!("{model_id}: model download cancelled"),
        }
    }
}

#[derive(Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("model acquisition cancelled")]
    Cancelled,
    #[error(transparent)]
    Failed(#[from] anyhow::Error),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestRoot {
    schema_version: u32,
    models: Vec<ModelManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelManifest {
    id: String,
    version: String,
    local_dir: String,
    architecture: String,
    #[serde(default)]
    files: Vec<DirectFile>,
    archive: Option<ArchiveArtifact>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DirectFile {
    path: String,
    url: String,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchiveArtifact {
    url: String,
    bytes: u64,
    sha256: String,
    files: Vec<ArchiveFile>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchiveFile {
    archive_path: String,
    path: String,
    bytes: u64,
    sha256: String,
}

struct SourceStream {
    reader: Box<dyn Read>,
    content_length: Option<u64>,
}

trait ArtifactSource: Send + Sync {
    fn open(&self, url: &str) -> anyhow::Result<SourceStream>;
}

struct HttpArtifactSource {
    client: reqwest::blocking::Client,
}

impl HttpArtifactSource {
    fn new() -> anyhow::Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(60 * 60))
            .user_agent(concat!("OpenWritr/", env!("CARGO_PKG_VERSION")))
            .build()
            .context("build model download client")?;
        Ok(Self { client })
    }
}

impl ArtifactSource for HttpArtifactSource {
    fn open(&self, url: &str) -> anyhow::Result<SourceStream> {
        let response = self
            .client
            .get(url)
            .send()
            .with_context(|| format!("request {url}"))?
            .error_for_status()
            .with_context(|| format!("download {url}"))?;
        let content_length = response.content_length();
        Ok(SourceStream {
            reader: Box::new(response),
            content_length,
        })
    }
}

type SpaceProbe = dyn Fn(&Path) -> anyhow::Result<u64> + Send + Sync;

pub struct ModelManager {
    root: PathBuf,
    manifest: ManifestRoot,
    source: Arc<dyn ArtifactSource>,
    space_probe: Arc<SpaceProbe>,
}

impl ModelManager {
    pub fn new() -> Result<Self, ModelError> {
        let manifest = parse_embedded_manifest()?;
        let root = paths::models_dir();
        cleanup_stale_workdirs(&root, &manifest)?;
        Ok(Self {
            root,
            manifest,
            source: Arc::new(HttpArtifactSource::new()?),
            space_probe: Arc::new(available_space),
        })
    }

    pub fn verify_embedded_manifest() -> Result<(), ModelError> {
        parse_embedded_manifest().map(|_| ())
    }

    pub fn ensure<F>(
        &self,
        model_id: &str,
        cancellation: &CancellationToken,
        mut emit: F,
    ) -> Result<PathBuf, ModelError>
    where
        F: FnMut(ModelState),
    {
        let result = self.ensure_inner(model_id, cancellation, &mut emit);
        match &result {
            Ok(_) => emit(ModelState::Ready),
            Err(ModelError::Cancelled) => emit(ModelState::Cancelled),
            Err(error) => emit(ModelState::Failed {
                message: error.to_string(),
            }),
        }
        result
    }

    fn ensure_inner<F>(
        &self,
        model_id: &str,
        cancellation: &CancellationToken,
        emit: &mut F,
    ) -> Result<PathBuf, ModelError>
    where
        F: FnMut(ModelState),
    {
        let model = self
            .manifest
            .models
            .iter()
            .find(|model| model.id == model_id)
            .ok_or_else(|| anyhow!("unknown model {model_id}"))?;
        if model.architecture != "any" && model.architecture != std::env::consts::ARCH {
            bail!(
                "model {} requires {}, current architecture is {}",
                model.id,
                model.architecture,
                std::env::consts::ARCH
            );
        }

        check_cancelled(cancellation)?;
        fs::create_dir_all(&self.root)
            .with_context(|| format!("create model directory {}", self.root.display()))?;
        let target = self.root.join(&model.local_dir);

        if target.exists() {
            emit(ModelState::Verifying);
            match validate_directory(model, &target, cancellation) {
                Ok(()) => {
                    info!(
                        model = model.id,
                        version = model.version,
                        "model cache verified"
                    );
                    return Ok(target);
                }
                Err(ModelError::Cancelled) => return Err(ModelError::Cancelled),
                Err(error) => {
                    warn!(
                        model = model.id,
                        error = %error,
                        "model cache is incomplete or corrupt; rebuilding"
                    );
                }
            }
        } else {
            emit(ModelState::Missing);
        }

        let required = required_staging_bytes(model)?;
        let available = (self.space_probe)(&self.root)?;
        if available < required {
            bail!(
                "not enough disk space for {}: need {} bytes, {} bytes available",
                model.id,
                required,
                available
            );
        }

        check_cancelled(cancellation)?;
        let staging = self.staging_path(model);
        remove_path_if_exists(&staging)?;
        fs::create_dir_all(&staging)
            .with_context(|| format!("create staging directory {}", staging.display()))?;

        let staged = self.stage_model(model, &staging, cancellation, emit);
        if let Err(error) = staged {
            let _ = remove_path_if_exists(&staging);
            return Err(error);
        }

        emit(ModelState::Verifying);
        if let Err(error) = validate_directory(model, &staging, cancellation) {
            let _ = remove_path_if_exists(&staging);
            return Err(error);
        }
        write_receipt(model, &staging, self.manifest.schema_version)?;
        promote_directory(&staging, &target)?;
        info!(model = model.id, version = model.version, "model promoted");
        Ok(target)
    }

    fn stage_model<F>(
        &self,
        model: &ModelManifest,
        staging: &Path,
        cancellation: &CancellationToken,
        emit: &mut F,
    ) -> Result<(), ModelError>
    where
        F: FnMut(ModelState),
    {
        let total_bytes = network_bytes(model)?;
        let mut downloaded_bytes = 0_u64;

        if let Some(archive) = &model.archive {
            let archive_path = staging.join(".model-download.zip");
            self.download(
                &archive.url,
                archive.bytes,
                &archive.sha256,
                &archive_path,
                cancellation,
                &mut downloaded_bytes,
                total_bytes,
                emit,
            )?;
            emit(ModelState::Verifying);
            extract_archive(archive, &archive_path, staging, cancellation)?;
            fs::remove_file(&archive_path)
                .with_context(|| format!("remove {}", archive_path.display()))?;
        }

        for file in &model.files {
            let destination = staging.join(&file.path);
            self.download(
                &file.url,
                file.bytes,
                &file.sha256,
                &destination,
                cancellation,
                &mut downloaded_bytes,
                total_bytes,
                emit,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn download<F>(
        &self,
        url: &str,
        expected_bytes: u64,
        expected_sha256: &str,
        destination: &Path,
        cancellation: &CancellationToken,
        downloaded_bytes: &mut u64,
        total_bytes: u64,
        emit: &mut F,
    ) -> Result<(), ModelError>
    where
        F: FnMut(ModelState),
    {
        check_cancelled(cancellation)?;
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }

        let mut source = self.source.open(url)?;
        if let Some(content_length) = source.content_length {
            if content_length != expected_bytes {
                bail!(
                    "unexpected content length for {url}: expected {expected_bytes}, got {content_length}"
                );
            }
        }

        let mut output = File::create(destination)
            .with_context(|| format!("create {}", destination.display()))?;
        let mut hasher = Sha256::new();
        let mut file_bytes = 0_u64;
        let mut buffer = vec![0_u8; DOWNLOAD_BUFFER_BYTES];
        loop {
            check_cancelled(cancellation)?;
            let read = source
                .reader
                .read(&mut buffer)
                .with_context(|| format!("read {url}"))?;
            if read == 0 {
                break;
            }
            output
                .write_all(&buffer[..read])
                .with_context(|| format!("write {}", destination.display()))?;
            hasher.update(&buffer[..read]);
            file_bytes = file_bytes
                .checked_add(read as u64)
                .ok_or_else(|| anyhow!("download size overflow"))?;
            *downloaded_bytes = downloaded_bytes
                .checked_add(read as u64)
                .ok_or_else(|| anyhow!("download progress overflow"))?;
            emit(ModelState::Downloading {
                downloaded_bytes: *downloaded_bytes,
                total_bytes,
            });
        }
        output
            .sync_all()
            .with_context(|| format!("flush {}", destination.display()))?;

        if file_bytes != expected_bytes {
            bail!(
                "size mismatch for {}: expected {}, got {}",
                destination.display(),
                expected_bytes,
                file_bytes
            );
        }
        let actual_sha256 = format!("{:x}", hasher.finalize());
        if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
            bail!(
                "SHA-256 mismatch for {}: expected {}, got {}",
                destination.display(),
                expected_sha256,
                actual_sha256
            );
        }
        Ok(())
    }

    fn staging_path(&self, model: &ModelManifest) -> PathBuf {
        self.root.join(format!(".{}.staging", model.local_dir))
    }

    #[cfg(test)]
    fn from_parts(
        root: PathBuf,
        manifest: ManifestRoot,
        source: Arc<dyn ArtifactSource>,
        space_probe: Arc<SpaceProbe>,
    ) -> Result<Self, ModelError> {
        validate_manifest(&manifest)?;
        cleanup_stale_workdirs(&root, &manifest)?;
        Ok(Self {
            root,
            manifest,
            source,
            space_probe,
        })
    }
}

pub fn diagnostic_status(root: &Path) -> anyhow::Result<serde_json::Value> {
    let manifest = parse_embedded_manifest().map_err(|error| anyhow!(error))?;
    let models = manifest
        .models
        .iter()
        .map(|model| {
            let directory = root.join(&model.local_dir);
            let files = expected_files(model)
                .into_iter()
                .map(|(path, expected_bytes, expected_sha256)| {
                    let metadata = directory.join(path).metadata().ok();
                    json!({
                        "path": path,
                        "present": metadata.is_some(),
                        "bytes": metadata.as_ref().map(|value| value.len()),
                        "expected_bytes": expected_bytes,
                        "size_matches": metadata
                            .as_ref()
                            .map(|value| value.is_file() && value.len() == expected_bytes)
                            .unwrap_or(false),
                        "expected_sha256": expected_sha256,
                    })
                })
                .collect::<Vec<_>>();
            let receipt = directory
                .join("model-receipt.json")
                .is_file()
                .then(|| fs::read(directory.join("model-receipt.json")).ok())
                .flatten()
                .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
            json!({
                "id": model.id,
                "version": model.version,
                "architecture": model.architecture,
                "architecture_supported": model.architecture == "any"
                    || model.architecture == std::env::consts::ARCH,
                "directory": model.local_dir,
                "directory_present": directory.is_dir(),
                "receipt": receipt,
                "files": files,
            })
        })
        .collect::<Vec<_>>();
    let known_directories = manifest
        .models
        .iter()
        .map(|model| model.local_dir.as_str())
        .collect::<HashSet<_>>();
    let unknown_directories = if root.is_dir() {
        let mut names = fs::read_dir(root)
            .with_context(|| format!("read {}", root.display()))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false))
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| !name.starts_with('.') && !known_directories.contains(name.as_str()))
            .collect::<Vec<_>>();
        names.sort();
        names
    } else {
        Vec::new()
    };
    Ok(json!({
        "schema_version": manifest.schema_version,
        "models_directory_present": root.is_dir(),
        "models": models,
        "unknown_directories": unknown_directories,
    }))
}

fn parse_embedded_manifest() -> Result<ManifestRoot, ModelError> {
    let manifest = serde_json::from_str(MANIFEST_JSON).context("parse models-manifest.json")?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

fn validate_manifest(manifest: &ManifestRoot) -> Result<(), ModelError> {
    if manifest.schema_version != 1 {
        bail!(
            "unsupported model manifest schema {}",
            manifest.schema_version
        );
    }
    let mut ids = HashSet::new();
    let mut directories = HashSet::new();
    for model in &manifest.models {
        if !ids.insert(model.id.as_str()) {
            bail!("duplicate model id {}", model.id);
        }
        if !directories.insert(model.local_dir.as_str()) {
            bail!("duplicate model directory {}", model.local_dir);
        }
        validate_relative_path(&model.local_dir)?;
        if !matches!(model.architecture.as_str(), "any" | "aarch64" | "x86_64") {
            bail!("unsupported architecture {}", model.architecture);
        }
        if model.files.is_empty() && model.archive.is_none() {
            bail!("model {} has no artifacts", model.id);
        }

        let mut target_paths = HashSet::new();
        for file in &model.files {
            validate_file_spec(&model.id, &file.path, &file.url, file.bytes, &file.sha256)?;
            if !target_paths.insert(file.path.as_str()) {
                bail!("duplicate target path {} in {}", file.path, model.id);
            }
        }
        if let Some(archive) = &model.archive {
            validate_url(&archive.url)?;
            validate_sha256(&archive.sha256)?;
            if archive.bytes == 0 {
                bail!("archive for {} has zero length", model.id);
            }
            if archive.files.is_empty() {
                bail!("archive for {} has no files", model.id);
            }
            for file in &archive.files {
                validate_relative_path(&file.archive_path)?;
                validate_file_spec(
                    &model.id,
                    &file.path,
                    &archive.url,
                    file.bytes,
                    &file.sha256,
                )?;
                if !target_paths.insert(file.path.as_str()) {
                    bail!("duplicate target path {} in {}", file.path, model.id);
                }
            }
        }
    }
    Ok(())
}

fn validate_file_spec(
    model_id: &str,
    path: &str,
    url: &str,
    bytes: u64,
    sha256: &str,
) -> Result<(), ModelError> {
    validate_relative_path(path)?;
    validate_url(url)?;
    validate_sha256(sha256)?;
    if bytes == 0 {
        bail!("file {path} in {model_id} has zero length");
    }
    Ok(())
}

fn validate_relative_path(path: &str) -> Result<(), ModelError> {
    let path = Path::new(path);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        bail!("unsafe relative path {}", path.display());
    }
    Ok(())
}

fn validate_url(url: &str) -> Result<(), ModelError> {
    if !url.starts_with("https://") {
        bail!("model URL must use HTTPS: {url}");
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), ModelError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("invalid SHA-256 value {value}");
    }
    Ok(())
}

fn expected_files(model: &ModelManifest) -> Vec<(&str, u64, &str)> {
    let mut files = Vec::with_capacity(
        model.files.len()
            + model
                .archive
                .as_ref()
                .map(|archive| archive.files.len())
                .unwrap_or_default(),
    );
    if let Some(archive) = &model.archive {
        files.extend(
            archive
                .files
                .iter()
                .map(|file| (file.path.as_str(), file.bytes, file.sha256.as_str())),
        );
    }
    files.extend(
        model
            .files
            .iter()
            .map(|file| (file.path.as_str(), file.bytes, file.sha256.as_str())),
    );
    files
}

fn validate_directory(
    model: &ModelManifest,
    directory: &Path,
    cancellation: &CancellationToken,
) -> Result<(), ModelError> {
    for (relative, expected_bytes, expected_sha256) in expected_files(model) {
        check_cancelled(cancellation)?;
        let path = directory.join(relative);
        let metadata = path
            .metadata()
            .with_context(|| format!("missing required model file {}", path.display()))?;
        if !metadata.is_file() {
            bail!("required model path is not a file: {}", path.display());
        }
        if metadata.len() != expected_bytes {
            bail!(
                "size mismatch for {}: expected {}, got {}",
                path.display(),
                expected_bytes,
                metadata.len()
            );
        }
        let actual_sha256 = sha256_file(&path, cancellation)?;
        if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
            bail!(
                "SHA-256 mismatch for {}: expected {}, got {}",
                path.display(),
                expected_sha256,
                actual_sha256
            );
        }
    }
    Ok(())
}

fn sha256_file(path: &Path, cancellation: &CancellationToken) -> Result<String, ModelError> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; DOWNLOAD_BUFFER_BYTES];
    loop {
        check_cancelled(cancellation)?;
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn extract_archive(
    archive: &ArchiveArtifact,
    archive_path: &Path,
    staging: &Path,
    cancellation: &CancellationToken,
) -> Result<(), ModelError> {
    let file =
        File::open(archive_path).with_context(|| format!("open {}", archive_path.display()))?;
    let mut zip = zip::ZipArchive::new(file).context("open model ZIP")?;
    for expected in &archive.files {
        check_cancelled(cancellation)?;
        let mut entry = zip
            .by_name(&expected.archive_path)
            .with_context(|| format!("missing ZIP entry {}", expected.archive_path))?;
        if entry.size() != expected.bytes {
            bail!(
                "size mismatch for ZIP entry {}: expected {}, got {}",
                expected.archive_path,
                expected.bytes,
                entry.size()
            );
        }
        let destination = staging.join(&expected.path);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
        let mut output = File::create(&destination)
            .with_context(|| format!("create {}", destination.display()))?;
        let mut hasher = Sha256::new();
        let mut written = 0_u64;
        let mut buffer = vec![0_u8; DOWNLOAD_BUFFER_BYTES];
        loop {
            check_cancelled(cancellation)?;
            let read = entry
                .read(&mut buffer)
                .with_context(|| format!("read ZIP entry {}", expected.archive_path))?;
            if read == 0 {
                break;
            }
            output
                .write_all(&buffer[..read])
                .with_context(|| format!("write {}", destination.display()))?;
            hasher.update(&buffer[..read]);
            written = written
                .checked_add(read as u64)
                .ok_or_else(|| anyhow!("extracted size overflow"))?;
        }
        output
            .sync_all()
            .with_context(|| format!("flush {}", destination.display()))?;
        if written != expected.bytes {
            bail!(
                "size mismatch for extracted {}: expected {}, got {}",
                destination.display(),
                expected.bytes,
                written
            );
        }
        let actual_sha256 = format!("{:x}", hasher.finalize());
        if !actual_sha256.eq_ignore_ascii_case(&expected.sha256) {
            bail!(
                "SHA-256 mismatch for extracted {}: expected {}, got {}",
                destination.display(),
                expected.sha256,
                actual_sha256
            );
        }
    }
    Ok(())
}

fn required_staging_bytes(model: &ModelManifest) -> Result<u64, ModelError> {
    let direct = model
        .files
        .iter()
        .try_fold(0_u64, |total, file| -> Result<u64, ModelError> {
            total
                .checked_add(file.bytes)
                .ok_or_else(|| ModelError::from(anyhow!("model size overflow")))
        })?;
    let archive = model.archive.as_ref().map_or(Ok(0_u64), |archive| {
        archive
            .files
            .iter()
            .try_fold(archive.bytes, |total, file| -> Result<u64, ModelError> {
                total
                    .checked_add(file.bytes)
                    .ok_or_else(|| ModelError::from(anyhow!("model size overflow")))
            })
    })?;
    direct
        .checked_add(archive)
        .and_then(|total| total.checked_add(STAGING_MARGIN_BYTES))
        .ok_or_else(|| anyhow!("model staging size overflow").into())
}

fn network_bytes(model: &ModelManifest) -> Result<u64, ModelError> {
    let direct = model
        .files
        .iter()
        .try_fold(0_u64, |total, file| -> Result<u64, ModelError> {
            total
                .checked_add(file.bytes)
                .ok_or_else(|| ModelError::from(anyhow!("model download size overflow")))
        })?;
    direct
        .checked_add(
            model
                .archive
                .as_ref()
                .map(|archive| archive.bytes)
                .unwrap_or_default(),
        )
        .ok_or_else(|| anyhow!("model download size overflow").into())
}

fn write_receipt(
    model: &ModelManifest,
    staging: &Path,
    schema_version: u32,
) -> Result<(), ModelError> {
    let receipt = json!({
        "schema_version": schema_version,
        "model_id": model.id,
        "version": model.version,
        "files": expected_files(model)
            .into_iter()
            .map(|(path, bytes, sha256)| json!({
                "path": path,
                "bytes": bytes,
                "sha256": sha256,
            }))
            .collect::<Vec<_>>(),
    });
    let path = staging.join("model-receipt.json");
    let bytes = serde_json::to_vec_pretty(&receipt).context("serialize model receipt")?;
    fs::write(&path, bytes).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn promote_directory(staging: &Path, target: &Path) -> Result<(), ModelError> {
    let backup = target.with_file_name(format!(
        ".{}.replaced",
        target
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model")
    ));
    remove_path_if_exists(&backup)?;
    let had_target = target.exists();
    if had_target {
        fs::rename(target, &backup)
            .with_context(|| format!("quarantine invalid model {}", target.display()))?;
    }

    if let Err(error) = fs::rename(staging, target) {
        if had_target {
            fs::rename(&backup, target).with_context(|| {
                format!(
                    "restore previous model {} after promotion failure",
                    target.display()
                )
            })?;
        }
        return Err(error)
            .with_context(|| format!("promote verified model {}", target.display()))
            .map_err(ModelError::from);
    }

    if had_target {
        if let Err(error) = remove_path_if_exists(&backup) {
            warn!(
                path = %backup.display(),
                error = %error,
                "could not remove quarantined model directory"
            );
        }
    }
    Ok(())
}

fn cleanup_stale_workdirs(root: &Path, manifest: &ManifestRoot) -> Result<(), ModelError> {
    if !root.exists() {
        return Ok(());
    }
    for model in &manifest.models {
        remove_path_if_exists(&root.join(format!(".{}.staging", model.local_dir)))?;
        remove_path_if_exists(&root.join(format!(".{}.replaced", model.local_dir)))?;
    }
    Ok(())
}

fn remove_path_if_exists(path: &Path) -> Result<(), ModelError> {
    if !path.exists() {
        return Ok(());
    }
    let metadata =
        fs::symlink_metadata(path).with_context(|| format!("inspect {}", path.display()))?;
    if metadata.is_dir() {
        fs::remove_dir_all(path).with_context(|| format!("remove {}", path.display()))?;
    } else {
        fs::remove_file(path).with_context(|| format!("remove {}", path.display()))?;
    }
    Ok(())
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<(), ModelError> {
    if cancellation.is_cancelled() {
        Err(ModelError::Cancelled)
    } else {
        Ok(())
    }
}

fn available_space(path: &Path) -> anyhow::Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    let wide = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let mut available = 0_u64;
    let success = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if success == 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("query disk space for {}", path.display()));
    }
    Ok(available)
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::io::{Cursor, ErrorKind};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct MemorySource {
        files: Mutex<HashMap<String, Vec<u8>>>,
        failures_remaining: AtomicUsize,
        fail_after: usize,
    }

    impl MemorySource {
        fn with_file(url: &str, bytes: &[u8]) -> Self {
            let mut files = HashMap::new();
            files.insert(url.to_string(), bytes.to_vec());
            Self {
                files: Mutex::new(files),
                failures_remaining: AtomicUsize::new(0),
                fail_after: 0,
            }
        }

        fn fail_once(url: &str, bytes: &[u8], fail_after: usize) -> Self {
            let mut source = Self::with_file(url, bytes);
            source.failures_remaining = AtomicUsize::new(1);
            source.fail_after = fail_after;
            source
        }
    }

    impl ArtifactSource for MemorySource {
        fn open(&self, url: &str) -> anyhow::Result<SourceStream> {
            let bytes = self
                .files
                .lock()
                .get(url)
                .cloned()
                .ok_or_else(|| anyhow!("missing test URL {url}"))?;
            let fail = self
                .failures_remaining
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok();
            let content_length = Some(bytes.len() as u64);
            let reader: Box<dyn Read> = if fail {
                Box::new(FailingReader {
                    inner: Cursor::new(bytes),
                    fail_after: self.fail_after,
                    failed: false,
                })
            } else {
                Box::new(Cursor::new(bytes))
            };
            Ok(SourceStream {
                reader,
                content_length,
            })
        }
    }

    struct FailingReader {
        inner: Cursor<Vec<u8>>,
        fail_after: usize,
        failed: bool,
    }

    impl Read for FailingReader {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            let position = self.inner.position() as usize;
            if !self.failed && position >= self.fail_after {
                self.failed = true;
                return Err(std::io::Error::new(
                    ErrorKind::ConnectionReset,
                    "injected interruption",
                ));
            }
            let allowed = self.fail_after.saturating_sub(position);
            if !self.failed && allowed < buffer.len() {
                return self.inner.read(&mut buffer[..allowed]);
            }
            self.inner.read(buffer)
        }
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn direct_manifest(url: &str, content: &[u8], expected_sha256: String) -> ManifestRoot {
        ManifestRoot {
            schema_version: 1,
            models: vec![ModelManifest {
                id: "test".into(),
                version: "1".into(),
                local_dir: "test-model".into(),
                architecture: "any".into(),
                files: vec![DirectFile {
                    path: "model.bin".into(),
                    url: url.into(),
                    bytes: content.len() as u64,
                    sha256: expected_sha256,
                }],
                archive: None,
            }],
        }
    }

    fn manager(
        root: &Path,
        manifest: ManifestRoot,
        source: Arc<dyn ArtifactSource>,
    ) -> ModelManager {
        ModelManager::from_parts(
            root.to_path_buf(),
            manifest,
            source,
            Arc::new(|_| Ok(u64::MAX)),
        )
        .unwrap()
    }

    #[test]
    fn downloads_verifies_and_promotes_a_model() {
        let temp = tempfile::tempdir().unwrap();
        let content = b"verified model";
        let url = "https://example.test/model";
        let source = Arc::new(MemorySource::with_file(url, content));
        let manager = manager(
            temp.path(),
            direct_manifest(url, content, sha256(content)),
            source,
        );
        let states = Mutex::new(Vec::new());

        let target = manager
            .ensure("test", &CancellationToken::default(), |state| {
                states.lock().push(state)
            })
            .unwrap();

        assert_eq!(fs::read(target.join("model.bin")).unwrap(), content);
        assert!(target.join("model-receipt.json").is_file());
        assert!(states
            .lock()
            .iter()
            .any(|state| matches!(state, ModelState::Downloading { .. })));
        assert_eq!(states.lock().last(), Some(&ModelState::Ready));
    }

    #[test]
    fn partial_non_empty_cache_is_replaced() {
        let temp = tempfile::tempdir().unwrap();
        let content = b"complete model";
        let url = "https://example.test/model";
        let target = temp.path().join("test-model");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("model.bin"), b"x").unwrap();
        let manager = manager(
            temp.path(),
            direct_manifest(url, content, sha256(content)),
            Arc::new(MemorySource::with_file(url, content)),
        );

        manager
            .ensure("test", &CancellationToken::default(), |_| {})
            .unwrap();

        assert_eq!(fs::read(target.join("model.bin")).unwrap(), content);
    }

    #[test]
    fn hash_mismatch_preserves_the_existing_cache() {
        let temp = tempfile::tempdir().unwrap();
        let content = b"new model";
        let url = "https://example.test/model";
        let target = temp.path().join("test-model");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("model.bin"), b"old model").unwrap();
        let manager = manager(
            temp.path(),
            direct_manifest(url, content, sha256(b"different")),
            Arc::new(MemorySource::with_file(url, content)),
        );

        assert!(manager
            .ensure("test", &CancellationToken::default(), |_| {})
            .is_err());
        assert_eq!(fs::read(target.join("model.bin")).unwrap(), b"old model");
    }

    #[test]
    fn cancellation_removes_staging_without_promoting() {
        let temp = tempfile::tempdir().unwrap();
        let content = vec![7_u8; DOWNLOAD_BUFFER_BYTES * 2];
        let url = "https://example.test/model";
        let manager = manager(
            temp.path(),
            direct_manifest(url, &content, sha256(&content)),
            Arc::new(MemorySource::with_file(url, &content)),
        );
        let cancellation = CancellationToken::default();
        let cancel_from_event = cancellation.clone();

        let result = manager.ensure("test", &cancellation, |state| {
            if matches!(state, ModelState::Downloading { .. }) {
                cancel_from_event.cancel();
            }
        });

        assert!(matches!(result, Err(ModelError::Cancelled)));
        assert!(!temp.path().join("test-model").exists());
        assert!(!temp.path().join(".test-model.staging").exists());
    }

    #[test]
    fn interrupted_download_can_be_retried_atomically() {
        let temp = tempfile::tempdir().unwrap();
        let content = b"new verified model";
        let url = "https://example.test/model";
        let target = temp.path().join("test-model");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("model.bin"), b"previous model").unwrap();
        let manager = manager(
            temp.path(),
            direct_manifest(url, content, sha256(content)),
            Arc::new(MemorySource::fail_once(url, content, 4)),
        );

        assert!(manager
            .ensure("test", &CancellationToken::default(), |_| {})
            .is_err());
        assert_eq!(
            fs::read(target.join("model.bin")).unwrap(),
            b"previous model"
        );

        manager
            .ensure("test", &CancellationToken::default(), |_| {})
            .unwrap();
        assert_eq!(fs::read(target.join("model.bin")).unwrap(), content);
    }

    #[test]
    fn extracts_and_verifies_archive_files() {
        let temp = tempfile::tempdir().unwrap();
        let file_content = b"archive model";
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        writer
            .start_file("root/model.bin", zip::write::SimpleFileOptions::default())
            .unwrap();
        writer.write_all(file_content).unwrap();
        let archive_bytes = writer.finish().unwrap().into_inner();
        let url = "https://example.test/model.zip";
        let manifest = ManifestRoot {
            schema_version: 1,
            models: vec![ModelManifest {
                id: "test".into(),
                version: "1".into(),
                local_dir: "test-model".into(),
                architecture: "any".into(),
                files: Vec::new(),
                archive: Some(ArchiveArtifact {
                    url: url.into(),
                    bytes: archive_bytes.len() as u64,
                    sha256: sha256(&archive_bytes),
                    files: vec![ArchiveFile {
                        archive_path: "root/model.bin".into(),
                        path: "model.bin".into(),
                        bytes: file_content.len() as u64,
                        sha256: sha256(file_content),
                    }],
                }),
            }],
        };
        let manager = manager(
            temp.path(),
            manifest,
            Arc::new(MemorySource::with_file(url, &archive_bytes)),
        );

        let target = manager
            .ensure("test", &CancellationToken::default(), |_| {})
            .unwrap();

        assert_eq!(fs::read(target.join("model.bin")).unwrap(), file_content);
        assert!(!target.join(".model-download.zip").exists());
    }

    #[test]
    fn refuses_download_when_disk_space_is_insufficient() {
        let temp = tempfile::tempdir().unwrap();
        let content = b"model";
        let url = "https://example.test/model";
        let manager = ModelManager::from_parts(
            temp.path().to_path_buf(),
            direct_manifest(url, content, sha256(content)),
            Arc::new(MemorySource::with_file(url, content)),
            Arc::new(|_| Ok(0)),
        )
        .unwrap();

        let error = manager
            .ensure("test", &CancellationToken::default(), |_| {})
            .unwrap_err();

        assert!(error.to_string().contains("not enough disk space"));
    }

    #[test]
    fn removes_stale_work_directories_from_a_previous_process() {
        let temp = tempfile::tempdir().unwrap();
        let content = b"model";
        let url = "https://example.test/model";
        let staging = temp.path().join(".test-model.staging");
        let replaced = temp.path().join(".test-model.replaced");
        fs::create_dir_all(&staging).unwrap();
        fs::create_dir_all(&replaced).unwrap();
        fs::write(staging.join("partial.bin"), b"partial").unwrap();
        fs::write(replaced.join("old.bin"), b"old").unwrap();

        let _manager = manager(
            temp.path(),
            direct_manifest(url, content, sha256(content)),
            Arc::new(MemorySource::with_file(url, content)),
        );

        assert!(!staging.exists());
        assert!(!replaced.exists());
    }

    #[test]
    #[ignore = "validates or downloads the real pinned Parakeet model cache"]
    fn validates_real_parakeet_manifests() {
        let manager = ModelManager::new().unwrap();
        manager
            .ensure("parakeet_cpu", &CancellationToken::default(), |_| {})
            .unwrap();
        if cfg!(target_arch = "aarch64") {
            manager
                .ensure("parakeet_npu", &CancellationToken::default(), |_| {})
                .unwrap();
        }
    }
}
