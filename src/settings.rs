use crate::credentials::{
    store_verified, CredentialError, CredentialStore, WindowsCredentialStore,
};
use crate::paths::settings_path;
use crate::single_instance::SettingsTransaction;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

const MAX_RECORD_SECONDS: f32 = 60.0 * 60.0;
const VALID_MODIFIERS: &[&str] = &["ctrl", "shift", "alt", "win"];
const VALID_TRIGGERS: &[&str] = &[
    "none",
    "space",
    "tab",
    "caps_lock",
    "scroll_lock",
    "pause",
    "insert",
    "right_ctrl",
    "f13",
    "f14",
    "f15",
    "f16",
    "f17",
    "f18",
    "f19",
    "f20",
];
const VALID_ENGINES: &[&str] = &["parakeet_cpu", "parakeet_npu", "whisper_npu"];
const VALID_ENHANCE_PROVIDERS: &[&str] = &["github_copilot", "openai_compatible"];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnhanceMode {
    #[default]
    Never,
    WithShift,
    Always,
}

impl EnhanceMode {
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Never)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Enhance {
    pub mode: EnhanceMode,
    pub provider: String,
    pub base_url: String,
    pub model: String,
}

impl Default for Enhance {
    fn default() -> Self {
        Self {
            mode: EnhanceMode::Never,
            provider: "github_copilot".into(),
            base_url: "https://api.openai.com/v1".into(),
            model: "claude-haiku-4.5".into(),
        }
    }
}

#[derive(Deserialize)]
struct EnhanceDocument {
    mode: Option<EnhanceMode>,
    provider: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
}

impl<'de> Deserialize<'de> for Enhance {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let document = EnhanceDocument::deserialize(deserializer)?;
        let defaults = Self::default();
        let legacy_document = document.mode.is_none();
        let legacy_provider = document.provider.as_deref();
        let mode = match (document.mode, legacy_provider) {
            (_, Some("off")) => EnhanceMode::Never,
            (Some(mode), _) => mode,
            (None, Some(_)) => EnhanceMode::WithShift,
            (None, None) => defaults.mode,
        };
        let provider = match document.provider.as_deref() {
            Some("off") | None => defaults.provider,
            Some(_) => document.provider.expect("provider was checked"),
        };
        let model = match document.model {
            Some(model) if !legacy_document || !model.trim().is_empty() => model,
            _ => defaults.model,
        };
        Ok(Self {
            mode,
            provider,
            base_url: document.base_url.unwrap_or(defaults.base_url),
            model,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub hotkey_modifiers: Vec<String>,
    pub hotkey_trigger: String,
    pub engine: String,
    pub auto_paste: bool,
    pub overlay: bool,
    pub sounds: bool,
    pub min_record_seconds: f32,
    pub max_record_seconds: f32,
    pub enhance: Enhance,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            hotkey_modifiers: vec!["ctrl".into(), "win".into()],
            hotkey_trigger: "none".into(),
            engine: "parakeet_cpu".into(),
            auto_paste: true,
            overlay: true,
            sounds: true,
            min_record_seconds: 0.25,
            max_record_seconds: 60.0,
            enhance: Enhance::default(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CredentialHealth {
    pub enhancement_disabled: bool,
    pub requires_user_resolution: bool,
    pub message: Option<String>,
}

#[derive(Clone, Debug)]
pub struct LoadedSettings {
    pub settings: Settings,
    pub credential_health: CredentialHealth,
    pub settings_error: Option<String>,
    pub revision: SettingsRevision,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SettingsRevision(Option<[u8; 32]>);

impl SettingsRevision {
    fn from_bytes(bytes: Option<&[u8]>) -> Self {
        Self(bytes.map(|bytes| Sha256::digest(bytes).into()))
    }
}

#[derive(Debug, Error)]
pub enum SettingsError {
    #[error("{operation} {path}: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("parse {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("serialize settings: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("invalid settings: {0}")]
    Validation(String),
    #[error("settings transaction lock failed: {0}")]
    Transaction(String),
}

impl Settings {
    pub fn load() -> Result<Self, SettingsError> {
        Self::load_from(&settings_path())
    }

    pub fn load_from(path: &Path) -> Result<Self, SettingsError> {
        let document = read_document(path)?;
        settings_from_document(path, document.as_ref())
    }

    pub fn load_runtime() -> Result<LoadedSettings, SettingsError> {
        load_with_migration(&settings_path(), &WindowsCredentialStore)
    }

    pub fn revision() -> Result<SettingsRevision, SettingsError> {
        settings_revision(&settings_path())
    }

    pub(crate) fn revision_from(path: &Path) -> Result<SettingsRevision, SettingsError> {
        settings_revision(path)
    }

    fn validate_compatible_shortcut(&self) -> Result<(), SettingsError> {
        if self.hotkey_modifiers.is_empty() && self.hotkey_trigger == "none" {
            return Err(SettingsError::Validation(
                "select at least one modifier or trigger key".into(),
            ));
        }
        for modifier in &self.hotkey_modifiers {
            if !VALID_MODIFIERS.contains(&modifier.as_str()) {
                return Err(SettingsError::Validation(format!(
                    "unsupported hotkey modifier `{modifier}`"
                )));
            }
        }
        let mut unique = self.hotkey_modifiers.clone();
        unique.sort();
        unique.dedup();
        if unique.len() != self.hotkey_modifiers.len() {
            return Err(SettingsError::Validation(
                "hotkey modifiers must not contain duplicates".into(),
            ));
        }
        if !VALID_TRIGGERS.contains(&self.hotkey_trigger.as_str()) {
            return Err(SettingsError::Validation(format!(
                "unsupported trigger key `{}`",
                self.hotkey_trigger
            )));
        }
        Ok(())
    }

    pub fn validate_shortcut(&self) -> Result<(), SettingsError> {
        self.validate_compatible_shortcut()?;
        if self.hotkey_trigger == "none" && self.hotkey_modifiers.len() < 2 {
            return Err(SettingsError::Validation(
                "a modifier-only shortcut requires at least two modifier keys".into(),
            ));
        }
        if self.hotkey_trigger != "none"
            && self.hotkey_modifiers.is_empty()
            && !is_standalone_function_trigger(&self.hotkey_trigger)
        {
            return Err(SettingsError::Validation(
                "a shortcut without modifiers may use only F13 through F20".into(),
            ));
        }
        if self.enhance.mode == EnhanceMode::WithShift
            && self
                .hotkey_modifiers
                .iter()
                .any(|modifier| modifier == "shift")
        {
            return Err(SettingsError::Validation(
                "Shift cannot be both part of the recording shortcut and the additional enhancement key"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn validate_enhancement(&self) -> Result<(), SettingsError> {
        if !VALID_ENHANCE_PROVIDERS.contains(&self.enhance.provider.as_str()) {
            return Err(SettingsError::Validation(format!(
                "unsupported enhancement provider `{}`",
                self.enhance.provider
            )));
        }
        if !self.enhance.mode.is_enabled() {
            return Ok(());
        }
        if self.enhance.model.trim().is_empty() {
            return Err(SettingsError::Validation(
                "enhancement model ID must not be empty".into(),
            ));
        }
        if self.enhance.provider == "openai_compatible" {
            let url = reqwest::Url::parse(self.enhance.base_url.trim()).map_err(|error| {
                SettingsError::Validation(format!("OpenAI-compatible base URL is invalid: {error}"))
            })?;
            if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
                return Err(SettingsError::Validation(
                    "OpenAI-compatible base URL must be an HTTP(S) URL with a host".into(),
                ));
            }
        }
        Ok(())
    }

    fn validate_non_shortcut_fields(&self) -> Result<(), SettingsError> {
        if !VALID_ENGINES.contains(&self.engine.as_str()) {
            return Err(SettingsError::Validation(format!(
                "unsupported transcription engine `{}`",
                self.engine
            )));
        }
        self.validate_enhancement()?;
        if !self.min_record_seconds.is_finite() || self.min_record_seconds < 0.0 {
            return Err(SettingsError::Validation(
                "minimum recording duration must be a finite non-negative number".into(),
            ));
        }
        if !self.max_record_seconds.is_finite()
            || self.max_record_seconds <= 0.0
            || self.max_record_seconds > MAX_RECORD_SECONDS
        {
            return Err(SettingsError::Validation(format!(
                "maximum recording duration must be between 0 and {MAX_RECORD_SECONDS} seconds"
            )));
        }
        if self.min_record_seconds > self.max_record_seconds {
            return Err(SettingsError::Validation(
                "minimum recording duration cannot exceed the maximum".into(),
            ));
        }
        Ok(())
    }

    fn validate_for_load(&self) -> Result<(), SettingsError> {
        self.validate_compatible_shortcut()?;
        self.validate_non_shortcut_fields()
    }

    pub fn validate(&self) -> Result<(), SettingsError> {
        self.validate_shortcut()?;
        self.validate_non_shortcut_fields()
    }

    pub fn save_to(&self, path: &Path) -> Result<(), SettingsError> {
        self.validate()?;
        let document = serde_json::to_value(self)?;
        atomic_write_json(path, &document)
    }

    #[allow(dead_code)]
    pub fn save(&self) -> Result<(), SettingsError> {
        self.save_to(&settings_path())
    }
}

fn is_standalone_function_trigger(trigger: &str) -> bool {
    matches!(
        trigger,
        "f13" | "f14" | "f15" | "f16" | "f17" | "f18" | "f19" | "f20"
    )
}

fn load_with_migration(
    path: &Path,
    store: &dyn CredentialStore,
) -> Result<LoadedSettings, SettingsError> {
    load_with_migration_and_writer(path, store, &atomic_write_json)
}

fn load_with_migration_and_writer(
    path: &Path,
    store: &dyn CredentialStore,
    writer: &dyn Fn(&Path, &Value) -> Result<(), SettingsError>,
) -> Result<LoadedSettings, SettingsError> {
    let mut snapshot = read_runtime_snapshot(path)?;
    let mut credential_health = CredentialHealth::default();
    if snapshot
        .document
        .as_ref()
        .and_then(legacy_api_key)
        .is_some()
    {
        match SettingsTransaction::try_acquire(path)
            .map_err(|error| SettingsError::Transaction(error.to_string()))?
        {
            Some(_transaction) => {
                snapshot = read_runtime_snapshot(path)?;
                if let Some(value) = snapshot.document.as_mut() {
                    let expected_revision = snapshot.revision.clone();
                    let guarded_writer = |path: &Path, document: &Value| {
                        if settings_revision(path)? != expected_revision {
                            return Err(SettingsError::Transaction(
                                "settings changed while credential migration was running".into(),
                            ));
                        }
                        writer(path, document)
                    };
                    credential_health = migrate_legacy_key(path, value, store, &guarded_writer);
                }
                snapshot = read_runtime_snapshot(path)?;
            }
            None => {
                credential_health = CredentialHealth {
                    enhancement_disabled: true,
                    requires_user_resolution: true,
                    message: Some(
                        "API key migration is waiting for another settings transaction; enhancement is disabled until migration can be retried"
                            .into(),
                    ),
                };
            }
        }
    }
    loaded_from_snapshot(path, snapshot, credential_health)
}

fn migrate_legacy_key(
    path: &Path,
    document: &mut Value,
    store: &dyn CredentialStore,
    writer: &dyn Fn(&Path, &Value) -> Result<(), SettingsError>,
) -> CredentialHealth {
    let Some(legacy_key) = legacy_api_key(document).map(str::to_string) else {
        return CredentialHealth::default();
    };
    if legacy_key.is_empty() {
        remove_legacy_api_key(document);
        return match writer(path, document) {
            Ok(()) => CredentialHealth::default(),
            Err(error) => CredentialHealth {
                enhancement_disabled: false,
                requires_user_resolution: false,
                message: Some(format!(
                    "Removing an obsolete empty API-key field from settings.json failed: {error}"
                )),
            },
        };
    }

    let verified = match store.read() {
        Ok(Some(existing)) if existing == legacy_key => Ok(()),
        Ok(Some(_)) => Err(CredentialError::Conflict),
        Ok(None) => store_verified(store, &legacy_key),
        Err(error) => Err(error),
    };
    if let Err(error) = verified {
        return CredentialHealth {
            enhancement_disabled: true,
            requires_user_resolution: true,
            message: Some(format!(
                "API key migration failed; enhancement is disabled and the original key remains in settings.json: {error}"
            )),
        };
    }

    remove_legacy_api_key(document);
    match writer(path, document) {
        Ok(()) => CredentialHealth::default(),
        Err(error) => CredentialHealth {
            enhancement_disabled: false,
            requires_user_resolution: false,
            message: Some(format!(
                "API key is secured, but removing the plaintext copy from settings.json failed: {error}"
            )),
        },
    }
}

struct RuntimeSnapshot {
    document: Option<Value>,
    parse_error: Option<String>,
    revision: SettingsRevision,
}

fn loaded_from_snapshot(
    path: &Path,
    snapshot: RuntimeSnapshot,
    mut credential_health: CredentialHealth,
) -> Result<LoadedSettings, SettingsError> {
    let (settings, settings_error) = if let Some(error) = snapshot.parse_error {
        if credential_health.message.is_none() {
            credential_health = CredentialHealth {
                enhancement_disabled: true,
                requires_user_resolution: false,
                message: Some(
                    "Enhancement is disabled because the unreadable settings file could contain an unverified legacy API key"
                        .into(),
                ),
            };
        }
        (Settings::default(), Some(error))
    } else {
        match settings_from_document(path, snapshot.document.as_ref()) {
            Ok(settings) => (settings, None),
            Err(error) => (Settings::default(), Some(error.to_string())),
        }
    };
    Ok(LoadedSettings {
        settings,
        credential_health,
        settings_error,
        revision: snapshot.revision,
    })
}

fn read_runtime_snapshot(path: &Path) -> Result<RuntimeSnapshot, SettingsError> {
    match fs::read(path) {
        Ok(bytes) => {
            let revision = SettingsRevision::from_bytes(Some(&bytes));
            match serde_json::from_slice(&bytes) {
                Ok(document) => Ok(RuntimeSnapshot {
                    document: Some(document),
                    parse_error: None,
                    revision,
                }),
                Err(source) => Ok(RuntimeSnapshot {
                    document: None,
                    parse_error: Some(
                        SettingsError::Parse {
                            path: path.to_path_buf(),
                            source,
                        }
                        .to_string(),
                    ),
                    revision,
                }),
            }
        }
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(RuntimeSnapshot {
            document: None,
            parse_error: None,
            revision: SettingsRevision::from_bytes(None),
        }),
        Err(source) => Err(io_error("read", path, source)),
    }
}

fn read_document(path: &Path) -> Result<Option<Value>, SettingsError> {
    match fs::read(path) {
        Ok(bytes) => {
            serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(|source| SettingsError::Parse {
                    path: path.to_path_buf(),
                    source,
                })
        }
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(io_error("read", path, source)),
    }
}

fn settings_revision(path: &Path) -> Result<SettingsRevision, SettingsError> {
    match fs::read(path) {
        Ok(bytes) => Ok(SettingsRevision::from_bytes(Some(&bytes))),
        Err(source) if source.kind() == io::ErrorKind::NotFound => {
            Ok(SettingsRevision::from_bytes(None))
        }
        Err(source) => Err(io_error("read settings revision", path, source)),
    }
}

fn settings_from_document(
    path: &Path,
    document: Option<&Value>,
) -> Result<Settings, SettingsError> {
    let settings = match document {
        Some(document) => {
            serde_json::from_value(document.clone()).map_err(|source| SettingsError::Parse {
                path: path.to_path_buf(),
                source,
            })?
        }
        None => Settings::default(),
    };
    settings.validate_for_load()?;
    Ok(settings)
}

fn legacy_api_key(document: &Value) -> Option<&str> {
    document
        .get("enhance")
        .and_then(Value::as_object)
        .and_then(|enhance| enhance.get("api_key"))
        .and_then(Value::as_str)
}

fn remove_legacy_api_key(document: &mut Value) {
    if let Some(enhance) = document.get_mut("enhance").and_then(Value::as_object_mut) {
        enhance.remove("api_key");
    }
}

fn atomic_write_json(path: &Path, document: &Value) -> Result<(), SettingsError> {
    let mut bytes = serde_json::to_vec_pretty(document)?;
    bytes.push(b'\n');
    atomic_write_bytes_with(path, &bytes, replace_file)
}

fn atomic_write_bytes_with<F>(path: &Path, bytes: &[u8], replace: F) -> Result<(), SettingsError>
where
    F: FnOnce(&Path, &Path) -> io::Result<()>,
{
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|source| io_error("create directory", parent, source))?;
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);
    let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("settings.json");
    let temporary = parent.join(format!(".{name}.{}.{}.tmp", std::process::id(), sequence));

    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|source| io_error("create temporary settings file", &temporary, source))?;
        file.write_all(bytes)
            .map_err(|source| io_error("write temporary settings file", &temporary, source))?;
        file.sync_all()
            .map_err(|source| io_error("flush temporary settings file", &temporary, source))?;
        drop(file);
        replace(&temporary, path).map_err(|source| io_error("replace settings file", path, source))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let success = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if success == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn io_error(operation: &'static str, path: &Path, source: io::Error) -> SettingsError {
    SettingsError::Io {
        operation,
        path: path.to_path_buf(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    #[derive(Default)]
    struct FakeStore {
        secret: Mutex<Option<String>>,
        reads: AtomicU64,
        writes: AtomicU64,
        fail_read: Mutex<Option<String>>,
        fail_read_at: Mutex<Option<u64>>,
        fail_write: Mutex<Option<String>>,
    }

    impl CredentialStore for FakeStore {
        fn read(&self) -> Result<Option<String>, CredentialError> {
            let read_number = self.reads.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(message) = self.fail_read.lock().clone() {
                return Err(CredentialError::Backend {
                    operation: "read",
                    message,
                });
            }
            if self.fail_read_at.lock().as_ref() == Some(&read_number) {
                return Err(CredentialError::Backend {
                    operation: "read",
                    message: "injected read-back failure".into(),
                });
            }
            Ok(self.secret.lock().clone())
        }

        fn write(&self, secret: &str) -> Result<(), CredentialError> {
            self.writes.fetch_add(1, Ordering::Relaxed);
            if let Some(message) = self.fail_write.lock().clone() {
                return Err(CredentialError::Backend {
                    operation: "write",
                    message,
                });
            }
            *self.secret.lock() = Some(secret.to_string());
            Ok(())
        }

        fn delete(&self) -> Result<(), CredentialError> {
            *self.secret.lock() = None;
            Ok(())
        }
    }

    fn legacy_document(key: &str) -> Value {
        let mut value = serde_json::to_value(Settings::default()).unwrap();
        value["enhance"]["api_key"] = Value::String(key.into());
        value
    }

    #[test]
    fn migrates_legacy_enhancement_provider_semantics() {
        let mut off = serde_json::to_value(Settings::default()).unwrap();
        off["enhance"].as_object_mut().unwrap().remove("mode");
        off["enhance"]["provider"] = Value::String("off".into());
        let off: Settings = serde_json::from_value(off).unwrap();
        assert_eq!(off.enhance.mode, EnhanceMode::Never);
        assert_eq!(off.enhance.provider, "github_copilot");

        let mut enabled = serde_json::to_value(Settings::default()).unwrap();
        enabled["enhance"].as_object_mut().unwrap().remove("mode");
        enabled["enhance"]["provider"] = Value::String("openai_compatible".into());
        enabled["enhance"]["model"] = Value::String(String::new());
        let enabled: Settings = serde_json::from_value(enabled).unwrap();
        assert_eq!(enabled.enhance.mode, EnhanceMode::WithShift);
        assert_eq!(enabled.enhance.provider, "openai_compatible");
        assert_eq!(enabled.enhance.model, "claude-haiku-4.5");
    }

    #[test]
    fn enhancement_mode_round_trips() {
        let mut settings = Settings::default();
        settings.enhance.mode = EnhanceMode::Always;

        let reloaded: Settings =
            serde_json::from_value(serde_json::to_value(&settings).unwrap()).unwrap();

        assert_eq!(reloaded.enhance.mode, EnhanceMode::Always);
    }

    #[test]
    fn validates_safe_shortcut_classes() {
        let mut settings = Settings::default();
        assert!(settings.validate_shortcut().is_ok());

        settings.hotkey_modifiers = vec!["ctrl".into()];
        assert!(settings.validate_shortcut().is_err());

        settings.hotkey_modifiers.clear();
        settings.hotkey_trigger = "space".into();
        assert!(settings.validate_shortcut().is_err());

        settings.hotkey_trigger = "f13".into();
        assert!(settings.validate_shortcut().is_ok());
    }

    #[test]
    fn loads_legacy_shortcuts_but_requires_a_safe_replacement_before_saving() {
        let mut single_modifier = Settings::default();
        single_modifier.hotkey_modifiers = vec!["ctrl".into()];

        let mut bare_trigger = Settings::default();
        bare_trigger.hotkey_modifiers.clear();
        bare_trigger.hotkey_trigger = "space".into();

        let mut shift_conflict = Settings::default();
        shift_conflict.hotkey_modifiers = vec!["ctrl".into(), "shift".into()];
        shift_conflict.enhance.mode = EnhanceMode::WithShift;

        let temp = tempfile::tempdir().unwrap();
        for (index, settings) in [single_modifier, bare_trigger, shift_conflict]
            .into_iter()
            .enumerate()
        {
            let path = temp.path().join(format!("settings-{index}.json"));
            fs::write(&path, serde_json::to_vec_pretty(&settings).unwrap()).unwrap();

            let loaded = Settings::load_from(&path).unwrap();

            assert_eq!(loaded.hotkey_modifiers, settings.hotkey_modifiers);
            assert_eq!(loaded.hotkey_trigger, settings.hotkey_trigger);
            assert_eq!(loaded.enhance.mode, settings.enhance.mode);
            assert!(loaded.validate().is_err());
        }
    }

    #[test]
    fn rejects_shift_enhancement_conflicts() {
        let mut settings = Settings::default();
        settings.hotkey_modifiers = vec!["ctrl".into(), "shift".into()];
        settings.enhance.mode = EnhanceMode::WithShift;
        assert!(settings.validate_shortcut().is_err());

        settings.enhance.mode = EnhanceMode::Always;
        assert!(settings.validate_shortcut().is_ok());
    }

    #[test]
    fn validates_openai_compatible_endpoint() {
        let mut settings = Settings::default();
        settings.enhance.mode = EnhanceMode::Always;
        settings.enhance.provider = "openai_compatible".into();
        settings.enhance.base_url = "ftp://example.com".into();
        assert!(settings.validate_enhancement().is_err());

        settings.enhance.base_url = "http://localhost:11434/v1".into();
        assert!(settings.validate_enhancement().is_ok());
    }

    #[test]
    fn rejects_empty_hotkey_and_invalid_limits() {
        let mut settings = Settings::default();
        settings.hotkey_modifiers.clear();
        assert!(settings.validate().is_err());

        settings.hotkey_trigger = "space".into();
        settings.max_record_seconds = f32::NAN;
        assert!(settings.validate().is_err());

        settings.max_record_seconds = 1.0;
        settings.min_record_seconds = 2.0;
        assert!(settings.validate().is_err());
    }

    #[test]
    fn failed_atomic_replace_preserves_previous_file() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        fs::write(&path, b"previous").unwrap();

        let error = atomic_write_bytes_with(&path, b"replacement", |_, _| {
            Err(io::Error::new(io::ErrorKind::PermissionDenied, "injected"))
        })
        .unwrap_err();

        assert!(error.to_string().contains("replace settings file"));
        assert_eq!(fs::read(&path).unwrap(), b"previous");
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 1);
    }

    #[test]
    fn legacy_key_is_verified_then_removed_atomically() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        atomic_write_json(&path, &legacy_document("legacy-secret")).unwrap();
        let store = FakeStore::default();

        let loaded = load_with_migration(&path, &store).unwrap();

        assert_eq!(store.secret.lock().as_deref(), Some("legacy-secret"));
        assert_eq!(store.writes.load(Ordering::Relaxed), 1);
        assert!(!loaded.credential_health.enhancement_disabled);
        let saved: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert!(legacy_api_key(&saved).is_none());
    }

    #[test]
    fn migration_preserves_the_legacy_secret_exactly() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        atomic_write_json(&path, &legacy_document("  exact-secret  ")).unwrap();
        let store = FakeStore::default();

        load_with_migration(&path, &store).unwrap();

        assert_eq!(store.secret.lock().as_deref(), Some("  exact-secret  "));
    }

    #[test]
    fn invalid_settings_do_not_bypass_legacy_key_migration() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        let mut document = legacy_document("only-copy");
        document["max_record_seconds"] = Value::from(-1.0);
        atomic_write_json(&path, &document).unwrap();
        let store = FakeStore::default();

        let loaded = load_with_migration(&path, &store).unwrap();

        assert!(loaded.settings_error.is_some());
        assert_eq!(store.secret.lock().as_deref(), Some("only-copy"));
        let saved: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert!(legacy_api_key(&saved).is_none());
    }

    #[test]
    fn invalid_settings_keep_legacy_key_when_migration_fails() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        let mut document = legacy_document("only-copy");
        document["max_record_seconds"] = Value::from(-1.0);
        let original = serde_json::to_vec_pretty(&document).unwrap();
        fs::write(&path, &original).unwrap();
        let store = FakeStore::default();
        *store.fail_write.lock() = Some("injected".into());

        let loaded = load_with_migration(&path, &store).unwrap();

        assert!(loaded.settings_error.is_some());
        assert!(loaded.credential_health.enhancement_disabled);
        assert_eq!(fs::read(path).unwrap(), original);
    }

    #[test]
    fn loaded_settings_and_revision_come_from_the_same_post_migration_snapshot() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        atomic_write_json(&path, &legacy_document("legacy-secret")).unwrap();
        let store = FakeStore::default();

        let loaded = load_with_migration_and_writer(&path, &store, &|path, document| {
            let mut replacement = document.clone();
            replacement["overlay"] = Value::Bool(false);
            atomic_write_json(path, &replacement)
        })
        .unwrap();

        assert!(!loaded.settings.overlay);
        assert_eq!(loaded.revision, settings_revision(&path).unwrap());
    }

    #[test]
    fn malformed_settings_preserve_their_exact_revision_and_disable_enhancement() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        fs::write(&path, br#"{"enhance":{"api_key":"only-copy"}"#).unwrap();
        let store = FakeStore::default();

        let loaded = load_with_migration(&path, &store).unwrap();

        assert!(loaded.settings_error.is_some());
        assert!(loaded.credential_health.enhancement_disabled);
        assert!(!loaded.credential_health.requires_user_resolution);
        assert_eq!(loaded.revision, settings_revision(&path).unwrap());
        assert_eq!(store.reads.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn deferred_migration_blocks_saving_without_a_credential_decision() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        atomic_write_json(&path, &legacy_document("only-copy")).unwrap();
        let transaction = SettingsTransaction::acquire(&path).unwrap();
        let store = FakeStore::default();

        let loaded = load_with_migration(&path, &store).unwrap();

        assert!(loaded.credential_health.enhancement_disabled);
        assert!(loaded.credential_health.requires_user_resolution);
        assert_eq!(store.reads.load(Ordering::Relaxed), 0);
        let saved: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(legacy_api_key(&saved), Some("only-copy"));
        drop(transaction);
    }

    #[test]
    fn failed_credential_write_leaves_legacy_json_intact() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        let original = serde_json::to_vec_pretty(&legacy_document("only-copy")).unwrap();
        fs::write(&path, &original).unwrap();
        let store = FakeStore::default();
        *store.fail_write.lock() = Some("injected".into());

        let loaded = load_with_migration(&path, &store).unwrap();

        assert!(loaded.credential_health.enhancement_disabled);
        assert_eq!(fs::read(path).unwrap(), original);
    }

    #[test]
    fn failed_credential_read_back_leaves_legacy_json_intact() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        let original = serde_json::to_vec_pretty(&legacy_document("only-copy")).unwrap();
        fs::write(&path, &original).unwrap();
        let store = FakeStore::default();
        *store.fail_read_at.lock() = Some(2);

        let loaded = load_with_migration(&path, &store).unwrap();

        assert!(loaded.credential_health.enhancement_disabled);
        assert_eq!(store.secret.lock().as_deref(), Some("only-copy"));
        assert_eq!(fs::read(path).unwrap(), original);
    }

    #[test]
    fn failed_cleanup_retries_without_rewriting_the_credential() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        atomic_write_json(&path, &legacy_document("legacy-secret")).unwrap();
        let store = FakeStore::default();

        let failed = load_with_migration_and_writer(&path, &store, &|_, _| {
            Err(SettingsError::Validation("injected cleanup failure".into()))
        })
        .unwrap();
        assert!(!failed.credential_health.enhancement_disabled);
        assert!(failed.credential_health.message.is_some());
        assert_eq!(store.writes.load(Ordering::Relaxed), 1);

        let retried = load_with_migration(&path, &store).unwrap();
        assert!(retried.credential_health.message.is_none());
        assert_eq!(store.writes.load(Ordering::Relaxed), 1);
        let saved: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert!(legacy_api_key(&saved).is_none());
    }

    #[test]
    fn no_legacy_key_does_not_access_the_credential_store() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        Settings::default().save_to(&path).unwrap();
        let store = FakeStore::default();

        load_with_migration(&path, &store).unwrap();

        assert_eq!(store.reads.load(Ordering::Relaxed), 0);
        assert_eq!(store.writes.load(Ordering::Relaxed), 0);
    }
}
