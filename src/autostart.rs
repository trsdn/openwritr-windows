//! "Run at startup" backend — manages the OS autostart entry for OpenWritr.
//!
//! This deliberately mirrors the `CredentialStore` trait shape in
//! `src/credentials.rs`: a small backend trait, a concrete Windows
//! implementation, and an injectable fake used by the unit tests. The rest of
//! the app (settings UI, diagnostics, self-check) only ever sees
//! [`AutostartBackend`].
//!
//! Two mutually exclusive backends exist, picked at runtime by [`backend`]:
//!
//! * **Unpackaged** (installer + portable): a value named `OpenWritr` under
//!   `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` pointing at the
//!   current executable. Task Manager / Settings → Apps → Startup can disable
//!   it; that disabled state is recorded under `...\Explorer\StartupApproved\Run`
//!   and reported as [`AutostartState::DisabledByOs`].
//! * **Packaged** (MSIX / Store): the `Windows.ApplicationModel.StartupTask`
//!   WinRT API against the `OpenWritrStartup` task declared in the manifest.
//!
//! A login start passes no arguments, and an argless launch already lands in
//! tray mode (see `src/main.rs`), so autostart is silent by construction — no
//! extra mode flag is needed.

use std::path::PathBuf;
use thiserror::Error;
use tracing::{info, warn};

/// Registry value / manifest task id. Both backends use the same name so the
/// installer, the in-app toggle, and the MSIX manifest all agree.
pub const AUTOSTART_ENTRY_NAME: &str = "OpenWritr";

#[derive(Debug, Error)]
pub enum AutostartError {
    #[error("resolve current executable failed: {0}")]
    CurrentExe(String),
    #[error("autostart {operation} failed: {message}")]
    Backend {
        operation: &'static str,
        message: String,
    },
    #[error("autostart verification failed after {0}")]
    VerificationFailed(&'static str),
}

impl AutostartError {
    fn backend(operation: &'static str, message: impl Into<String>) -> Self {
        Self::Backend {
            operation,
            message: message.into(),
        }
    }
}

/// The autostart state as reported by the OS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AutostartState {
    Enabled,
    Disabled,
    /// The entry exists but Windows overrides it (user turned it off in Task
    /// Manager, or policy forbids it). The app must not fight this by
    /// re-enabling on launch.
    DisabledByOs {
        reason: String,
    },
}

impl AutostartState {
    pub fn is_enabled(&self) -> bool {
        matches!(self, AutostartState::Enabled)
    }

    /// Short, log/diagnostics-friendly label.
    pub fn label(&self) -> &'static str {
        match self {
            AutostartState::Enabled => "enabled",
            AutostartState::Disabled => "disabled",
            AutostartState::DisabledByOs { .. } => "disabled_by_os",
        }
    }
}

pub trait AutostartBackend {
    /// A stable identifier for the resolved backend, for diagnostics.
    fn kind(&self) -> &'static str;
    fn state(&self) -> Result<AutostartState, AutostartError>;
    fn enable(&self) -> Result<(), AutostartError>;
    fn disable(&self) -> Result<(), AutostartError>;
}

/// Enable then read back, so a UI toggle never claims success on a silent
/// no-op. `DisabledByOs` after an enable is *not* success — Windows overrode us.
pub fn enable_verified(backend: &dyn AutostartBackend) -> Result<(), AutostartError> {
    backend.enable()?;
    match backend.state()? {
        AutostartState::Enabled => Ok(()),
        _ => Err(AutostartError::VerificationFailed("enable")),
    }
}

/// Disable then read back and confirm it is no longer enabled.
pub fn disable_verified(backend: &dyn AutostartBackend) -> Result<(), AutostartError> {
    backend.disable()?;
    match backend.state()? {
        AutostartState::Enabled => Err(AutostartError::VerificationFailed("disable")),
        _ => Ok(()),
    }
}

/// Resolve the backend for the current process: packaged builds use the WinRT
/// StartupTask, everything else uses the registry Run value.
pub fn backend() -> Box<dyn AutostartBackend> {
    #[cfg(windows)]
    {
        if packaged::is_packaged() {
            Box::new(packaged::StartupTaskBackend::new())
        } else {
            Box::new(RegistryRunBackend::for_current_exe())
        }
    }
    #[cfg(not(windows))]
    {
        Box::new(NoopBackend)
    }
}

/// Non-Windows placeholder so the crate still builds off-Windows (tests/CI).
#[cfg(not(windows))]
struct NoopBackend;

#[cfg(not(windows))]
impl AutostartBackend for NoopBackend {
    fn kind(&self) -> &'static str {
        "noop"
    }
    fn state(&self) -> Result<AutostartState, AutostartError> {
        Ok(AutostartState::Disabled)
    }
    fn enable(&self) -> Result<(), AutostartError> {
        Ok(())
    }
    fn disable(&self) -> Result<(), AutostartError> {
        Ok(())
    }
}

fn quoted_exe(path: &std::path::Path) -> String {
    format!("\"{}\"", path.display())
}

/// The path Windows Explorer would launch is the exe wrapped in quotes, no
/// args. Used for stale-path comparison and for writing the value.
fn current_exe_command() -> Result<String, AutostartError> {
    let exe =
        std::env::current_exe().map_err(|error| AutostartError::CurrentExe(error.to_string()))?;
    Ok(quoted_exe(&exe))
}

// ---------------------------------------------------------------------------
// Unpackaged backend: HKCU Run value.
// ---------------------------------------------------------------------------

/// Abstraction over the two registry locations we touch, so the disabled-blob
/// parsing, stale-path repair, and legacy migration logic can be unit-tested
/// against an in-memory fake instead of the live registry.
pub trait RunRegistry {
    /// The `Run` value data, or `None` if the value is absent.
    fn run_value(&self) -> Result<Option<String>, AutostartError>;
    fn set_run_value(&self, data: &str) -> Result<(), AutostartError>;
    fn delete_run_value(&self) -> Result<(), AutostartError>;
    /// The raw `StartupApproved\Run` blob for our entry, if present.
    fn startup_approved(&self) -> Result<Option<Vec<u8>>, AutostartError>;
}

/// Task Manager records a disabled startup entry as a 12-byte blob whose first
/// byte is 2 or 3. Enabled entries start with a low even value (commonly 2 for
/// the header but with a distinct layout); empirically the disabled flag lives
/// in the first byte being 3 (disabled) vs 2 (enabled). We treat first byte
/// odd / >= 3 as disabled to be robust.
pub fn startup_approved_is_disabled(blob: &[u8]) -> bool {
    match blob.first() {
        Some(&first) => first != 2,
        None => false,
    }
}

pub struct RegistryRunBackend<R: RunRegistry> {
    registry: R,
    desired_command: Result<String, AutostartError>,
    legacy_shortcut: Option<PathBuf>,
}

impl RegistryRunBackend<WindowsRunRegistry> {
    pub fn for_current_exe() -> Self {
        Self {
            registry: WindowsRunRegistry,
            desired_command: current_exe_command(),
            legacy_shortcut: legacy_startup_shortcut(),
        }
    }
}

impl<R: RunRegistry> RegistryRunBackend<R> {
    #[cfg(test)]
    fn with_parts(registry: R, command: String, legacy_shortcut: Option<PathBuf>) -> Self {
        Self {
            registry,
            desired_command: Ok(command),
            legacy_shortcut,
        }
    }

    fn desired(&self) -> Result<&str, AutostartError> {
        match &self.desired_command {
            Ok(command) => Ok(command.as_str()),
            Err(error) => Err(AutostartError::backend("resolve exe", error.to_string())),
        }
    }

    /// Migrate a legacy `{userstartup}\OpenWritr.lnk` to the registry once: if
    /// the shortcut exists, treat autostart as enabled, write the Run value,
    /// delete the shortcut, and log it. Returns true if a migration happened.
    fn migrate_legacy_shortcut(&self) -> Result<bool, AutostartError> {
        let Some(shortcut) = self.legacy_shortcut.as_ref() else {
            return Ok(false);
        };
        if !shortcut.exists() {
            return Ok(false);
        }
        let desired = self.desired()?.to_string();
        self.registry.set_run_value(&desired)?;
        match std::fs::remove_file(shortcut) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(AutostartError::backend(
                    "remove legacy shortcut",
                    error.to_string(),
                ));
            }
        }
        info!(
            shortcut = %shortcut.display(),
            "migrated legacy startup-folder shortcut to the registry Run value"
        );
        Ok(true)
    }
}

impl<R: RunRegistry> AutostartBackend for RegistryRunBackend<R> {
    fn kind(&self) -> &'static str {
        "registry_run"
    }

    fn state(&self) -> Result<AutostartState, AutostartError> {
        // Legacy migration happens first so a fresh install that used the old
        // shortcut mechanism reports enabled and converges on the registry.
        self.migrate_legacy_shortcut()?;

        let Some(current) = self.registry.run_value()? else {
            return Ok(AutostartState::Disabled);
        };

        // Stale-path repair: portable users move the folder, so the stored
        // path drifts from the current exe. Rewrite it on read so the entry
        // keeps working. Only attempt when we could resolve our own exe.
        if let Ok(desired) = self.desired() {
            if current != desired {
                if let Err(error) = self.registry.set_run_value(desired) {
                    warn!(%error, "failed to repair stale autostart path");
                } else {
                    info!(
                        old = %current,
                        new = %desired,
                        "repaired stale autostart Run value path"
                    );
                }
            }
        }

        if let Some(blob) = self.registry.startup_approved()? {
            if startup_approved_is_disabled(&blob) {
                return Ok(AutostartState::DisabledByOs {
                    reason: "Turned off in Windows (Task Manager or Settings → Apps → Startup)."
                        .to_string(),
                });
            }
        }

        Ok(AutostartState::Enabled)
    }

    fn enable(&self) -> Result<(), AutostartError> {
        let desired = self.desired()?.to_string();
        self.registry.set_run_value(&desired)
    }

    fn disable(&self) -> Result<(), AutostartError> {
        self.registry.delete_run_value()
    }
}

/// `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\OpenWritr.lnk`.
fn legacy_startup_shortcut() -> Option<PathBuf> {
    let appdata = std::env::var_os("APPDATA")?;
    Some(
        PathBuf::from(appdata)
            .join("Microsoft")
            .join("Windows")
            .join("Start Menu")
            .join("Programs")
            .join("Startup")
            .join("OpenWritr.lnk"),
    )
}

// ---------------------------------------------------------------------------
// Live Windows registry implementation of RunRegistry.
// ---------------------------------------------------------------------------

#[cfg(windows)]
pub struct WindowsRunRegistry;

#[cfg(not(windows))]
pub struct WindowsRunRegistry;

#[cfg(windows)]
mod win_registry {
    use super::{AutostartError, RunRegistry, WindowsRunRegistry, AUTOSTART_ENTRY_NAME};
    use windows::core::PCWSTR;
    use windows::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_SUCCESS, WIN32_ERROR};
    use windows::Win32::System::Registry::{
        RegCloseKey, RegDeleteValueW, RegOpenKeyExW, RegQueryValueExW, RegSetValueExW, HKEY,
        HKEY_CURRENT_USER, KEY_QUERY_VALUE, KEY_SET_VALUE, REG_BINARY, REG_SZ, REG_VALUE_TYPE,
    };

    const RUN_SUBKEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";
    const STARTUP_APPROVED_SUBKEY: &str =
        r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run";

    fn wide(value: &str) -> Vec<u16> {
        value.encode_utf16().chain(std::iter::once(0)).collect()
    }

    struct OpenKey(HKEY);

    impl Drop for OpenKey {
        fn drop(&mut self) {
            unsafe {
                let _ = RegCloseKey(self.0);
            }
        }
    }

    fn open(
        subkey: &str,
        access: windows::Win32::System::Registry::REG_SAM_FLAGS,
    ) -> Result<Option<OpenKey>, AutostartError> {
        let subkey_wide = wide(subkey);
        let mut handle = HKEY::default();
        let status = unsafe {
            RegOpenKeyExW(
                HKEY_CURRENT_USER,
                PCWSTR(subkey_wide.as_ptr()),
                None,
                access,
                &mut handle,
            )
        };
        if status == ERROR_SUCCESS {
            Ok(Some(OpenKey(handle)))
        } else if status == ERROR_FILE_NOT_FOUND {
            Ok(None)
        } else {
            Err(AutostartError::backend(
                "open registry key",
                format!("{subkey}: {}", status.0),
            ))
        }
    }

    fn open_run_for_write() -> Result<OpenKey, AutostartError> {
        // The Run key always exists on a normal profile, but be defensive.
        match open(RUN_SUBKEY, KEY_SET_VALUE | KEY_QUERY_VALUE)? {
            Some(key) => Ok(key),
            None => Err(AutostartError::backend(
                "open registry key",
                format!("{RUN_SUBKEY}: missing"),
            )),
        }
    }

    fn ok(status: WIN32_ERROR, operation: &'static str) -> Result<(), AutostartError> {
        if status == ERROR_SUCCESS {
            Ok(())
        } else {
            Err(AutostartError::backend(operation, status.0.to_string()))
        }
    }

    impl RunRegistry for WindowsRunRegistry {
        fn run_value(&self) -> Result<Option<String>, AutostartError> {
            let Some(key) = open(RUN_SUBKEY, KEY_QUERY_VALUE)? else {
                return Ok(None);
            };
            let name = wide(AUTOSTART_ENTRY_NAME);
            let mut kind = REG_VALUE_TYPE::default();
            let mut size = 0u32;
            let status = unsafe {
                RegQueryValueExW(
                    key.0,
                    PCWSTR(name.as_ptr()),
                    None,
                    Some(&mut kind),
                    None,
                    Some(&mut size),
                )
            };
            if status == ERROR_FILE_NOT_FOUND {
                return Ok(None);
            }
            ok(status, "query Run value size")?;
            if size == 0 {
                return Ok(Some(String::new()));
            }
            let mut buffer = vec![0u8; size as usize];
            let mut size_out = size;
            let status = unsafe {
                RegQueryValueExW(
                    key.0,
                    PCWSTR(name.as_ptr()),
                    None,
                    Some(&mut kind),
                    Some(buffer.as_mut_ptr()),
                    Some(&mut size_out),
                )
            };
            ok(status, "query Run value")?;
            buffer.truncate(size_out as usize);
            Ok(Some(decode_wide_sz(&buffer)))
        }

        fn set_run_value(&self, data: &str) -> Result<(), AutostartError> {
            let key = open_run_for_write()?;
            let name = wide(AUTOSTART_ENTRY_NAME);
            let value = wide(data);
            let bytes =
                unsafe { std::slice::from_raw_parts(value.as_ptr().cast::<u8>(), value.len() * 2) };
            let status =
                unsafe { RegSetValueExW(key.0, PCWSTR(name.as_ptr()), None, REG_SZ, Some(bytes)) };
            ok(status, "set Run value")
        }

        fn delete_run_value(&self) -> Result<(), AutostartError> {
            let Some(key) = open(RUN_SUBKEY, KEY_SET_VALUE)? else {
                return Ok(());
            };
            let name = wide(AUTOSTART_ENTRY_NAME);
            let status = unsafe { RegDeleteValueW(key.0, PCWSTR(name.as_ptr())) };
            if status == ERROR_FILE_NOT_FOUND {
                return Ok(());
            }
            ok(status, "delete Run value")
        }

        fn startup_approved(&self) -> Result<Option<Vec<u8>>, AutostartError> {
            let Some(key) = open(STARTUP_APPROVED_SUBKEY, KEY_QUERY_VALUE)? else {
                return Ok(None);
            };
            let name = wide(AUTOSTART_ENTRY_NAME);
            let mut kind = REG_VALUE_TYPE::default();
            let mut size = 0u32;
            let status = unsafe {
                RegQueryValueExW(
                    key.0,
                    PCWSTR(name.as_ptr()),
                    None,
                    Some(&mut kind),
                    None,
                    Some(&mut size),
                )
            };
            if status == ERROR_FILE_NOT_FOUND {
                return Ok(None);
            }
            ok(status, "query StartupApproved size")?;
            if kind != REG_BINARY || size == 0 {
                return Ok(None);
            }
            let mut buffer = vec![0u8; size as usize];
            let mut size_out = size;
            let status = unsafe {
                RegQueryValueExW(
                    key.0,
                    PCWSTR(name.as_ptr()),
                    None,
                    Some(&mut kind),
                    Some(buffer.as_mut_ptr()),
                    Some(&mut size_out),
                )
            };
            ok(status, "query StartupApproved")?;
            buffer.truncate(size_out as usize);
            Ok(Some(buffer))
        }
    }

    fn decode_wide_sz(bytes: &[u8]) -> String {
        let units: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .take_while(|unit| *unit != 0)
            .collect();
        String::from_utf16_lossy(&units)
    }
}

// ---------------------------------------------------------------------------
// Packaged (MSIX) backend: WinRT StartupTask.
// ---------------------------------------------------------------------------

#[cfg(windows)]
mod packaged {
    use super::{AutostartBackend, AutostartError, AutostartState, AUTOSTART_ENTRY_NAME};
    use windows::core::HSTRING;
    use windows::ApplicationModel::{StartupTask, StartupTaskState};
    use windows::Win32::Foundation::ERROR_INSUFFICIENT_BUFFER;
    use windows::Win32::Storage::Packaging::Appx::GetCurrentPackageFullName;
    use windows_future::{AsyncStatus, IAsyncOperation};

    /// True when running from an MSIX package (has a package identity).
    pub fn is_packaged() -> bool {
        let mut length: u32 = 0;
        let status = unsafe { GetCurrentPackageFullName(&mut length, None) };
        // With a null buffer, a packaged process returns ERROR_INSUFFICIENT_BUFFER;
        // an unpackaged one returns APPMODEL_ERROR_NO_PACKAGE.
        status == ERROR_INSUFFICIENT_BUFFER
    }

    /// Block on a WinRT async operation without pulling in an async runtime.
    /// StartupTask operations complete effectively immediately, so a short
    /// bounded spin is sufficient and avoids the `windows-future` blocking
    /// helper (not exposed by this crate version).
    fn block_on<T: windows::core::RuntimeType>(
        operation: IAsyncOperation<T>,
        context: &'static str,
    ) -> Result<T, AutostartError> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let status = operation
                .Status()
                .map_err(|error| AutostartError::backend(context, error.message()))?;
            match status {
                AsyncStatus::Completed => {
                    return operation
                        .GetResults()
                        .map_err(|error| AutostartError::backend(context, error.message()));
                }
                AsyncStatus::Started => {
                    if std::time::Instant::now() >= deadline {
                        return Err(AutostartError::backend(context, "timed out"));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                AsyncStatus::Canceled => {
                    return Err(AutostartError::backend(context, "operation canceled"));
                }
                _ => {
                    // Error status: surface the underlying failure via GetResults.
                    return operation
                        .GetResults()
                        .map_err(|error| AutostartError::backend(context, error.message()));
                }
            }
        }
    }

    pub struct StartupTaskBackend;

    impl StartupTaskBackend {
        pub fn new() -> Self {
            Self
        }

        fn task(&self) -> Result<StartupTask, AutostartError> {
            let operation =
                StartupTask::GetAsync(&HSTRING::from(AUTOSTART_ENTRY_NAME)).map_err(|error| {
                    AutostartError::backend("StartupTask.GetAsync", error.message())
                })?;
            block_on(operation, "StartupTask.GetAsync")
        }
    }

    impl AutostartBackend for StartupTaskBackend {
        fn kind(&self) -> &'static str {
            "startup_task"
        }

        fn state(&self) -> Result<AutostartState, AutostartError> {
            let task = self.task()?;
            let state = task
                .State()
                .map_err(|error| AutostartError::backend("StartupTask.State", error.message()))?;
            Ok(map_state(state))
        }

        fn enable(&self) -> Result<(), AutostartError> {
            let task = self.task()?;
            // RequestEnableAsync may still return a DisabledBy* state if the OS
            // or user overrides us; the verified wrapper re-reads and surfaces it.
            let operation = task.RequestEnableAsync().map_err(|error| {
                AutostartError::backend("StartupTask.RequestEnableAsync", error.message())
            })?;
            block_on(operation, "StartupTask.RequestEnableAsync")?;
            Ok(())
        }

        fn disable(&self) -> Result<(), AutostartError> {
            let task = self.task()?;
            task.Disable()
                .map_err(|error| AutostartError::backend("StartupTask.Disable", error.message()))
        }
    }

    fn map_state(state: StartupTaskState) -> AutostartState {
        if state == StartupTaskState::Enabled || state == StartupTaskState::EnabledByPolicy {
            AutostartState::Enabled
        } else if state == StartupTaskState::DisabledByUser {
            AutostartState::DisabledByOs {
                reason: "Turned off in Windows (Task Manager or Settings → Apps → Startup)."
                    .to_string(),
            }
        } else if state == StartupTaskState::DisabledByPolicy {
            AutostartState::DisabledByOs {
                reason: "Blocked by Windows policy on this device.".to_string(),
            }
        } else {
            // StartupTaskState::Disabled and any future variant.
            AutostartState::Disabled
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    #[derive(Default)]
    struct FakeRegistry {
        run: Mutex<Option<String>>,
        approved: Mutex<Option<Vec<u8>>>,
    }

    impl RunRegistry for FakeRegistry {
        fn run_value(&self) -> Result<Option<String>, AutostartError> {
            Ok(self.run.lock().clone())
        }
        fn set_run_value(&self, data: &str) -> Result<(), AutostartError> {
            *self.run.lock() = Some(data.to_string());
            Ok(())
        }
        fn delete_run_value(&self) -> Result<(), AutostartError> {
            *self.run.lock() = None;
            Ok(())
        }
        fn startup_approved(&self) -> Result<Option<Vec<u8>>, AutostartError> {
            Ok(self.approved.lock().clone())
        }
    }

    fn backend_with(registry: FakeRegistry, command: &str) -> RegistryRunBackend<FakeRegistry> {
        RegistryRunBackend::with_parts(registry, command.to_string(), None)
    }

    #[test]
    fn enable_disable_read_round_trip() {
        let backend = backend_with(FakeRegistry::default(), "\"C:\\app\\openwritr.exe\"");
        assert_eq!(backend.state().unwrap(), AutostartState::Disabled);

        enable_verified(&backend).unwrap();
        assert_eq!(backend.state().unwrap(), AutostartState::Enabled);
        assert_eq!(
            backend.registry.run.lock().as_deref(),
            Some("\"C:\\app\\openwritr.exe\"")
        );

        disable_verified(&backend).unwrap();
        assert_eq!(backend.state().unwrap(), AutostartState::Disabled);
        assert!(backend.registry.run.lock().is_none());
    }

    #[test]
    fn stale_path_is_repaired_on_read() {
        let registry = FakeRegistry::default();
        *registry.run.lock() = Some("\"C:\\old\\location\\openwritr.exe\"".to_string());
        let backend = backend_with(registry, "\"C:\\new\\location\\openwritr.exe\"");

        // Reading reports enabled and rewrites the drifted path in place.
        assert_eq!(backend.state().unwrap(), AutostartState::Enabled);
        assert_eq!(
            backend.registry.run.lock().as_deref(),
            Some("\"C:\\new\\location\\openwritr.exe\"")
        );
    }

    #[test]
    fn startup_approved_disabled_blob_is_reported() {
        let registry = FakeRegistry::default();
        *registry.run.lock() = Some("\"C:\\app\\openwritr.exe\"".to_string());
        // Task Manager "disabled" blob: first byte 3.
        *registry.approved.lock() = Some(vec![3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let backend = backend_with(registry, "\"C:\\app\\openwritr.exe\"");

        match backend.state().unwrap() {
            AutostartState::DisabledByOs { reason } => assert!(!reason.is_empty()),
            other => panic!("expected DisabledByOs, got {other:?}"),
        }
    }

    #[test]
    fn startup_approved_enabled_blob_reports_enabled() {
        let registry = FakeRegistry::default();
        *registry.run.lock() = Some("\"C:\\app\\openwritr.exe\"".to_string());
        // Enabled blob: first byte 2.
        *registry.approved.lock() = Some(vec![2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let backend = backend_with(registry, "\"C:\\app\\openwritr.exe\"");

        assert_eq!(backend.state().unwrap(), AutostartState::Enabled);
    }

    #[test]
    fn disabled_blob_first_byte_parsing() {
        assert!(!startup_approved_is_disabled(&[2, 0, 0, 0]));
        assert!(startup_approved_is_disabled(&[3, 0, 0, 0]));
        assert!(startup_approved_is_disabled(&[6, 0, 0, 0]));
        assert!(!startup_approved_is_disabled(&[]));
    }

    #[test]
    fn legacy_shortcut_migration_writes_registry_and_deletes_shortcut() {
        let temp = tempfile::tempdir().unwrap();
        let shortcut = temp.path().join("OpenWritr.lnk");
        std::fs::write(&shortcut, b"legacy").unwrap();

        let backend = RegistryRunBackend::with_parts(
            FakeRegistry::default(),
            "\"C:\\app\\openwritr.exe\"".to_string(),
            Some(shortcut.clone()),
        );

        // First read migrates: enabled, value written, shortcut gone.
        assert_eq!(backend.state().unwrap(), AutostartState::Enabled);
        assert_eq!(
            backend.registry.run.lock().as_deref(),
            Some("\"C:\\app\\openwritr.exe\"")
        );
        assert!(!shortcut.exists());
    }

    #[test]
    fn no_legacy_shortcut_leaves_state_disabled() {
        let temp = tempfile::tempdir().unwrap();
        let shortcut = temp.path().join("OpenWritr.lnk");
        // Do not create the file.
        let backend = RegistryRunBackend::with_parts(
            FakeRegistry::default(),
            "\"C:\\app\\openwritr.exe\"".to_string(),
            Some(shortcut),
        );
        assert_eq!(backend.state().unwrap(), AutostartState::Disabled);
        assert!(backend.registry.run.lock().is_none());
    }
}
