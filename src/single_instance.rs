use anyhow::{anyhow, Context, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::{
    CloseHandle, GetLastError, SetLastError, ERROR_ALREADY_EXISTS, ERROR_LOCK_VIOLATION,
    ERROR_SHARING_VIOLATION, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Storage::FileSystem::{CreateFileW, FILE_ATTRIBUTE_HIDDEN, OPEN_ALWAYS};
use windows_sys::Win32::System::Threading::CreateMutexW;

const TRAY_MUTEX_PURPOSE: &str = "tray";
const SETTINGS_MUTEX_PURPOSE: &str = "settings";
const SETTINGS_TRANSACTION_LOCK_FILE: &str = ".settings-transaction.lock";

pub struct SingleInstance {
    handle: HANDLE,
}

impl SingleInstance {
    pub fn acquire_tray() -> Result<Option<Self>> {
        Self::acquire_named(&mutex_name(TRAY_MUTEX_PURPOSE))
    }

    pub fn acquire_settings() -> Result<Option<Self>> {
        Self::acquire_named(&mutex_name(SETTINGS_MUTEX_PURPOSE))
    }

    fn acquire_named(name: &str) -> Result<Option<Self>> {
        use std::os::windows::ffi::OsStrExt;

        let wide = std::ffi::OsStr::new(name)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>();
        unsafe {
            SetLastError(0);
        }
        let handle = unsafe { CreateMutexW(std::ptr::null(), 0, wide.as_ptr()) };
        if handle.is_null() {
            let error = unsafe { GetLastError() };
            return Err(anyhow!(
                "create single-instance mutex {name}: {}",
                std::io::Error::from_raw_os_error(error as i32)
            ));
        }
        if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
            unsafe {
                CloseHandle(handle);
            }
            Ok(None)
        } else {
            Ok(Some(Self { handle }))
        }
    }
}

impl Drop for SingleInstance {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.handle);
        }
    }
}

fn mutex_name(purpose: &str) -> String {
    mutex_name_for_identity(purpose, &crate::paths::data_dir().to_string_lossy())
}

fn mutex_name_for_identity(purpose: &str, identity: &str) -> String {
    let digest = Sha256::digest(identity.to_lowercase().as_bytes());
    let suffix = digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!(r"Global\OpenWritr.{purpose}.{suffix}")
}

pub struct SettingsTransaction {
    handle: HANDLE,
}

impl SettingsTransaction {
    pub fn try_acquire(settings_path: &Path) -> Result<Option<Self>> {
        let path = transaction_lock_path(settings_path);
        try_open_exclusive(&path).map(|handle| handle.map(|handle| Self { handle }))
    }

    pub fn acquire(settings_path: &Path) -> Result<Self> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(transaction) = Self::try_acquire(settings_path)? {
                return Ok(transaction);
            }
            if Instant::now() >= deadline {
                return Err(anyhow!(
                    "timed out waiting for another OpenWritr settings transaction"
                ));
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }
}

impl Drop for SettingsTransaction {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.handle);
        }
    }
}

fn transaction_lock_path(settings_path: &Path) -> PathBuf {
    settings_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(SETTINGS_TRANSACTION_LOCK_FILE)
}

fn try_open_exclusive(path: &Path) -> Result<Option<HANDLE>> {
    use std::os::windows::ffi::OsStrExt;

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create lock directory {}", parent.display()))?;
    let wide = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let handle = unsafe {
        CreateFileW(
            wide.as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            0,
            std::ptr::null(),
            OPEN_ALWAYS,
            FILE_ATTRIBUTE_HIDDEN,
            std::ptr::null_mut(),
        )
    };
    if handle != INVALID_HANDLE_VALUE {
        return Ok(Some(handle));
    }
    let error = unsafe { GetLastError() };
    if error == ERROR_SHARING_VIOLATION || error == ERROR_LOCK_VIOLATION {
        Ok(None)
    } else {
        Err(anyhow!(
            "open lock file {}: {}",
            path.display(),
            std::io::Error::from_raw_os_error(error as i32)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::windows::process::CommandExt;
    use std::process::Command;

    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    #[test]
    fn rejects_a_second_live_instance() {
        let name = format!(
            r"Global\OpenWritr.test.{}.{}",
            std::process::id(),
            Instant::now().elapsed().as_nanos()
        );
        let first = SingleInstance::acquire_named(&name).unwrap();
        assert!(first.is_some());
        assert!(SingleInstance::acquire_named(&name).unwrap().is_none());
        drop(first);
        assert!(SingleInstance::acquire_named(&name).unwrap().is_some());
    }

    #[test]
    fn tray_and_settings_locks_are_distinct() {
        let tray = mutex_name_for_identity(TRAY_MUTEX_PURPOSE, r"C:\Users\Example");
        let settings = mutex_name_for_identity(SETTINGS_MUTEX_PURPOSE, r"C:\Users\Example");
        assert!(tray.starts_with(r"Global\OpenWritr."));
        assert_ne!(tray, settings);
        assert_ne!(
            tray,
            mutex_name_for_identity(TRAY_MUTEX_PURPOSE, r"C:\Users\Other")
        );
    }

    #[test]
    fn settings_transaction_excludes_other_threads() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        let first = SettingsTransaction::acquire(&path).unwrap();
        let other_path = path.clone();
        let blocked = std::thread::spawn(move || {
            SettingsTransaction::try_acquire(&other_path)
                .unwrap()
                .is_none()
        })
        .join()
        .unwrap();
        assert!(blocked);
        drop(first);
        assert!(SettingsTransaction::try_acquire(&path).unwrap().is_some());
    }

    #[test]
    fn rejects_a_second_process() {
        const CHILD_MUTEX: &str = "OPENWRITR_TEST_MUTEX_NAME";
        if let Ok(name) = std::env::var(CHILD_MUTEX) {
            assert!(SingleInstance::acquire_named(&name).unwrap().is_none());
            return;
        }

        let name = format!(
            r"Global\OpenWritr.test-process.{}.{}",
            std::process::id(),
            Instant::now().elapsed().as_nanos()
        );
        let first = SingleInstance::acquire_named(&name).unwrap().unwrap();
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "single_instance::tests::rejects_a_second_process",
                "--nocapture",
            ])
            .env(CHILD_MUTEX, &name)
            .creation_flags(CREATE_NO_WINDOW)
            .output()
            .unwrap();
        drop(first);

        assert!(
            output.status.success(),
            "child test failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
