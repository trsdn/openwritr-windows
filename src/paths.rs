use std::path::PathBuf;
pub const APP_NAME: &str = "OpenWritr";

fn base() -> PathBuf {
    if let Some(p) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(p).join(APP_NAME);
    }
    directories::BaseDirs::new()
        .map(|d| d.data_local_dir().join(APP_NAME))
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn log_dir() -> PathBuf { base().join("logs") }
pub fn models_dir() -> PathBuf { base().join("models") }
pub fn settings_path() -> PathBuf { base().join("settings.json") }
#[allow(dead_code)]
pub fn data_dir() -> PathBuf { base() }
