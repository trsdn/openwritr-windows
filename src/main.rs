//! OpenWritr — push-to-talk voice-to-text for Windows on ARM (native build).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod asr;
mod audio;
mod credentials;
mod diagnostics;
mod enhance;
mod hotkey;
mod key_hook;
mod model_manager;
mod overlay;
mod paste;
mod paths;
mod self_check;
mod settings;
mod settings_ui;
mod single_instance;
mod sounds;
mod tray;
mod worker;

use anyhow::{Context, Result};
use std::ffi::OsStr;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if is_self_check_mode(arguments.iter()) {
        return self_check::run();
    }
    let settings_mode = is_settings_mode(arguments.iter());

    // If invoked with `--settings`, render the egui settings dialog and exit.
    if settings_mode {
        let settings_instance = single_instance::SingleInstance::acquire_settings()?;
        let _log_guard = init_tracing()?;
        info!(
            version = env!("CARGO_PKG_VERSION"),
            pid = std::process::id(),
            architecture = std::env::consts::ARCH,
            mode = "settings",
            "process starting"
        );
        let Some(_settings_instance) = settings_instance else {
            info!("settings window already running; exiting");
            return Ok(());
        };
        let result = settings_ui::run_dialog();
        info!("settings process stopping");
        return result;
    }

    let tray_instance = single_instance::SingleInstance::acquire_tray()?;
    let _log_guard = init_tracing()?;
    let Some(_instance) = tray_instance else {
        info!("tray instance already running; exiting");
        return Ok(());
    };

    info!(
        version = env!("CARGO_PKG_VERSION"),
        pid = std::process::id(),
        architecture = std::env::consts::ARCH,
        mode = "tray",
        "process starting"
    );
    // Install the global low-level keyboard hook before any subsystem starts
    // polling for keys. It tracks physical key state across focus changes,
    // which GetAsyncKeyState cannot.
    key_hook::install_once();
    let result = app::run();
    if let Err(ref failure) = result {
        error!(error = %failure, "tray process stopping after error");
    } else {
        info!("tray process stopping");
    }
    result
}

fn is_settings_mode<T: AsRef<OsStr>>(args: impl IntoIterator<Item = T>) -> bool {
    args.into_iter()
        .any(|arg| arg.as_ref().eq_ignore_ascii_case(OsStr::new("--settings")))
}

fn is_self_check_mode<T: AsRef<OsStr>>(args: impl IntoIterator<Item = T>) -> bool {
    args.into_iter().any(|arg| {
        arg.as_ref()
            .eq_ignore_ascii_case(OsStr::new("--self-check"))
    })
}

fn init_tracing() -> Result<tracing_appender::non_blocking::WorkerGuard> {
    let log_dir = paths::log_dir();
    std::fs::create_dir_all(&log_dir)
        .with_context(|| format!("create log directory {}", log_dir.display()))?;
    diagnostics::prune_logs()?;

    let file_appender = tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_prefix("openwritr")
        .filename_suffix("log")
        .max_log_files(7)
        .build(&log_dir)
        .with_context(|| format!("initialize log appender in {}", log_dir.display()))?;
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(non_blocking)
        .with_ansi(false)
        .with_thread_ids(true)
        .with_thread_names(true);
    let console_layer =
        cfg!(debug_assertions).then(|| tracing_subscriber::fmt::layer().with_ansi(false));

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "openwritr=info".into()))
        .with(file_layer)
        .with(console_layer)
        .try_init()
        .context("initialize tracing subscriber")?;

    Ok(guard)
}

#[cfg(test)]
mod tests {
    use super::{is_self_check_mode, is_settings_mode};
    use std::ffi::OsString;

    #[test]
    fn settings_mode_is_selected_before_tray_startup() {
        assert!(is_settings_mode([OsString::from("--settings")]));
        assert!(is_settings_mode([OsString::from("--SETTINGS")]));
        assert!(!is_settings_mode([OsString::from("--unknown")]));
    }

    #[test]
    fn self_check_mode_is_selected_before_tray_startup() {
        assert!(is_self_check_mode([OsString::from("--self-check")]));
        assert!(is_self_check_mode([OsString::from("--SELF-CHECK")]));
        assert!(!is_self_check_mode([OsString::from("--settings")]));
    }
}
