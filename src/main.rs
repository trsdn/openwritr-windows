//! OpenWritr — push-to-talk voice-to-text for Windows on ARM (native build).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod asr;
mod audio;
mod download;
mod enhance;
mod hotkey;
mod paths;
mod settings;
mod settings_ui;
mod sounds;
mod tray;

use anyhow::Result;
use tracing::info;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    // If invoked with `--settings`, render the egui settings dialog and exit.
    if std::env::args().any(|a| a == "--settings") {
        init_tracing();
        info!("opening settings UI subprocess");
        return settings_ui::run_dialog();
    }

    init_tracing();
    info!("OpenWritr native v{} starting", env!("CARGO_PKG_VERSION"));
    app::run()
}

fn init_tracing() {
    let log_dir = paths::log_dir();
    std::fs::create_dir_all(&log_dir).ok();
    let log_path = log_dir.join("openwritr.log");

    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let file_writer = tracing_subscriber::fmt::layer()
        .with_writer(move || {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)
                .unwrap_or_else(|_| std::fs::File::create("openwritr-fallback.log").unwrap())
        })
        .with_ansi(false);

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "openwritr=info".into()))
        .with(file_writer)
        .with(tracing_subscriber::fmt::layer().with_ansi(false))
        .init();
}
