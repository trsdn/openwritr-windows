// OpenWritr — Windows ARM tray app entry point.
//
// State machine:
//   Idle  --hotkey down-->  Recording  --hotkey up-->  Transcribing
//   Transcribing  --done-->  (Enhancing)?  -->  Pasting  -->  Idle
//
// All long-running work is offloaded onto Tokio. The audio thread is owned
// by cpal and pushes samples into a SPSC ring buffer drained by the ASR task.

#![cfg_attr(all(not(debug_assertions), windows), windows_subsystem = "windows")]

mod asr;
mod audio;
mod enhance;
mod hotkey;
mod paste;
mod state;

use std::sync::Arc;
use tauri::Manager;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tauri::command]
fn get_active_ep(state: tauri::State<'_, Arc<state::App>>) -> String {
    state.active_ep.read().clone()
}

#[tauri::command]
fn open_settings(app: tauri::AppHandle) -> Result<(), String> {
    if let Some(w) = app.get_webview_window("settings") {
        let _ = w.show();
        let _ = w.set_focus();
    }
    Ok(())
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "openwritr=info".into()))
        .init();

    let app_state = Arc::new(state::App::new());

    tauri::Builder::default()
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_notification::init())
        .manage(app_state.clone())
        .setup(move |app| {
            info!("OpenWritr starting");

            // Spawn the audio + hotkey + asr supervisor.
            let handle = app.handle().clone();
            let s = app_state.clone();
            tauri::async_runtime::spawn(async move {
                if let Err(e) = supervisor::run(handle, s).await {
                    tracing::error!(error = %e, "supervisor exited");
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_active_ep, open_settings])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

mod supervisor {
    use crate::{asr, audio, hotkey, paste, state};
    use anyhow::Result;
    use std::sync::Arc;
    use tauri::AppHandle;
    use tracing::{info, warn};

    pub async fn run(app: AppHandle, st: Arc<state::App>) -> Result<()> {
        // 1. Bring up the ASR engine (model load + EP probe).
        let engine = match asr::Engine::load().await {
            Ok(e) => {
                let ep = e.active_ep().to_string();
                info!(ep = %ep, "ASR engine ready");
                *st.active_ep.write() = ep;
                e
            }
            Err(e) => {
                warn!(error = %e, "ASR engine failed to load — running degraded");
                *st.active_ep.write() = format!("unavailable: {e}");
                return Ok(());
            }
        };
        let engine = Arc::new(engine);

        // 2. Set up audio capture (16 kHz mono, WASAPI shared).
        let audio_handle = audio::Capture::spawn()?;

        // 3. Wire global hotkey -> FSM events.
        hotkey::run(app, st, audio_handle, engine, paste::Paster::new()).await
    }
}
