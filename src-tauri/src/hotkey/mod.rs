// Global hotkey + finite state machine.
//
// Default binding: hold Ctrl+Space to record; hold Shift+Ctrl+Space for enhanced.
// We listen for press/release transitions and gate the audio recorder accordingly.

use crate::{asr, audio::CaptureHandle, enhance, paste::Paster, state};
use anyhow::Result;
use global_hotkey::{hotkey::{Code, HotKey, Modifiers}, GlobalHotKeyEvent, GlobalHotKeyManager};
use std::sync::{atomic::Ordering, Arc};
use tauri::AppHandle;
use tracing::{info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Normal,
    Enhanced,
}

pub async fn run(
    _app: AppHandle,
    _st: Arc<state::App>,
    mut capture: CaptureHandle,
    engine: Arc<asr::Engine>,
    paster: Paster,
) -> Result<()> {
    let manager = GlobalHotKeyManager::new()?;
    let hk_normal = HotKey::new(Some(Modifiers::CONTROL), Code::Space);
    let hk_enhanced = HotKey::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Space);
    manager.register(hk_normal)?;
    manager.register(hk_enhanced)?;
    info!("hotkeys registered: Ctrl+Space, Ctrl+Shift+Space");

    let rx = GlobalHotKeyEvent::receiver();
    let mut mode: Option<Mode> = None;

    loop {
        // global-hotkey is sync; poll on a small interval. Cheap on Windows.
        if let Ok(ev) = rx.try_recv() {
            let chosen = if ev.id == hk_enhanced.id() {
                Mode::Enhanced
            } else {
                Mode::Normal
            };
            match ev.state {
                global_hotkey::HotKeyState::Pressed => {
                    if mode.is_none() {
                        mode = Some(chosen);
                        capture.recording.store(true, Ordering::Relaxed);
                        info!(?chosen, "recording started");
                    }
                }
                global_hotkey::HotKeyState::Released => {
                    if let Some(m) = mode.take() {
                        capture.recording.store(false, Ordering::Relaxed);
                        let samples = (capture.take_samples)();
                        info!(samples = samples.len(), "recording stopped");
                        let engine = engine.clone();
                        let paster = paster.clone();
                        let sr = capture.sample_rate;
                        tauri::async_runtime::spawn(async move {
                            match engine.transcribe(&samples, sr).await {
                                Ok(text) => {
                                    let final_text = if m == Mode::Enhanced {
                                        enhance::enhance(&text).await.unwrap_or(text)
                                    } else {
                                        text
                                    };
                                    if let Err(e) = paster.paste(&final_text) {
                                        warn!(error = %e, "paste failed");
                                    }
                                }
                                Err(e) => warn!(error = %e, "transcription failed"),
                            }
                        });
                    }
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(8)).await;
    }
}
