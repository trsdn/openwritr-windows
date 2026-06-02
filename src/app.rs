//! App orchestration: tray + winit event loop + hotkey polling + ASR.

use crate::{asr, audio::Recorder, hotkey, settings::Settings, tray};
use anyhow::Result;
use arboard::Clipboard;
use enigo::{Direction, Enigo, Key, Keyboard, Settings as EnigoSettings};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{info, warn};
use tray_icon::menu::MenuEvent;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::WindowId;

struct State {
    settings: Settings,
    recorder: Recorder,
    hotkey_mgr: hotkey::HotkeyManager,
    tray: tray::Tray,
    pressed: bool,
    record_started: Option<Instant>,
    engine: Option<Arc<dyn asr::Engine>>,
    engine_loading: Arc<AtomicBool>,
}

pub fn run() -> Result<()> {
    let settings = Settings::load();
    let recorder = Recorder::new()?;
    let hotkey_mgr = hotkey::HotkeyManager::register(&settings)?;
    let tray = tray::Tray::new(&settings)?;
    let engine_loading = Arc::new(AtomicBool::new(false));
    let state = State {
        settings,
        recorder,
        hotkey_mgr,
        tray,
        pressed: false,
        record_started: None,
        engine: None,
        engine_loading,
    };

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = AppHandler { state };
    app.start_engine_load();
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct AppHandler { state: State }

impl AppHandler {
    fn start_engine_load(&mut self) {
        if self.state.engine.is_some() || self.state.engine_loading.load(Ordering::Relaxed) {
            return;
        }
        self.state.engine_loading.store(true, Ordering::Relaxed);
        let engine_name = self.state.settings.engine.clone();
        let loading_flag = self.state.engine_loading.clone();
        thread::spawn(move || {
            match asr::load(&engine_name) {
                Ok(e) => {
                    info!("engine loaded: {}", e.label());
                    STAGED_ENGINE.lock().replace(Arc::from(e));
                }
                Err(e) => warn!(error = %e, "engine load failed"),
            }
            loading_flag.store(false, Ordering::Relaxed);
        });
    }
}

// Shared drop-box for the engine handle once the background thread has
// finished loading it. The main thread picks it up on the next pump cycle.
static STAGED_ENGINE: parking_lot::Mutex<Option<Arc<dyn asr::Engine>>> =
    parking_lot::Mutex::new(None);

impl ApplicationHandler for AppHandler {
    fn resumed(&mut self, _el: &ActiveEventLoop) {
        info!("event loop ready");
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, _ev: WindowEvent) {}

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        // Reset tray color when a transcription thread signals done.
        if DONE_FLAG.swap(false, Ordering::Relaxed) {
            self.state.tray.set_color(tray::IconColor::Idle);
        }

        // Pick up the engine if the background loader produced one.
        if self.state.engine.is_none() {
            if let Some(e) = STAGED_ENGINE.lock().take() {
                self.state.engine = Some(e);
            }
        }

        // Tray menu events.
        if let Ok(ev) = MenuEvent::receiver().try_recv() {
            if ev.id == self.state.tray.menu_quit_id {
                info!("quit via tray");
                el.exit();
                return;
            }
            if ev.id == self.state.tray.menu_settings_id {
                info!("settings clicked (UI not implemented in v0.2 skeleton)");
            }
        }

        // Hotkey edge polling.
        if let Some(event) = hotkey::poll_state(&self.state.hotkey_mgr, &mut self.state.pressed) {
            match event {
                hotkey::Event::Press => {
                    if self.state.record_started.is_none() {
                        self.state.recorder.start();
                        self.state.tray.set_color(tray::IconColor::Recording);
                        self.state.record_started = Some(Instant::now());
                        info!("recording start");
                    }
                }
                hotkey::Event::Release => {
                    if let Some(started) = self.state.record_started.take() {
                        let samples = self.state.recorder.stop();
                        self.state.tray.set_color(tray::IconColor::Idle);
                        let dur = started.elapsed();
                        let min = self.state.settings.min_record_seconds;
                        let sr = self.state.recorder.sample_rate;
                        if dur.as_secs_f32() < min {
                            info!(secs = dur.as_secs_f32(), "below min — discarded");
                        } else {
                            self.dispatch_transcribe(samples, sr);
                        }
                    }
                }
            }
        }

        hotkey::poll_sleep();
        el.set_control_flow(ControlFlow::Poll);
    }
}

impl AppHandler {
    fn dispatch_transcribe(&mut self, samples: Vec<f32>, sr: u32) {
        let Some(engine) = self.state.engine.clone() else {
            warn!("engine not yet ready, discarding {} samples", samples.len());
            return;
        };
        self.state.tray.set_color(tray::IconColor::Transcribing);
        let auto_paste = self.state.settings.auto_paste;
        thread::spawn(move || {
            let text = match engine.transcribe(&samples, sr) {
                Ok(t) => t,
                Err(e) => {
                    warn!(error = %e, "transcription failed");
                    return;
                }
            };
            if text.is_empty() {
                return;
            }
            info!(chars = text.len(), "transcript ready");
            if auto_paste {
                paste(&text);
            }
            // Tell the main thread we're done so it can flip the tray back.
            DONE_FLAG.store(true, Ordering::Relaxed);
        });
    }
}

static DONE_FLAG: AtomicBool = AtomicBool::new(false);

fn paste(text: &str) {
    // Save & restore clipboard around the synthesized Ctrl+V.
    let mut clip = match Clipboard::new() {
        Ok(c) => c,
        Err(e) => {
            warn!(error = %e, "clipboard open failed");
            return;
        }
    };
    let saved = clip.get_text().ok();
    if clip.set_text(text.to_string()).is_err() {
        warn!("clipboard write failed");
        return;
    }
    if let Ok(mut enigo) = Enigo::new(&EnigoSettings::default()) {
        let _ = enigo.key(Key::Control, Direction::Press);
        let _ = enigo.key(Key::Unicode('v'), Direction::Click);
        let _ = enigo.key(Key::Control, Direction::Release);
    } else {
        warn!("enigo init failed");
    }
    if let Some(prev) = saved {
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(400));
            if let Ok(mut c) = Clipboard::new() {
                let _ = c.set_text(prev);
            }
        });
    }
}
