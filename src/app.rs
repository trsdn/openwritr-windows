//! App orchestration: tray + winit event loop + hotkey thread + ASR.
//!
//! The hotkey FSM runs on its own background thread now, completely
//! independent of the winit event loop. That way the tray menu spawning
//! the settings subprocess, or any other event loop weirdness, cannot
//! stall recording. The hotkey thread sends `Event::Start` / `Event::Stop`
//! over a crossbeam-style channel into the winit loop, which translates
//! them into recorder/tray/engine actions.

use crate::{asr, audio::Recorder, enhance, hotkey, settings::Settings, sounds, tray};
use anyhow::Result;
use arboard::Clipboard;
use enigo::{Direction, Enigo, Key, Keyboard, Settings as EnigoSettings};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver, Sender},
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

enum HkEvent {
    Press,
    Release,
}

struct State {
    settings: Settings,
    recorder: Recorder,
    tray: tray::Tray,
    record_started: Option<Instant>,
    engine: Option<Arc<dyn asr::Engine>>,
    engine_loading: Arc<AtomicBool>,
    hk_rx: Receiver<HkEvent>,
    hk_stop: Arc<AtomicBool>,
}

pub fn run() -> Result<()> {
    let settings = Settings::load();
    let recorder = Recorder::new()?;
    let tray = tray::Tray::new(&settings)?;

    let (hk_tx, hk_rx) = mpsc::channel::<HkEvent>();
    let hk_stop = Arc::new(AtomicBool::new(false));
    spawn_hotkey_thread(settings.clone(), hk_tx, hk_stop.clone())?;

    let engine_loading = Arc::new(AtomicBool::new(false));
    let state = State {
        settings,
        recorder,
        tray,
        record_started: None,
        engine: None,
        engine_loading,
        hk_rx,
        hk_stop,
    };

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = AppHandler { state };
    app.start_engine_load();
    event_loop.run_app(&mut app)?;
    Ok(())
}

fn spawn_hotkey_thread(
    initial: Settings,
    tx: Sender<HkEvent>,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    // The thread re-reads settings.json after a small debounce so changes
    // from the settings subprocess take effect without needing a restart.
    thread::Builder::new()
        .name("hotkey".into())
        .spawn(move || hotkey_loop(initial, tx, stop))?;
    Ok(())
}

fn hotkey_loop(initial: Settings, tx: Sender<HkEvent>, stop: Arc<AtomicBool>) {
    let mut settings = initial;
    let mut mgr = match hotkey::HotkeyManager::register(&settings) {
        Ok(m) => Some(m),
        Err(e) => {
            warn!(error = %e, "hotkey register failed; polling without OS reservation");
            None
        }
    };
    let mut pressed = false;
    let mut last_check = Instant::now();

    while !stop.load(Ordering::Relaxed) {
        if let Some(m) = mgr.as_ref() {
            if let Some(ev) = hotkey::poll_state(m, &mut pressed) {
                let _ = tx.send(match ev {
                    hotkey::Event::Press => HkEvent::Press,
                    hotkey::Event::Release => HkEvent::Release,
                });
            }
        } else {
            // Even without RegisterHotKey we can still drive press/release
            // from raw GetAsyncKeyState.
            let dummy = match hotkey::HotkeyManager::register(&settings) {
                Ok(m) => Some(m),
                Err(_) => None,
            };
            if let Some(m) = dummy.as_ref() {
                if let Some(ev) = hotkey::poll_state(m, &mut pressed) {
                    let _ = tx.send(match ev {
                        hotkey::Event::Press => HkEvent::Press,
                        hotkey::Event::Release => HkEvent::Release,
                    });
                }
                mgr = dummy;
            }
        }

        // Every 500 ms, check whether settings.json changed and re-register
        // the hotkey if the combo changed.
        if last_check.elapsed() >= Duration::from_millis(500) {
            last_check = Instant::now();
            let new = Settings::load();
            if new.hotkey_modifiers != settings.hotkey_modifiers
                || new.hotkey_trigger != settings.hotkey_trigger
            {
                info!(
                    "hotkey changed: {:?}+{} -> {:?}+{}",
                    settings.hotkey_modifiers, settings.hotkey_trigger,
                    new.hotkey_modifiers, new.hotkey_trigger
                );
                drop(mgr.take());      // unregister old
                pressed = false;
                match hotkey::HotkeyManager::register(&new) {
                    Ok(m) => mgr = Some(m),
                    Err(e) => warn!(error = %e, "re-register failed"),
                }
            }
            settings = new;
        }

        thread::sleep(Duration::from_millis(8));
    }
    info!("hotkey thread exiting");
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

static STAGED_ENGINE: parking_lot::Mutex<Option<Arc<dyn asr::Engine>>> =
    parking_lot::Mutex::new(None);

impl ApplicationHandler for AppHandler {
    fn resumed(&mut self, _el: &ActiveEventLoop) {
        info!("event loop ready");
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, _ev: WindowEvent) {}

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        // Engine ready?
        if self.state.engine.is_none() {
            if let Some(e) = STAGED_ENGINE.lock().take() {
                self.state.engine = Some(e);
            }
        }

        // Drain tray menu events.
        while let Ok(ev) = MenuEvent::receiver().try_recv() {
            if ev.id == self.state.tray.menu_quit_id {
                info!("quit via tray");
                self.state.hk_stop.store(true, Ordering::Relaxed);
                el.exit();
                return;
            }
            if ev.id == self.state.tray.menu_settings_id {
                info!("opening settings dialog");
                if let Ok(exe) = std::env::current_exe() {
                    // Spawn detached so we never get blocked by it.
                    let _ = std::process::Command::new(exe).arg("--settings").spawn();
                }
                RELOAD_AT.lock().replace(Instant::now() + Duration::from_secs(2));
            }
        }

        // Periodic settings reload (engine hot-swap; hotkey reload happens
        // inside the hotkey thread independently).
        if let Some(at) = *RELOAD_AT.lock() {
            if Instant::now() >= at {
                *RELOAD_AT.lock() = None;
                let new = Settings::load();
                let old_engine = self.state.settings.engine.clone();
                self.state.settings = new;
                let new_engine = self.state.settings.engine.clone();
                if new_engine != old_engine {
                    info!("engine changed: {old_engine} -> {new_engine}; reloading");
                    self.state.engine = None;
                    self.start_engine_load();
                }
            }
        }

        // Drain hotkey events from the background thread.
        while let Ok(ev) = self.state.hk_rx.try_recv() {
            match ev {
                HkEvent::Press => {
                    if self.state.record_started.is_none() {
                        self.state.recorder.start();
                        self.state.tray.set_color(tray::IconColor::Recording);
                        self.state.record_started = Some(Instant::now());
                        if self.state.settings.sounds {
                            sounds::play_start();
                        }
                        info!("recording start");
                    }
                }
                HkEvent::Release => {
                    if let Some(started) = self.state.record_started.take() {
                        let samples = self.state.recorder.stop();
                        self.state.tray.set_color(tray::IconColor::Idle);
                        if self.state.settings.sounds {
                            sounds::play_stop();
                        }
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

        // Reset tray colour when a transcription thread signals done.
        if DONE_FLAG.swap(false, Ordering::Relaxed) {
            self.state.tray.set_color(tray::IconColor::Idle);
        }

        // Small sleep to avoid pegging the CPU — winit's Poll mode would
        // otherwise spin tight. 16 ms ~= 60 fps which is plenty for tray.
        thread::sleep(Duration::from_millis(16));
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
        let settings = self.state.settings.clone();
        thread::spawn(move || {
            let text = match engine.transcribe(&samples, sr) {
                Ok(t) => t,
                Err(e) => {
                    warn!(error = %e, "transcription failed");
                    DONE_FLAG.store(true, Ordering::Relaxed);
                    return;
                }
            };
            if text.is_empty() {
                DONE_FLAG.store(true, Ordering::Relaxed);
                return;
            }
            let final_text = if settings.enhance.provider != "off" {
                match enhance::enhance(&text, &settings) {
                    Ok(t) if !t.trim().is_empty() => t,
                    Ok(_) => text,
                    Err(e) => {
                        warn!(error = %e, "enhance failed; using raw transcript");
                        text
                    }
                }
            } else {
                text
            };
            info!(chars = final_text.len(), "transcript ready");
            if auto_paste {
                paste(&final_text);
            }
            DONE_FLAG.store(true, Ordering::Relaxed);
        });
    }
}

static DONE_FLAG: AtomicBool = AtomicBool::new(false);
static RELOAD_AT: parking_lot::Mutex<Option<Instant>> = parking_lot::Mutex::new(None);

fn paste(text: &str) {
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
