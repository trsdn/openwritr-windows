//! App orchestration: tray + winit event loop + hotkey thread + ASR.
//!
//! The hotkey FSM runs on its own background thread now, completely
//! independent of the winit event loop. That way the tray menu spawning
//! the settings subprocess, or any other event loop weirdness, cannot
//! stall recording. The hotkey thread sends `Event::Start` / `Event::Stop`
//! over a crossbeam-style channel into the winit loop, which translates
//! them into recorder/tray/engine actions.

use crate::overlay;
use crate::{asr, audio::Recorder, enhance, hotkey, paths, settings::Settings, sounds, tray};
use anyhow::Result;
use arboard::Clipboard;
use enigo::{Direction, Enigo, Key, Keyboard, Settings as EnigoSettings};
use std::os::windows::process::CommandExt;
use std::process::Stdio;
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
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::WindowId;

#[derive(Debug, Clone, Copy)]
pub enum UserEvent {
    HotkeyPress,
    HotkeyRelease { enhance: bool },
    TranscribeDone,
    Tick,
}

// DETACHED_PROCESS | CREATE_NO_WINDOW — child fully decoupled from parent
const DETACHED_PROCESS: u32 = 0x0000_0008;
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

struct State {
    settings: Settings,
    recorder: Recorder,
    tray: tray::Tray,
    record_started: Option<Instant>,
    engine: Option<Arc<dyn asr::Engine>>,
    engine_loading: Arc<AtomicBool>,
    hk_stop: Arc<AtomicBool>,
}

pub fn run() -> Result<()> {
    let settings = Settings::load();
    let recorder = Recorder::new()?;
    let tray = tray::Tray::new(&settings)?;

    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();

    let hk_stop = Arc::new(AtomicBool::new(false));
    spawn_hotkey_thread(settings.clone(), proxy.clone(), hk_stop.clone())?;
    spawn_tick_thread(proxy);

    // Visual recording indicator on its own thread + own Win32 message loop.
    // It only reads atomics from the recorder, so there's no shared state with
    // the main winit/tray loop that could deadlock.
    overlay::spawn(overlay::OverlayHandles {
        recording: recorder.recording.clone(),
        level_x10000: recorder.last_rms_x10000.clone(),
        stop: hk_stop.clone(),
    });

    let engine_loading = Arc::new(AtomicBool::new(false));
    let state = State {
        settings,
        recorder,
        tray,
        record_started: None,
        engine: None,
        engine_loading,
        hk_stop,
    };

    // Wait mode: loop sleeps until a message arrives (tray click, user event,
    // window event). The hotkey + tick threads wake it via EventLoopProxy.
    // No more thread::sleep inside the message pump → tray stays responsive.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = AppHandler { state };
    app.start_engine_load();
    event_loop.run_app(&mut app)?;
    Ok(())
}

fn spawn_hotkey_thread(
    initial: Settings,
    proxy: EventLoopProxy<UserEvent>,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    thread::Builder::new()
        .name("hotkey".into())
        .spawn(move || hotkey_loop(initial, proxy, stop))?;
    Ok(())
}

fn spawn_tick_thread(proxy: EventLoopProxy<UserEvent>) {
    // Tick the event loop every 100 ms so periodic tasks (tray menu poll,
    // staged engine, settings reload, transcribe-done flag) get serviced
    // even when no input is happening.
    thread::Builder::new()
        .name("tick".into())
        .spawn(move || loop {
            thread::sleep(Duration::from_millis(100));
            if proxy.send_event(UserEvent::Tick).is_err() {
                break;
            }
        })
        .ok();
}

fn hotkey_loop(initial: Settings, proxy: EventLoopProxy<UserEvent>, stop: Arc<AtomicBool>) {
    let mut settings = initial;
    // Try OS-level registration so other apps know the combo is taken.
    // If that fails (e.g. Windows already reserved it), fall through to
    // pure GetAsyncKeyState polling — that always works regardless of
    // whether RegisterHotKey accepted us.
    let mut _mgr: Option<hotkey::HotkeyManager> = match hotkey::HotkeyManager::register(&settings) {
        Ok(m) => Some(m),
        Err(e) => {
            warn!(error = %e, "RegisterHotKey failed; using key-state polling only");
            None
        }
    };
    // Track combo vk codes manually so polling works without a HotkeyManager.
    let mut trigger_vk = hotkey::trigger_vk_for(&settings.hotkey_trigger);
    let mut mod_vks: Vec<u32> = settings
        .hotkey_modifiers
        .iter()
        .map(|m| hotkey::mod_vk_for(m))
        .collect();
    let mut poll_state = hotkey::PollState::default();
    let mut last_check = Instant::now();

    while !stop.load(Ordering::Relaxed) {
        if let Some(ev) = hotkey::poll_combo(trigger_vk, &mod_vks, &mut poll_state) {
            let user_ev = match ev {
                hotkey::Event::Press => UserEvent::HotkeyPress,
                hotkey::Event::Release => {
                    let shift_is_modifier = settings
                        .hotkey_modifiers
                        .iter()
                        .any(|m| m == "shift");
                    let enhance = !shift_is_modifier && shift_currently_down();
                    UserEvent::HotkeyRelease { enhance }
                }
            };
            if proxy.send_event(user_ev).is_err() {
                break;
            }
        }

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
                drop(_mgr.take());
                poll_state = hotkey::PollState::default();
                _mgr = hotkey::HotkeyManager::register(&new).ok();
                trigger_vk = hotkey::trigger_vk_for(&new.hotkey_trigger);
                mod_vks = new.hotkey_modifiers.iter().map(|m| hotkey::mod_vk_for(m)).collect();
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
        // The default Windows worker-thread stack is 1 MiB. QnnHtp.dll's
        // EPContext binary loader needs more and triggers
        // STATUS_STACK_BUFFER_OVERRUN (0xC0000409, /GS cookie corruption
        // from stack overflow). Give it a generous 32 MiB. Cheap on x64.
        thread::Builder::new()
            .name("engine-loader".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                match asr::load(&engine_name) {
                    Ok(e) => {
                        info!("engine loaded: {}", e.label());
                        STAGED_ENGINE.lock().replace(Arc::from(e));
                    }
                    Err(e) => warn!(error = %e, "engine load failed"),
                }
                loading_flag.store(false, Ordering::Relaxed);
            })
            .expect("spawn engine-loader thread");
    }
}

static STAGED_ENGINE: parking_lot::Mutex<Option<Arc<dyn asr::Engine>>> =
    parking_lot::Mutex::new(None);

impl ApplicationHandler<UserEvent> for AppHandler {
    fn resumed(&mut self, _el: &ActiveEventLoop) {
        info!("event loop ready");
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, _ev: WindowEvent) {}

    fn user_event(&mut self, el: &ActiveEventLoop, ev: UserEvent) {
        match ev {
            UserEvent::HotkeyPress => self.on_press(),
            UserEvent::HotkeyRelease { enhance } => self.on_release(enhance),
            UserEvent::TranscribeDone => {
                self.state.tray.set_color(tray::IconColor::Idle);
            }
            UserEvent::Tick => self.tick(el),
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        // Wait mode: don't sleep here; tick thread + user events wake us.
        self.tick(el);
        el.set_control_flow(ControlFlow::Wait);
    }
}

impl AppHandler {
    fn tick(&mut self, el: &ActiveEventLoop) {
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
                info!("opening settings UI subprocess");
                // CRITICAL: spawn from a background thread. CreateProcessW on
                // Windows ARM64 (especially with Defender real-time scanning)
                // can block for several seconds. If we call it from inside the
                // winit pump, the tray's message queue stalls → app goes
                // "Not Responding" → hotkey dies. From a worker thread the
                // main pump keeps draining messages while the child boots.
                if let Ok(exe) = std::env::current_exe() {
                    thread::spawn(move || {
                        let _ = std::process::Command::new(exe)
                            .arg("--settings")
                            .stdin(Stdio::null())
                            .stdout(Stdio::null())
                            .stderr(Stdio::null())
                            .creation_flags(DETACHED_PROCESS | CREATE_NO_WINDOW)
                            .spawn();
                    });
                }
            }
        }

        // Settings hot-reload: poll settings.json mtime; reload if changed.
        // (Replaces the old fixed 2-second debounce.)
        {
            let path = paths::settings_path();
            let mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
            let mut last = LAST_SETTINGS_MTIME.lock();
            let changed = match (*last, mtime) {
                (Some(a), Some(b)) => a != b,
                (None, Some(_)) => true,
                _ => false,
            };
            if changed {
                *last = mtime;
                drop(last);
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
        // Keep RELOAD_AT path as a no-op safety net (cleared if ever set).
        if let Some(at) = *RELOAD_AT.lock() {
            if Instant::now() >= at {
                *RELOAD_AT.lock() = None;
            }
        }

        if DONE_FLAG.swap(false, Ordering::Relaxed) {
            self.state.tray.set_color(tray::IconColor::Idle);
        }
    }

    fn on_press(&mut self) {
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

    fn on_release(&mut self, enhance: bool) {
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
                self.dispatch_transcribe(samples, sr, enhance);
            }
        }
    }
}

fn shift_currently_down() -> bool {
    use windows::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_SHIFT};
    unsafe { (GetAsyncKeyState(VK_SHIFT.0 as i32) as u32) & 0x8000 != 0 }
}

impl AppHandler {
    fn dispatch_transcribe(&mut self, samples: Vec<f32>, sr: u32, enhance_requested: bool) {
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
            // Enhance only when the user explicitly requested it by holding
            // Shift at release time AND a provider is configured.
            let final_text = if enhance_requested && settings.enhance.provider != "off" {
                info!("enhance requested (shift held)");
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
static LAST_SETTINGS_MTIME: parking_lot::Mutex<Option<std::time::SystemTime>> = parking_lot::Mutex::new(None);

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
