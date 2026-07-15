//! App orchestration: tray + winit event loop + hotkey thread + ASR.
//!
//! The hotkey FSM runs on its own background thread now, completely
//! independent of the winit event loop. That way the tray menu spawning
//! the settings subprocess, or any other event loop weirdness, cannot
//! stall recording. The hotkey thread sends `Event::Start` / `Event::Stop`
//! over a crossbeam-style channel into the winit loop, which translates
//! them into recorder/tray/engine actions.

use crate::{
    audio::Recorder,
    diagnostics, hotkey, key_hook,
    model_manager::ModelState,
    overlay, paste, paths,
    settings::{CredentialHealth, Settings},
    sounds, tray,
    worker::{ShutdownMode, Worker, WorkerEvent},
};
use anyhow::Result;
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

#[derive(Debug, Clone)]
pub enum UserEvent {
    HotkeyPress,
    HotkeyRelease { enhance: bool },
    DiagnosticsExported,
    DiagnosticsExportFailed(String),
    Tick,
}

// DETACHED_PROCESS | CREATE_NO_WINDOW — child fully decoupled from parent
const DETACHED_PROCESS: u32 = 0x0000_0008;
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

enum EngineState {
    NotStarted,
    Loading,
    Ready { label: String },
    Failed { error: String },
}

struct State {
    settings: Settings,
    credential_health: CredentialHealth,
    settings_error: Option<String>,
    recorder: Recorder,
    tray: tray::Tray,
    overlay: overlay::OverlayController,
    record_started: Option<Instant>,
    worker: Worker,
    engine_state: EngineState,
    model_state: Option<ModelState>,
    load_generation: u64,
    pending_jobs: usize,
    active_job: Option<u64>,
    shutting_down: bool,
    hk_stop: Arc<AtomicBool>,
    proxy: EventLoopProxy<UserEvent>,
    diagnostics_exporting: bool,
}

pub fn run() -> Result<()> {
    let (settings, credential_health, settings_error) = match Settings::load_runtime() {
        Ok(loaded) => (
            loaded.settings,
            loaded.credential_health,
            loaded.settings_error,
        ),
        Err(error) => {
            warn!(error = %error, "settings load failed; using defaults until a valid file is saved");
            (
                Settings::default(),
                CredentialHealth::default(),
                Some(format!("settings could not be loaded: {error}")),
            )
        }
    };
    if let Some(message) = &credential_health.message {
        warn!(message, "credential migration needs attention");
    }
    if let Some(error) = &settings_error {
        warn!(
            error,
            "settings validation failed; using defaults until a valid file is saved"
        );
    }
    info!(
        engine = %settings.engine,
        auto_paste = settings.auto_paste,
        overlay = settings.overlay,
        sounds = settings.sounds,
        "settings loaded"
    );
    let recorder = Recorder::new()?;
    let tray = tray::Tray::new(&settings)?;
    let worker = Worker::spawn()?;

    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();

    let hk_stop = Arc::new(AtomicBool::new(false));
    spawn_hotkey_thread(settings.clone(), proxy.clone(), hk_stop.clone())?;
    spawn_tick_thread(proxy.clone());

    // Visual recording indicator on its own thread + own Win32 message loop.
    // It only reads atomics from the recorder, so there's no shared state with
    // the main winit/tray loop that could deadlock.
    let overlay = overlay::spawn(
        overlay::OverlayHandles {
            recording: recorder.recording.clone(),
            level_x10000: recorder.last_rms_x10000.clone(),
            stop: hk_stop.clone(),
        },
        settings.overlay,
    )?;

    let state = State {
        settings,
        credential_health,
        settings_error,
        recorder,
        tray,
        overlay,
        record_started: None,
        worker,
        engine_state: EngineState::NotStarted,
        model_state: None,
        load_generation: 0,
        pending_jobs: 0,
        active_job: None,
        shutting_down: false,
        hk_stop,
        proxy,
        diagnostics_exporting: false,
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
    let mut last_settings_error: Option<String> = None;
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
    let mut hook_health = key_hook::HealthMonitor::new(
        hotkey::configured_vks(trigger_vk, &mod_vks),
        hotkey::secondary_key_down,
    );
    let mut last_check = Instant::now();

    while !stop.load(Ordering::Relaxed) {
        if let Some(ev) = hotkey::poll_combo(trigger_vk, &mod_vks, &mut poll_state) {
            let user_ev = match ev {
                hotkey::Event::Press => UserEvent::HotkeyPress,
                hotkey::Event::Release => {
                    let shift_is_modifier = settings.hotkey_modifiers.iter().any(|m| m == "shift");
                    let enhance = !shift_is_modifier && shift_currently_down();
                    UserEvent::HotkeyRelease { enhance }
                }
            };
            if proxy.send_event(user_ev).is_err() {
                break;
            }
        }
        if hook_health.observe(
            Instant::now(),
            hotkey::secondary_key_down,
            hotkey::configured_key_down,
        ) {
            warn!(
                events_seen = key_hook::events_seen(),
                "configured key transitions were repeatedly absent from the keyboard hook; requesting reinstall"
            );
            key_hook::request_reinstall();
            poll_state = hotkey::PollState::default();
        }

        if last_check.elapsed() >= Duration::from_millis(500) {
            last_check = Instant::now();
            match Settings::load() {
                Ok(new) => {
                    last_settings_error = None;
                    if new.hotkey_modifiers != settings.hotkey_modifiers
                        || new.hotkey_trigger != settings.hotkey_trigger
                    {
                        info!(
                            "hotkey changed: {:?}+{} -> {:?}+{}",
                            settings.hotkey_modifiers,
                            settings.hotkey_trigger,
                            new.hotkey_modifiers,
                            new.hotkey_trigger
                        );
                        drop(_mgr.take());
                        poll_state = hotkey::PollState::default();
                        _mgr = hotkey::HotkeyManager::register(&new).ok();
                        trigger_vk = hotkey::trigger_vk_for(&new.hotkey_trigger);
                        mod_vks = new
                            .hotkey_modifiers
                            .iter()
                            .map(|modifier| hotkey::mod_vk_for(modifier))
                            .collect();
                        hook_health = key_hook::HealthMonitor::new(
                            hotkey::configured_vks(trigger_vk, &mod_vks),
                            hotkey::secondary_key_down,
                        );
                    }
                    settings = new;
                }
                Err(error) => {
                    let message = error.to_string();
                    if last_settings_error.as_deref() != Some(message.as_str()) {
                        warn!(error = %message, "hotkey settings reload failed; keeping the last valid hotkey");
                        last_settings_error = Some(message);
                    }
                }
            }
        }

        thread::sleep(Duration::from_millis(8));
    }
    info!("hotkey thread exiting");
}

struct AppHandler {
    state: State,
}

impl AppHandler {
    fn start_engine_load(&mut self) {
        if self.state.shutting_down {
            return;
        }
        let engine_name = self.state.settings.engine.clone();
        match self.state.worker.load(engine_name.clone()) {
            Ok(generation) => {
                self.state.load_generation = generation;
                self.state.engine_state = EngineState::Loading;
                self.state.model_state = None;
                self.state.tray.set_status(
                    tray::IconColor::Transcribing,
                    &format!("OpenWritr — loading {engine_name}"),
                );
            }
            Err(error) => {
                self.state.engine_state = EngineState::Failed {
                    error: error.to_string(),
                };
                self.state.tray.set_status(
                    tray::IconColor::Error,
                    "OpenWritr — could not start engine loader (see log)",
                );
                warn!(error = %error, "failed to start engine load");
            }
        }
    }
}

impl ApplicationHandler<UserEvent> for AppHandler {
    fn resumed(&mut self, _el: &ActiveEventLoop) {
        info!("event loop ready");
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, _ev: WindowEvent) {}

    fn user_event(&mut self, el: &ActiveEventLoop, ev: UserEvent) {
        match ev {
            UserEvent::HotkeyPress => self.on_press(),
            UserEvent::HotkeyRelease { enhance } => self.on_release(enhance),
            UserEvent::DiagnosticsExported => {
                self.state.diagnostics_exporting = false;
                self.state
                    .tray
                    .set_tooltip("OpenWritr — diagnostics exported");
            }
            UserEvent::DiagnosticsExportFailed(error) => {
                self.state.diagnostics_exporting = false;
                warn!(error = %error, "diagnostics export failed");
                self.state
                    .tray
                    .set_tooltip("OpenWritr — diagnostics export failed (see log)");
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
        while let Some(event) = self.state.worker.try_recv() {
            self.handle_worker_event(el, event);
        }

        if self.state.record_started.is_some() {
            if self.state.recorder.stream_failed() {
                self.finish_recording(false, false);
            } else {
                let max_seconds = self.state.settings.max_record_seconds;
                let timer_limit_reached = self
                    .state
                    .record_started
                    .map(|started| started.elapsed().as_secs_f32() >= max_seconds)
                    .unwrap_or(false);
                if self.state.recorder.limit_reached() || timer_limit_reached {
                    self.finish_recording(false, true);
                }
            }
        }

        // Drain tray menu events.
        while let Ok(ev) = MenuEvent::receiver().try_recv() {
            if ev.id == self.state.tray.menu_quit_id {
                self.request_shutdown(el);
                return;
            }
            if ev.id == self.state.tray.menu_cancel_model_id {
                if matches!(self.state.engine_state, EngineState::Loading) {
                    self.state.load_generation = self.state.worker.cancel_load();
                    self.state.engine_state = EngineState::NotStarted;
                    self.state.model_state = Some(ModelState::Cancelled);
                    self.state.tray.set_status(
                        tray::IconColor::Error,
                        "OpenWritr — model download cancelled; choose Retry",
                    );
                    info!("model acquisition cancelled by user");
                }
                continue;
            }
            if ev.id == self.state.tray.menu_retry_engine_id {
                self.start_engine_load();
                continue;
            }
            if ev.id == self.state.tray.menu_open_logs_id {
                if let Err(e) = diagnostics::open_logs_dir() {
                    warn!(error = %e, "failed to open logs directory");
                    self.state.tray.set_status(
                        tray::IconColor::Error,
                        "OpenWritr — could not open logs (see log)",
                    );
                }
                continue;
            }
            if ev.id == self.state.tray.menu_export_diagnostics_id {
                if !self.state.diagnostics_exporting {
                    self.state.diagnostics_exporting = true;
                    self.state
                        .tray
                        .set_tooltip("OpenWritr — exporting diagnostics…");
                    let settings = self.state.settings.clone();
                    let proxy = self.state.proxy.clone();
                    let spawn = thread::Builder::new()
                        .name("diagnostics-export".into())
                        .spawn(move || {
                            let event = match diagnostics::export_bundle(&settings) {
                                Ok(path) => {
                                    info!(file = %path.display(), "diagnostics exported");
                                    if let Err(e) = diagnostics::reveal(&path) {
                                        warn!(error = %e, "failed to reveal diagnostics bundle");
                                    }
                                    UserEvent::DiagnosticsExported
                                }
                                Err(e) => UserEvent::DiagnosticsExportFailed(e.to_string()),
                            };
                            let _ = proxy.send_event(event);
                        });
                    if let Err(e) = spawn {
                        self.state.diagnostics_exporting = false;
                        warn!(error = %e, "failed to start diagnostics export");
                        self.state
                            .tray
                            .set_tooltip("OpenWritr — diagnostics export failed (see log)");
                    }
                } else {
                    self.state
                        .tray
                        .set_tooltip("OpenWritr — diagnostics export already running");
                }
                continue;
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
                match Settings::load_runtime() {
                    Ok(loaded) => {
                        self.state.credential_health = loaded.credential_health;
                        if let Some(message) = &self.state.credential_health.message {
                            warn!(message, "credential migration needs attention");
                        }
                        if let Some(error) = loaded.settings_error {
                            warn!(error, "settings reload failed validation; keeping the last valid settings");
                            self.state.settings_error = Some(error);
                            self.update_job_status();
                        } else {
                            let old_engine = self.state.settings.engine.clone();
                            let old_overlay = self.state.settings.overlay;
                            self.state.settings = loaded.settings;
                            self.state.settings_error = None;
                            if self.state.settings.overlay != old_overlay {
                                if let Err(error) =
                                    self.state.overlay.set_enabled(self.state.settings.overlay)
                                {
                                    warn!(error, "failed to update overlay setting");
                                }
                            }
                            let new_engine = self.state.settings.engine.clone();
                            if new_engine != old_engine {
                                info!("engine changed: {old_engine} -> {new_engine}; reloading");
                                self.start_engine_load();
                            } else {
                                self.update_job_status();
                            }
                        }
                    }
                    Err(error) => {
                        warn!(error = %error, "settings reload failed; keeping the last valid settings");
                        self.state.settings_error =
                            Some(format!("settings could not be reloaded: {error}"));
                        self.update_job_status();
                    }
                }
            }
        }
    }

    fn on_press(&mut self) {
        if self.state.shutting_down || self.state.record_started.is_some() {
            return;
        }
        if !matches!(self.state.engine_state, EngineState::Ready { .. }) {
            let (color, status) = self.blocked_recording_status();
            self.state.tray.set_status(color, &status);
            if let Err(error) = self.state.overlay.show_status(
                status
                    .strip_prefix("OpenWritr — ")
                    .unwrap_or(&status)
                    .to_string(),
            ) {
                warn!(error, "failed to show overlay status");
            }
            info!(status, "recording blocked while engine is unavailable");
            return;
        }

        if self.state.record_started.is_none() {
            match self
                .state
                .recorder
                .start(self.state.settings.max_record_seconds)
            {
                Ok(capture) => {
                    self.state.tray.set_status(
                        tray::IconColor::Recording,
                        &format!("OpenWritr — recording from {}", capture.device_name),
                    );
                    self.state.record_started = Some(Instant::now());
                    if self.state.settings.sounds {
                        sounds::play_start();
                    }
                    info!(
                        device = %capture.device_name,
                        sample_rate = capture.sample_rate,
                        channels = capture.channels,
                        "recording start"
                    );
                }
                Err(e) => {
                    self.state.tray.set_status(
                        tray::IconColor::Error,
                        "OpenWritr — microphone unavailable (see log)",
                    );
                    warn!(error = %e, "failed to start recording");
                }
            }
        }
    }

    fn on_release(&mut self, enhance: bool) {
        if self.state.record_started.is_some() {
            self.finish_recording(enhance, false);
        }
    }

    fn finish_recording(&mut self, enhance: bool, timer_limit_reached: bool) {
        let Some(started) = self.state.record_started.take() else {
            return;
        };
        let recording = match self.state.recorder.stop() {
            Ok(recording) => recording,
            Err(e) => {
                self.state.tray.set_status(
                    tray::IconColor::Error,
                    "OpenWritr — microphone error (see log)",
                );
                warn!(error = %e, "failed to stop recording");
                return;
            }
        };
        if self.state.settings.sounds {
            sounds::play_stop();
        }
        if let Some(error) = recording.stream_error {
            self.state.tray.set_status(
                tray::IconColor::Error,
                "OpenWritr — microphone stream failed (see log)",
            );
            warn!(error = %error, "recording aborted after stream failure");
            return;
        }

        self.state
            .tray
            .set_status(tray::IconColor::Idle, "OpenWritr");
        let reached_limit = timer_limit_reached || recording.reached_limit;
        if reached_limit {
            info!(
                max_seconds = self.state.settings.max_record_seconds,
                "maximum recording duration reached"
            );
        }

        let dur = started.elapsed();
        let min = self.state.settings.min_record_seconds;
        if dur.as_secs_f32() < min {
            info!(secs = dur.as_secs_f32(), "below min — discarded");
        } else {
            info!(
                device = %recording.device_name,
                sample_rate = recording.sample_rate,
                channels = recording.channels,
                samples = recording.samples.len(),
                reached_limit,
                "recording stop"
            );
            self.dispatch_transcribe(recording.samples, recording.sample_rate, enhance);
        }
    }
}

fn shift_currently_down() -> bool {
    use windows::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_SHIFT};
    unsafe { (GetAsyncKeyState(VK_SHIFT.0 as i32) as u32) & 0x8000 != 0 }
}

impl AppHandler {
    fn dispatch_transcribe(&mut self, samples: Vec<f32>, sr: u32, enhance_requested: bool) {
        if self.state.shutting_down {
            warn!("discarding completed recording because shutdown has started");
            return;
        }
        let enhance_requested = if enhance_requested
            && self.state.settings.enhance.provider == "openai_compatible"
            && self.state.credential_health.enhancement_disabled
        {
            warn!("enhancement skipped because credential migration is incomplete");
            if let Err(error) = self
                .state
                .overlay
                .show_status("Enhancement disabled: secure the API key in Settings")
            {
                warn!(error, "failed to show overlay status");
            }
            false
        } else {
            enhance_requested
        };
        match self
            .state
            .worker
            .enqueue(samples, sr, enhance_requested, self.state.settings.clone())
        {
            Ok(id) => {
                self.state.pending_jobs = self.state.pending_jobs.saturating_add(1);
                info!(
                    id,
                    queue_depth = self.state.pending_jobs,
                    "transcription job queued"
                );
                self.update_job_status();
            }
            Err(error) => {
                warn!(error = %error, "failed to queue transcription");
                self.state.tray.set_status(
                    tray::IconColor::Error,
                    "OpenWritr — could not queue transcription (see log)",
                );
            }
        }
    }

    fn handle_worker_event(&mut self, el: &ActiveEventLoop, event: WorkerEvent) {
        match event {
            WorkerEvent::ModelState {
                generation,
                engine,
                state,
            } => {
                if generation != self.state.load_generation || engine != self.state.settings.engine
                {
                    return;
                }
                let color = match &state {
                    ModelState::Failed { .. } | ModelState::Cancelled => tray::IconColor::Error,
                    ModelState::Ready => tray::IconColor::Transcribing,
                    ModelState::Missing
                    | ModelState::Downloading { .. }
                    | ModelState::Verifying => tray::IconColor::Transcribing,
                };
                let status = state.status_text(&engine);
                self.state.model_state = Some(state);
                self.state
                    .tray
                    .set_status(color, &format!("OpenWritr — {status}"));
            }
            WorkerEvent::EngineLoading { generation, engine } => {
                if generation != self.state.load_generation || engine != self.state.settings.engine
                {
                    return;
                }
                self.state.engine_state = EngineState::Loading;
                self.state.tray.set_status(
                    tray::IconColor::Transcribing,
                    &format!("OpenWritr — loading {engine}"),
                );
            }
            WorkerEvent::EngineReady {
                generation,
                engine,
                label,
            } => {
                if generation != self.state.load_generation || engine != self.state.settings.engine
                {
                    return;
                }
                info!(engine, label, "selected engine ready");
                self.state.engine_state = EngineState::Ready {
                    label: label.clone(),
                };
                self.state.model_state = Some(ModelState::Ready);
                if self.state.pending_jobs == 0 {
                    self.update_job_status();
                }
            }
            WorkerEvent::EngineFailed {
                generation,
                engine,
                error,
            } => {
                if generation != self.state.load_generation || engine != self.state.settings.engine
                {
                    return;
                }
                warn!(engine, error = %error, "selected engine failed to load");
                let short_error = short_error(&error);
                self.state.engine_state = EngineState::Failed { error };
                self.state.tray.set_status(
                    tray::IconColor::Error,
                    &format!("OpenWritr — {engine} failed: {short_error}; choose Retry"),
                );
            }
            WorkerEvent::JobStarted { id } => {
                self.state.active_job = Some(id);
                self.update_job_status();
            }
            WorkerEvent::JobCompleted {
                id,
                text,
                auto_paste,
            } => {
                self.finish_job(id);
                if auto_paste && !text.is_empty() {
                    paste::paste(&text);
                }
                self.update_job_status();
            }
            WorkerEvent::JobFailed { id, error } => {
                self.finish_job(id);
                warn!(id, error = %error, "transcription job failed");
                if self.state.pending_jobs == 0 {
                    self.state.tray.set_status(
                        tray::IconColor::Error,
                        &format!("OpenWritr — transcription failed: {}", short_error(&error)),
                    );
                } else {
                    self.update_job_status();
                }
            }
            WorkerEvent::JobDiscarded { id } => {
                self.finish_job(id);
                info!(id, "transcription job discarded");
                self.update_job_status();
            }
            WorkerEvent::ShutdownComplete => {
                info!("worker shutdown complete");
                el.exit();
            }
        }
    }

    fn finish_job(&mut self, id: u64) {
        self.state.pending_jobs = self.state.pending_jobs.saturating_sub(1);
        if self.state.active_job == Some(id) {
            self.state.active_job = None;
        }
    }

    fn update_job_status(&self) {
        if self.state.pending_jobs > 0 {
            let noun = if self.state.pending_jobs == 1 {
                "job"
            } else {
                "jobs"
            };
            self.state.tray.set_status(
                tray::IconColor::Transcribing,
                &format!(
                    "OpenWritr — transcribing; {} {noun} remaining",
                    self.state.pending_jobs
                ),
            );
        } else if !self.state.shutting_down {
            if let EngineState::Failed { error } = &self.state.engine_state {
                self.state.tray.set_status(
                    tray::IconColor::Error,
                    &format!("OpenWritr — engine failed: {}", short_error(error)),
                );
            } else if let Some(issue) = self.configuration_issue() {
                self.state.tray.set_status(
                    tray::IconColor::Error,
                    &format!("OpenWritr — {}", short_error(issue)),
                );
            } else {
                match &self.state.engine_state {
                    EngineState::Ready { label } => self.state.tray.set_status(
                        tray::IconColor::Idle,
                        &format!("OpenWritr — ready: {label}"),
                    ),
                    EngineState::Loading => self
                        .state
                        .tray
                        .set_status(tray::IconColor::Transcribing, "OpenWritr — loading engine"),
                    EngineState::NotStarted => self
                        .state
                        .tray
                        .set_status(tray::IconColor::Error, "OpenWritr — engine not ready"),
                    EngineState::Failed { .. } => unreachable!(),
                }
            }
        }
    }

    fn configuration_issue(&self) -> Option<&str> {
        self.state
            .settings_error
            .as_deref()
            .or(self.state.credential_health.message.as_deref())
    }

    fn blocked_recording_status(&self) -> (tray::IconColor, String) {
        if let Some(state) = &self.state.model_state {
            let color = match state {
                ModelState::Failed { .. } | ModelState::Cancelled => tray::IconColor::Error,
                _ => tray::IconColor::Transcribing,
            };
            return (
                color,
                format!(
                    "OpenWritr — recording blocked: {}",
                    state.status_text(&self.state.settings.engine)
                ),
            );
        }
        match &self.state.engine_state {
            EngineState::Loading => (
                tray::IconColor::Transcribing,
                "OpenWritr — recording blocked: engine is loading".into(),
            ),
            EngineState::Failed { error } => (
                tray::IconColor::Error,
                format!(
                    "OpenWritr — recording blocked: engine failed: {}",
                    short_error(error)
                ),
            ),
            EngineState::NotStarted => (
                tray::IconColor::Error,
                "OpenWritr — recording blocked: engine is not ready".into(),
            ),
            EngineState::Ready { .. } => (tray::IconColor::Idle, "OpenWritr — engine ready".into()),
        }
    }

    fn request_shutdown(&mut self, el: &ActiveEventLoop) {
        if self.state.shutting_down {
            return;
        }
        info!(pending_jobs = self.state.pending_jobs, "quit requested");
        self.state.shutting_down = true;
        self.state.hk_stop.store(true, Ordering::Relaxed);
        if self.state.record_started.take().is_some() {
            let _ = self.state.recorder.stop();
            info!("active recording discarded during shutdown");
        }

        let mode = if self.state.pending_jobs > 0 {
            prompt_shutdown_mode(self.state.pending_jobs)
        } else {
            ShutdownMode::Discard
        };
        let status = match mode {
            ShutdownMode::Wait => format!(
                "OpenWritr — finishing {} queued job(s) before exit",
                self.state.pending_jobs
            ),
            ShutdownMode::Discard => "OpenWritr — discarding queued work and exiting".into(),
        };
        self.state
            .tray
            .set_status(tray::IconColor::Transcribing, &status);
        if let Err(error) = self.state.worker.shutdown(mode) {
            warn!(error = %error, "worker shutdown command failed");
            el.exit();
        }
    }
}

static LAST_SETTINGS_MTIME: parking_lot::Mutex<Option<std::time::SystemTime>> =
    parking_lot::Mutex::new(None);

fn short_error(error: &str) -> String {
    const MAX_CHARS: usize = 140;
    let mut shortened = error.chars().take(MAX_CHARS).collect::<String>();
    if error.chars().count() > MAX_CHARS {
        shortened.push('…');
    }
    shortened
}

fn prompt_shutdown_mode(pending_jobs: usize) -> ShutdownMode {
    use windows::core::HSTRING;
    use windows::Win32::UI::WindowsAndMessaging::{
        MessageBoxW, IDNO, MB_DEFBUTTON1, MB_ICONQUESTION, MB_SETFOREGROUND, MB_YESNO,
    };

    let message = HSTRING::from(format!(
        "OpenWritr is still processing {pending_jobs} transcription job(s).\n\n\
         Choose Yes to wait for them to finish before exiting.\n\
         Choose No to discard queued results and exit after the current native call reaches a safe boundary."
    ));
    let title = HSTRING::from("OpenWritr");
    let response = unsafe {
        MessageBoxW(
            None,
            &message,
            &title,
            MB_YESNO | MB_ICONQUESTION | MB_DEFBUTTON1 | MB_SETFOREGROUND,
        )
    };
    if response == IDNO {
        ShutdownMode::Discard
    } else {
        ShutdownMode::Wait
    }
}
