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
    cleanup::FallbackReason,
    diagnostics, hotkey, key_hook,
    model_manager::ModelState,
    overlay::{self, ListeningIntent, NoticeKind, OverlayViewState, ProcessingPhase},
    paste,
    settings::{CredentialHealth, EnhanceMode, Settings, SettingsRevision},
    sounds, tray,
    worker::{DeliveryTarget, JobConfig, RecordingIntent, ShutdownMode, Worker, WorkerEvent},
};
use anyhow::Result;
use std::collections::BTreeMap;
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
    HotkeyPress {
        epoch: u64,
        shift_down: bool,
        delivery_target: Option<DeliveryTarget>,
    },
    HotkeyRelease,
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

struct ActiveRecording {
    started: Instant,
    settings: Settings,
    intent: RecordingIntent,
    delivery_target: Option<DeliveryTarget>,
}

impl ActiveRecording {
    fn new(
        settings: Settings,
        intent: RecordingIntent,
        delivery_target: Option<DeliveryTarget>,
    ) -> Self {
        Self {
            started: Instant::now(),
            settings,
            intent,
            delivery_target,
        }
    }
}

#[derive(Clone)]
struct OverlayNotice {
    kind: NoticeKind,
    message: String,
    expires_at: Instant,
}

struct OverlayReducer {
    recording: Option<ListeningIntent>,
    jobs: BTreeMap<u64, ProcessingPhase>,
    active_job: Option<u64>,
    notice: Option<OverlayNotice>,
    last_view: OverlayViewState,
}

impl Default for OverlayReducer {
    fn default() -> Self {
        Self {
            recording: None,
            jobs: BTreeMap::new(),
            active_job: None,
            notice: None,
            last_view: OverlayViewState::Hidden,
        }
    }
}

impl OverlayReducer {
    fn recording_started(&mut self, intent: RecordingIntent) {
        self.recording = Some(match intent {
            RecordingIntent::Raw => ListeningIntent::Raw,
            RecordingIntent::Enhance => ListeningIntent::Enhance,
        });
    }

    fn recording_finished(&mut self) {
        self.recording = None;
    }

    fn job_queued(&mut self, id: u64) {
        self.jobs.entry(id).or_insert(ProcessingPhase::Queued);
    }

    fn job_started(&mut self, id: u64) -> bool {
        if self.active_job.is_some_and(|active| active != id) {
            return false;
        }
        let Some(phase) = self.jobs.get_mut(&id) else {
            return false;
        };
        if *phase != ProcessingPhase::Queued {
            return false;
        }
        *phase = ProcessingPhase::Transcribing;
        self.active_job = Some(id);
        true
    }

    fn enhancement_started(&mut self, id: u64) -> bool {
        if self.active_job != Some(id) {
            return false;
        }
        let Some(phase) = self.jobs.get_mut(&id) else {
            return false;
        };
        if *phase != ProcessingPhase::Transcribing {
            return false;
        }
        *phase = ProcessingPhase::Enhancing;
        true
    }

    fn job_finished(&mut self, id: u64) -> bool {
        if self.jobs.remove(&id).is_none() {
            return false;
        }
        if self.active_job == Some(id) {
            self.active_job = None;
        }
        true
    }

    fn show_notice(&mut self, kind: NoticeKind, message: impl Into<String>, now: Instant) {
        let duration = match kind {
            NoticeKind::Success => Duration::from_millis(2500),
            NoticeKind::Info => Duration::from_secs(3),
            NoticeKind::Warning
            | NoticeKind::RawFallback
            | NoticeKind::ProviderWarning
            | NoticeKind::DeliveryWarning => Duration::from_secs(4),
            NoticeKind::Error => Duration::from_secs(5),
        };
        self.notice = Some(OverlayNotice {
            kind,
            message: message.into(),
            expires_at: now + duration,
        });
    }

    fn desired_view(&mut self, now: Instant) -> OverlayViewState {
        if self
            .notice
            .as_ref()
            .is_some_and(|notice| now >= notice.expires_at)
        {
            self.notice = None;
        }
        if let Some(intent) = self.recording {
            return OverlayViewState::listening(intent);
        }
        if let Some(id) = self.active_job {
            if let Some(phase) = self.jobs.get(&id) {
                let queued = self
                    .jobs
                    .values()
                    .filter(|phase| **phase == ProcessingPhase::Queued)
                    .count();
                return OverlayViewState::processing(id, *phase, queued);
            }
        }
        let queued = self
            .jobs
            .iter()
            .filter(|(_, phase)| **phase == ProcessingPhase::Queued)
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        if let Some(id) = queued.first() {
            return OverlayViewState::processing(
                *id,
                ProcessingPhase::Queued,
                queued.len().saturating_sub(1),
            );
        }
        if let Some(notice) = &self.notice {
            return OverlayViewState::notice(notice.kind, notice.message.clone());
        }
        OverlayViewState::Hidden
    }

    fn sync(&mut self, controller: &overlay::OverlayController, now: Instant) {
        let view = self.desired_view(now);
        if view != self.last_view {
            if let Err(error) = controller.set_state(view.clone()) {
                warn!(error, "failed to update overlay state");
            } else {
                self.last_view = view;
            }
        }
    }
}

struct State {
    settings: Settings,
    credential_health: CredentialHealth,
    settings_error: Option<String>,
    recorder: Recorder,
    tray: tray::Tray,
    overlay: overlay::OverlayController,
    overlay_state: OverlayReducer,
    active_recording: Option<ActiveRecording>,
    worker: Worker,
    engine_state: EngineState,
    model_state: Option<ModelState>,
    load_generation: u64,
    pending_jobs: usize,
    active_job: Option<u64>,
    shutting_down: bool,
    discarding_jobs: bool,
    hk_stop: Arc<AtomicBool>,
    delivery_interlock: Arc<paste::DeliveryInterlock>,
    settings_revision: Option<SettingsRevision>,
    proxy: EventLoopProxy<UserEvent>,
    diagnostics_exporting: bool,
}

pub fn run() -> Result<()> {
    let (settings, credential_health, settings_error, settings_revision) =
        match Settings::load_runtime() {
            Ok(loaded) => (
                loaded.settings,
                loaded.credential_health,
                loaded.settings_error,
                Some(loaded.revision),
            ),
            Err(error) => {
                warn!(error = %error, "settings load failed; using defaults until a valid file is saved");
                (
                    Settings::default(),
                    CredentialHealth::default(),
                    Some(format!("settings could not be loaded: {error}")),
                    Settings::revision().ok(),
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
    let delivery_interlock = Arc::new(paste::DeliveryInterlock::default());
    spawn_hotkey_thread(
        settings.clone(),
        settings_revision.clone(),
        proxy.clone(),
        hk_stop.clone(),
        delivery_interlock.clone(),
    )?;
    spawn_tick_thread(proxy.clone());

    // Visual recording indicator on its own thread + own Win32 message loop.
    // It only reads atomics from the recorder, so there's no shared state with
    // the main winit/tray loop that could deadlock.
    let overlay = overlay::spawn(
        overlay::OverlayHandles {
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
        overlay_state: OverlayReducer::default(),
        active_recording: None,
        worker,
        engine_state: EngineState::NotStarted,
        model_state: None,
        load_generation: 0,
        pending_jobs: 0,
        active_job: None,
        shutting_down: false,
        discarding_jobs: false,
        hk_stop,
        delivery_interlock,
        settings_revision,
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
    initial_revision: Option<SettingsRevision>,
    proxy: EventLoopProxy<UserEvent>,
    stop: Arc<AtomicBool>,
    delivery_interlock: Arc<paste::DeliveryInterlock>,
) -> Result<()> {
    thread::Builder::new()
        .name("hotkey".into())
        .spawn(move || hotkey_loop(initial, initial_revision, proxy, stop, delivery_interlock))?;
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

fn hotkey_loop(
    initial: Settings,
    initial_revision: Option<SettingsRevision>,
    proxy: EventLoopProxy<UserEvent>,
    stop: Arc<AtomicBool>,
    delivery_interlock: Arc<paste::DeliveryInterlock>,
) {
    let mut settings = initial;
    let mut settings_revision = initial_revision;
    let mut last_settings_error: Option<String> = None;
    let mut next_press_epoch = 0_u64;
    // Try OS-level registration so other apps know the combo is taken.
    // If that fails (e.g. Windows already reserved it), fall through to
    // physical key-hook polling, which does not require RegisterHotKey.
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

    'polling: while !stop.load(Ordering::Relaxed) {
        if let Some(ev) = hotkey::poll_combo(trigger_vk, &mod_vks, &mut poll_state) {
            let mut posted_press_epoch = None;
            let user_ev = match ev {
                hotkey::Event::Press { shift_down } => {
                    let epoch =
                        mark_press_pending(delivery_interlock.as_ref(), &mut next_press_epoch);
                    posted_press_epoch = Some(epoch);
                    UserEvent::HotkeyPress {
                        epoch,
                        shift_down,
                        delivery_target: capture_delivery_target(),
                    }
                }
                hotkey::Event::Release => UserEvent::HotkeyRelease,
            };
            if proxy.send_event(user_ev).is_err() {
                if let Some(epoch) = posted_press_epoch {
                    consume_press_pending(delivery_interlock.as_ref(), epoch);
                }
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
            if !emit_release_before_poll_reset(&proxy, &mut poll_state) {
                break;
            }
            key_hook::request_reinstall();
        }

        if last_check.elapsed() >= Duration::from_millis(500) {
            last_check = Instant::now();
            match Settings::revision() {
                Ok(revision) if settings_revision.as_ref() != Some(&revision) => {
                    settings_revision = Some(revision);
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
                                if !emit_release_before_poll_reset(&proxy, &mut poll_state) {
                                    break 'polling;
                                }
                                drop(_mgr.take());
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
                Ok(_) => {}
                Err(error) => {
                    let message = error.to_string();
                    if last_settings_error.as_deref() != Some(message.as_str()) {
                        warn!(error = %message, "hotkey settings revision check failed; keeping the last valid hotkey");
                        last_settings_error = Some(message);
                    }
                }
            }
        }

        thread::sleep(Duration::from_millis(8));
    }
    info!("hotkey thread exiting");
}

fn emit_release_before_poll_reset(
    proxy: &EventLoopProxy<UserEvent>,
    poll_state: &mut hotkey::PollState,
) -> bool {
    match hotkey::release_before_reset(poll_state) {
        Some(hotkey::Event::Release) => proxy.send_event(UserEvent::HotkeyRelease).is_ok(),
        Some(hotkey::Event::Press { .. }) => unreachable!("reset can only synthesize release"),
        None => true,
    }
}

fn mark_press_pending(interlock: &paste::DeliveryInterlock, next_epoch: &mut u64) -> u64 {
    interlock.mark_press_pending(next_epoch)
}

fn consume_press_pending(interlock: &paste::DeliveryInterlock, epoch: u64) {
    interlock.consume_press(epoch);
}

fn recording_or_press_pending(
    recording_active: bool,
    interlock: &paste::DeliveryInterlock,
) -> bool {
    recording_active || interlock.press_pending()
}

fn capture_delivery_target() -> Option<DeliveryTarget> {
    use windows::Win32::UI::WindowsAndMessaging::{
        GetAncestor, GetForegroundWindow, GetWindowThreadProcessId, GA_ROOT,
    };

    let foreground = unsafe { GetForegroundWindow() };
    if foreground.0.is_null() {
        return None;
    }
    let hwnd = unsafe { GetAncestor(foreground, GA_ROOT) };
    if hwnd.0.is_null() {
        return None;
    }
    let mut process_id = 0;
    unsafe {
        GetWindowThreadProcessId(hwnd, Some(&mut process_id));
    }
    if process_id == 0 {
        return None;
    }
    let process_creation_time_100ns = process_creation_time_100ns(process_id)?;
    Some(DeliveryTarget {
        hwnd: hwnd.0 as isize,
        process_id,
        process_creation_time_100ns: Some(process_creation_time_100ns),
    })
}

fn delivery_target_is_current(target: DeliveryTarget) -> bool {
    capture_delivery_target().is_some_and(|current| current == target)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeliveryBlockReason {
    RecordingActive,
    TargetChanged,
    Cancelled,
    ClipboardChanged,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeliveryPlan {
    Paste,
    ClipboardExplicit,
    ClipboardSafety(DeliveryBlockReason),
}

fn decide_delivery(
    auto_paste: bool,
    recording_active: bool,
    target_present: bool,
    target_matches: bool,
    cancelled_or_tombstoned: bool,
) -> DeliveryPlan {
    if !auto_paste {
        return DeliveryPlan::ClipboardExplicit;
    }
    if cancelled_or_tombstoned {
        return DeliveryPlan::ClipboardSafety(DeliveryBlockReason::Cancelled);
    }
    if recording_active {
        return DeliveryPlan::ClipboardSafety(DeliveryBlockReason::RecordingActive);
    }
    if !target_present || !target_matches {
        return DeliveryPlan::ClipboardSafety(DeliveryBlockReason::TargetChanged);
    }
    DeliveryPlan::Paste
}

fn cleanup_notice_kind(reason: &FallbackReason) -> NoticeKind {
    match reason {
        FallbackReason::UnknownProvider
        | FallbackReason::MissingCredential
        | FallbackReason::CredentialTargetChanged
        | FallbackReason::InvalidEndpoint
        | FallbackReason::EmptyModelId
        | FallbackReason::RequestFailed
        | FallbackReason::ResponseUnparseable => NoticeKind::ProviderWarning,
        FallbackReason::EmptyCandidate
        | FallbackReason::IntegrityRejected(_)
        | FallbackReason::ValidatorError => NoticeKind::RawFallback,
    }
}

fn process_creation_time_100ns(process_id: u32) -> Option<u64> {
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
    use windows_sys::Win32::System::Threading::{
        GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, process_id) };
    if process.is_null() {
        return None;
    }
    let empty_time = || FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let mut creation = empty_time();
    let mut exit = empty_time();
    let mut kernel = empty_time();
    let mut user = empty_time();
    let success =
        unsafe { GetProcessTimes(process, &mut creation, &mut exit, &mut kernel, &mut user) } != 0;
    unsafe {
        let _ = CloseHandle(process);
    }
    success.then(|| ((creation.dwHighDateTime as u64) << 32) | creation.dwLowDateTime as u64)
}

struct AppHandler {
    state: State,
}

impl AppHandler {
    fn sync_overlay(&mut self) {
        self.state
            .overlay_state
            .sync(&self.state.overlay, Instant::now());
    }

    fn show_overlay_notice(&mut self, kind: NoticeKind, message: impl Into<String>) {
        self.state
            .overlay_state
            .show_notice(kind, message, Instant::now());
        self.sync_overlay();
    }

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
            UserEvent::HotkeyPress {
                epoch,
                shift_down,
                delivery_target,
            } => {
                consume_press_pending(self.state.delivery_interlock.as_ref(), epoch);
                self.on_press(shift_down, delivery_target);
            }
            UserEvent::HotkeyRelease => self.on_release(),
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
        self.sync_overlay();

        if self.state.active_recording.is_some() {
            let stream_failed = self.state.recorder.stream_failed();
            let stream_stalled = self
                .state
                .recorder
                .callback_stalled(Duration::from_millis(500));
            if stream_failed || stream_stalled {
                let max_record_seconds = self
                    .state
                    .active_recording
                    .as_ref()
                    .map(|recording| recording.settings.max_record_seconds)
                    .expect("active recording disappeared");
                match self.state.recorder.recover_stream(max_record_seconds) {
                    Ok(capture) => {
                        warn!(
                            stream_failed,
                            stream_stalled,
                            device = %capture.device_name,
                            "capture stream recovered during active recording"
                        );
                    }
                    Err(error) => {
                        warn!(
                            stream_failed,
                            stream_stalled,
                            error = %error,
                            "capture stream recovery failed"
                        );
                        self.state.recorder.mark_stream_failed(error.to_string());
                        self.finish_recording(false);
                    }
                }
            } else {
                let active_recording = self
                    .state
                    .active_recording
                    .as_ref()
                    .expect("active recording disappeared");
                let max_seconds = active_recording.settings.max_record_seconds;
                let timer_limit_reached = self
                    .state
                    .active_recording
                    .as_ref()
                    .map(|recording| recording.started.elapsed().as_secs_f32() >= max_seconds)
                    .unwrap_or(false);
                if self.state.recorder.limit_reached() || timer_limit_reached {
                    self.finish_recording(true);
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
                continue;
            }
            if ev.id == self.state.tray.menu_about_id {
                info!("opening About UI subprocess");
                if let Ok(exe) = std::env::current_exe() {
                    thread::spawn(move || {
                        let _ = std::process::Command::new(exe)
                            .arg("--about")
                            .stdin(Stdio::null())
                            .stdout(Stdio::null())
                            .stderr(Stdio::null())
                            .creation_flags(DETACHED_PROCESS | CREATE_NO_WINDOW)
                            .spawn();
                    });
                }
            }
        }

        // Settings hot-reload compares content revisions so deletion is a
        // first-class change and both loops return to the same defaults.
        {
            match Settings::revision() {
                Ok(revision) if self.state.settings_revision.as_ref() != Some(&revision) => {
                    match Settings::load_runtime() {
                        Ok(loaded) => {
                            self.state.settings_revision = Some(loaded.revision.clone());
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
                                    self.sync_overlay();
                                }
                                let new_engine = self.state.settings.engine.clone();
                                if new_engine != old_engine {
                                    info!(
                                        "engine changed: {old_engine} -> {new_engine}; reloading"
                                    );
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
                Ok(_) => {}
                Err(error) => {
                    let message = format!("settings revision could not be read: {error}");
                    if self.state.settings_error.as_deref() != Some(message.as_str()) {
                        warn!(error = %error, "settings revision check failed; keeping the last valid settings");
                        self.state.settings_error = Some(message);
                        self.update_job_status();
                    }
                }
            }
        }
    }

    fn on_press(&mut self, shift_down: bool, delivery_target: Option<DeliveryTarget>) {
        if self.state.shutting_down || self.state.active_recording.is_some() {
            return;
        }
        if !matches!(self.state.engine_state, EngineState::Ready { .. }) {
            let (color, status) = self.blocked_recording_status();
            self.state.tray.set_status(color, &status);
            self.show_overlay_notice(
                NoticeKind::Error,
                status
                    .strip_prefix("OpenWritr — ")
                    .unwrap_or(&status)
                    .to_string(),
            );
            info!(status, "recording blocked while engine is unavailable");
            return;
        }

        if self.state.active_recording.is_none() {
            let settings = self.state.settings.clone();
            let shift_is_modifier = settings
                .hotkey_modifiers
                .iter()
                .any(|modifier| modifier == "shift");
            let intent = recording_intent(settings.enhance.mode, shift_is_modifier, shift_down);
            match self.state.recorder.start(settings.max_record_seconds) {
                Ok(capture) => {
                    self.state.tray.set_status(
                        tray::IconColor::Recording,
                        &format!("OpenWritr — recording from {}", capture.device_name),
                    );
                    let active_recording = ActiveRecording::new(settings, intent, delivery_target);
                    if active_recording.settings.sounds {
                        sounds::play_start();
                    }
                    self.state.active_recording = Some(active_recording);
                    self.state.overlay_state.recording_started(intent);
                    self.sync_overlay();
                    info!(
                        device = %capture.device_name,
                        sample_rate = capture.sample_rate,
                        channels = capture.channels,
                        ?intent,
                        ?delivery_target,
                        "recording start"
                    );
                }
                Err(e) => {
                    self.state.tray.set_status(
                        tray::IconColor::Error,
                        "OpenWritr — microphone unavailable (see log)",
                    );
                    self.show_overlay_notice(NoticeKind::Error, "Microphone unavailable");
                    warn!(error = %e, "failed to start recording");
                }
            }
        }
    }

    fn on_release(&mut self) {
        if self.state.active_recording.is_some() {
            self.finish_recording(false);
        }
    }

    fn finish_recording(&mut self, timer_limit_reached: bool) {
        let Some(active_recording) = self.state.active_recording.take() else {
            return;
        };
        self.state.overlay_state.recording_finished();
        let recording = match self.state.recorder.stop() {
            Ok(recording) => recording,
            Err(e) => {
                self.state.tray.set_status(
                    tray::IconColor::Error,
                    "OpenWritr — microphone error (see log)",
                );
                self.show_overlay_notice(NoticeKind::Error, "Microphone error");
                warn!(error = %e, "failed to stop recording");
                return;
            }
        };
        if active_recording.settings.sounds {
            sounds::play_stop();
        }
        if let Some(error) = recording.stream_error {
            self.state.tray.set_status(
                tray::IconColor::Error,
                "OpenWritr — microphone stream failed (see log)",
            );
            self.show_overlay_notice(NoticeKind::Error, "Microphone stream failed");
            warn!(error = %error, "recording aborted after stream failure");
            return;
        }

        self.state
            .tray
            .set_status(tray::IconColor::Idle, "OpenWritr");
        let reached_limit = timer_limit_reached || recording.reached_limit;
        if reached_limit {
            info!(
                max_seconds = active_recording.settings.max_record_seconds,
                "maximum recording duration reached"
            );
        }

        let dur = active_recording.started.elapsed();
        let min = active_recording.settings.min_record_seconds;
        if dur.as_secs_f32() < min {
            info!(secs = dur.as_secs_f32(), "below min — discarded");
            self.sync_overlay();
        } else {
            info!(
                device = %recording.device_name,
                sample_rate = recording.sample_rate,
                channels = recording.channels,
                samples = recording.samples.len(),
                drained_samples = recording.drained_samples,
                stream_recoveries = recording.stream_recoveries,
                reached_limit,
                "recording stop"
            );
            self.dispatch_transcribe(recording.samples, recording.sample_rate, active_recording);
        }
    }
}

fn recording_intent(
    mode: EnhanceMode,
    shift_is_modifier: bool,
    shift_is_down: bool,
) -> RecordingIntent {
    let additional_shift_is_down = shift_is_down && !shift_is_modifier;
    match mode {
        EnhanceMode::Never => RecordingIntent::Raw,
        EnhanceMode::WithShift if additional_shift_is_down => RecordingIntent::Enhance,
        EnhanceMode::WithShift => RecordingIntent::Raw,
        EnhanceMode::Always if additional_shift_is_down => RecordingIntent::Raw,
        EnhanceMode::Always => RecordingIntent::Enhance,
    }
}

impl AppHandler {
    fn dispatch_transcribe(
        &mut self,
        samples: Vec<f32>,
        sr: u32,
        active_recording: ActiveRecording,
    ) {
        if self.state.shutting_down {
            warn!("discarding completed recording because shutdown has started");
            self.sync_overlay();
            return;
        }
        let config = JobConfig::new(
            active_recording.settings,
            active_recording.intent,
            active_recording.delivery_target,
            false,
        );
        match self.state.worker.enqueue(samples, sr, config) {
            Ok(id) => {
                self.state.pending_jobs = self.state.pending_jobs.saturating_add(1);
                self.state.overlay_state.job_queued(id);
                info!(
                    id,
                    queue_depth = self.state.pending_jobs,
                    "transcription job queued"
                );
                self.update_job_status();
                self.sync_overlay();
            }
            Err(error) => {
                warn!(error = %error, "failed to queue transcription");
                self.state.tray.set_status(
                    tray::IconColor::Error,
                    "OpenWritr — could not queue transcription (see log)",
                );
                self.show_overlay_notice(NoticeKind::Error, "Could not queue transcription");
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
                if !self.state.overlay_state.job_started(id) {
                    warn!(id, "ignoring start event for unknown or inactive job");
                    return;
                }
                self.state.active_job = Some(id);
                self.update_job_status();
                self.sync_overlay();
            }
            WorkerEvent::EnhancementStarted { id } => {
                if !self.state.overlay_state.enhancement_started(id) {
                    warn!(id, "ignoring enhancement phase for inactive job");
                    return;
                }
                self.state.tray.set_status(
                    tray::IconColor::Transcribing,
                    &format!(
                        "OpenWritr — enhancing; {} job(s) remaining",
                        self.state.pending_jobs
                    ),
                );
                self.sync_overlay();
            }
            WorkerEvent::JobCompleted {
                id,
                text,
                auto_paste,
                enhancement_warning,
                delivery_target,
            } => {
                if !self.finish_job(id) {
                    warn!(id, "ignoring completion for tombstoned or unknown job");
                    return;
                }
                self.update_job_status();
                self.sync_overlay();
                if self.state.discarding_jobs {
                    warn!(id, "suppressing completed job during discard shutdown");
                    return;
                }
                if text.is_empty() {
                    return;
                }
                let target_matches = delivery_target
                    .map(delivery_target_is_current)
                    .unwrap_or(false);
                let recording_active = recording_or_press_pending(
                    self.state.active_recording.is_some(),
                    self.state.delivery_interlock.as_ref(),
                );
                let plan = decide_delivery(
                    auto_paste,
                    recording_active,
                    delivery_target.is_some(),
                    target_matches,
                    false,
                );
                let (mode, safety_warning) = match plan {
                    DeliveryPlan::Paste => (paste::DeliveryMode::Paste, None),
                    DeliveryPlan::ClipboardExplicit => (paste::DeliveryMode::Clipboard, None),
                    DeliveryPlan::ClipboardSafety(reason) => {
                        (paste::DeliveryMode::Clipboard, Some(reason))
                    }
                };
                let delivery = match mode {
                    paste::DeliveryMode::Paste => paste::deliver_guarded(
                        &text,
                        mode,
                        Some(self.state.delivery_interlock.as_ref()),
                        || {
                            delivery_target
                                .map(delivery_target_is_current)
                                .unwrap_or(false)
                        },
                    ),
                    paste::DeliveryMode::Clipboard => paste::deliver(&text, mode),
                };
                match delivery {
                    Ok(paste::DeliveryOutcome::Pasted) => {
                        if let Some(reason) = enhancement_warning {
                            self.show_cleanup_fallback(reason, "pasted");
                        } else {
                            self.show_job_success("Transcript pasted");
                        }
                    }
                    Ok(paste::DeliveryOutcome::Copied) => {
                        let guarded_paste_rejected =
                            auto_paste && matches!(plan, DeliveryPlan::Paste);
                        if let Some(reason) = safety_warning.or_else(|| {
                            guarded_paste_rejected.then(|| {
                                if recording_or_press_pending(
                                    self.state.active_recording.is_some(),
                                    self.state.delivery_interlock.as_ref(),
                                ) {
                                    DeliveryBlockReason::RecordingActive
                                } else {
                                    DeliveryBlockReason::TargetChanged
                                }
                            })
                        }) {
                            self.show_delivery_fallback(reason);
                        } else if let Some(reason) = enhancement_warning {
                            self.show_cleanup_fallback(reason, "copied");
                        } else {
                            self.state.tray.set_status(
                                tray::IconColor::Idle,
                                "OpenWritr — copied transcript to clipboard",
                            );
                            self.show_overlay_notice(
                                NoticeKind::Success,
                                "Copied transcript to clipboard",
                            );
                        }
                    }
                    Ok(paste::DeliveryOutcome::CopiedWithWarning { warning, detail }) => {
                        warn!(?warning, detail, "automatic paste fell back to clipboard");
                        let summary = match warning {
                            paste::DeliveryWarning::ClipboardPreparationFailed => {
                                "Clipboard preservation was unavailable"
                            }
                            paste::DeliveryWarning::KeyInjectionFailed => {
                                "Keyboard paste was unavailable"
                            }
                        };
                        self.show_job_warning_kind(
                            NoticeKind::DeliveryWarning,
                            &format!(
                                "{summary}; transcript copied to clipboard: {}",
                                short_error(&detail)
                            ),
                        );
                    }
                    Ok(paste::DeliveryOutcome::CancelledClipboardChanged) => {
                        self.show_delivery_fallback(DeliveryBlockReason::ClipboardChanged);
                    }
                    Err(error) => self.show_job_warning(&format!(
                        "Transcript delivery failed: {}",
                        short_error(&error.to_string())
                    )),
                }
            }
            WorkerEvent::JobFailed { id, error } => {
                if !self.finish_job(id) {
                    warn!(id, "ignoring failure for tombstoned or unknown job");
                    return;
                }
                warn!(id, error = %error, "transcription job failed");
                if self.state.pending_jobs == 0 {
                    self.state.tray.set_status(
                        tray::IconColor::Error,
                        &format!("OpenWritr — transcription failed: {}", short_error(&error)),
                    );
                } else {
                    self.update_job_status();
                }
                self.show_overlay_notice(
                    NoticeKind::Error,
                    format!("Transcription failed: {}", short_error(&error)),
                );
            }
            WorkerEvent::JobDiscarded { id } => {
                if !self.finish_job(id) {
                    return;
                }
                info!(id, "transcription job discarded");
                self.update_job_status();
                self.sync_overlay();
            }
            WorkerEvent::ShutdownComplete => {
                info!("worker shutdown complete");
                self.exit_after_clipboard_settles(el);
            }
        }
    }

    fn finish_job(&mut self, id: u64) -> bool {
        if !self.state.overlay_state.job_finished(id) {
            return false;
        }
        self.state.pending_jobs = self.state.pending_jobs.saturating_sub(1);
        if self.state.active_job == Some(id) {
            self.state.active_job = None;
        }
        true
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

    fn show_job_success(&mut self, message: &str) {
        self.state
            .tray
            .set_status(tray::IconColor::Idle, &format!("OpenWritr — {message}"));
        self.show_overlay_notice(NoticeKind::Success, message);
    }

    fn show_job_warning(&mut self, message: &str) {
        self.show_job_warning_kind(NoticeKind::Warning, message);
    }

    fn show_job_warning_kind(&mut self, kind: NoticeKind, message: &str) {
        self.state
            .tray
            .set_status(tray::IconColor::Error, &format!("OpenWritr — {message}"));
        self.show_overlay_notice(kind, message);
    }

    fn show_cleanup_fallback(&mut self, reason: FallbackReason, action: &str) {
        let kind = cleanup_notice_kind(&reason);
        let message = match reason {
            FallbackReason::UnknownProvider => {
                format!("Provider is invalid; raw transcript was {action}")
            }
            FallbackReason::MissingCredential => {
                format!("Credential unavailable; raw transcript was {action}")
            }
            FallbackReason::CredentialTargetChanged => {
                format!("Provider changed after recording; raw transcript was {action}")
            }
            FallbackReason::InvalidEndpoint => {
                format!("Provider endpoint is invalid; raw transcript was {action}")
            }
            FallbackReason::EmptyModelId => {
                format!("Provider model is invalid; raw transcript was {action}")
            }
            FallbackReason::RequestFailed => {
                format!("Provider request failed; raw transcript was {action}")
            }
            FallbackReason::ResponseUnparseable => {
                format!("Provider response was invalid; raw transcript was {action}")
            }
            FallbackReason::EmptyCandidate => {
                format!("Cleanup returned no text; raw transcript was {action}")
            }
            FallbackReason::IntegrityRejected(_) => {
                format!("Cleanup changed critical content; raw transcript was {action}")
            }
            FallbackReason::ValidatorError => {
                format!("Cleanup validation failed; raw transcript was {action}")
            }
        };
        self.show_job_warning_kind(kind, &message);
    }

    fn show_delivery_fallback(&mut self, reason: DeliveryBlockReason) {
        let message = match reason {
            DeliveryBlockReason::RecordingActive => {
                "Another recording is active; transcript copied to clipboard"
            }
            DeliveryBlockReason::TargetChanged => {
                "Target window changed; transcript copied to clipboard"
            }
            DeliveryBlockReason::Cancelled => {
                "Delivery was cancelled; transcript copied to clipboard"
            }
            DeliveryBlockReason::ClipboardChanged => {
                "Delivery cancelled because the clipboard changed; newer clipboard contents were preserved"
            }
        };
        self.show_job_warning_kind(NoticeKind::DeliveryWarning, message);
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
        if self.state.active_recording.take().is_some() {
            let _ = self.state.recorder.stop();
            self.state.overlay_state.recording_finished();
            info!("active recording discarded during shutdown");
        }

        let mode = if self.state.pending_jobs > 0 {
            prompt_shutdown_mode(self.state.pending_jobs)
        } else {
            ShutdownMode::Discard
        };
        self.state.discarding_jobs = mode == ShutdownMode::Discard;
        if mode == ShutdownMode::Discard {
            paste::cancel_pending_restorations();
        }
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
            self.exit_after_clipboard_settles(el);
        }
    }

    fn exit_after_clipboard_settles(&self, el: &ActiveEventLoop) {
        if !paste::wait_for_pending_restorations(Duration::from_secs(2)) {
            warn!("clipboard restoration did not settle before the shutdown timeout");
        }
        el.exit();
    }
}

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

#[cfg(test)]
mod tests {
    use super::{
        cleanup_notice_kind, consume_press_pending, decide_delivery, mark_press_pending,
        recording_intent, recording_or_press_pending, ActiveRecording, DeliveryBlockReason,
        DeliveryPlan, OverlayReducer,
    };
    use crate::cleanup::{EndpointScope, FallbackReason, PromptSource, PromptTarget};
    use crate::overlay::{ListeningIntent, NoticeKind, OverlayViewState, ProcessingPhase};
    use crate::paste::DeliveryInterlock;
    use crate::settings::{EnhanceMode, Settings};
    use crate::worker::RecordingIntent;
    use std::time::Instant;

    #[test]
    fn recording_intent_covers_the_full_shift_truth_table() {
        use RecordingIntent::{Enhance, Raw};

        let cases = [
            (EnhanceMode::Never, false, false, Raw),
            (EnhanceMode::Never, false, true, Raw),
            (EnhanceMode::Never, true, false, Raw),
            (EnhanceMode::Never, true, true, Raw),
            (EnhanceMode::WithShift, false, false, Raw),
            (EnhanceMode::WithShift, false, true, Enhance),
            (EnhanceMode::WithShift, true, false, Raw),
            (EnhanceMode::WithShift, true, true, Raw),
            (EnhanceMode::Always, false, false, Enhance),
            (EnhanceMode::Always, false, true, Raw),
            (EnhanceMode::Always, true, false, Enhance),
            (EnhanceMode::Always, true, true, Enhance),
        ];

        for (mode, shift_is_modifier, shift_is_down, expected) in cases {
            assert_eq!(
                recording_intent(mode, shift_is_modifier, shift_is_down),
                expected
            );
        }
    }

    #[test]
    fn active_recording_keeps_the_press_time_settings_snapshot() {
        let mut live = Settings::default();
        live.auto_paste = false;
        live.min_record_seconds = 0.5;
        live.max_record_seconds = 12.0;
        live.enhance.mode = EnhanceMode::Always;
        live.enhance.provider = "openai_compatible".into();
        live.enhance.base_url = "https://snapshot.example/v1".into();
        live.enhance.model = "snapshot-model".into();
        let target = PromptTarget::openai_compatible(
            EndpointScope::parse(&live.enhance.base_url).unwrap(),
            &live.enhance.model,
        )
        .unwrap();
        live.prompt_overrides
            .set(target.clone(), "snapshot prompt".into());

        let active = ActiveRecording::new(live.clone(), RecordingIntent::Enhance, None);
        live.auto_paste = true;
        live.min_record_seconds = 2.0;
        live.max_record_seconds = 30.0;
        live.enhance.mode = EnhanceMode::Never;
        live.enhance.provider = "github_copilot".into();
        live.enhance.base_url = "https://changed.example/v1".into();
        live.enhance.model = "changed-model".into();

        assert!(!active.settings.auto_paste);
        assert_eq!(active.settings.min_record_seconds, 0.5);
        assert_eq!(active.settings.max_record_seconds, 12.0);
        assert_eq!(active.settings.enhance.mode, EnhanceMode::Always);
        assert_eq!(active.settings.prompt_target().unwrap(), target);
        let resolved = active.settings.resolve_prompt(&target);
        assert_eq!(resolved.source, PromptSource::CustomOverride);
        assert_eq!(resolved.system, "snapshot prompt");
        assert_eq!(active.intent, RecordingIntent::Enhance);
    }

    #[test]
    fn overlay_reducer_enforces_recording_job_queue_notice_priority() {
        let now = Instant::now();
        let mut reducer = OverlayReducer::default();
        reducer.show_notice(NoticeKind::Warning, "old notice", now);
        reducer.job_queued(10);
        reducer.job_queued(11);
        assert_eq!(
            reducer.desired_view(now),
            OverlayViewState::processing(10, ProcessingPhase::Queued, 1)
        );

        assert!(reducer.job_started(10));
        assert_eq!(
            reducer.desired_view(now),
            OverlayViewState::processing(10, ProcessingPhase::Transcribing, 1)
        );

        reducer.recording_started(RecordingIntent::Enhance);
        assert_eq!(
            reducer.desired_view(now),
            OverlayViewState::listening(ListeningIntent::Enhance)
        );
        reducer.recording_finished();
        assert!(reducer.enhancement_started(10));
        assert_eq!(
            reducer.desired_view(now),
            OverlayViewState::processing(10, ProcessingPhase::Enhancing, 1)
        );

        assert!(reducer.job_finished(10));
        assert_eq!(
            reducer.desired_view(now),
            OverlayViewState::processing(11, ProcessingPhase::Queued, 0)
        );
        assert!(reducer.job_finished(11));
        assert_eq!(
            reducer.desired_view(now),
            OverlayViewState::notice(NoticeKind::Warning, "old notice")
        );
    }

    #[test]
    fn old_enhancement_and_terminal_events_cannot_clobber_newer_activity() {
        let now = Instant::now();
        let mut reducer = OverlayReducer::default();
        reducer.job_queued(1);
        reducer.job_queued(2);
        assert!(reducer.job_started(1));
        assert!(reducer.job_finished(1));
        assert!(reducer.job_started(2));

        assert!(!reducer.enhancement_started(1));
        assert!(!reducer.job_finished(1));
        assert_eq!(
            reducer.desired_view(now),
            OverlayViewState::processing(2, ProcessingPhase::Transcribing, 0)
        );
    }

    #[test]
    fn delivery_safety_falls_back_for_target_change_and_cancellation() {
        assert_eq!(
            decide_delivery(true, false, true, false, false),
            DeliveryPlan::ClipboardSafety(DeliveryBlockReason::TargetChanged)
        );
        assert_eq!(
            decide_delivery(true, false, true, true, true),
            DeliveryPlan::ClipboardSafety(DeliveryBlockReason::Cancelled)
        );
        assert_eq!(
            decide_delivery(true, true, true, true, false),
            DeliveryPlan::ClipboardSafety(DeliveryBlockReason::RecordingActive)
        );
        assert_eq!(
            decide_delivery(false, true, false, false, true),
            DeliveryPlan::ClipboardExplicit
        );
        assert_eq!(
            decide_delivery(true, false, true, true, false),
            DeliveryPlan::Paste
        );
    }

    #[test]
    fn pending_press_blocks_older_completion_until_app_consumes_its_epoch() {
        let pending = DeliveryInterlock::default();
        let mut next_epoch = 0;
        let epoch = mark_press_pending(&pending, &mut next_epoch);

        assert!(recording_or_press_pending(false, &pending));
        assert_eq!(
            decide_delivery(
                true,
                recording_or_press_pending(false, &pending),
                true,
                true,
                false,
            ),
            DeliveryPlan::ClipboardSafety(DeliveryBlockReason::RecordingActive)
        );

        consume_press_pending(&pending, epoch);
        assert!(!recording_or_press_pending(false, &pending));
    }

    #[test]
    fn consuming_an_older_press_epoch_does_not_clear_a_newer_pending_press() {
        let pending = DeliveryInterlock::default();
        let mut next_epoch = 0;
        let first = mark_press_pending(&pending, &mut next_epoch);
        let second = mark_press_pending(&pending, &mut next_epoch);

        consume_press_pending(&pending, first);

        assert!(pending.press_pending());
        consume_press_pending(&pending, second);
        assert!(!pending.press_pending());
    }

    #[test]
    fn cleanup_fallback_notices_distinguish_provider_and_validator_failures() {
        assert_eq!(
            cleanup_notice_kind(&FallbackReason::MissingCredential),
            NoticeKind::ProviderWarning
        );
        assert_eq!(
            cleanup_notice_kind(&FallbackReason::RequestFailed),
            NoticeKind::ProviderWarning
        );
        assert_eq!(
            cleanup_notice_kind(&FallbackReason::ValidatorError),
            NoticeKind::RawFallback
        );
        assert_eq!(
            cleanup_notice_kind(&FallbackReason::IntegrityRejected(vec![])),
            NoticeKind::RawFallback
        );
    }
}
