//! Visual recording indicator (macOS-style centered meter).
//!
//! Runs on its own dedicated thread with its own Win32 message loop.
//! Presentation is driven entirely by a small typed API
//! (`OverlayCommand::SetState` / `SetEnabled`) plus a single RMS atomic
//! (`level_x10000`) for waveform animation — the renderer never inspects
//! recorder/app lifecycle atomics to decide what to show. It shares NO other
//! state with the tray's winit loop — so it can never deadlock the main app,
//! no matter what happens here.
//!
//! Look: a horizontal pill near the bottom-center. While listening or
//! processing a job it shows a row of vertical bars whose heights breathe
//! with the audio level (or a steady pulse once there's no live audio),
//! plus a small caption. Transient notices instead show word-wrapped text.
//! Gaussian envelope makes the center bars react strongest, with a per-bar
//! wave shimmer.

use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    mpsc::{self, Receiver, Sender, TryRecvError},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::{COLORREF, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::{
    BeginPaint, BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, CreatePen, CreateSolidBrush,
    DeleteDC, DeleteObject, DrawTextW, EndPaint, FillRect, InvalidateRect, Rectangle, RoundRect,
    SelectObject, SetBkMode, SetTextColor, DT_CALCRECT, DT_CENTER, DT_NOPREFIX, DT_SINGLELINE,
    DT_VCENTER, DT_WORDBREAK, HDC, PAINTSTRUCT, PS_SOLID, SRCCOPY, TRANSPARENT,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DispatchMessageW, GetClientRect, GetMessageW,
    GetSystemMetrics, LoadCursorW, PostMessageW, PostQuitMessage, RegisterClassExW,
    SetLayeredWindowAttributes, SetWindowPos, ShowWindow, TranslateMessage, CS_HREDRAW, CS_VREDRAW,
    HCURSOR, HMENU, HWND_TOPMOST, IDC_ARROW, LWA_COLORKEY, MSG, SM_CXSCREEN, SM_CYSCREEN,
    SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SW_HIDE, SW_SHOWNOACTIVATE, WM_CLOSE, WM_CREATE,
    WM_DESTROY, WM_PAINT, WM_USER, WNDCLASSEXW, WS_EX_LAYERED, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW,
    WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_POPUP,
};

const WIN_W: i32 = 320;
const WIN_H: i32 = 60;
const WM_APP_TICK: u32 = WM_USER + 1;
const NBARS: usize = 22;
const BAR_W: i32 = 4;
const BAR_GAP: i32 = 3;
const WAVEFORM_LABEL_GAP: i32 = 14;
const WAVEFORM_LABEL_PADDING: i32 = 4;
/// Hard cap on notice text so a runaway error message can never blow up
/// layout or `DrawTextW`; longer text is truncated with an ellipsis.
const MAX_NOTICE_CHARS: usize = 160;

pub struct OverlayHandles {
    pub level_x10000: Arc<AtomicU32>,
    pub stop: Arc<AtomicBool>,
}

/// What the in-flight recording intends to do with the transcript. Mirrors
/// `worker::RecordingIntent` (Raw/Enhance) in shape, but is defined locally —
/// under a distinct name — so this module has no compile-time dependency on
/// `worker`. The integration layer added later is expected to map
/// `worker::RecordingIntent` to this type when calling `set_state`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListeningIntent {
    Raw,
    Enhance,
}

impl ListeningIntent {
    pub fn is_enhance(self) -> bool {
        matches!(self, ListeningIntent::Enhance)
    }
}

/// Stage of a queued transcription/enhancement job, used by
/// `OverlayViewState::Processing`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingPhase {
    Queued,
    Transcribing,
    Enhancing,
}

/// Visual/semantic category of a transient notice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoticeKind {
    Success,
    Info,
    Warning,
    RawFallback,
    ProviderWarning,
    DeliveryWarning,
    Error,
}

/// The overlay's entire presentation state. The renderer is a pure function
/// of the most recently applied `OverlayViewState` (plus live RMS while
/// listening) — it never infers anything from recorder/app atomics.
#[derive(Debug, Clone, PartialEq)]
pub enum OverlayViewState {
    Hidden,
    Listening {
        enhanced: bool,
    },
    Processing {
        job_id: u64,
        phase: ProcessingPhase,
        /// Jobs still waiting behind this one (0 if it's the only job).
        queue_depth: usize,
    },
    Notice {
        kind: NoticeKind,
        message: String,
    },
}

impl OverlayViewState {
    pub fn listening(intent: ListeningIntent) -> Self {
        OverlayViewState::Listening {
            enhanced: intent.is_enhance(),
        }
    }

    pub fn processing(job_id: u64, phase: ProcessingPhase, queue_depth: usize) -> Self {
        OverlayViewState::Processing {
            job_id,
            phase,
            queue_depth,
        }
    }

    pub fn notice(kind: NoticeKind, message: impl Into<String>) -> Self {
        OverlayViewState::Notice {
            kind,
            message: message.into(),
        }
    }

    pub fn is_hidden(&self) -> bool {
        matches!(self, OverlayViewState::Hidden)
    }
}

/// Commands accepted by the overlay's dedicated thread.
#[derive(Debug, Clone)]
pub enum OverlayCommand {
    SetEnabled(bool),
    SetState(OverlayViewState),
}

#[derive(Clone)]
pub struct OverlayController {
    sender: Sender<OverlayCommand>,
}

impl OverlayController {
    pub fn set_enabled(&self, enabled: bool) -> Result<(), &'static str> {
        self.sender
            .send(OverlayCommand::SetEnabled(enabled))
            .map_err(|_| "overlay command channel is closed")
    }

    pub fn set_state(&self, state: OverlayViewState) -> Result<(), &'static str> {
        self.sender
            .send(OverlayCommand::SetState(state))
            .map_err(|_| "overlay command channel is closed")
    }
}

pub fn spawn(handles: OverlayHandles, enabled: bool) -> std::io::Result<OverlayController> {
    let (sender, receiver) = mpsc::channel();
    thread::Builder::new()
        .name("overlay".into())
        .spawn(move || overlay_main(handles, receiver, enabled))?;
    Ok(OverlayController { sender })
}

fn overlay_main(handles: OverlayHandles, commands: Receiver<OverlayCommand>, enabled: bool) {
    unsafe {
        let hinst = match GetModuleHandleW(None) {
            Ok(h) => h,
            Err(_) => return,
        };
        let class_name = w!("OpenWritrOverlay");
        let wc = WNDCLASSEXW {
            cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(wnd_proc),
            hInstance: hinst.into(),
            hCursor: LoadCursorW(None, IDC_ARROW).unwrap_or(HCURSOR(std::ptr::null_mut())),
            lpszClassName: PCWSTR(class_name.as_ptr()),
            ..Default::default()
        };
        RegisterClassExW(&wc);

        let sw = GetSystemMetrics(SM_CXSCREEN);
        let sh = GetSystemMetrics(SM_CYSCREEN);
        let x = (sw - WIN_W) / 2;
        let y = sh - WIN_H - 120;

        let hwnd = match CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            PCWSTR(class_name.as_ptr()),
            w!("OpenWritr"),
            WS_POPUP,
            x,
            y,
            WIN_W,
            WIN_H,
            None,
            Some(HMENU(std::ptr::null_mut())),
            Some(hinst.into()),
            None,
        ) {
            Ok(h) => h,
            Err(_) => return,
        };

        // Color-key transparency: any pixel painted in pure magenta becomes
        // fully transparent. Lets the pill have a true rounded shape with no
        // visible rectangle around it.
        let _ = SetLayeredWindowAttributes(hwnd, COLORREF(0x00FF00FF), 0, LWA_COLORKEY);

        let level = handles.level_x10000.clone();
        let stop = handles.stop.clone();
        let hwnd_u = hwnd.0 as usize;
        thread::Builder::new()
            .name("overlay-tick".into())
            .spawn(move || {
                let mut last_visible = false;
                // Owns typed state for this thread: enabled flag, current
                // view, and notice-expiry bookkeeping. Visibility and what
                // gets rendered are entirely a function of this state plus
                // the live RMS atomic — never of recorder/app lifecycle bits.
                let mut state = TickState::new(enabled);
                let started = Instant::now();
                'tick: while !stop.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(33));
                    loop {
                        match commands.try_recv() {
                            Ok(command) => state.apply_command(command, Instant::now()),
                            Err(TryRecvError::Empty) => break,
                            Err(TryRecvError::Disconnected) => break 'tick,
                        }
                    }
                    let hwnd = HWND(hwnd_u as *mut _);
                    let visible = state.tick(Instant::now());
                    if visible != last_visible {
                        last_visible = visible;
                        if visible {
                            let _ = SetWindowPos(
                                hwnd,
                                Some(HWND_TOPMOST),
                                0,
                                0,
                                0,
                                0,
                                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
                            );
                            let _ = ShowWindow(hwnd, SW_SHOWNOACTIVATE);
                        } else {
                            let _ = ShowWindow(hwnd, SW_HIDE);
                        }
                    }
                    if visible {
                        let lvl = level.load(Ordering::Relaxed);
                        *RENDER.lock() = RenderFrame::for_view(&state.view, lvl);
                        let phase = started.elapsed().as_millis() as u32;
                        let _ = PostMessageW(
                            Some(hwnd),
                            WM_APP_TICK,
                            WPARAM(0),
                            LPARAM(phase as isize),
                        );
                    }
                }
                let _ = PostMessageW(Some(HWND(hwnd_u as *mut _)), WM_CLOSE, WPARAM(0), LPARAM(0));
            })
            .ok();

        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).into() {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

static PHASE_MS: AtomicU32 = AtomicU32::new(0);
static RENDER: parking_lot::Mutex<RenderFrame> = parking_lot::Mutex::new(RenderFrame::Waveform {
    color: 0x00FF_FFFF,
    amplitude: 0.0,
    label: String::new(),
});

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_CREATE => LRESULT(0),
        WM_APP_TICK => {
            PHASE_MS.store(lparam.0 as u32, Ordering::Relaxed);
            let _ = InvalidateRect(Some(hwnd), None, false);
            LRESULT(0)
        }
        WM_PAINT => {
            let mut ps = PAINTSTRUCT::default();
            let hdc = BeginPaint(hwnd, &mut ps);
            let mut rc = RECT::default();
            let _ = GetClientRect(hwnd, &mut rc);
            let w = rc.right - rc.left;
            let h = rc.bottom - rc.top;

            // Double-buffer.
            let mem_dc = CreateCompatibleDC(Some(hdc));
            let bm = CreateCompatibleBitmap(hdc, w, h);
            let old_bm = SelectObject(mem_dc, bm.into());

            // Fill entire window with the color-key (magenta) → transparent.
            let bg = CreateSolidBrush(COLORREF(0x00FF00FF));
            FillRect(mem_dc, &rc, bg);
            let _ = DeleteObject(bg.into());

            // Dark pill body (rounded). Corner radius = full height for a
            // proper capsule shape.
            let pill = CreateSolidBrush(COLORREF(0x002A2A2A));
            let old_brush = SelectObject(mem_dc, pill.into());
            let pen = CreatePen(PS_SOLID, 1, COLORREF(0x00444444));
            let old_pen = SelectObject(mem_dc, pen.into());
            let _ = RoundRect(mem_dc, 0, 0, w, h, h, h);
            SelectObject(mem_dc, old_pen);
            let _ = DeleteObject(pen.into());
            SelectObject(mem_dc, old_brush);
            let _ = DeleteObject(pill.into());

            match RENDER.lock().clone() {
                RenderFrame::Text { color, message } => {
                    draw_notice_text(mem_dc, w, h, color, &message);
                }
                RenderFrame::Waveform {
                    color,
                    amplitude,
                    label,
                } => {
                    if !label.is_empty() {
                        let label_width =
                            measure_single_line_width(mem_dc, &label) + WAVEFORM_LABEL_PADDING;
                        let layout = waveform_layout(w, h, label_width);
                        draw_waveform_bars(mem_dc, &layout, color, amplitude);
                        draw_caption(mem_dc, layout.label_rect, color, &label);
                    }
                }
            }

            let _ = BitBlt(hdc, 0, 0, w, h, Some(mem_dc), 0, 0, SRCCOPY);

            SelectObject(mem_dc, old_bm);
            let _ = DeleteObject(bm.into());
            let _ = DeleteDC(mem_dc);

            let _ = EndPaint(hwnd, &ps);
            LRESULT(0)
        }
        WM_DESTROY => {
            PostQuitMessage(0);
            LRESULT(0)
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

#[derive(Debug, Clone, Copy)]
struct WaveformLayout {
    bars_left: i32,
    bars_center_y: i32,
    max_bar_height: i32,
    label_rect: RECT,
}

fn waveform_layout(w: i32, h: i32, label_width: i32) -> WaveformLayout {
    let bars_width = NBARS as i32 * BAR_W + (NBARS as i32 - 1) * BAR_GAP;
    let content_width = bars_width + WAVEFORM_LABEL_GAP + label_width;
    let content_left = ((w - content_width) / 2).max(8);

    WaveformLayout {
        bars_left: content_left,
        bars_center_y: h / 2,
        max_bar_height: (h - 18).max(6),
        label_rect: RECT {
            left: content_left + bars_width + WAVEFORM_LABEL_GAP,
            top: 4,
            right: content_left + content_width,
            bottom: h - 4,
        },
    }
}

/// Renders the 22-bar Gaussian-envelope waveform. `amplitude` (0.0-1.0)
/// drives overall bar height — computed from live RMS while listening, or a
/// steady synthetic pulse while processing a job (see `waveform_amplitude`).
/// This is the same bar math the overlay has always used; only the color
/// and amplitude source are now parameterized by typed state.
unsafe fn draw_waveform_bars(mem_dc: HDC, layout: &WaveformLayout, color: u32, amplitude: f32) {
    let amp = amplitude.clamp(0.0, 1.0);
    let phase = PHASE_MS.load(Ordering::Relaxed) as f32 / 1000.0;

    let bar_brush = CreateSolidBrush(COLORREF(color));
    let old_brush = SelectObject(mem_dc, bar_brush.into());
    let pen = CreatePen(PS_SOLID, 0, COLORREF(color));
    let old_pen = SelectObject(mem_dc, pen.into());

    for i in 0..NBARS {
        let t = i as f32 / (NBARS - 1) as f32;
        let centered = (t - 0.5).abs() * 2.0;
        let envelope = (-centered * centered * 2.2).exp();
        let w1 = (phase * 5.5 + t * 7.0).sin();
        let w2 = (phase * 3.2 - t * 4.0).sin();
        let wobble = (w1 * 0.6 + w2 * 0.4) * 0.5 + 0.5;
        let mixed = envelope * (0.35 + 0.65 * wobble) * amp;
        // Round to even so the bar is symmetric around the center line.
        let mut bar_h = (mixed * layout.max_bar_height as f32).max(6.0) as i32;
        if bar_h % 2 != 0 {
            bar_h += 1;
        }
        let x = layout.bars_left + i as i32 * (BAR_W + BAR_GAP);
        let top = layout.bars_center_y - bar_h / 2;
        let bot = top + bar_h;
        let _ = Rectangle(mem_dc, x, top, x + BAR_W, bot);
    }

    SelectObject(mem_dc, old_pen);
    let _ = DeleteObject(pen.into());
    SelectObject(mem_dc, old_brush);
    let _ = DeleteObject(bar_brush.into());
}

/// Small single-line caption beside the waveform bars (e.g. "Listening",
/// "Enhanced", "Queued", "Writing", "Polishing").
unsafe fn draw_caption(mem_dc: HDC, rect: RECT, color: u32, label: &str) {
    let mut text = label.encode_utf16().collect::<Vec<_>>();
    let mut rect = rect;
    let _ = SetBkMode(mem_dc, TRANSPARENT);
    let _ = SetTextColor(mem_dc, COLORREF(color));
    let _ = DrawTextW(
        mem_dc,
        &mut text,
        &mut rect,
        DT_SINGLELINE | DT_VCENTER | DT_NOPREFIX,
    );
}

unsafe fn measure_single_line_width(mem_dc: HDC, text: &str) -> i32 {
    if text.is_empty() {
        return 0;
    }
    let mut text = text.encode_utf16().collect::<Vec<_>>();
    let mut rect = RECT::default();
    let _ = DrawTextW(
        mem_dc,
        &mut text,
        &mut rect,
        DT_CALCRECT | DT_SINGLELINE | DT_NOPREFIX,
    );
    (rect.right - rect.left).max(0)
}

/// Word-wrapped notice text filling the whole pill (success/info/warning/
/// error messages, which can be arbitrarily long free-form strings).
unsafe fn draw_notice_text(mem_dc: HDC, w: i32, h: i32, color: u32, message: &str) {
    let mut text = message.encode_utf16().collect::<Vec<_>>();
    let horizontal_padding = 18;
    let available_width = w - horizontal_padding * 2;
    let single_line_width = measure_single_line_width(mem_dc, message);
    let mut rect = RECT {
        left: horizontal_padding,
        top: 0,
        right: w - horizontal_padding,
        bottom: h,
    };
    let _ = SetBkMode(mem_dc, TRANSPARENT);
    let _ = SetTextColor(mem_dc, COLORREF(color));
    if single_line_width <= available_width {
        let _ = DrawTextW(
            mem_dc,
            &mut text,
            &mut rect,
            DT_CENTER | DT_SINGLELINE | DT_VCENTER | DT_NOPREFIX,
        );
    } else {
        let mut measured = RECT {
            left: horizontal_padding,
            top: 0,
            right: w - horizontal_padding,
            bottom: 0,
        };
        let _ = DrawTextW(
            mem_dc,
            &mut text,
            &mut measured,
            DT_CALCRECT | DT_CENTER | DT_WORDBREAK | DT_NOPREFIX,
        );
        let text_height = (measured.bottom - measured.top).min(h - 8);
        rect.top = (h - text_height) / 2;
        rect.bottom = rect.top + text_height;
        let _ = DrawTextW(
            mem_dc,
            &mut text,
            &mut rect,
            DT_CENTER | DT_WORDBREAK | DT_NOPREFIX,
        );
    }
}

/// Owns the overlay's current typed state (enabled flag, view, and notice
/// expiry bookkeeping) and reduces incoming commands into it. Pure aside
/// from taking `now` explicitly wherever time matters, so it can be unit
/// tested without any Win32 dependency or real sleeps.
struct TickState {
    enabled: bool,
    view: OverlayViewState,
    notice_deadline: Option<Instant>,
}

impl TickState {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            view: OverlayViewState::Hidden,
            notice_deadline: None,
        }
    }

    /// Applies one command to the state.
    fn apply_command(&mut self, command: OverlayCommand, now: Instant) {
        match command {
            OverlayCommand::SetEnabled(enabled) => self.enabled = enabled,
            OverlayCommand::SetState(view) => {
                self.notice_deadline = match &view {
                    OverlayViewState::Notice { kind, .. } => Some(now + notice_duration(*kind)),
                    _ => None,
                };
                self.view = view;
            }
        }
    }

    /// Called every tick: expires a stale notice (collapsing it to
    /// `Hidden`) and returns whether the window should currently be shown.
    fn tick(&mut self, now: Instant) -> bool {
        if let Some(deadline) = self.notice_deadline {
            if now >= deadline {
                self.view = OverlayViewState::Hidden;
                self.notice_deadline = None;
            }
        }
        self.is_visible()
    }

    fn is_visible(&self) -> bool {
        self.enabled && !self.view.is_hidden()
    }
}

/// How long a notice stays up before the overlay auto-hides. More severe
/// notices linger longer so they're easier to read.
fn notice_duration(kind: NoticeKind) -> Duration {
    match kind {
        NoticeKind::Success => Duration::from_millis(2500),
        NoticeKind::Info => Duration::from_secs(3),
        NoticeKind::Warning
        | NoticeKind::RawFallback
        | NoticeKind::ProviderWarning
        | NoticeKind::DeliveryWarning => Duration::from_secs(4),
        NoticeKind::Error => Duration::from_secs(5),
    }
}

/// Defends the renderer against unexpectedly long notice text (e.g. a raw
/// error message) by capping it to a sane length before it ever reaches
/// `DrawTextW`.
fn truncate_notice(message: &str) -> String {
    if message.chars().count() <= MAX_NOTICE_CHARS {
        message.to_string()
    } else {
        let mut truncated: String = message.chars().take(MAX_NOTICE_CHARS - 1).collect();
        truncated.push('…');
        truncated
    }
}

fn listening_label(enhanced: bool) -> &'static str {
    if enhanced {
        "Enhanced"
    } else {
        "Listening"
    }
}

fn phase_label(phase: ProcessingPhase, queue_depth: usize) -> String {
    let base = match phase {
        ProcessingPhase::Queued => "Queued",
        ProcessingPhase::Transcribing => "Writing",
        ProcessingPhase::Enhancing => "Polishing",
    };
    if phase == ProcessingPhase::Queued && queue_depth > 1 {
        format!("{base} · {queue_depth}")
    } else {
        base.to_string()
    }
}

/// Packs an 8-bit RGB triple into the BGR-in-low-24-bits layout `COLORREF`
/// expects, so the palette below can be written in plain RGB.
fn rgb(r: u8, g: u8, b: u8) -> u32 {
    ((b as u32) << 16) | ((g as u32) << 8) | r as u32
}

fn listening_color(enhanced: bool) -> u32 {
    if enhanced {
        rgb(168, 85, 247) // violet — the "enhanced" accent used across the Mac v1.5 port
    } else {
        rgb(255, 255, 255) // plain white waveform for raw dictation
    }
}

fn phase_color(phase: ProcessingPhase) -> u32 {
    match phase {
        ProcessingPhase::Queued => rgb(156, 163, 175), // neutral gray
        ProcessingPhase::Transcribing => rgb(96, 165, 250), // light blue
        ProcessingPhase::Enhancing => rgb(168, 85, 247), // same purple as "enhanced" listening
    }
}

fn notice_color(kind: NoticeKind) -> u32 {
    match kind {
        NoticeKind::Success => rgb(52, 211, 153), // green
        NoticeKind::Info => rgb(226, 232, 240),   // neutral/white
        NoticeKind::Warning
        | NoticeKind::RawFallback
        | NoticeKind::ProviderWarning
        | NoticeKind::DeliveryWarning => rgb(251, 191, 36), // yellow/orange
        NoticeKind::Error => rgb(248, 113, 113),  // orange/red
    }
}

/// Overall bar height (0.0-1.0) for the waveform. While listening this
/// reacts to the live RMS atomic; once capture has stopped there's no more
/// audio to show, so processing states get a steady synthetic pulse instead
/// (still animated via the phase/wobble math in `draw_waveform_bars`).
fn waveform_amplitude(view: &OverlayViewState, level_x10000: u32) -> f32 {
    match view {
        OverlayViewState::Listening { .. } => {
            let level = (level_x10000 as f32 / 10_000.0).clamp(0.0, 1.0);
            // sqrt() compresses dynamic range so quiet speech still shows
            // strong movement, then a generous multiplier saturates loud speech.
            (0.20 + level.sqrt() * 2.2).min(1.0)
        }
        OverlayViewState::Processing { .. } => 0.55,
        OverlayViewState::Hidden | OverlayViewState::Notice { .. } => 0.0,
    }
}

/// Everything `WM_PAINT` needs to draw one frame, computed once per tick on
/// the background thread and handed to the message-loop thread via `RENDER`.
#[derive(Clone)]
enum RenderFrame {
    Waveform {
        color: u32,
        amplitude: f32,
        label: String,
    },
    Text {
        color: u32,
        message: String,
    },
}

impl RenderFrame {
    fn for_view(view: &OverlayViewState, level_x10000: u32) -> Self {
        match view {
            OverlayViewState::Listening { enhanced } => RenderFrame::Waveform {
                color: listening_color(*enhanced),
                amplitude: waveform_amplitude(view, level_x10000),
                label: listening_label(*enhanced).to_string(),
            },
            OverlayViewState::Processing {
                phase, queue_depth, ..
            } => RenderFrame::Waveform {
                color: phase_color(*phase),
                amplitude: waveform_amplitude(view, level_x10000),
                label: phase_label(*phase, *queue_depth),
            },
            OverlayViewState::Notice { kind, message } => RenderFrame::Text {
                color: notice_color(*kind),
                message: truncate_notice(message),
            },
            OverlayViewState::Hidden => RenderFrame::Waveform {
                color: listening_color(false),
                amplitude: 0.0,
                label: String::new(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn notice(kind: NoticeKind, message: &str) -> OverlayViewState {
        OverlayViewState::notice(kind, message)
    }

    // --- visibility -----------------------------------------------------

    #[test]
    fn hidden_state_is_never_visible_even_when_enabled() {
        let mut state = TickState::new(true);
        assert!(!state.tick(Instant::now()));
    }

    #[test]
    fn listening_and_processing_states_are_visible_when_enabled() {
        let now = Instant::now();

        let mut listening = TickState::new(true);
        listening.apply_command(
            OverlayCommand::SetState(OverlayViewState::listening(ListeningIntent::Raw)),
            now,
        );
        assert!(listening.tick(now));

        let mut processing = TickState::new(true);
        processing.apply_command(
            OverlayCommand::SetState(OverlayViewState::processing(
                1,
                ProcessingPhase::Transcribing,
                0,
            )),
            now,
        );
        assert!(processing.tick(now));
    }

    #[test]
    fn disabled_overlay_immediately_hides_regardless_of_view() {
        let now = Instant::now();
        let mut state = TickState::new(true);
        state.apply_command(
            OverlayCommand::SetState(OverlayViewState::listening(ListeningIntent::Enhance)),
            now,
        );
        assert!(state.tick(now));

        state.apply_command(OverlayCommand::SetEnabled(false), now);
        assert!(!state.tick(now));
    }

    // --- enabled/disabled restoration ------------------------------------

    #[test]
    fn re_enabling_restores_the_current_view_instead_of_losing_it() {
        let now = Instant::now();
        let mut state = TickState::new(true);
        let view = OverlayViewState::processing(7, ProcessingPhase::Enhancing, 2);
        state.apply_command(OverlayCommand::SetState(view.clone()), now);

        state.apply_command(OverlayCommand::SetEnabled(false), now);
        assert!(!state.tick(now));
        assert_eq!(state.view, view, "view must be preserved while disabled");

        state.apply_command(OverlayCommand::SetEnabled(true), now);
        assert!(state.tick(now));
        assert_eq!(state.view, view);
    }

    #[test]
    fn disabling_does_not_clear_a_pending_notice_deadline() {
        let now = Instant::now();
        let mut state = TickState::new(true);
        state.apply_command(
            OverlayCommand::SetState(notice(NoticeKind::Info, "hi")),
            now,
        );
        state.apply_command(OverlayCommand::SetEnabled(false), now);
        state.apply_command(OverlayCommand::SetEnabled(true), now);

        // Still within the notice window: visible again after re-enabling.
        assert!(state.tick(now));
        // Past the notice window: expires even though it was toggled off/on.
        let after = now + notice_duration(NoticeKind::Info) + Duration::from_millis(1);
        assert!(!state.tick(after));
    }

    // --- notice expiry ----------------------------------------------------

    #[test]
    fn notice_stays_visible_until_its_deadline_then_hides() {
        let now = Instant::now();
        let mut state = TickState::new(true);
        state.apply_command(
            OverlayCommand::SetState(notice(NoticeKind::Warning, "careful")),
            now,
        );

        let just_before = now + notice_duration(NoticeKind::Warning) - Duration::from_millis(1);
        assert!(state.tick(just_before));

        let just_after = now + notice_duration(NoticeKind::Warning) + Duration::from_millis(1);
        assert!(!state.tick(just_after));
        assert!(state.view.is_hidden());
    }

    #[test]
    fn more_severe_notices_stay_up_longer() {
        assert!(notice_duration(NoticeKind::Error) > notice_duration(NoticeKind::Warning));
        assert!(notice_duration(NoticeKind::Warning) > notice_duration(NoticeKind::Info));
        assert!(notice_duration(NoticeKind::Info) >= notice_duration(NoticeKind::Success));
    }

    // --- typed state transitions / reducer helpers ------------------------

    #[test]
    fn set_state_overwrites_the_previous_view() {
        let now = Instant::now();
        let mut state = TickState::new(true);
        state.apply_command(
            OverlayCommand::SetState(OverlayViewState::listening(ListeningIntent::Raw)),
            now,
        );
        state.apply_command(
            OverlayCommand::SetState(OverlayViewState::processing(3, ProcessingPhase::Queued, 0)),
            now,
        );
        assert_eq!(
            state.view,
            OverlayViewState::processing(3, ProcessingPhase::Queued, 0)
        );
    }

    #[test]
    fn setting_a_non_notice_state_clears_any_pending_deadline() {
        let now = Instant::now();
        let mut state = TickState::new(true);
        state.apply_command(
            OverlayCommand::SetState(notice(NoticeKind::Error, "boom")),
            now,
        );
        assert!(state.notice_deadline.is_some());

        state.apply_command(
            OverlayCommand::SetState(OverlayViewState::listening(ListeningIntent::Raw)),
            now,
        );
        assert!(state.notice_deadline.is_none());
    }

    #[test]
    fn listening_intent_maps_to_the_enhanced_flag() {
        assert_eq!(
            OverlayViewState::listening(ListeningIntent::Raw),
            OverlayViewState::Listening { enhanced: false }
        );
        assert_eq!(
            OverlayViewState::listening(ListeningIntent::Enhance),
            OverlayViewState::Listening { enhanced: true }
        );
        assert!(!ListeningIntent::Raw.is_enhance());
        assert!(ListeningIntent::Enhance.is_enhance());
    }

    // --- labels ------------------------------------------------------------

    #[test]
    fn listening_labels_match_mac_v15_wording() {
        assert_eq!(listening_label(false), "Listening");
        assert_eq!(listening_label(true), "Enhanced");
    }

    #[test]
    fn waveform_and_label_share_one_centered_row() {
        let label_width = 64;
        let layout = waveform_layout(WIN_W, WIN_H, label_width);
        let bars_width = NBARS as i32 * BAR_W + (NBARS as i32 - 1) * BAR_GAP;
        let content_width = bars_width + WAVEFORM_LABEL_GAP + label_width;

        assert_eq!(
            layout.label_rect.left,
            layout.bars_left + bars_width + WAVEFORM_LABEL_GAP
        );
        assert!((layout.bars_left * 2 + content_width - WIN_W).abs() <= 1);
        assert_eq!(layout.bars_center_y, WIN_H / 2);
        assert!(layout.label_rect.right <= WIN_W - 8);
        assert!(layout.label_rect.top < layout.bars_center_y);
        assert!(layout.label_rect.bottom > layout.bars_center_y);
    }

    #[test]
    fn phase_labels_match_mac_v15_wording() {
        assert_eq!(phase_label(ProcessingPhase::Queued, 0), "Queued");
        assert_eq!(phase_label(ProcessingPhase::Transcribing, 0), "Writing");
        assert_eq!(phase_label(ProcessingPhase::Enhancing, 0), "Polishing");
    }

    #[test]
    fn queued_label_shows_depth_only_when_backed_up() {
        assert_eq!(phase_label(ProcessingPhase::Queued, 1), "Queued");
        assert_eq!(phase_label(ProcessingPhase::Queued, 3), "Queued · 3");
        // Depth is irrelevant once actively transcribing/enhancing.
        assert_eq!(phase_label(ProcessingPhase::Transcribing, 5), "Writing");
    }

    // --- colors --------------------------------------------------------------

    #[test]
    fn listening_colors_differ_between_raw_and_enhanced() {
        assert_ne!(listening_color(false), listening_color(true));
        assert_eq!(listening_color(false), rgb(255, 255, 255));
    }

    #[test]
    fn enhanced_listening_and_enhancing_phase_share_the_purple_accent() {
        assert_eq!(
            listening_color(true),
            phase_color(ProcessingPhase::Enhancing)
        );
    }

    #[test]
    fn phase_colors_are_distinct_per_phase() {
        let queued = phase_color(ProcessingPhase::Queued);
        let transcribing = phase_color(ProcessingPhase::Transcribing);
        let enhancing = phase_color(ProcessingPhase::Enhancing);
        assert_ne!(queued, transcribing);
        assert_ne!(transcribing, enhancing);
        assert_ne!(queued, enhancing);
    }

    #[test]
    fn notice_colors_are_distinct_per_kind() {
        let colors = [
            notice_color(NoticeKind::Success),
            notice_color(NoticeKind::Info),
            notice_color(NoticeKind::Warning),
            notice_color(NoticeKind::Error),
        ];
        for i in 0..colors.len() {
            for j in (i + 1)..colors.len() {
                assert_ne!(
                    colors[i], colors[j],
                    "notice colors must be distinguishable"
                );
            }
        }
    }

    // --- amplitude -----------------------------------------------------------

    #[test]
    fn listening_amplitude_reacts_to_rms_level() {
        let view = OverlayViewState::listening(ListeningIntent::Raw);
        let quiet = waveform_amplitude(&view, 0);
        let loud = waveform_amplitude(&view, 10_000);
        assert!(quiet > 0.0, "baseline keeps some motion even at silence");
        assert!(loud > quiet);
        assert!((0.0..=1.0).contains(&quiet));
        assert!((0.0..=1.0).contains(&loud));
    }

    #[test]
    fn processing_amplitude_is_a_steady_pulse_independent_of_rms() {
        let view = OverlayViewState::processing(1, ProcessingPhase::Queued, 0);
        assert_eq!(
            waveform_amplitude(&view, 0),
            waveform_amplitude(&view, 9_999)
        );
    }

    #[test]
    fn hidden_and_notice_states_have_zero_waveform_amplitude() {
        assert_eq!(waveform_amplitude(&OverlayViewState::Hidden, 10_000), 0.0);
        assert_eq!(
            waveform_amplitude(&notice(NoticeKind::Success, "done"), 10_000),
            0.0
        );
    }

    // --- render frame selection -----------------------------------------------

    #[test]
    fn render_frame_selects_waveform_for_listening_and_processing() {
        assert!(matches!(
            RenderFrame::for_view(&OverlayViewState::listening(ListeningIntent::Raw), 0),
            RenderFrame::Waveform { .. }
        ));
        assert!(matches!(
            RenderFrame::for_view(
                &OverlayViewState::processing(1, ProcessingPhase::Queued, 0),
                0
            ),
            RenderFrame::Waveform { .. }
        ));
    }

    #[test]
    fn render_frame_selects_text_for_notices() {
        let frame = RenderFrame::for_view(&notice(NoticeKind::Error, "oh no"), 0);
        match frame {
            RenderFrame::Text { color, message } => {
                assert_eq!(color, notice_color(NoticeKind::Error));
                assert_eq!(message, "oh no");
            }
            RenderFrame::Waveform { .. } => panic!("notices must render as text"),
        }
    }

    // --- long notice text robustness -------------------------------------

    #[test]
    fn truncate_notice_preserves_short_text() {
        assert_eq!(truncate_notice("Ready"), "Ready");
    }

    #[test]
    fn truncate_notice_caps_long_text_with_an_ellipsis() {
        let long = "x".repeat(500);
        let truncated = truncate_notice(&long);
        assert_eq!(truncated.chars().count(), MAX_NOTICE_CHARS);
        assert!(truncated.ends_with('…'));
    }

    #[test]
    fn render_frame_for_notice_truncates_long_messages() {
        let long = "boom ".repeat(100);
        let frame = RenderFrame::for_view(&notice(NoticeKind::Error, &long), 0);
        match frame {
            RenderFrame::Text { message, .. } => {
                assert!(message.chars().count() <= MAX_NOTICE_CHARS);
            }
            RenderFrame::Waveform { .. } => panic!("notices must render as text"),
        }
    }
}
