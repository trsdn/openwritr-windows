//! Visual recording indicator (macOS-style centered meter).
//!
//! Runs on its own dedicated thread with its own Win32 message loop.
//! Reads only two atomics from the recorder (`recording`, `last_rms_x10000`)
//! and shares NO state with the tray's winit loop — so it can never deadlock
//! the main app, no matter what happens here.
//!
//! Look: a horizontal pill near the bottom-center with a row of vertical
//! bars whose heights breathe with the audio level. Gaussian envelope makes
//! the center bars react strongest, with a per-bar wave shimmer.

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
    SelectObject, SetBkMode, SetTextColor, DT_CENTER, DT_NOPREFIX, DT_VCENTER, DT_WORDBREAK,
    PAINTSTRUCT, PS_SOLID, SRCCOPY, TRANSPARENT,
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

pub struct OverlayHandles {
    pub recording: Arc<AtomicBool>,
    pub level_x10000: Arc<AtomicU32>,
    pub stop: Arc<AtomicBool>,
}

enum OverlayCommand {
    SetEnabled(bool),
    ShowStatus(String),
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

    pub fn show_status(&self, message: impl Into<String>) -> Result<(), &'static str> {
        self.sender
            .send(OverlayCommand::ShowStatus(message.into()))
            .map_err(|_| "overlay command channel is closed")
    }
}

static STATUS_MESSAGE: parking_lot::Mutex<Option<(String, Instant)>> =
    parking_lot::Mutex::new(None);
static STATUS_MODE: AtomicBool = AtomicBool::new(false);

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

        let recording = handles.recording.clone();
        let level = handles.level_x10000.clone();
        let stop = handles.stop.clone();
        let hwnd_u = hwnd.0 as usize;
        thread::Builder::new()
            .name("overlay-tick".into())
            .spawn(move || {
                let mut last_visible = false;
                let mut enabled = enabled;
                let started = Instant::now();
                'tick: while !stop.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(33));
                    loop {
                        match commands.try_recv() {
                            Ok(OverlayCommand::SetEnabled(next)) => {
                                enabled = next;
                                if !enabled {
                                    *STATUS_MESSAGE.lock() = None;
                                    STATUS_MODE.store(false, Ordering::Relaxed);
                                }
                            }
                            Ok(OverlayCommand::ShowStatus(message)) => {
                                if enabled {
                                    *STATUS_MESSAGE.lock() =
                                        Some((message, Instant::now() + Duration::from_secs(3)));
                                }
                            }
                            Err(TryRecvError::Empty) => break,
                            Err(TryRecvError::Disconnected) => break 'tick,
                        }
                    }
                    let hwnd = HWND(hwnd_u as *mut _);
                    let now_rec = enabled && recording.load(Ordering::Relaxed);
                    let status_visible = enabled && status_is_visible();
                    let visible = should_be_visible(enabled, now_rec, status_visible);
                    STATUS_MODE.store(!now_rec && status_visible, Ordering::Relaxed);
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
                        let phase = started.elapsed().as_millis() as u32;
                        let _ = PostMessageW(
                            Some(hwnd),
                            WM_APP_TICK,
                            WPARAM(lvl as usize),
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

static LAST_LEVEL: AtomicU32 = AtomicU32::new(0);
static PHASE_MS: AtomicU32 = AtomicU32::new(0);

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_CREATE => LRESULT(0),
        WM_APP_TICK => {
            LAST_LEVEL.store(wparam.0 as u32, Ordering::Relaxed);
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

            if STATUS_MODE.load(Ordering::Relaxed) {
                if let Some(message) = current_status_message() {
                    let mut text = message.encode_utf16().collect::<Vec<_>>();
                    let mut text_rect = RECT {
                        left: 18,
                        top: 8,
                        right: w - 18,
                        bottom: h - 8,
                    };
                    let _ = SetBkMode(mem_dc, TRANSPARENT);
                    let _ = SetTextColor(mem_dc, COLORREF(0x00_FFFFFF));
                    let _ = DrawTextW(
                        mem_dc,
                        &mut text,
                        &mut text_rect,
                        DT_CENTER | DT_VCENTER | DT_WORDBREAK | DT_NOPREFIX,
                    );
                }
            } else {
                let level = (LAST_LEVEL.load(Ordering::Relaxed) as f32 / 10_000.0).clamp(0.0, 1.0);
                // sqrt() compresses dynamic range so quiet speech still shows
                // strong movement, then a generous multiplier saturates loud speech.
                let baseline = 0.20;
                let amp = (baseline + level.sqrt() * 2.2).min(1.0);
                let phase = PHASE_MS.load(Ordering::Relaxed) as f32 / 1000.0;

                let bar_w: i32 = 4;
                let gap: i32 = 3;
                // Total width of all bars + gaps. Center that block horizontally.
                let block_w = NBARS as i32 * bar_w + (NBARS as i32 - 1) * gap;
                let bar_area_left = (w - block_w) / 2;
                let max_bar_h = h - 18;
                let cy = h / 2;

                let bar_brush = CreateSolidBrush(COLORREF(0x00_FFFFFF)); // clean white
                let old_brush = SelectObject(mem_dc, bar_brush.into());
                let pen = CreatePen(PS_SOLID, 0, COLORREF(0x00_FFFFFF));
                let old_pen = SelectObject(mem_dc, pen.into());

                for i in 0..NBARS {
                    let t = i as f32 / (NBARS - 1) as f32;
                    let centered = (t - 0.5).abs() * 2.0;
                    let envelope = (-centered * centered * 2.2).exp();
                    let w1 = (phase * 5.5 + t * 7.0).sin();
                    let w2 = (phase * 3.2 - t * 4.0).sin();
                    let wobble = (w1 * 0.6 + w2 * 0.4) * 0.5 + 0.5;
                    let mixed = envelope * (0.35 + 0.65 * wobble) * amp;
                    // Round to even so the bar is symmetric around cy.
                    let mut bar_h = (mixed * max_bar_h as f32).max(6.0) as i32;
                    if bar_h % 2 != 0 {
                        bar_h += 1;
                    }
                    let x = bar_area_left + i as i32 * (bar_w + gap);
                    let top = cy - bar_h / 2;
                    let bot = top + bar_h;
                    let _ = Rectangle(mem_dc, x, top, x + bar_w, bot);
                }

                SelectObject(mem_dc, old_pen);
                let _ = DeleteObject(pen.into());
                SelectObject(mem_dc, old_brush);
                let _ = DeleteObject(bar_brush.into());
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

fn status_is_visible() -> bool {
    let mut status = STATUS_MESSAGE.lock();
    match status.as_ref() {
        Some((_, until)) if Instant::now() < *until => true,
        Some(_) => {
            *status = None;
            false
        }
        None => false,
    }
}

fn current_status_message() -> Option<String> {
    STATUS_MESSAGE
        .lock()
        .as_ref()
        .map(|(message, _)| message.clone())
}

fn should_be_visible(enabled: bool, recording: bool, status_visible: bool) -> bool {
    enabled && (recording || status_visible)
}

#[cfg(test)]
mod tests {
    use super::should_be_visible;

    #[test]
    fn disabled_overlay_never_becomes_visible() {
        assert!(!should_be_visible(false, true, false));
        assert!(!should_be_visible(false, false, true));
        assert!(!should_be_visible(false, true, true));
    }
}
