//! Global low-level keyboard hook for **physical** key state.
//!
//! `GetAsyncKeyState` is what you'd reach for, but Windows lies to it during
//! focus changes: when a foreground window pops up (PowerShell launched by a
//! Win-key shortcut, UAC prompt, anything that grabs input), the OS synthesises
//! key-up events to "clean up" pending modifier state for the new foreground.
//! For a push-to-talk app that means the recording aborts the moment any tool
//! command shells out — even though the user is physically still holding the
//! hotkey down.
//!
//! `SetWindowsHookExW(WH_KEYBOARD_LL, ...)` runs callbacks for every physical
//! key event in the system before any window receives them. We use that to
//! maintain a shared bitmap of currently-held virtual-key codes; the hotkey
//! poll loop reads from it instead of calling `GetAsyncKeyState`.
//!
//! Caveats:
//! - The hook callback must run on a thread with a Windows message loop.
//!   We spin one up dedicated thread for that, separate from the winit event
//!   loop, so the hook can't be starved by app rendering.
//! - The callback must return within ~300 ms or Windows silently unhooks us.
//!   We do a constant-time atomic update — well under the limit.
//! - The hook is process-local for `WH_KEYBOARD_LL`; it sees every keystroke
//!   that goes through SendInput / the keyboard driver, regardless of focus.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use tracing::{info, warn};

static EVENTS_SEEN: AtomicU64 = AtomicU64::new(0);

/// Diagnostic: how many keyboard events the LL hook has processed since
/// install. 0 after a press → hook isn't firing. Read from anywhere.
pub fn events_seen() -> u64 {
    EVENTS_SEEN.load(Ordering::Relaxed)
}

#[cfg(windows)]
use windows::Win32::{
    Foundation::{LPARAM, LRESULT, WPARAM},
    UI::WindowsAndMessaging::{
        CallNextHookEx, DispatchMessageW, GetMessageW, SetWindowsHookExW, TranslateMessage,
        UnhookWindowsHookEx, KBDLLHOOKSTRUCT, MSG, WH_KEYBOARD_LL, WM_KEYDOWN, WM_KEYUP,
        WM_SYSKEYDOWN, WM_SYSKEYUP,
    },
};

/// 256-bit bitmap (one bit per virtual-key code, 0..=255), held as
/// 4 × AtomicU64 so the LL hook callback can fetch_or / fetch_and without
/// acquiring any lock. WH_KEYBOARD_LL callbacks have a hard 300 ms time
/// budget (HKCU\Control Panel\Desktop\LowLevelHooksTimeout) — exceeding it
/// silently unhooks us. parking_lot::Mutex acquisition + lazy init was over
/// that on first call, which is why the previous Mutex<u64> version logged
/// zero events even though the install succeeded.
struct State {
    bits: [AtomicU64; 4],
}

impl State {
    const fn new() -> Self {
        Self {
            bits: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
        }
    }

    fn set(&self, vk: u32, down: bool) {
        if vk > 255 { return; }
        let word = (vk / 64) as usize;
        let bit = 1u64 << (vk % 64);
        if down {
            self.bits[word].fetch_or(bit, Ordering::Relaxed);
        } else {
            self.bits[word].fetch_and(!bit, Ordering::Relaxed);
        }
    }

    fn is_down(&self, vk: u32) -> bool {
        if vk > 255 { return false; }
        let word = (vk / 64) as usize;
        let bit = 1u64 << (vk % 64);
        self.bits[word].load(Ordering::Relaxed) & bit != 0
    }
}

static STATE: State = State::new();
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// Install the global low-level keyboard hook on a dedicated thread.
/// Idempotent: subsequent calls are no-ops. Returns the hook thread handle
/// (which we leak — the hook lives for the lifetime of the process).
pub fn install_once() {
    if INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }
    // Spawn a thread that installs the hook and runs the Windows message
    // loop. The hook is bound to the calling thread, so we must process
    // messages here for callbacks to fire.
    let _ = thread::Builder::new()
        .name("kbd-hook".into())
        .spawn(|| run_hook_thread());
}

#[cfg(not(windows))]
fn run_hook_thread() {}

#[cfg(windows)]
fn run_hook_thread() {
    info!("kbd-hook thread starting, about to SetWindowsHookExW");
    let hook = unsafe {
        SetWindowsHookExW(WH_KEYBOARD_LL, Some(low_level_kbd_proc), None, 0)
    };
    let hook = match hook {
        Ok(h) => h,
        Err(e) => {
            warn!(error = ?e, "SetWindowsHookExW(WH_KEYBOARD_LL) failed");
            INSTALLED.store(false, Ordering::SeqCst);
            return;
        }
    };
    info!("LL keyboard hook installed");

    // PeekMessage + small sleep instead of GetMessage. This lets us:
    //   (a) Keep the thread responsive enough for the hook to fire (the system
    //       gates LL hook callbacks on the thread being in a "ready" message
    //       state, and PeekMessage qualifies just like GetMessage).
    //   (b) Periodically detect "Windows silently unhooked us" — happens when
    //       a single callback exceeds the LowLevelHooksTimeout (default 300 ms,
    //       HKCU\Control Panel\Desktop). Symptom: hook handle still valid but
    //       EVENTS_SEEN stays at 0 forever even though keys are being pressed
    //       elsewhere in the system. We rearm by re-calling SetWindowsHookExW.
    use windows::Win32::UI::WindowsAndMessaging::{PeekMessageW, PM_REMOVE};
    unsafe {
        let mut msg = MSG::default();
        let mut current_hook = hook;
        let mut last_check = std::time::Instant::now();
        let mut last_events = 0u64;
        loop {
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            std::thread::sleep(std::time::Duration::from_millis(2));

            // Every ~5 s, if the event counter has not advanced at all
            // (no key has hit the LL hook in 5 s — extremely unlikely on a
            // machine the user is actively interacting with), the hook is
            // probably dead. Reinstall.
            if last_check.elapsed().as_secs() >= 5 {
                let now = EVENTS_SEEN.load(Ordering::Relaxed);
                if now == last_events {
                    warn!("kbd-hook quiet for 5 s — Windows likely unhooked us; reinstalling");
                    let _ = UnhookWindowsHookEx(current_hook);
                    match SetWindowsHookExW(WH_KEYBOARD_LL, Some(low_level_kbd_proc), None, 0) {
                        Ok(h) => { current_hook = h; info!("LL keyboard hook re-installed"); }
                        Err(e) => { warn!(error = ?e, "re-install failed"); }
                    }
                }
                last_events = now;
                last_check = std::time::Instant::now();
            }
        }
    }
}

#[cfg(windows)]
unsafe extern "system" fn low_level_kbd_proc(
    n_code: i32,
    w_param: WPARAM,
    l_param: LPARAM,
) -> LRESULT {
    // Count EVERY invocation up front (independent of the n_code filter) so
    // we can tell "callback never fires" from "callback fires but we filter
    // out the events" during diagnosis.
    EVENTS_SEEN.fetch_add(1, Ordering::Relaxed);
    // Microsoft docs: if nCode < 0, we MUST pass the event on without
    // inspecting it.
    if n_code < 0 {
        return CallNextHookEx(None, n_code, w_param, l_param);
    }

    // l_param points to a KBDLLHOOKSTRUCT.
    let kb = &*(l_param.0 as *const KBDLLHOOKSTRUCT);
    let msg = w_param.0 as u32;
    let down = matches!(msg, x if x == WM_KEYDOWN || x == WM_SYSKEYDOWN);
    let up   = matches!(msg, x if x == WM_KEYUP   || x == WM_SYSKEYUP);
    if down || up {
        STATE.set(kb.vkCode, down);
    }

    CallNextHookEx(None, n_code, w_param, l_param)
}

/// Cheap atomic read of "is this virtual-key code currently held down".
/// On Windows uses the low-level hook state. On other platforms (we ship
/// only Windows, but cargo check builds on dev machines) returns false.
pub fn is_down(vk: u32) -> bool {
    #[cfg(windows)]
    {
        STATE.is_down(vk)
    }
    #[cfg(not(windows))]
    {
        let _ = vk;
        false
    }
}

// We keep `Arc<Mutex<_>>` imports usable even if the LL-hook path is the only
// real consumer of this module — the cfg-not-windows fallback would otherwise
// produce dead-code warnings.
#[allow(dead_code)]
fn _silence_dead_code(_: Arc<Mutex<()>>) {}
