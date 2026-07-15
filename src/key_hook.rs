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

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{info, warn};

static EVENTS_SEEN: AtomicU64 = AtomicU64::new(0);
static REINSTALL_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Diagnostic: how many keyboard events the LL hook has processed since
/// install. 0 after a press → hook isn't firing. Read from anywhere.
pub fn events_seen() -> u64 {
    EVENTS_SEEN.load(Ordering::Relaxed)
}

#[cfg(windows)]
use windows::Win32::{
    Foundation::{LPARAM, LRESULT, WPARAM},
    UI::WindowsAndMessaging::{
        CallNextHookEx, DispatchMessageW, SetWindowsHookExW, TranslateMessage, UnhookWindowsHookEx,
        KBDLLHOOKSTRUCT, MSG, WH_KEYBOARD_LL, WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP,
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
        if vk > 255 {
            return;
        }
        let word = (vk / 64) as usize;
        let bit = 1u64 << (vk % 64);
        if down {
            self.bits[word].fetch_or(bit, Ordering::Relaxed);
        } else {
            self.bits[word].fetch_and(!bit, Ordering::Relaxed);
        }
    }

    fn is_down(&self, vk: u32) -> bool {
        if vk > 255 {
            return false;
        }
        let word = (vk / 64) as usize;
        let bit = 1u64 << (vk % 64);
        self.bits[word].load(Ordering::Relaxed) & bit != 0
    }

    fn clear(&self) {
        for word in &self.bits {
            word.store(0, Ordering::Relaxed);
        }
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
    if let Err(error) = thread::Builder::new()
        .name("kbd-hook".into())
        .spawn(run_hook_thread)
    {
        INSTALLED.store(false, Ordering::SeqCst);
        warn!(error = %error, "failed to spawn keyboard hook thread");
    }
}

#[cfg(not(windows))]
fn run_hook_thread() {}

#[cfg(windows)]
fn run_hook_thread() {
    info!("kbd-hook thread starting, about to SetWindowsHookExW");
    let hook = unsafe { SetWindowsHookExW(WH_KEYBOARD_LL, Some(low_level_kbd_proc), None, 0) };
    let hook = match hook {
        Ok(h) => h,
        Err(e) => {
            warn!(error = ?e, "SetWindowsHookExW(WH_KEYBOARD_LL) failed");
            INSTALLED.store(false, Ordering::SeqCst);
            return;
        }
    };
    info!("LL keyboard hook installed");

    // PeekMessage + a small sleep keeps the thread in a ready message state
    // while also allowing the hotkey health monitor to request a controlled
    // reinstall after repeated, observed key-transition mismatches.
    use windows::Win32::UI::WindowsAndMessaging::{PeekMessageW, PM_REMOVE};
    unsafe {
        let mut msg = MSG::default();
        let mut current_hook = hook;
        loop {
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            std::thread::sleep(Duration::from_millis(2));

            if REINSTALL_REQUESTED.swap(false, Ordering::AcqRel) {
                match SetWindowsHookExW(WH_KEYBOARD_LL, Some(low_level_kbd_proc), None, 0) {
                    Ok(replacement) => {
                        let _ = UnhookWindowsHookEx(current_hook);
                        current_hook = replacement;
                        STATE.clear();
                        info!("LL keyboard hook re-installed after confirmed mismatch");
                    }
                    Err(error) => {
                        warn!(error = ?error, "keyboard hook re-install failed");
                    }
                }
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
    let up = matches!(msg, x if x == WM_KEYUP   || x == WM_SYSKEYUP);
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

pub fn request_reinstall() {
    clear_held_state();
    REINSTALL_REQUESTED.store(true, Ordering::Release);
}

pub fn clear_held_state() {
    STATE.clear();
}

const MISSED_TRANSITIONS_BEFORE_RECOVERY: u8 = 3;
const RECOVERY_COOLDOWN: Duration = Duration::from_secs(60);

pub struct HealthMonitor {
    tracked: Vec<TrackedKey>,
    missed_transitions: u8,
    last_recovery: Option<Instant>,
}

struct TrackedKey {
    vk: u32,
    secondary_down: bool,
}

impl HealthMonitor {
    pub fn new(
        keys: impl IntoIterator<Item = u32>,
        mut secondary_down: impl FnMut(u32) -> bool,
    ) -> Self {
        let mut unique = Vec::new();
        for vk in keys {
            if vk != 0 && !unique.contains(&vk) {
                unique.push(vk);
            }
        }
        let tracked = unique
            .into_iter()
            .map(|vk| TrackedKey {
                vk,
                secondary_down: secondary_down(vk),
            })
            .collect();
        Self {
            tracked,
            missed_transitions: 0,
            last_recovery: None,
        }
    }

    pub fn observe(
        &mut self,
        now: Instant,
        mut secondary_down: impl FnMut(u32) -> bool,
        mut hook_down: impl FnMut(u32) -> bool,
    ) -> bool {
        let mut matched_transition = false;
        for tracked in &mut self.tracked {
            let secondary = secondary_down(tracked.vk);
            if secondary == tracked.secondary_down {
                continue;
            }
            tracked.secondary_down = secondary;
            if hook_down(tracked.vk) == secondary {
                matched_transition = true;
            } else {
                self.missed_transitions = self.missed_transitions.saturating_add(1);
            }
        }
        if matched_transition {
            self.missed_transitions = 0;
        }
        let cooldown_elapsed = self
            .last_recovery
            .map(|last| now.saturating_duration_since(last) >= RECOVERY_COOLDOWN)
            .unwrap_or(true);
        if self.missed_transitions >= MISSED_TRANSITIONS_BEFORE_RECOVERY && cooldown_elapsed {
            self.missed_transitions = 0;
            self.last_recovery = Some(now);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HealthMonitor;
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    #[test]
    fn long_keyboard_inactivity_never_requests_reinstall() {
        let start = Instant::now();
        let mut monitor = HealthMonitor::new([17], |_| false);
        assert!(!monitor.observe(start + Duration::from_secs(600), |_| false, |_| false));
    }

    #[test]
    fn repeated_missing_transitions_request_one_rate_limited_reinstall() {
        let start = Instant::now();
        let mut secondary = HashMap::from([(17_u32, false)]);
        let mut monitor = HealthMonitor::new([17], |vk| secondary[&vk]);

        for index in 0..2 {
            secondary.insert(17, index % 2 == 0);
            assert!(!monitor.observe(
                start + Duration::from_secs(index + 1),
                |vk| secondary[&vk],
                |vk| !secondary[&vk],
            ));
        }
        secondary.insert(17, true);
        assert!(monitor.observe(
            start + Duration::from_secs(3),
            |vk| secondary[&vk],
            |vk| !secondary[&vk],
        ));

        for index in 0..3 {
            secondary.insert(17, index % 2 == 0);
            assert!(!monitor.observe(
                start + Duration::from_secs(4 + index),
                |vk| secondary[&vk],
                |vk| !secondary[&vk],
            ));
        }
    }

    #[test]
    fn matching_transition_clears_suspicion() {
        let start = Instant::now();
        let mut secondary = false;
        let mut monitor = HealthMonitor::new([17], |_| secondary);

        secondary = true;
        assert!(!monitor.observe(start, |_| secondary, |_| false));
        secondary = false;
        assert!(!monitor.observe(start + Duration::from_secs(1), |_| secondary, |_| false,));
        secondary = true;
        assert!(!monitor.observe(start + Duration::from_secs(2), |_| secondary, |_| true,));
        secondary = false;
        assert!(!monitor.observe(start + Duration::from_secs(3), |_| secondary, |_| true,));
    }
}
