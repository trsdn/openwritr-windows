//! Push-to-talk hotkey state machine.
//!
//! global-hotkey on Windows only fires on key-down, so for the hold-to-record
//! UX we register the combo and then poll the OS key state to detect the
//! release edge ourselves.

use crate::settings::Settings;
use anyhow::Result;
use global_hotkey::{
    hotkey::{Code, HotKey, Modifiers},
    GlobalHotKeyManager,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use tracing::{info, warn};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event { Press, Release }

pub struct HotkeyManager {
    _mgr: GlobalHotKeyManager,
    _hk: HotKey,
    pub trigger_vk: u32,
    pub required_mods_vk: Vec<u32>,
    pub is_pressed: Arc<AtomicBool>,
}

impl HotkeyManager {
    pub fn register(s: &Settings) -> Result<Self> {
        // Modifiers-only mode: skip OS RegisterHotKey (which needs a key).
        // The poll loop handles detection via GetAsyncKeyState.
        if s.hotkey_trigger == "none" || s.hotkey_trigger.is_empty() {
            return Err(anyhow::anyhow!("modifiers-only mode (no OS reservation)"));
        }
        let mgr = GlobalHotKeyManager::new()?;
        let mods = build_modifiers(&s.hotkey_modifiers);
        let code = trigger_to_code(&s.hotkey_trigger);
        let hk = HotKey::new(Some(mods), code);
        mgr.register(hk)?;
        info!(modifiers = ?s.hotkey_modifiers, trigger = %s.hotkey_trigger, "hotkey registered");
        Ok(Self {
            _mgr: mgr,
            _hk: hk,
            trigger_vk: trigger_to_vk(&s.hotkey_trigger),
            required_mods_vk: s.hotkey_modifiers.iter().map(|m| mod_to_vk(m)).collect(),
            is_pressed: Arc::new(AtomicBool::new(false)),
        })
    }
}

fn build_modifiers(names: &[String]) -> Modifiers {
    let mut m = Modifiers::empty();
    for n in names {
        match n.as_str() {
            "ctrl" => m |= Modifiers::CONTROL,
            "shift" => m |= Modifiers::SHIFT,
            "alt" => m |= Modifiers::ALT,
            "win" => m |= Modifiers::SUPER,
            other => warn!("unknown modifier {other}"),
        }
    }
    m
}

fn trigger_to_code(name: &str) -> Code {
    match name {
        "space" => Code::Space,
        "tab" => Code::Tab,
        "caps_lock" => Code::CapsLock,
        "scroll_lock" => Code::ScrollLock,
        "pause" => Code::Pause,
        "insert" => Code::Insert,
        "right_ctrl" => Code::ControlRight,
        "f13" => Code::F13, "f14" => Code::F14, "f15" => Code::F15,
        "f16" => Code::F16, "f17" => Code::F17, "f18" => Code::F18,
        "f19" => Code::F19, "f20" => Code::F20,
        other => { warn!("unknown trigger {other}, falling back to Space"); Code::Space }
    }
}

fn trigger_to_vk(name: &str) -> u32 {
    use windows::Win32::UI::Input::KeyboardAndMouse::*;
    let vk: VIRTUAL_KEY = match name {
        "space" => VK_SPACE,
        "tab" => VK_TAB,
        "caps_lock" => VK_CAPITAL,
        "scroll_lock" => VK_SCROLL,
        "pause" => VK_PAUSE,
        "insert" => VK_INSERT,
        "right_ctrl" => VK_RCONTROL,
        "f13" => VK_F13, "f14" => VK_F14, "f15" => VK_F15,
        "f16" => VK_F16, "f17" => VK_F17, "f18" => VK_F18,
        "f19" => VK_F19, "f20" => VK_F20,
        _ => VK_SPACE,
    };
    vk.0 as u32
}

fn mod_to_vk(name: &str) -> u32 {
    use windows::Win32::UI::Input::KeyboardAndMouse::*;
    let vk: VIRTUAL_KEY = match name {
        "ctrl" => VK_CONTROL,
        "shift" => VK_SHIFT,
        "alt" => VK_MENU,
        "win" => VK_LWIN,
        _ => VK_SPACE,
    };
    vk.0 as u32
}

/// Public helpers for callers that want to poll a combo without a HotkeyManager.
/// Returns 0 for "no trigger" — combo fires on modifiers alone.
pub fn trigger_vk_for(name: &str) -> u32 {
    if name == "none" || name.is_empty() { 0 } else { trigger_to_vk(name) }
}
pub fn mod_vk_for(name: &str) -> u32 { mod_to_vk(name) }

/// State carried between successive `poll_combo` calls. Just a boolean —
/// the low-level keyboard hook in `key_hook` already gives us reliable
/// physical state across focus changes, so no debounce is needed.
#[derive(Default)]
pub struct PollState {
    pub pressed: bool,
}

/// Edge-detecting combo poll. Reads physical key state from the global
/// low-level keyboard hook (see `crate::key_hook`), which is immune to
/// the focus-change synthesised key-ups that fool `GetAsyncKeyState`.
/// Returns Press on the 0→1 edge and Release on the 1→0 edge. Works
/// without RegisterHotKey. If `trigger_vk` is 0, the combo fires on
/// modifiers-only.
pub fn poll_combo(trigger_vk: u32, mod_vks: &[u32], state: &mut PollState) -> Option<Event> {
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        VK_CONTROL, VK_LCONTROL, VK_RCONTROL, VK_LWIN, VK_RWIN,
        VK_SHIFT, VK_LSHIFT, VK_RSHIFT, VK_MENU, VK_LMENU, VK_RMENU,
    };
    let down = |vk: u32| crate::key_hook::is_down(vk);
    let trigger_down = trigger_vk == 0 || down(trigger_vk);
    // LL keyboard hooks deliver side-specific virtual-key codes
    // (VK_LCONTROL / VK_RCONTROL etc., NOT the generic VK_CONTROL). The
    // settings layer maps "ctrl" / "shift" / "alt" to the generic codes,
    // so we must accept either side here.
    let mods_down = !mod_vks.is_empty() && mod_vks.iter().all(|&vk| {
        let v = vk as u16;
        match v {
            x if x == VK_CONTROL.0 => down(VK_LCONTROL.0 as u32) || down(VK_RCONTROL.0 as u32) || down(VK_CONTROL.0 as u32),
            x if x == VK_SHIFT.0   => down(VK_LSHIFT.0   as u32) || down(VK_RSHIFT.0   as u32) || down(VK_SHIFT.0   as u32),
            x if x == VK_MENU.0    => down(VK_LMENU.0    as u32) || down(VK_RMENU.0    as u32) || down(VK_MENU.0    as u32),
            x if x == VK_LWIN.0    => down(VK_LWIN.0     as u32) || down(VK_RWIN.0     as u32),
            _ => down(vk),
        }
    });
    let combo = trigger_down && mods_down;

    // Once-a-second-ish heartbeat for diagnosing missing key events.
    use std::sync::atomic::{AtomicU64, Ordering};
    static LAST_HEARTBEAT_MS: AtomicU64 = AtomicU64::new(0);
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
    let last = LAST_HEARTBEAT_MS.load(Ordering::Relaxed);
    if now_ms.saturating_sub(last) > 5000 {
        LAST_HEARTBEAT_MS.store(now_ms, Ordering::Relaxed);
        tracing::info!(
            events_seen = crate::key_hook::events_seen(),
            trigger_down,
            mods_down,
            combo,
            "hotkey poll heartbeat"
        );
    }

    if combo && !state.pressed {
        state.pressed = true;
        return Some(Event::Press);
    }
    if !combo && state.pressed {
        state.pressed = false;
        return Some(Event::Release);
    }
    None
}

/// Returns Some(Press)/Some(Release) on edges, otherwise None.
pub fn poll_state(mgr: &HotkeyManager, last_state: &mut bool) -> Option<Event> {
    use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
    let trigger_down = unsafe { GetAsyncKeyState(mgr.trigger_vk as i32) as u32 & 0x8000 != 0 };
    let mods_down = mgr
        .required_mods_vk
        .iter()
        .all(|&vk| unsafe { GetAsyncKeyState(vk as i32) as u32 & 0x8000 != 0 });

    let combo = trigger_down && mods_down;
    if combo && !*last_state {
        *last_state = true;
        mgr.is_pressed.store(true, Ordering::Relaxed);
        return Some(Event::Press);
    }
    if !combo && *last_state {
        *last_state = false;
        mgr.is_pressed.store(false, Ordering::Relaxed);
        return Some(Event::Release);
    }
    None
}

pub fn poll_sleep() {
    std::thread::sleep(Duration::from_millis(8));
}
