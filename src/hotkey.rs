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
use tracing::{info, warn};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Press,
    Release,
}

pub struct HotkeyManager {
    _mgr: GlobalHotKeyManager,
    _hk: HotKey,
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
        Ok(Self { _mgr: mgr, _hk: hk })
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
        "f13" => Code::F13,
        "f14" => Code::F14,
        "f15" => Code::F15,
        "f16" => Code::F16,
        "f17" => Code::F17,
        "f18" => Code::F18,
        "f19" => Code::F19,
        "f20" => Code::F20,
        other => {
            warn!("unknown trigger {other}, falling back to Space");
            Code::Space
        }
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
        "f13" => VK_F13,
        "f14" => VK_F14,
        "f15" => VK_F15,
        "f16" => VK_F16,
        "f17" => VK_F17,
        "f18" => VK_F18,
        "f19" => VK_F19,
        "f20" => VK_F20,
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
    if name == "none" || name.is_empty() {
        0
    } else {
        trigger_to_vk(name)
    }
}
pub fn mod_vk_for(name: &str) -> u32 {
    mod_to_vk(name)
}

pub fn configured_vks(trigger_vk: u32, mod_vks: &[u32]) -> Vec<u32> {
    let mut keys = Vec::with_capacity(mod_vks.len() + usize::from(trigger_vk != 0));
    if trigger_vk != 0 {
        keys.push(trigger_vk);
    }
    for &vk in mod_vks {
        if vk != 0 && !keys.contains(&vk) {
            keys.push(vk);
        }
    }
    keys
}

pub fn secondary_key_down(vk: u32) -> bool {
    use windows::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_LWIN, VK_RWIN};
    let down = |key: u32| unsafe { GetAsyncKeyState(key as i32) as u32 & 0x8000 != 0 };
    if vk as u16 == VK_LWIN.0 {
        down(VK_LWIN.0 as u32) || down(VK_RWIN.0 as u32)
    } else {
        down(vk)
    }
}

pub fn configured_key_down(vk: u32) -> bool {
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        VK_CONTROL, VK_LCONTROL, VK_LMENU, VK_LSHIFT, VK_LWIN, VK_MENU, VK_RCONTROL, VK_RMENU,
        VK_RSHIFT, VK_RWIN, VK_SHIFT,
    };
    let down = |key: u32| crate::key_hook::is_down(key);
    let key = vk as u16;
    match key {
        value if value == VK_CONTROL.0 => {
            down(VK_LCONTROL.0 as u32) || down(VK_RCONTROL.0 as u32) || down(VK_CONTROL.0 as u32)
        }
        value if value == VK_SHIFT.0 => {
            down(VK_LSHIFT.0 as u32) || down(VK_RSHIFT.0 as u32) || down(VK_SHIFT.0 as u32)
        }
        value if value == VK_MENU.0 => {
            down(VK_LMENU.0 as u32) || down(VK_RMENU.0 as u32) || down(VK_MENU.0 as u32)
        }
        value if value == VK_LWIN.0 => down(VK_LWIN.0 as u32) || down(VK_RWIN.0 as u32),
        _ => down(vk),
    }
}

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
    let trigger_down = trigger_vk == 0 || configured_key_down(trigger_vk);
    let mods_down = !mod_vks.is_empty() && mod_vks.iter().all(|&vk| configured_key_down(vk));
    let combo = trigger_down && mods_down;
    update_combo_state(combo, state)
}

fn update_combo_state(combo: bool, state: &mut PollState) -> Option<Event> {
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

#[cfg(test)]
mod tests {
    use super::{update_combo_state, Event, PollState};

    #[test]
    fn combo_edges_remain_correct_after_recovery_reset() {
        let mut state = PollState::default();
        assert_eq!(update_combo_state(true, &mut state), Some(Event::Press));
        assert_eq!(update_combo_state(true, &mut state), None);

        state = PollState::default();
        assert_eq!(update_combo_state(false, &mut state), None);
        assert_eq!(update_combo_state(true, &mut state), Some(Event::Press));
        assert_eq!(update_combo_state(false, &mut state), Some(Event::Release));
    }
}
