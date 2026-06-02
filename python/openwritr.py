"""OpenWritr for Windows — push-to-talk voice-to-text tray app."""
from __future__ import annotations
import os
import sys
import json
import time
import queue
import threading
import logging
import tkinter as tk
from pathlib import Path

import numpy as np
import sounddevice as sd
from pynput import keyboard
import pyperclip
from PIL import Image, ImageDraw
import pystray

from engine import load_engine, AVAILABLE as ENGINES
from overlay import Overlay, enable_dpi_awareness
from settings_launcher import SettingsLauncher
import sounds
import enhance as enhance_mod

APP_NAME = "OpenWritr"
APPDATA = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME
MODEL_DIR = APPDATA / "models" / "parakeet-tdt-0.6b-v3"  # legacy alias
LOG_DIR = APPDATA / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_PATH = APPDATA / "settings.json"
SAMPLE_RATE = 16_000
CHANNELS = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "openwritr.log", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger("openwritr")


DEFAULTS = {
    "hotkey_modifiers": ["ctrl", "win"],
    "hotkey_trigger": "space",
    "engine": "parakeet_cpu",
    "auto_paste": True,
    "overlay": True,
    "sounds": True,
    "min_record_seconds": 0.25,
    "max_record_seconds": 60,
    "enhance": {"provider": "off", "base_url": "https://api.openai.com/v1", "api_key": "", "model": "gpt-4o-mini"},
}


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text("utf-8"))
            merged = {**DEFAULTS, **data}
            merged["enhance"] = {**DEFAULTS["enhance"], **(data.get("enhance") or {})}
            return merged
        except Exception as e:
            log.warning("settings load failed: %s", e)
    return {**DEFAULTS, "enhance": dict(DEFAULTS["enhance"])}


def save_settings(s: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(s, indent=2), "utf-8")


# ---------- Audio capture ----------
class AudioRecorder:
    def __init__(self, samplerate: int = SAMPLE_RATE, on_level=None) -> None:
        self.samplerate = samplerate
        self.on_level = on_level
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None

    def _cb(self, indata, frames, time_info, status):
        if status:
            log.debug("audio status: %s", status)
        self._q.put(indata.copy())
        if self.on_level is not None:
            rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)) + 1e-9)
            level = min(1.0, (rms / 0.08) ** 0.6)
            try:
                self.on_level(level)
            except Exception:
                pass

    def start(self) -> None:
        if self._stream is not None:
            return
        while not self._q.empty():
            self._q.get_nowait()
        self._stream = sd.InputStream(
            samplerate=self.samplerate, channels=CHANNELS, dtype="float32",
            callback=self._cb, blocksize=1600,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        chunks: list[np.ndarray] = []
        while not self._q.empty():
            chunks.append(self._q.get_nowait())
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks, axis=0).reshape(-1)


# ---------- ASR ----------
class Transcriber:
    def __init__(self, engine_name: str) -> None:
        log.info("loading engine: %s", engine_name)
        self._engine = load_engine(engine_name)
        self._engine_name = engine_name

    @property
    def engine_name(self) -> str:
        return self._engine_name

    def transcribe(self, audio: np.ndarray) -> str:
        return self._engine.transcribe(audio)


# ---------- Paste ----------
def paste_text(text: str) -> None:
    if not text:
        return
    saved = None
    try:
        saved = pyperclip.paste()
    except Exception:
        pass
    pyperclip.copy(text)
    kb = keyboard.Controller()
    with kb.pressed(keyboard.Key.ctrl):
        kb.press("v")
        kb.release("v")
    if saved is not None:
        def _restore():
            time.sleep(0.4)
            try:
                pyperclip.copy(saved)
            except Exception:
                pass
        threading.Thread(target=_restore, daemon=True).start()


# ---------- Hotkey FSM ----------
MOD_MAP = {
    "ctrl": {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r},
    "shift": {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r},
    "alt": {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr},
    "win": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
}

TRIGGER_KEYS = {
    "space":       keyboard.Key.space,
    "tab":         keyboard.Key.tab,
    "caps_lock":   keyboard.Key.caps_lock,
    "scroll_lock": keyboard.Key.scroll_lock,
    "pause":       keyboard.Key.pause,
    "insert":      keyboard.Key.insert,
    "right_ctrl":  keyboard.Key.ctrl_r,
    "f13":         keyboard.Key.f13,
    "f14":         keyboard.Key.f14,
    "f15":         keyboard.Key.f15,
    "f16":         keyboard.Key.f16,
    "f17":         keyboard.Key.f17,
    "f18":         keyboard.Key.f18,
    "f19":         keyboard.Key.f19,
    "f20":         keyboard.Key.f20,
}


def trigger_key_for(name: str):
    return TRIGGER_KEYS.get((name or "space").lower(), keyboard.Key.space)


class HotkeyEngine:
    def __init__(self, settings, recorder, transcriber, on_state) -> None:
        self.settings = settings
        self.recorder = recorder
        self.transcriber = transcriber
        self.on_state = on_state
        self._pressed_mods = {m: False for m in MOD_MAP}
        self._recording = False
        self._record_started_at = 0.0
        self._enhanced_for_this_press = False
        self._lock = threading.Lock()
        self._safety_timer: threading.Timer | None = None

    def _required_mods(self) -> set[str]:
        return set(self.settings.get("hotkey_modifiers", ["ctrl", "win"]))

    def _trigger(self):
        return trigger_key_for(self.settings.get("hotkey_trigger", "space"))

    def _mods_ok(self) -> bool:
        return all(self._pressed_mods.get(m, False) for m in self._required_mods())

    def _enhance_active(self) -> bool:
        if "alt" not in self._required_mods():
            return self._pressed_mods.get("alt", False)
        return self._pressed_mods.get("win", False)

    def _stop_recording_locked(self) -> None:
        """Caller holds self._lock. Stop capture + dispatch the transcribe job."""
        if not self._recording:
            return
        if self._safety_timer is not None:
            self._safety_timer.cancel()
            self._safety_timer = None
        self._recording = False
        dur = time.time() - self._record_started_at
        audio = self.recorder.stop()
        if self.settings.get("sounds", True):
            sounds.play_stop()
        if dur < float(self.settings.get("min_record_seconds", 0.25)):
            log.info("recording too short (%.2fs), discarding", dur)
            self.on_state("idle", None, False)
            return
        enhanced = self._enhanced_for_this_press
        threading.Thread(target=self._run_transcribe, args=(audio, enhanced), daemon=True).start()

    def _on_press(self, key):
        # Track modifier state.
        for name, keys in MOD_MAP.items():
            if key in keys:
                self._pressed_mods[name] = True
                return
        # Trigger key.
        if key == self._trigger() and self._mods_ok():
            with self._lock:
                if not self._recording:
                    self._recording = True
                    self._record_started_at = time.time()
                    self._enhanced_for_this_press = self._enhance_active()
                    try:
                        self.recorder.start()
                        self.on_state("listening", None, self._enhanced_for_this_press)
                        if self.settings.get("sounds", True):
                            sounds.play_start()
                        # Safety auto-stop in case we never see the release event.
                        max_s = float(self.settings.get("max_record_seconds", 60))
                        self._safety_timer = threading.Timer(max_s, self._safety_stop)
                        self._safety_timer.daemon = True
                        self._safety_timer.start()
                        log.info("recording started (enhanced=%s)", self._enhanced_for_this_press)
                    except Exception:
                        log.exception("recorder start failed")
                        self._recording = False

    def _on_release(self, key):
        # Track modifier release.
        was_required_mod_release = False
        for name, keys in MOD_MAP.items():
            if key in keys:
                self._pressed_mods[name] = False
                if name in self._required_mods():
                    was_required_mod_release = True
        # Trigger release -> stop.
        if key == self._trigger():
            with self._lock:
                self._stop_recording_locked()
            return
        # Required modifier released while we are recording -> also stop, so
        # the user doesn't get stuck listening when Windows steals focus and
        # eats the trigger-release event.
        if was_required_mod_release:
            with self._lock:
                if self._recording:
                    log.info("required modifier released — stopping safely")
                    self._stop_recording_locked()

    def _safety_stop(self) -> None:
        with self._lock:
            if self._recording:
                log.warning("safety auto-stop after max_record_seconds")
                self._stop_recording_locked()

    def _run_transcribe(self, audio: np.ndarray, enhanced: bool):
        try:
            self.on_state("transcribing", None, enhanced)
            text = self.transcriber.transcribe(audio)
            if text and enhanced:
                self.on_state("enhancing", None, enhanced)
                t0 = time.time()
                text = enhance_mod.enhance(text, self.settings)
                log.info("enhanced in %.2fs", time.time() - t0)
            if text and self.settings.get("auto_paste", True):
                paste_text(text)
            self.on_state("done", text or None, enhanced)
        except Exception:
            log.exception("transcription failed")
            self.on_state("error", None, False)

    def run(self):
        listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release, suppress=False
        )
        listener.start()
        return listener


# ---------- Tray ----------
TRAY_COLORS = {
    "idle": (74, 144, 226, 255),
    "listening": (220, 38, 38, 255),
    "transcribing": (245, 158, 11, 255),
    "enhancing": (37, 99, 235, 255),
    "done": (22, 163, 74, 255),
    "error": (107, 114, 128, 255),
}


def make_tray_icon(state: str = "idle") -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = TRAY_COLORS.get(state, TRAY_COLORS["idle"])
    d.rounded_rectangle((24, 10, 40, 38), radius=8, fill=color)
    d.rectangle((31, 38, 33, 50), fill=color)
    d.rectangle((20, 50, 44, 52), fill=color)
    return img


class TrayController:
    def __init__(self, settings: dict, on_open_settings, on_quit) -> None:
        self.settings = settings
        self.on_open_settings = on_open_settings
        self.on_quit = on_quit
        self.icon = pystray.Icon(
            APP_NAME, make_tray_icon("idle"), APP_NAME,
            menu=pystray.Menu(
                pystray.MenuItem(lambda _: f"Hotkey: {self._hotkey_label()}", None, enabled=False),
                pystray.MenuItem(lambda _: f"Engine: {self._engine_label()}", None, enabled=False),
                pystray.MenuItem(lambda _: f"Enhance: {self._enhance_label()}", None, enabled=False),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Settings…", lambda *_: self.on_open_settings()),
                pystray.MenuItem("Open log folder", lambda *_: os.startfile(LOG_DIR)),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", lambda *_: self.on_quit()),
            ),
        )

    def _hotkey_label(self) -> str:
        mods = "+".join(m.capitalize() for m in self.settings.get("hotkey_modifiers", []))
        trig = (self.settings.get("hotkey_trigger") or "space").replace("_", " ").title()
        return f"{mods}+{trig}" if mods else trig

    def _engine_label(self) -> str:
        eid = self.settings.get("engine", "parakeet_cpu")
        return {"parakeet_cpu": "Parakeet CPU",
                "parakeet_npu": "Parakeet NPU",
                "whisper_npu":  "Whisper NPU"}.get(eid, eid)

    def _enhance_label(self) -> str:
        prov = (self.settings.get("enhance") or {}).get("provider", "off")
        return {"off": "off", "github_copilot": "GitHub Copilot", "openai_compatible": "OpenAI API"}.get(prov, prov)

    def set_state(self, state: str) -> None:
        try:
            self.icon.icon = make_tray_icon(state)
        except Exception:
            pass

    def refresh_menu(self) -> None:
        try:
            self.icon.update_menu()
        except Exception:
            pass

    def run_detached(self) -> None:
        self.icon.run_detached()

    def stop(self) -> None:
        try:
            self.icon.stop()
        except Exception:
            pass


# ---------- App ----------
class App:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.transcriber = Transcriber(self.settings.get("engine", "parakeet_cpu"))
        self.dpi = enable_dpi_awareness()
        log.info("DPI scale: %.2fx", self.dpi)
        self.root = tk.Tk()
        try:
            self.root.tk.call("tk", "scaling", self.dpi)
        except tk.TclError:
            pass
        self.root.withdraw()
        self.overlay = Overlay(self.root, dpi_scale=self.dpi)
        self.recorder = AudioRecorder(on_level=self.overlay.set_level)
        self.settings_ui = SettingsLauncher(self._reload_settings_from_disk)
        self.tray = TrayController(self.settings, self._open_settings, self.quit)
        self.engine = HotkeyEngine(self.settings, self.recorder, self.transcriber, self._on_state)
        self._listener = None

    def _open_settings(self) -> None:
        self.settings_ui.open()

    def _reload_settings_from_disk(self) -> None:
        new = load_settings()
        old_engine = self.settings.get("engine", "parakeet_cpu")
        self.settings.clear()
        self.settings.update(new)
        self.tray.refresh_menu()
        if new.get("engine") != old_engine:
            log.info("engine changed: %s -> %s (hot-swapping)", old_engine, new.get("engine"))
            try:
                self.transcriber = Transcriber(new.get("engine", "parakeet_cpu"))
                # Rebind so the in-flight HotkeyEngine instance sees the new transcriber.
                self.engine.transcriber = self.transcriber
            except Exception:
                log.exception("engine swap failed; reverting to previous")
        log.info("settings reloaded: hotkey=%s engine=%s enhance=%s",
                 "+".join(self.settings["hotkey_modifiers"]),
                 self.settings.get("engine"),
                 (self.settings.get("enhance") or {}).get("provider"))

    def _on_settings_saved(self, new: dict) -> None:
        self.settings.clear()
        self.settings.update(new)
        save_settings(self.settings)
        self.tray.refresh_menu()
        log.info("settings saved: hotkey=%s enhance=%s",
                 "+".join(self.settings["hotkey_modifiers"]),
                 (self.settings.get("enhance") or {}).get("provider"))

    def _on_state(self, state: str, text: str | None, enhanced: bool) -> None:
        self.root.after(0, lambda: self._apply_state(state, text, enhanced))

    def _apply_state(self, state: str, text: str | None, enhanced: bool) -> None:
        self.tray.set_state(state if state != "done" else "idle")
        if not self.settings.get("overlay", True):
            self.overlay.hide()
            return
        if state in ("listening", "transcribing", "enhancing"):
            self.overlay.show(state)
        elif state == "done":
            self.overlay.show_briefly("done", 700)
        elif state in ("idle", "error"):
            self.overlay.hide()

    def run(self) -> None:
        log.info("OpenWritr ready — %s (hold)", self.tray._hotkey_label())
        self.tray.run_detached()
        self._listener = self.engine.run()
        try:
            self.root.mainloop()
        finally:
            self.tray.stop()

    def quit(self) -> None:
        log.info("OpenWritr quitting")
        try:
            if self._listener is not None:
                self._listener.stop()
        except Exception:
            pass
        self.root.after(0, self.root.destroy)


def main() -> int:
    from engine import PARAKEET_CPU_DIR
    if not PARAKEET_CPU_DIR.exists():
        log.error("Parakeet base model not found at %s", PARAKEET_CPU_DIR)
        log.error("Run: python python/fetch_model.py")
        return 2
    try:
        App().run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
