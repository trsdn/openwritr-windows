"""
OpenWritr for Windows — push-to-talk voice-to-text tray app.

Architecture:
- Tk root on the MAIN thread owns the floating overlay window
- pystray runs detached on a background thread (Windows tray icon)
- pynput keyboard listener runs on its own thread (hotkey FSM)
- sounddevice InputStream owns the audio thread (16 kHz mono ring buffer)
- onnx-asr (Parakeet TDT v3 INT8) on a worker thread, results marshalled
  back to the main thread via root.after()

Default hotkey: hold Ctrl+Shift+Space to record. Releasing transcribes the
captured audio and pastes it at the caret.
"""
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
import onnx_asr

from overlay import Overlay
import sounds

APP_NAME = "OpenWritr"
APPDATA = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME
MODEL_DIR = APPDATA / "models" / "parakeet-tdt-0.6b-v3"
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
    "hotkey_modifiers": ["ctrl", "shift"],
    "auto_paste": True,
    "overlay": True,
    "sounds": True,
    "min_record_seconds": 0.25,
    "max_record_seconds": 60,
}


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return {**DEFAULTS, **json.loads(SETTINGS_PATH.read_text("utf-8"))}
        except Exception as e:
            log.warning("settings load failed: %s", e)
    return dict(DEFAULTS)


def save_settings(s: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(s, indent=2), "utf-8")


# ---------- Audio capture ----------
class AudioRecorder:
    def __init__(self, samplerate: int = SAMPLE_RATE) -> None:
        self.samplerate = samplerate
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None

    def _cb(self, indata, frames, time_info, status):
        if status:
            log.debug("audio status: %s", status)
        self._q.put(indata.copy())

    def start(self) -> None:
        if self._stream is not None:
            return
        while not self._q.empty():
            self._q.get_nowait()
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=CHANNELS,
            dtype="float32",
            callback=self._cb,
            blocksize=1600,
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
    def __init__(self, model_dir: Path) -> None:
        log.info("loading Parakeet TDT v3 from %s", model_dir)
        t0 = time.time()
        self.model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3", str(model_dir), quantization="int8"
        )
        log.info("model ready in %.2fs", time.time() - t0)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        t0 = time.time()
        text = self.model.recognize(audio)
        log.info(
            "transcribed %.1fs audio in %.2fs -> %r",
            len(audio) / SAMPLE_RATE, time.time() - t0, text,
        )
        return text.strip() if isinstance(text, str) else ""


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
TRIGGER = keyboard.Key.space


class HotkeyEngine:
    def __init__(self, settings, recorder, transcriber, on_state) -> None:
        self.settings = settings
        self.recorder = recorder
        self.transcriber = transcriber
        self.on_state = on_state   # called as on_state(new_state, text_or_none)
        self._pressed_mods = {m: False for m in MOD_MAP}
        self._recording = False
        self._record_started_at = 0.0
        self._lock = threading.Lock()

    def _mods_ok(self) -> bool:
        wanted = set(self.settings.get("hotkey_modifiers", []))
        return all(self._pressed_mods.get(m, False) for m in wanted)

    def _on_press(self, key):
        for name, keys in MOD_MAP.items():
            if key in keys:
                self._pressed_mods[name] = True
                return
        if key == TRIGGER and self._mods_ok():
            with self._lock:
                if not self._recording:
                    self._recording = True
                    self._record_started_at = time.time()
                    try:
                        self.recorder.start()
                        self.on_state("listening", None)
                        if self.settings.get("sounds", True):
                            sounds.play_start()
                        log.info("recording started")
                    except Exception:
                        log.exception("recorder start failed")
                        self._recording = False

    def _on_release(self, key):
        for name, keys in MOD_MAP.items():
            if key in keys:
                self._pressed_mods[name] = False
        if key == TRIGGER:
            with self._lock:
                if not self._recording:
                    return
                self._recording = False
                dur = time.time() - self._record_started_at
                audio = self.recorder.stop()
                if self.settings.get("sounds", True):
                    sounds.play_stop()
                if dur < float(self.settings.get("min_record_seconds", 0.25)):
                    log.info("recording too short (%.2fs), discarding", dur)
                    self.on_state("idle", None)
                    return
                threading.Thread(target=self._run_transcribe, args=(audio,), daemon=True).start()

    def _run_transcribe(self, audio: np.ndarray):
        try:
            self.on_state("transcribing", None)
            text = self.transcriber.transcribe(audio)
            if text and self.settings.get("auto_paste", True):
                paste_text(text)
            self.on_state("done", text or None)
        except Exception:
            log.exception("transcription failed")
            self.on_state("error", None)

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
    def __init__(self, settings: dict, on_quit) -> None:
        self.settings = settings
        self.on_quit = on_quit
        self.icon = pystray.Icon(
            APP_NAME,
            make_tray_icon("idle"),
            APP_NAME,
            menu=pystray.Menu(
                pystray.MenuItem(lambda _: f"Hotkey: {self._hotkey_label()}", None, enabled=False),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Auto-paste", self._toggle("auto_paste"),
                                 checked=lambda i: self.settings.get("auto_paste", True)),
                pystray.MenuItem("Show overlay", self._toggle("overlay"),
                                 checked=lambda i: self.settings.get("overlay", True)),
                pystray.MenuItem("Play sounds", self._toggle("sounds"),
                                 checked=lambda i: self.settings.get("sounds", True)),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Open log folder", lambda *_: os.startfile(LOG_DIR)),
                pystray.MenuItem("Quit", lambda *_: self.on_quit()),
            ),
        )

    def _hotkey_label(self) -> str:
        mods = "+".join(m.capitalize() for m in self.settings.get("hotkey_modifiers", []))
        return f"{mods}+Space (hold)"

    def _toggle(self, key: str):
        def handler(*_):
            self.settings[key] = not self.settings.get(key, True)
            save_settings(self.settings)
        return handler

    def set_state(self, state: str) -> None:
        try:
            self.icon.icon = make_tray_icon(state)
        except Exception:
            pass

    def run_detached(self) -> None:
        self.icon.run_detached()

    def stop(self) -> None:
        try:
            self.icon.stop()
        except Exception:
            pass


# ---------- App orchestration ----------
class App:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.transcriber = Transcriber(MODEL_DIR)
        self.recorder = AudioRecorder()
        self.root = tk.Tk()
        self.root.withdraw()  # hide invisible root; only the overlay shows
        self.overlay = Overlay(self.root)
        self.tray = TrayController(self.settings, self.quit)
        self.engine = HotkeyEngine(self.settings, self.recorder, self.transcriber, self._on_state)
        self._listener = None

    def _on_state(self, state: str, text: str | None) -> None:
        # Marshal everything to the Tk main thread.
        self.root.after(0, lambda: self._apply_state(state, text))

    def _apply_state(self, state: str, text: str | None) -> None:
        self.tray.set_state(state if state != "done" else "idle")
        if not self.settings.get("overlay", True):
            self.overlay.hide()
            return
        if state == "listening":
            self.overlay.show("listening")
        elif state == "transcribing":
            self.overlay.show("transcribing")
        elif state == "enhancing":
            self.overlay.show("enhancing")
        elif state == "done":
            self.overlay.show_briefly("done", 900)
        elif state in ("idle", "error"):
            self.overlay.hide()

    def run(self) -> None:
        log.info("OpenWritr ready — Hotkey: Ctrl+Shift+Space (hold)")
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
    if not MODEL_DIR.exists():
        log.error("Model directory not found: %s", MODEL_DIR)
        log.error("Run: python python/fetch_model.py")
        return 2
    try:
        App().run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
