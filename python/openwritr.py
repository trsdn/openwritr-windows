"""
OpenWritr for Windows — push-to-talk voice-to-text tray app.

- Tray icon (pystray)
- Hold Ctrl+Shift+Space to record; release to transcribe + paste at caret.
- Local ASR: NVIDIA Parakeet TDT 0.6B v3 (INT8) via onnx-asr.
- Model dir: %LOCALAPPDATA%\\OpenWritr\\models\\parakeet-tdt-0.6b-v3
"""
from __future__ import annotations
import os
import sys
import json
import time
import queue
import threading
import logging
from pathlib import Path

import numpy as np
import sounddevice as sd
from pynput import keyboard
import pyperclip
from PIL import Image, ImageDraw
import pystray
import onnx_asr

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


# ---------- Settings ----------
DEFAULTS = {
    "hotkey_modifiers": ["ctrl", "shift"],  # plus 'space'
    "auto_paste": True,
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
            blocksize=1600,  # 100 ms
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
    def __init__(self, settings, recorder, transcriber, status_cb) -> None:
        self.settings = settings
        self.recorder = recorder
        self.transcriber = transcriber
        self.status_cb = status_cb
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
                        self.status_cb("recording")
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
                self.status_cb("idle")
                if dur < float(self.settings.get("min_record_seconds", 0.25)):
                    log.info("recording too short (%.2fs), discarding", dur)
                    return
                threading.Thread(target=self._run_transcribe, args=(audio,), daemon=True).start()

    def _run_transcribe(self, audio: np.ndarray):
        try:
            self.status_cb("transcribing")
            text = self.transcriber.transcribe(audio)
            if text and self.settings.get("auto_paste", True):
                paste_text(text)
            self.status_cb("idle")
        except Exception:
            log.exception("transcription failed")
            self.status_cb("error")

    def run(self):
        listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release, suppress=False
        )
        listener.start()
        return listener


# ---------- Tray ----------
def make_icon(state: str = "idle") -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = {
        "idle": (74, 144, 226, 255),
        "recording": (220, 38, 38, 255),
        "transcribing": (245, 158, 11, 255),
        "error": (107, 114, 128, 255),
    }.get(state, (74, 144, 226, 255))
    d.rounded_rectangle((24, 10, 40, 38), radius=8, fill=color)
    d.rectangle((31, 38, 33, 50), fill=color)
    d.rectangle((20, 50, 44, 52), fill=color)
    return img


class TrayApp:
    def __init__(self, settings, transcriber):
        self.settings = settings
        self.recorder = AudioRecorder()
        self.transcriber = transcriber
        self.icon = pystray.Icon(
            APP_NAME,
            make_icon("idle"),
            APP_NAME,
            menu=pystray.Menu(
                pystray.MenuItem(lambda _: f"Hotkey: {self._hotkey_label()}", None, enabled=False),
                pystray.MenuItem(
                    "Auto-paste",
                    self._toggle_paste,
                    checked=lambda item: self.settings.get("auto_paste", True),
                ),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Open log folder", self._open_logs),
                pystray.MenuItem("Quit", self._quit),
            ),
        )
        self.engine = HotkeyEngine(settings, self.recorder, self.transcriber, self._set_state)

    def _hotkey_label(self) -> str:
        mods = "+".join(m.capitalize() for m in self.settings.get("hotkey_modifiers", []))
        return f"{mods}+Space (hold)"

    def _toggle_paste(self, _icon, _item):
        self.settings["auto_paste"] = not self.settings.get("auto_paste", True)
        save_settings(self.settings)

    def _open_logs(self, _icon, _item):
        os.startfile(LOG_DIR)

    def _quit(self, _icon, _item):
        self.icon.stop()

    def _set_state(self, state: str):
        try:
            self.icon.icon = make_icon(state)
        except Exception:
            pass

    def run(self):
        self.engine.run()
        self.icon.run()


def main() -> int:
    if not MODEL_DIR.exists():
        log.error("Model directory not found: %s", MODEL_DIR)
        log.error("Run: python python/fetch_model.py")
        return 2
    settings = load_settings()
    transcriber = Transcriber(MODEL_DIR)
    app = TrayApp(settings, transcriber)
    log.info("OpenWritr ready — %s", app._hotkey_label())
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
