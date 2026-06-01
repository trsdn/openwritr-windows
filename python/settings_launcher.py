"""Subprocess launcher for the WebView2 settings window.

The settings UI lives in its own process (settings_window.py) so that
pywebview's one-shot start() restriction doesn't collide with the main
tray app, and so that we can re-open the dialog as many times as the
user wants without state issues.
"""
from __future__ import annotations
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

APPDATA = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "OpenWritr"
SETTINGS_PATH = APPDATA / "settings.json"


class SettingsLauncher:
    def __init__(self, on_changed: Callable[[], None]) -> None:
        self.on_changed = on_changed
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def open(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                # Already open — bring to front by re-launching another instance
                # is awkward; we just no-op and the existing window stays.
                return
            here = Path(__file__).resolve().parent
            script = here / "settings_window.py"
            mtime_before = SETTINGS_PATH.stat().st_mtime if SETTINGS_PATH.exists() else 0
            self._proc = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(here),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            threading.Thread(target=self._watch, args=(mtime_before,), daemon=True).start()

    def _watch(self, mtime_before: float) -> None:
        proc = self._proc
        if proc is None:
            return
        proc.wait()
        try:
            mtime_after = SETTINGS_PATH.stat().st_mtime if SETTINGS_PATH.exists() else 0
        except Exception:
            mtime_after = mtime_before
        if mtime_after > mtime_before:
            try:
                self.on_changed()
            except Exception:
                pass
        with self._lock:
            self._proc = None
