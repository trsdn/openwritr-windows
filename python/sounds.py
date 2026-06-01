"""Start/stop UI cues — Windows port of macOS SoundManager.swift.

Generates two short sine tones (ascending start ping, descending stop ping)
and plays them via sounddevice on a dedicated output stream. Same shape and
duration (~80 ms) as the macOS originals.
"""
from __future__ import annotations
import numpy as np
import sounddevice as sd

_SR = 44_100


def _tone(freq: float, duration: float, ascending: bool) -> np.ndarray:
    n = int(_SR * duration)
    t = np.arange(n, dtype=np.float32) / _SR
    progress = t / duration
    multiplier = 0.8 + 0.4 * progress if ascending else 1.2 - 0.4 * progress
    inst_freq = freq * multiplier
    attack = np.minimum(1.0, progress * 10)
    decay = (1.0 - progress) ** (2 if ascending else 3)
    env = attack * decay
    return (env * np.sin(2 * np.pi * inst_freq * t) * 0.3).astype(np.float32)


_START = _tone(440, 0.08, ascending=True)
_STOP = _tone(330, 0.08, ascending=False)


def play_start() -> None:
    try:
        sd.play(_START, _SR, blocking=False)
    except Exception:
        pass


def play_stop() -> None:
    try:
        sd.play(_STOP, _SR, blocking=False)
    except Exception:
        pass
