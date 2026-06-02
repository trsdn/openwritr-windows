"""Soft, warm UI cues — low fundamentals, gentle attack, long exponential
decay. Sounds like a muffled woodblock or felt-hammer tap, much more
mellow than the click-y Windows system sounds.

The start cue is a slightly higher pitch than the stop cue, so the two
events are still distinguishable without being sharp.
"""
from __future__ import annotations
import os
import tempfile
import wave
from pathlib import Path
import winsound

import numpy as np

_SR = 44_100
_DUR = 0.32  # long enough for the body of the tone to ring out


def _soft_tone(freq: float, duration: float = _DUR) -> np.ndarray:
    n = int(_SR * duration)
    t = np.arange(n, dtype=np.float32) / _SR

    # Body: fundamental + one octave above at -12 dB (warmer than 3rd-harmonic mix).
    body = (np.sin(2 * np.pi * freq * t)
            + 0.25 * np.sin(2 * np.pi * freq * 2 * t)).astype(np.float32)

    # Soft attack (~25 ms) and exponential decay.
    attack_n = int(_SR * 0.025)
    env = np.ones_like(body)
    env[:attack_n] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, attack_n))
    decay = np.exp(-t / (duration * 0.32)).astype(np.float32)

    # 80 Hz sub thump for warmth in the first 60 ms.
    thump_n = int(_SR * 0.06)
    thump = (np.sin(2 * np.pi * 80 * t[:thump_n])
             * np.linspace(1.0, 0.0, thump_n) * 0.18).astype(np.float32)
    body[:thump_n] += thump

    return (body * env * decay * 0.18).astype(np.float32)


_CACHE = Path(tempfile.gettempdir()) / "openwritr-cues"
_CACHE.mkdir(parents=True, exist_ok=True)
_START_PATH = _CACHE / "start.wav"
_STOP_PATH = _CACHE / "stop.wav"


def _write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SR)
        w.writeframes(pcm.tobytes())


# Warm low fundamentals.
#   start ≈ G3  (196 Hz)  — friendly upward feel
#   stop  ≈ E3  (165 Hz)  — settled, conclusive
_write_wav(_START_PATH, _soft_tone(196.0))
_write_wav(_STOP_PATH,  _soft_tone(164.81))

_FLAGS = winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT


def play_start() -> None:
    try:
        winsound.PlaySound(str(_START_PATH), _FLAGS)
    except Exception:
        pass


def play_stop() -> None:
    try:
        winsound.PlaySound(str(_STOP_PATH), _FLAGS)
    except Exception:
        pass
