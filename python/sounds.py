"""UI cues using Windows system sounds.

We play short professionally-designed WAVs from C:\\Windows\\Media via
winsound.PlaySound (async, non-blocking, no playback device kept open).
Falls back silently if the file is missing or audio is unavailable.
"""
from __future__ import annotations
import os
import winsound

MEDIA = r"C:\Windows\Media"
_START = os.path.join(MEDIA, "Windows Navigation Start.wav")
_STOP = os.path.join(MEDIA, "Windows Menu Command.wav")

_FLAGS = winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT


def _play(path: str) -> None:
    if not os.path.isfile(path):
        return
    try:
        winsound.PlaySound(path, _FLAGS)
    except Exception:
        pass


def play_start() -> None:
    _play(_START)


def play_stop() -> None:
    _play(_STOP)
