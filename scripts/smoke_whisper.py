"""Quick offline smoke test of the Whisper NPU engine on the bundled test WAVs."""
from __future__ import annotations
import os, sys, time, wave
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import numpy as np
from scipy.io import wavfile
from whisper_npu import WhisperNpuEngine

APPDATA = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "OpenWritr"
DIR = APPDATA / "models" / "whisper-large-v3-turbo-qnn"
WAVS = [APPDATA / "models" / "parakeet-tdt-0.6b-v3" / "test_en.wav",
        APPDATA / "models" / "parakeet-tdt-0.6b-v3" / "test_de.wav"]


def load_wav(p: Path) -> tuple[np.ndarray, int]:
    sr, a = wavfile.read(str(p))
    if a.dtype == np.int16: a = a.astype(np.float32) / 32768.0
    elif a.dtype == np.int32: a = a.astype(np.float32) / 2147483648.0
    elif a.dtype == np.uint8: a = (a.astype(np.float32) - 128.0) / 128.0
    else: a = a.astype(np.float32)
    if a.ndim == 2: a = a.mean(axis=1)
    return a.astype(np.float32), sr


def resample16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, 16000)
    return resample_poly(audio, 16000 // g, sr // g).astype(np.float32)


def main() -> int:
    import logging
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    eng = WhisperNpuEngine(DIR)
    for w in WAVS:
        audio, sr = load_wav(w)
        audio = resample16k(audio, sr)
        t0 = time.perf_counter()
        text = eng.transcribe(audio)
        print(f"{w.name}  ({(time.perf_counter()-t0)*1000:.0f} ms)  ->  {text!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
