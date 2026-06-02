"""Latency breakdown of the current Python ASR pipeline."""
from __future__ import annotations
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx_asr

APPDATA = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "OpenWritr"
MDL = APPDATA / "models" / "parakeet-tdt-0.6b-v3"
WAVS = [MDL / "test_en.wav", MDL / "test_de.wav"]
RUNS = 5


def main() -> int:
    t0 = time.perf_counter()
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", str(MDL), quantization="int8")
    load_s = time.perf_counter() - t0
    print(f"Cold load: {load_s*1000:7.1f} ms\n")

    # Warm up.
    _ = model.recognize(str(WAVS[0]))

    print(f"{'file':<14} {'min':>9} {'p50':>9} {'mean':>9} {'max':>9}")
    for w in WAVS:
        times = []
        for _ in range(RUNS):
            t = time.perf_counter()
            text = model.recognize(str(w))
            times.append((time.perf_counter() - t) * 1000)
        print(f"{w.name:<14} {min(times):>7.1f}ms "
              f"{statistics.median(times):>7.1f}ms {statistics.mean(times):>7.1f}ms "
              f"{max(times):>7.1f}ms   -> {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

