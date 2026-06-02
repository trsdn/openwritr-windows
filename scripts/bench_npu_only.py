"""Isolated NPU-encoder benchmark — must be run from a fresh Python process
so the QNN provider registration does not pollute the CPU-baseline sessions
that other benchmarks rely on."""
from __future__ import annotations
import os
import statistics
import sys
import time
from pathlib import Path

import onnx_asr
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

APPDATA = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "OpenWritr"
MDL_FP32 = APPDATA / "models" / "parakeet-tdt-0.6b-v3-fp32"
WAVS = [APPDATA / "models" / "parakeet-tdt-0.6b-v3" / "test_en.wav",
        APPDATA / "models" / "parakeet-tdt-0.6b-v3" / "test_de.wav"]
RUNS = 8


def main() -> int:
    ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    npu = [d for d in devs if d.device.type == ort.OrtHardwareDeviceType.NPU]

    so = ort.SessionOptions()
    so.add_provider_for_devices(npu, {
        "backend_path": qnn_ep.get_qnn_htp_path(),
        "htp_performance_mode": "burst",
        "enable_htp_fp16_precision": "1",
        "context_cache_enable": "1",
        "context_file_path": str(MDL_FP32 / "encoder.qnn_ctx.onnx"),
    })

    print("loading FP32 model wrapper (CPU sessions) …", flush=True)
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", str(MDL_FP32))
    print("loading NPU encoder context …", flush=True)
    ctx = MDL_FP32 / "encoder.qnn_ctx.onnx"
    src = ctx if ctx.exists() else (MDL_FP32 / "encoder-model.onnx")
    t0 = time.perf_counter()
    enc = ort.InferenceSession(str(src), sess_options=so)
    print(f"  encoder ready in {time.perf_counter()-t0:.1f}s, providers={enc.get_providers()}", flush=True)
    model._encoder = enc

    # Warm.
    for w in WAVS:
        _ = model.recognize(str(w))

    print(f"\n{'file':<14} {'min':>9} {'p50':>9} {'mean':>9} {'max':>9}", flush=True)
    for w in WAVS:
        ts = []
        for _ in range(RUNS):
            t = time.perf_counter()
            text = model.recognize(str(w))
            ts.append((time.perf_counter() - t) * 1000)
        print(f"{w.name:<14} {min(ts):>7.1f}ms "
              f"{statistics.median(ts):>7.1f}ms {statistics.mean(ts):>7.1f}ms "
              f"{max(ts):>7.1f}ms  -> {text!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
