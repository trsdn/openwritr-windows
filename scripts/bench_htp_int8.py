"""Bench the statically INT8-QDQ-quantized Parakeet encoder on QNN HTP."""
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
MDL_HTP = APPDATA / "models" / "parakeet-tdt-0.6b-v3-htp-int8"
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
        "context_cache_enable": "1",
        "context_file_path": str(MDL_HTP / "encoder.qnn_ctx.onnx"),
    })

    print("loading model wrapper …", flush=True)
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", str(MDL_HTP))

    ctx = MDL_HTP / "encoder.qnn_ctx.onnx"
    src = ctx if ctx.exists() else (MDL_HTP / "encoder-model.onnx")
    print(f"loading INT8-QDQ encoder on QNN HTP ({'cached ctx' if ctx.exists() else 'compiling'}) …", flush=True)
    t0 = time.perf_counter()
    try:
        enc = ort.InferenceSession(str(src), sess_options=so)
        print(f"  ready in {time.perf_counter()-t0:.1f}s, providers={enc.get_providers()}", flush=True)
    except Exception as e:
        print(f"  ENCODER LOAD FAILED: {type(e).__name__}: {e}", flush=True)
        return 2
    model._encoder = enc

    # Warm.
    for w in WAVS:
        _ = model.recognize(str(w))

    print(f"\n{'file':<14} {'min':>9} {'p50':>9} {'mean':>9} {'max':>9}", flush=True)
    for w in WAVS:
        ts = []
        text = ""
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
