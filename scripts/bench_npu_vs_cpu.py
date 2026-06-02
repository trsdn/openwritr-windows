"""Full E2E benchmark: NPU encoder + CPU decoder/joint, vs CPU-only baseline.

We replace onnx-asr's encoder session with one that uses QNN HTP, then run
the same recognize() calls and compare to the pure-CPU baseline.
"""
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
MDL_INT8 = APPDATA / "models" / "parakeet-tdt-0.6b-v3"
WAVS = [MDL_INT8 / "test_en.wav", MDL_INT8 / "test_de.wav"]
RUNS = 5


def npu_session(model_path: Path, ctx_path: Path) -> ort.InferenceSession:
    ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    npu = [d for d in devs if d.device.type == ort.OrtHardwareDeviceType.NPU]
    so = ort.SessionOptions()
    so.add_provider_for_devices(npu, {
        "backend_path": qnn_ep.get_qnn_htp_path(),
        "htp_performance_mode": "burst",
        "enable_htp_fp16_precision": "1",
        "context_cache_enable": "1",
        "context_file_path": str(ctx_path),
    })
    # If a compiled context already exists, load it directly (much faster).
    if ctx_path.exists():
        return ort.InferenceSession(str(ctx_path), sess_options=so)
    return ort.InferenceSession(str(model_path), sess_options=so)


def main() -> int:
    print("=== CPU baseline (INT8) ===", flush=True)
    cpu = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", str(MDL_INT8), quantization="int8")
    _ = cpu.recognize(str(WAVS[0]))  # warm
    cpu_times = {}
    for w in WAVS:
        ts = []
        for _ in range(RUNS):
            t = time.perf_counter()
            text = cpu.recognize(str(w))
            ts.append((time.perf_counter() - t) * 1000)
        cpu_times[w.name] = ts
        print(f"  {w.name}: mean={statistics.mean(ts):.1f} ms  min={min(ts):.1f}  ->  {text!r}", flush=True)

    print("\n=== NPU encoder (FP32 + HTP FP16 auto-cast) + CPU decoder/joint ===", flush=True)
    # Load FP32 model variant via onnx-asr to share its preprocessor+decoder,
    # then swap the encoder session for an NPU one.
    npu_model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", str(MDL_FP32))
    enc_ctx = MDL_FP32 / "encoder.qnn_ctx.onnx"
    enc_src = MDL_FP32 / "encoder-model.onnx"
    print(f"  compiling/loading NPU encoder (ctx={'cached' if enc_ctx.exists() else 'fresh'}) …", flush=True)
    t0 = time.perf_counter()
    npu_enc = npu_session(enc_src, enc_ctx)
    print(f"  NPU encoder ready in {time.perf_counter()-t0:.1f}s", flush=True)
    npu_model._encoder = npu_enc
    _ = npu_model.recognize(str(WAVS[0]))  # warm
    npu_times = {}
    for w in WAVS:
        ts = []
        for _ in range(RUNS):
            t = time.perf_counter()
            text = npu_model.recognize(str(w))
            ts.append((time.perf_counter() - t) * 1000)
        npu_times[w.name] = ts
        print(f"  {w.name}: mean={statistics.mean(ts):.1f} ms  min={min(ts):.1f}  ->  {text!r}", flush=True)

    print("\n=== summary ===")
    print(f"{'file':<14} {'CPU mean':>12} {'NPU mean':>12} {'speedup':>10}")
    for name in cpu_times:
        c = statistics.mean(cpu_times[name])
        n = statistics.mean(npu_times[name])
        sp = c / n if n else float("inf")
        print(f"{name:<14} {c:>10.1f}ms {n:>10.1f}ms {sp:>9.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
