"""
Validate that a freshly-built NPU encoder actually loads + runs on the
Qualcomm Hexagon HTP through ORT's QNN execution provider.

This is the gate the build_npu_encoder.py output has to pass before we
point the Rust app at it. Failure mode we explicitly check for is the
historic /Expand 'invalid expand shape' runtime crash.

Usage:
    python scripts/test_npu_encoder.py \
        --encoder C:/.../parakeet-htp-int8-v2/encoder-model.onnx
"""

import argparse
import sys
import time
from pathlib import Path

import os

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qep


MEL_BINS = 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--cpu", action="store_true", help="Force CPU EP — sanity check the surgery before risking NPU")
    args = ap.parse_args()

    enc = Path(args.encoder)
    print(f"validating: {enc}")
    if not enc.exists():
        print(f"  MISSING: {enc}", file=sys.stderr)
        return 2

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 2

    if args.cpu:
        print("EP: CPU (sanity)")
        sess = ort.InferenceSession(str(enc), sess_options=so, providers=["CPUExecutionProvider"])
    else:
        # Plugin-EP architecture (QNN EP 2.0+): register the EP library, then
        # bind via OrtEpDevice instead of providers=[...]. Mirrors how the
        # Rust app does it via with_devices().
        os.add_dll_directory(qep.LIB_DIR_FULL_PATH)
        ort.register_execution_provider_library("QNNExecutionProvider", qep.get_library_path())
        all_devs = ort.get_ep_devices()
        print(f"registered EPs see {len(all_devs)} device(s):")
        for d in all_devs:
            print(f"  ep={d.ep_name} type={d.device.type}")
        npu_devs = [d for d in all_devs
                    if d.device.type == ort.OrtHardwareDeviceType.NPU
                    and d.ep_name == "QNNExecutionProvider"]
        if not npu_devs:
            print("  no QNN-backed NPU device found", file=sys.stderr)
            return 3
        print(f"EP: QNN ({len(npu_devs)} NPU device(s), HTP backend, burst perf)")
        qnn_opts = {
            "backend_type": "htp",
            "htp_performance_mode": "burst",
            # Allow HTP to run unsupported-in-INT8 ops (e.g. LayerNorm) as FP16
            # instead of failing OpConfig validation. The standard escape hatch
            # when a QDQ pattern doesn't map cleanly to a single HTP kernel.
            "enable_htp_fp16_precision": "1",
            "htp_graph_finalization_optimization_mode": "3",
        }
        so.add_provider_for_devices(npu_devs, qnn_opts)
        sess = ort.InferenceSession(str(enc), sess_options=so)

    print(f"session ready. inputs: {[(i.name, i.shape, i.type) for i in sess.get_inputs()]}")
    print(f"               outputs: {[(o.name, o.shape, o.type) for o in sess.get_outputs()]}")

    # Auto-detect window length from the session's input shape (handles both
    # the 28-s and 8-s compiled binaries without hardcoding).
    T_FIXED = sess.get_inputs()[0].shape[-1]
    print(f"window: {T_FIXED} mel frames ({T_FIXED/100:.1f} s of audio)")

    # Synthetic mel features at the calibrated fixed length.
    rng = np.random.default_rng(0)
    audio_signal = rng.standard_normal((1, MEL_BINS, T_FIXED)).astype(np.float32)
    # length dtype: int64 for the original encoder-frozen.onnx; int32 for the
    # AI-Hub-compiled QNN context binary wrapper (--truncate_64bit_io rewrote it).
    length_dtype = np.int32 if sess.get_inputs()[1].type == 'tensor(int32)' else np.int64
    length = np.array([T_FIXED], dtype=length_dtype)

    print("running 1 inference...")
    t0 = time.time()
    out = sess.run(None, {"audio_signal": audio_signal, "length": length})
    dt = time.time() - t0
    print(f"  OK in {dt*1000:.0f} ms")
    print(f"  outputs[0] shape={out[0].shape} dtype={out[0].dtype}  min={out[0].min():.3f} max={out[0].max():.3f}")
    if len(out) > 1:
        print(f"  outputs[1] shape={out[1].shape} dtype={out[1].dtype}  value={out[1].tolist()}")

    # Warm + 3 timed runs for a latency feel.
    t0 = time.time(); sess.run(None, {"audio_signal": audio_signal, "length": length}); _ = time.time()-t0
    ts = []
    for _ in range(3):
        t = time.time()
        sess.run(None, {"audio_signal": audio_signal, "length": length})
        ts.append((time.time()-t)*1000)
    print(f"  steady-state: {[f'{x:.0f}ms' for x in ts]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
