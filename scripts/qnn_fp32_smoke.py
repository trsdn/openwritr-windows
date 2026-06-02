"""Try loading the FP32 Parakeet encoder on QNN HTP with FP16 precision."""
from __future__ import annotations
import os
import sys
import time
import traceback
from pathlib import Path

import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

MDL = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "OpenWritr" / "models" / "parakeet-tdt-0.6b-v3-fp32"
ENCODER = MDL / "encoder-model.onnx"


def main() -> int:
    print(f"ort {ort.__version__}, qnn {qnn_ep.__version__}", flush=True)
    ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    npu = [d for d in devs if d.device.type == ort.OrtHardwareDeviceType.NPU]
    if not npu:
        print("no NPU device — abort"); return 1
    print(f"NPU device: vendor={npu[0].device.vendor} id={npu[0].device.device_id}", flush=True)

    so = ort.SessionOptions()
    so.add_provider_for_devices(npu, {
        "backend_path": qnn_ep.get_qnn_htp_path(),
        "htp_performance_mode": "burst",
        "enable_htp_fp16_precision": "1",
        "context_cache_enable": "1",
        "context_file_path": str(MDL / "encoder.qnn_ctx.onnx"),
    })
    print(f"compiling encoder for HTP — first time can take several minutes …", flush=True)
    t0 = time.perf_counter()
    try:
        enc = ort.InferenceSession(str(ENCODER), sess_options=so)
        dt = time.perf_counter() - t0
        print(f"  loaded in {dt:.1f}s — providers: {enc.get_providers()}", flush=True)
    except Exception:
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
