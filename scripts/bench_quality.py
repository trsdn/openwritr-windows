"""Quality + latency check: NPU INT8 (FLEURS-calibrated) vs CPU INT8 baseline.

Runs both pipelines on a held-out subset of FLEURS samples and reports:
  - Word Error Rate (NPU vs CPU as the reference — we can't easily get
    FLEURS ground-truth transcripts without `datasets`, so this measures
    *drift* from quantization, not absolute WER).
  - Per-utterance latency, p50.

If the drift is low (<5 %) and the latency is faster, the NPU model is a
keeper.
"""
from __future__ import annotations
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx_asr
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

APPDATA = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "OpenWritr"
MDL_CPU = APPDATA / "models" / "parakeet-tdt-0.6b-v3"
MDL_NPU = APPDATA / "models" / "parakeet-tdt-0.6b-v3-htp-int8"
CALIB_ROOT = APPDATA / "calibration" / "fleurs"
LANGS = ["en_us", "de_de"]
N_PER_LANG = 10            # held-out subset, must not overlap calibration


def held_out_wavs() -> list[Path]:
    # The two Sherpa test clips are NOT in FLEURS — perfect held-out signal.
    test_dir = APPDATA / "models" / "parakeet-tdt-0.6b-v3"
    return [test_dir / "test_en.wav", test_dir / "test_de.wav"]


_NORM = re.compile(r"[^\w\s']")


def norm(text: str) -> list[str]:
    text = _NORM.sub(" ", (text or "").lower())
    return text.split()


def wer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    # Standard DP edit distance.
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): dp[i][0] = i
    for j in range(len(h) + 1): dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i-1] == h[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[len(r)][len(h)] / len(r)


def npu_session(model_path: Path, ctx_path: Path) -> ort.InferenceSession:
    ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    npu = [d for d in devs if d.device.type == ort.OrtHardwareDeviceType.NPU]
    so = ort.SessionOptions()
    so.add_provider_for_devices(npu, {
        "backend_path": qnn_ep.get_qnn_htp_path(),
        "htp_performance_mode": "burst",
        "context_cache_enable": "1",
        "context_file_path": str(ctx_path),
    })
    src = ctx_path if ctx_path.exists() else model_path
    return ort.InferenceSession(str(src), sess_options=so)


def run_cpu(wavs: list[Path]) -> tuple[list[str], list[float]]:
    cpu = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", str(MDL_CPU), quantization="int8")
    _ = cpu.recognize(str(wavs[0]))  # warm
    texts, times = [], []
    for w in wavs:
        t = time.perf_counter()
        s = cpu.recognize(str(w))
        times.append((time.perf_counter() - t) * 1000)
        texts.append(s or "")
    return texts, times


class _ClampingEncoder:
    """Wrap an ort InferenceSession to clamp encoded_lengths to the actual T_out."""

    def __init__(self, sess: ort.InferenceSession) -> None:
        self._sess = sess
        self._debug = True

    def run(self, output_names, feeds, run_options=None):
        outs = self._sess.run(output_names, feeds, run_options)
        try:
            i_out = output_names.index("outputs")
            i_len = output_names.index("encoded_lengths")
            enc = outs[i_out]
            lens = outs[i_len]
            t_out = enc.shape[2] if enc.ndim == 3 else enc.shape[-2]
            if self._debug:
                print(f"  [clamp] enc.shape={enc.shape}  raw_lens={lens.tolist()}  t_out={t_out}", flush=True)
                self._debug = False  # one-time debug only
            lens = np.minimum(lens, t_out)
            outs = list(outs)
            outs[i_len] = lens.astype(lens.dtype)
        except Exception as e:
            print(f"  [clamp] failed: {e}", flush=True)
        return outs

    def get_providers(self):
        return self._sess.get_providers()


def run_npu(wavs: list[Path]) -> tuple[list[str], list[float]]:
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", str(MDL_NPU))
    ctx = MDL_NPU / "encoder.qnn_ctx.onnx"
    raw = npu_session(MDL_NPU / "encoder-model.onnx", ctx)
    wrapped = _ClampingEncoder(raw)
    # `model` is an Adapter wrapper; the actual ASR is at `model.asr`.
    target = getattr(model, "asr", model)
    target._encoder = wrapped
    _ = model.recognize(str(wavs[0]))  # warm
    texts, times = [], []
    for w in wavs:
        t = time.perf_counter()
        s = model.recognize(str(w))
        times.append((time.perf_counter() - t) * 1000)
        texts.append(s or "")
    return texts, times


def main() -> int:
    wavs = held_out_wavs()
    if not wavs:
        print("no held-out wavs found — run scripts/fetch_fleurs.py first"); return 1
    print(f"held-out set: {len(wavs)} wavs", flush=True)

    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("cpu", "both"):
        print("\n--- CPU INT8 (reference) ---", flush=True)
        cpu_texts, cpu_times = run_cpu(wavs)
        for w, t, ms in zip(wavs, cpu_texts, cpu_times):
            print(f"  [{ms:5.0f}ms] {w.name:<14}  {t[:80]!r}", flush=True)
        print(f"CPU p50={statistics.median(cpu_times):.0f} ms  mean={statistics.mean(cpu_times):.0f} ms", flush=True)
        # Save for later comparison if running in parts.
        out = APPDATA / "calibration" / "cpu_ref.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for w, t, ms in zip(wavs, cpu_texts, cpu_times):
                f.write(f"{w.name}\t{ms:.1f}\t{t}\n")
        print(f"saved reference to {out}")

    if mode in ("npu", "both"):
        print("\n--- NPU INT8 (FLEURS calibration) ---", flush=True)
        npu_texts, npu_times = run_npu(wavs)
        for w, t, ms in zip(wavs, npu_texts, npu_times):
            print(f"  [{ms:5.0f}ms] {w.name:<14}  {t[:80]!r}", flush=True)
        print(f"NPU p50={statistics.median(npu_times):.0f} ms  mean={statistics.mean(npu_times):.0f} ms", flush=True)
        if mode == "both":
            print("\n--- drift (NPU vs CPU) ---", flush=True)
            wers = [wer(c, n) for c, n in zip(cpu_texts, npu_texts)]
            for w, c, n, e in zip(wavs, cpu_texts, npu_texts, wers):
                marker = " <-- drift" if e > 0.0 else ""
                print(f"  {e*100:5.1f}%  {w.name}{marker}")
                if e > 0:
                    print(f"        CPU: {c!r}")
                    print(f"        NPU: {n!r}")
            avg = statistics.mean(wers)
            print(f"\nmean drift WER vs CPU: {avg*100:.2f}%", flush=True)
            sp = statistics.median(cpu_times) / statistics.median(npu_times)
            print(f"speedup (p50): {sp:.2f}x", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
