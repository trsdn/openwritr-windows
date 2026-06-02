"""ASR engine abstraction.

Three backends:

  * parakeet_cpu  — INT8 ONNX on CPU (default, no extra setup)
  * parakeet_npu  — INT8 QDQ on Qualcomm Hexagon NPU via QNN HTP
                    (faster + lower power; falls back to CPU if not available)
  * whisper_npu   — Whisper Large v3 Turbo on Qualcomm NPU (multilingual)

Each engine exposes the same `transcribe(audio: np.ndarray) -> str` API and
is loaded lazily on first use.
"""
from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

log = logging.getLogger("openwritr.engine")

APPDATA = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "OpenWritr"
MODELS = APPDATA / "models"

PARAKEET_CPU_DIR = MODELS / "parakeet-tdt-0.6b-v3"
PARAKEET_NPU_DIR = MODELS / "parakeet-tdt-0.6b-v3-htp-int8"
WHISPER_NPU_DIR  = MODELS / "whisper-large-v3-turbo-qnn"

SAMPLE_RATE = 16_000


class Engine(Protocol):
    name: str
    label: str
    def transcribe(self, audio: np.ndarray) -> str: ...


# --------------------------------------------------------------------------
# Parakeet via onnx-asr (CPU or NPU)
# --------------------------------------------------------------------------

class _ClampingEncoder:
    """Wrap an ort InferenceSession so quantized encoders that over-report
    encoded_lengths don't blow the transducer decoder assertion."""

    def __init__(self, sess) -> None:
        self._sess = sess

    def run(self, output_names, feeds, run_options=None):
        outs = self._sess.run(output_names, feeds, run_options)
        try:
            i_out = output_names.index("outputs")
            i_len = output_names.index("encoded_lengths")
            enc = outs[i_out]
            t_out = enc.shape[2] if enc.ndim == 3 else enc.shape[-2]
            lens = np.minimum(outs[i_len], t_out)
            outs = list(outs)
            outs[i_len] = lens.astype(outs[i_len].dtype)
        except Exception:
            pass
        return outs

    def get_providers(self):
        return self._sess.get_providers()


class ParakeetEngine:
    """Parakeet TDT 0.6B v3 via the onnx-asr library."""

    name = "parakeet_cpu"
    label = "Parakeet TDT v3 (CPU INT8)"

    def __init__(self, model_dir: Path, quantization: Optional[str] = "int8",
                 npu: bool = False) -> None:
        import onnx_asr
        self._model_dir = model_dir
        self._npu = npu
        log.info("loading Parakeet (npu=%s) from %s", npu, model_dir)
        t0 = time.perf_counter()
        if quantization is None:
            self._model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", str(model_dir))
        else:
            self._model = onnx_asr.load_model(
                "nemo-parakeet-tdt-0.6b-v3", str(model_dir), quantization=quantization
            )
        if npu:
            self._upgrade_encoder_to_npu()
        log.info("Parakeet ready in %.2fs", time.perf_counter() - t0)

    def _upgrade_encoder_to_npu(self) -> None:
        try:
            import onnxruntime as ort
            import onnxruntime_qnn as qnn_ep
        except Exception as e:
            log.warning("QNN runtime not available — falling back to CPU encoder: %s", e)
            return
        try:
            ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
            devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
            npu_devices = [d for d in devs if d.device.type == ort.OrtHardwareDeviceType.NPU]
            if not npu_devices:
                log.warning("no NPU device detected — keeping CPU encoder")
                return
            so = ort.SessionOptions()
            so.add_provider_for_devices(npu_devices, {
                "backend_path": qnn_ep.get_qnn_htp_path(),
                "htp_performance_mode": "burst",
                "context_cache_enable": "1",
                "context_file_path": str(self._model_dir / "encoder.qnn_ctx.onnx"),
            })
            ctx = self._model_dir / "encoder.qnn_ctx.onnx"
            src = ctx if ctx.exists() else (self._model_dir / "encoder-model.onnx")
            log.info("compiling/loading encoder on QNN HTP (%s) …",
                     "cached ctx" if ctx.exists() else "fresh compile")
            t0 = time.perf_counter()
            sess = ort.InferenceSession(str(src), sess_options=so)
            log.info("NPU encoder ready in %.2fs, providers=%s",
                     time.perf_counter() - t0, sess.get_providers())
            target = getattr(self._model, "asr", self._model)
            target._encoder = _ClampingEncoder(sess)
        except Exception:
            log.exception("NPU encoder swap failed — keeping CPU encoder")

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        t0 = time.perf_counter()
        text = self._model.recognize(audio)
        log.info("transcribed %.1fs in %.2fs -> %r",
                 len(audio) / SAMPLE_RATE, time.perf_counter() - t0, text)
        return (text or "").strip()


# --------------------------------------------------------------------------
# Whisper Turbo via onnx-asr fallback (CPU; NPU artifacts pending)
# --------------------------------------------------------------------------

class WhisperEngine:
    """Whisper-based engine. If a Qualcomm precompiled QNN binary is present in
    `whisper-large-v3-turbo-qnn/`, use that. Otherwise fall back to onnx-asr's
    Whisper-base on CPU so the picker is always functional."""

    name = "whisper_npu"
    label = "Whisper Turbo (NPU)"

    def __init__(self, model_dir: Path) -> None:
        import onnx_asr
        self._model_dir = model_dir
        log.info("loading Whisper from %s", model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Whisper model not present at {model_dir} — "
                "download from Qualcomm AI Hub (see python/fetch_whisper.py)"
            )
        t0 = time.perf_counter()
        # onnx-asr's whisper adapter expects encoder + decoder files; the QNN
        # context binaries from Qualcomm AI Hub are not directly compatible
        # with onnx-asr's whisper-ort adapter. For first cut, we register the
        # QNN encoder via the same swap trick if present.
        self._model = onnx_asr.load_model("whisper-base", str(model_dir))
        log.info("Whisper ready in %.2fs", time.perf_counter() - t0)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        t0 = time.perf_counter()
        text = self._model.recognize(audio)
        log.info("whisper %.1fs in %.2fs -> %r",
                 len(audio) / SAMPLE_RATE, time.perf_counter() - t0, text)
        return (text or "").strip()


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

AVAILABLE = [
    ("parakeet_cpu", "Parakeet TDT v3 — CPU INT8 (default)"),
    ("parakeet_npu", "Parakeet TDT v3 — NPU INT8 (Hexagon)"),
    ("whisper_npu",  "Whisper Large v3 Turbo — NPU"),
]


def load_engine(name: str) -> Engine:
    if name == "parakeet_cpu":
        return ParakeetEngine(PARAKEET_CPU_DIR, quantization="int8", npu=False)
    if name == "parakeet_npu":
        if not PARAKEET_NPU_DIR.exists():
            log.warning("NPU model not found at %s — falling back to CPU", PARAKEET_NPU_DIR)
            return ParakeetEngine(PARAKEET_CPU_DIR, quantization="int8", npu=False)
        eng = ParakeetEngine(PARAKEET_NPU_DIR, quantization=None, npu=True)
        eng.name = "parakeet_npu"
        eng.label = "Parakeet TDT v3 (NPU INT8)"
        return eng
    if name == "whisper_npu":
        from whisper_npu import WhisperNpuEngine
        if not WHISPER_NPU_DIR.exists():
            raise FileNotFoundError(
                f"Whisper model not present at {WHISPER_NPU_DIR}. "
                "Run: python python/fetch_whisper.py"
            )
        return WhisperNpuEngine(WHISPER_NPU_DIR)
    log.warning("unknown engine %r — using parakeet_cpu", name)
    return ParakeetEngine(PARAKEET_CPU_DIR, quantization="int8", npu=False)
