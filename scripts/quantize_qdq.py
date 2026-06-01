"""Statically quantize the Parakeet TDT encoder to INT8 QDQ for QNN HTP.

Steps:
  1. Load the nemo128 ONNX preprocessor to convert raw audio to mel features.
  2. Build a CalibrationDataReader that yields (audio_signal, length) pairs.
  3. Run onnxruntime.quantization.quantize_static against encoder-model.onnx,
     writing encoder-model.htp_int8.onnx with QDQ ops.

Calibration set: the two test wavs (en + de) plus simple per-utterance gain
variants. Not ideal — production-quality QDQ wants ~100 diverse utterances —
but enough to validate the toolchain and get a working HTP graph.

HTP-friendly knobs:
  * QDQ format (HTP only accepts QDQ, not QOperator)
  * Symmetric activations and weights (HTP requirement)
  * Per-channel weights (recommended for accuracy)
  * Activation/weight type QUInt8/QInt8 — HTP commonly handles the standard
    8-bit shapes, fp16 fallback for the rest via the runtime flag.
"""
from __future__ import annotations
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    QuantFormat,
    QuantType,
    CalibrationDataReader,
    CalibrationMethod,
)

APPDATA = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "OpenWritr"
SRC_DIR = APPDATA / "models" / "parakeet-tdt-0.6b-v3-fp32"
DST_DIR = APPDATA / "models" / "parakeet-tdt-0.6b-v3-htp-int8"
CALIB_WAVS = [
    APPDATA / "models" / "parakeet-tdt-0.6b-v3" / "test_en.wav",
    APPDATA / "models" / "parakeet-tdt-0.6b-v3" / "test_de.wav",
]
GAIN_FACTORS = [0.5, 0.75, 1.0, 1.25]   # 8 calibration samples


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sample_width = w.getsampwidth()
        raw = w.readframes(n)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width, np.int16)
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    audio /= float(2 ** (8 * sample_width - 1))
    if ch == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    return audio, sr


def resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio.astype(np.float32)
    # Simple high-quality resample via scipy polyphase.
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, 16000)
    up, down = 16000 // g, sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def build_calibration_features(preproc_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run nemo128.onnx on raw audio and collect (features, lengths)."""
    sess = ort.InferenceSession(str(preproc_path), providers=["CPUExecutionProvider"])
    items: list[tuple[np.ndarray, np.ndarray]] = []
    for wav in CALIB_WAVS:
        a, sr = load_wav(wav)
        a16 = resample_to_16k(a, sr)
        for g in GAIN_FACTORS:
            scaled = np.clip(a16 * g, -1.0, 1.0).astype(np.float32)
            audio_batch = scaled[None, :]            # (1, N)
            lens = np.asarray([len(scaled)], dtype=np.int64)
            feats, feats_len = sess.run(["features", "features_lens"],
                                        {"waveforms": audio_batch, "waveforms_lens": lens})
            items.append((feats.astype(np.float32), feats_len.astype(np.int64)))
            print(f"  calib {wav.name} g={g:.2f}  feats={feats.shape} len={int(feats_len[0])}", flush=True)
    return items


class EncoderCalibReader(CalibrationDataReader):
    def __init__(self, items: list[tuple[np.ndarray, np.ndarray]]) -> None:
        # Encoder input names from istupakov export: audio_signal, length
        self._iter = iter([
            {"audio_signal": f, "length": L}
            for f, L in items
        ])

    def get_next(self):
        return next(self._iter, None)


def main() -> int:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    src = SRC_DIR / "encoder-model.onnx"
    dst = DST_DIR / "encoder-model.onnx"
    if not src.exists():
        print(f"missing FP32 encoder at {src}", flush=True); return 1

    # 1) Pre-process: shape-infer, optimize, write with external data so the
    # augmented graph stays under the 2 GB protobuf limit.
    pre_path = DST_DIR / "encoder-preproc.onnx"
    if not pre_path.exists():
        print(f"preprocessing model (shape inference, optimization) …", flush=True)
        from onnxruntime.quantization.shape_inference import quant_pre_process
        t0 = time.perf_counter()
        quant_pre_process(
            input_model_path=str(src),
            output_model_path=str(pre_path),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
            auto_merge=True,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            external_data_location="encoder-preproc.data",
            external_data_size_threshold=1024,
        )
        print(f"  preprocessed in {time.perf_counter()-t0:.1f}s -> {pre_path.name}", flush=True)

    print(f"building calibration features (preprocessor on CPU) …", flush=True)
    items = build_calibration_features(SRC_DIR / "nemo128.onnx")
    print(f"  {len(items)} calibration samples ready", flush=True)

    reader = EncoderCalibReader(items)
    print(f"running quantize_static (QDQ, per-channel, symmetric) …", flush=True)
    t0 = time.perf_counter()
    quantize_static(
        model_input=str(pre_path),
        model_output=str(dst),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        use_external_data_format=True,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
            "EnableSubgraph": False,
            "DisableShapeInference": False,
        },
    )
    print(f"  done in {time.perf_counter()-t0:.1f}s -> {dst}  ({dst.stat().st_size/1e6:.1f} MB)", flush=True)

    # Copy companion files for onnx-asr.
    import shutil
    for name in ("decoder_joint-model.onnx", "nemo128.onnx", "vocab.txt", "config.json"):
        s = SRC_DIR / name
        if s.exists():
            shutil.copy2(s, DST_DIR / name)
    print(f"companion files copied to {DST_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
