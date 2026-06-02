"""Re-quantize Parakeet encoder with 50 real FLEURS samples (EN + DE).

Replaces the 8-sample POC calibration with a realistic multilingual set.
Writes the new model into a separate dir so we can A/B compare.
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
    quantize_static, QuantFormat, QuantType,
    CalibrationDataReader, CalibrationMethod,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

APPDATA = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "OpenWritr"
SRC_DIR = APPDATA / "models" / "parakeet-tdt-0.6b-v3-fp32"
DST_DIR = APPDATA / "models" / "parakeet-tdt-0.6b-v3-htp-int8-fleurs"
CALIB_ROOT = APPDATA / "calibration" / "fleurs"
LANGS = ["en_us", "de_de"]
SAMPLES_PER_LANG = 25


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    from scipy.io import wavfile
    sr, audio = wavfile.read(str(path))
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio.astype(np.float32)
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, 16000)
    return resample_poly(audio, 16000 // g, sr // g).astype(np.float32)


def build_features(preproc_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    sess = ort.InferenceSession(str(preproc_path), providers=["CPUExecutionProvider"])
    items: list[tuple[np.ndarray, np.ndarray]] = []
    for lang in LANGS:
        wavs = sorted((CALIB_ROOT / lang).glob("*.wav"))[:SAMPLES_PER_LANG]
        print(f"  {lang}: {len(wavs)} wavs", flush=True)
        for w in wavs:
            a, sr = load_wav(w)
            a16 = resample(a, sr)
            # Cap at 30 s to keep calibration tractable.
            a16 = a16[: 30 * 16000]
            lens = np.asarray([len(a16)], dtype=np.int64)
            feats, feats_len = sess.run(
                ["features", "features_lens"],
                {"waveforms": a16[None, :], "waveforms_lens": lens},
            )
            items.append((feats.astype(np.float32), feats_len.astype(np.int64)))
    return items


class Reader(CalibrationDataReader):
    def __init__(self, items):
        self._iter = iter([{"audio_signal": f, "length": L} for f, L in items])

    def get_next(self):
        return next(self._iter, None)


def main() -> int:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    src = SRC_DIR / "encoder-model.onnx"
    dst = DST_DIR / "encoder-model.onnx"

    pre_path = DST_DIR / "encoder-preproc.onnx"
    if not pre_path.exists():
        print("pre-processing (shape inference + optimization) …", flush=True)
        t0 = time.perf_counter()
        quant_pre_process(
            input_model_path=str(src),
            output_model_path=str(pre_path),
            skip_optimization=False, skip_onnx_shape=False, skip_symbolic_shape=False,
            auto_merge=True,
            save_as_external_data=True, all_tensors_to_one_file=True,
            external_data_location="encoder-preproc.data",
            external_data_size_threshold=1024,
        )
        print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    else:
        print(f"reusing existing preprocessed graph: {pre_path}", flush=True)

    print("building calibration features from FLEURS …", flush=True)
    items = build_features(SRC_DIR / "nemo128.onnx")
    print(f"  {len(items)} calibration samples", flush=True)

    print("quantize_static (QDQ, per-channel, MinMax) …", flush=True)
    t0 = time.perf_counter()
    quantize_static(
        model_input=str(pre_path),
        model_output=str(dst),
        calibration_data_reader=Reader(items),
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
        },
    )
    print(f"  done in {time.perf_counter()-t0:.1f}s -> {dst}", flush=True)

    import shutil
    for name in ("decoder_joint-model.onnx", "nemo128.onnx", "vocab.txt", "config.json"):
        s = SRC_DIR / name
        if s.exists():
            shutil.copy2(s, DST_DIR / name)
    print(f"companion files copied", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
