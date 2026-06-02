"""Whisper Large v3 Turbo on Qualcomm Hexagon NPU.

Uses Qualcomm AI Hub's pre-compiled QAIRT context binaries (encoder +
decoder), wrapped with our own:
  - 128-bin log-mel preprocessor (numpy + scipy.fft)
  - HuggingFace `tokenizers` for detokenization
  - Autoregressive greedy decode loop with KV-cache management

Model artifact layout (under MDL_DIR):
  encoder.onnx                 — context-binary wrapper, loaded via QNN EP
  encoder_qairt_context.bin    — pre-compiled HTP encoder
  decoder.onnx                 — context-binary wrapper
  decoder_qairt_context.bin    — pre-compiled HTP decoder
  metadata.json
  tokenizer.json               — fetched separately (HF tokenizers format)
  added_tokens.json
"""
from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep
from tokenizers import Tokenizer

log = logging.getLogger("openwritr.engine.whisper")

# --- model constants ----------------------------------------------------
SAMPLE_RATE = 16_000
N_FFT = 400
HOP = 160
N_MELS = 128
N_FRAMES = 3000          # encoder expects exactly 30 s @ 100 fps
WIN_LENGTH = N_FFT       # Whisper uses N_FFT-sized Hann window
MAX_DECODE_STEPS = 200
N_DECODER_LAYERS = 4
N_HEADS = 20
HEAD_DIM = 64
SELF_CACHE_LEN = 199     # decoder.k_cache_self_*_in second-to-last dim
CROSS_CACHE_LEN = 1500
MEAN_DECODE_LEN = 200    # attention_mask width

# Whisper special-token IDs (Large v3 / Turbo, vocab 51866).
TOK_SOT          = 50258  # <|startoftranscript|>
TOK_EOT          = 50257  # <|endoftext|>


# --- mel preprocessor ---------------------------------------------------

_MEL_BASIS: np.ndarray | None = None
_HANN: np.ndarray | None = None


def _mel_basis() -> np.ndarray:
    global _MEL_BASIS
    if _MEL_BASIS is None:
        # Slaney-style 128-bin mel filterbank, 0..8000 Hz, 201-point FFT bin grid.
        n_bins = N_FFT // 2 + 1
        fft_freqs = np.linspace(0, SAMPLE_RATE / 2, n_bins, dtype=np.float64)
        f_min, f_max = 0.0, 8000.0

        def hz_to_mel(f):
            f_sp = 200.0 / 3.0
            min_log_hz = 1000.0
            min_log_mel = (min_log_hz - 0.0) / f_sp
            logstep = np.log(6.4) / 27.0
            mels = np.where(f >= min_log_hz, min_log_mel + np.log(f / min_log_hz) / logstep, f / f_sp)
            return mels

        def mel_to_hz(m):
            f_sp = 200.0 / 3.0
            min_log_hz = 1000.0
            min_log_mel = (min_log_hz - 0.0) / f_sp
            logstep = np.log(6.4) / 27.0
            freqs = np.where(m >= min_log_mel,
                             min_log_hz * np.exp(logstep * (m - min_log_mel)),
                             f_sp * m)
            return freqs

        m_min = hz_to_mel(np.array([f_min]))[0]
        m_max = hz_to_mel(np.array([f_max]))[0]
        mel_pts = mel_to_hz(np.linspace(m_min, m_max, N_MELS + 2))
        fb = np.zeros((N_MELS, n_bins), dtype=np.float32)
        for m in range(N_MELS):
            f_left, f_center, f_right = mel_pts[m], mel_pts[m + 1], mel_pts[m + 2]
            enorm = 2.0 / (f_right - f_left)
            for k in range(n_bins):
                f = fft_freqs[k]
                if f < f_left or f > f_right:
                    w = 0.0
                elif f <= f_center:
                    w = (f - f_left) / (f_center - f_left)
                else:
                    w = (f_right - f) / (f_right - f_center)
                fb[m, k] = w * enorm
        _MEL_BASIS = fb
    return _MEL_BASIS


def _hann() -> np.ndarray:
    global _HANN
    if _HANN is None:
        _HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN_LENGTH, dtype=np.float64) / WIN_LENGTH)).astype(np.float32)
    return _HANN


def log_mel_30s(audio: np.ndarray) -> np.ndarray:
    """Return (1, 128, 3000) float16 log-mel features padded/truncated to 30 s."""
    from scipy.fft import rfft

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    # Pad / truncate to exactly N_SAMPLES so we always emit 3000 frames.
    n_samples = (N_FRAMES - 1) * HOP + N_FFT
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)), mode="constant")
    else:
        audio = audio[:n_samples]

    # Center-padding (reflect) by N_FFT // 2 like torch.stft(center=True).
    pad = N_FFT // 2
    audio = np.pad(audio, (pad, pad), mode="reflect")

    win = _hann()
    n_frames = 1 + (len(audio) - N_FFT) // HOP
    n_frames = min(n_frames, N_FRAMES)
    spec = np.zeros((N_MELS, N_FRAMES), dtype=np.float32)
    fb = _mel_basis()
    for t in range(n_frames):
        seg = audio[t * HOP : t * HOP + N_FFT] * win
        s = rfft(seg)
        power = (s.real * s.real + s.imag * s.imag).astype(np.float32)
        mel = fb @ power
        spec[:, t] = np.maximum(mel, 1e-10)

    log_spec = np.log10(spec)
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float16)[None, :, :]


# --- engine -------------------------------------------------------------

_QNN_REGISTERED = False


def _qnn_session(model_path: Path, ctx_path: Path) -> ort.InferenceSession:
    global _QNN_REGISTERED
    if not _QNN_REGISTERED:
        try:
            ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
        except Exception as e:
            # Already registered by another module in this process — fine.
            if "already registered" not in str(e):
                raise
        _QNN_REGISTERED = True
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    npu = [d for d in devs if d.device.type == ort.OrtHardwareDeviceType.NPU]
    if not npu:
        raise RuntimeError("no NPU device available for Whisper engine")
    so = ort.SessionOptions()
    so.add_provider_for_devices(npu, {
        "backend_path": qnn_ep.get_qnn_htp_path(),
        "htp_performance_mode": "burst",
    })
    return ort.InferenceSession(str(model_path), sess_options=so)


class WhisperNpuEngine:
    name = "whisper_npu"
    label = "Whisper Turbo (NPU)"

    def __init__(self, model_dir: Path) -> None:
        self._dir = model_dir
        t0 = time.perf_counter()
        log.info("loading Whisper Turbo (NPU) from %s …", model_dir)
        self._enc = _qnn_session(model_dir / "encoder.onnx", model_dir / "encoder_qairt_context.bin")
        log.info("  encoder loaded (%.1fs)", time.perf_counter() - t0)
        t0 = time.perf_counter()
        self._dec = _qnn_session(model_dir / "decoder.onnx", model_dir / "decoder_qairt_context.bin")
        log.info("  decoder loaded (%.1fs)", time.perf_counter() - t0)
        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._enc_out_names = [o.name for o in self._enc.get_outputs()]
        self._dec_out_names = [o.name for o in self._dec.get_outputs()]
        log.info("Whisper NPU ready")

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        mel = log_mel_30s(audio)
        t0 = time.perf_counter()
        enc_out = self._enc.run(None, {"input_features": mel})
        cross_kv = dict(zip(self._enc_out_names, enc_out))
        log.info("  encoder %.0fms", (time.perf_counter() - t0) * 1000)

        # Initialise self-KV caches to zero.
        self_kv: dict[str, np.ndarray] = {}
        for i in range(N_DECODER_LAYERS):
            self_kv[f"k_cache_self_{i}_in"] = np.zeros(
                (N_HEADS, 1, HEAD_DIM, SELF_CACHE_LEN), dtype=np.float16,
            )
            self_kv[f"v_cache_self_{i}_in"] = np.zeros(
                (N_HEADS, 1, SELF_CACHE_LEN, HEAD_DIM), dtype=np.float16,
            )

        # Right-aligned attention mask: valid positions at the TAIL of a
        # 200-wide window. Each step we open one more slot from the right.
        attn_mask = np.full((1, 1, 1, MEAN_DECODE_LEN), -65504.0, dtype=np.float16)

        # Feed only SOT — the decoder auto-detects language and emits the
        # task / no-timestamps tokens itself.
        out_ids: list[int] = [TOK_SOT]
        t0 = time.perf_counter()
        for n in range(MEAN_DECODE_LEN - 1):
            input_id = out_ids[n]
            attn_mask[..., MEAN_DECODE_LEN - n - 1] = 0.0
            feeds = {
                "input_ids": np.array([[input_id]], dtype=np.int32),
                "attention_mask": attn_mask,
                "position_ids": np.array([n], dtype=np.int32),
            }
            feeds.update(self_kv)
            feeds.update(cross_kv)
            outs = self._dec.run(None, feeds)
            out = dict(zip(self._dec_out_names, outs))
            for i in range(N_DECODER_LAYERS):
                self_kv[f"k_cache_self_{i}_in"] = out[f"k_cache_self_{i}_out"]
                self_kv[f"v_cache_self_{i}_in"] = out[f"v_cache_self_{i}_out"]
            logits = out["logits"].reshape(-1)
            next_id = int(np.argmax(logits))
            if next_id == TOK_EOT:
                break
            out_ids.append(next_id)
        log.info("  decoder %d steps in %.0fms", n + 1, (time.perf_counter() - t0) * 1000)
        text = self._tok.decode(out_ids, skip_special_tokens=True).strip()
        return text
        text = self._tok.decode(emitted, skip_special_tokens=True).strip()
        return text
