"""
Submit the Parakeet TDT v3 encoder to Qualcomm AI Hub for HTP compilation.

We do this in two AI Hub jobs:
  1. submit_quantize_job: FP32 encoder + FLEURS calibration → INT8/INT16 QDQ
     ONNX, using AI Hub's own quantizer (aimet-onnx). Their recipe targets
     HTP specifically, so we side-step the LayerNorm OpConfig 3110 failures
     we hit doing it locally with onnxruntime.quantization.
  2. submit_compile_job: quantized ONNX → QNN context binary (.bin) for
     Snapdragon X Elite. This .bin loads directly via ORT's QNN EP with
     `qnn_context_binary_file=...` and skips graph partitioning entirely.

Setup (one time, on the dev box):
    pip install qai-hub
    qai-hub configure --api_token <TOKEN_FROM_AIHUB.QUALCOMM.COM>

Usage:
    python scripts/aihub_compile_encoder.py \\
        --fp32-encoder C:/.../parakeet-htp-int8-v2/encoder-frozen.onnx \\
        --preprocessor C:/.../parakeet-tdt-0.6b-v3-fp32/nemo128.onnx \\
        --calib-glob "C:/.../calibration/fleurs/*/*.wav" \\
        --max-calib 64 \\
        --out C:/.../parakeet-tdt-0.6b-v3-htp-int8/encoder-model.bin

The output .bin is what Rust's `build_npu_session` will consume — replaces the
ONNX-encoder-on-HTP-via-EP-partitioning path with a precompiled HTP binary.
"""

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile  # handles both int16 and float32 WAV


MEL_BINS = 128
SAMPLE_RATE = 16_000
# AUDIO_SECONDS, T_FIXED resolved from --seconds at runtime.
AUDIO_SECONDS = 28
T_FIXED = AUDIO_SECONDS * 100 + 1


def load_mel_calibration(preproc_path: str, wav_glob: str, max_samples: int):
    """Run the same nemo128 preprocessor the runtime uses, on real audio,
    to produce shape-aligned mel features for AI Hub calibration.

    Returns list[np.ndarray (1, 128, 2801)] — AI Hub's calibration_data format
    is dict[input_name, list[np.ndarray]].
    """
    import onnxruntime as ort
    sess = ort.InferenceSession(preproc_path, providers=["CPUExecutionProvider"])
    paths = sorted(glob.glob(wav_glob, recursive=True))[:max_samples]
    if not paths:
        raise SystemExit(f"no calibration WAVs at {wav_glob}")
    print(f"calibration: preprocessing {len(paths)} WAV files...")
    feats_list = []
    target_samples = AUDIO_SECONDS * SAMPLE_RATE
    t0 = time.time()
    for i, p in enumerate(paths):
        sr, pcm = wavfile.read(p)
        if pcm.dtype == np.int16:
            pcm = pcm.astype(np.float32) / 32768.0
        elif pcm.dtype == np.int32:
            pcm = pcm.astype(np.float32) / 2147483648.0
        elif pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        if sr != SAMPLE_RATE:
            idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * SAMPLE_RATE / sr)).astype(np.int64)
            pcm = pcm[idx]
        if pcm.size < target_samples:
            pcm = np.pad(pcm, (0, target_samples - pcm.size))
        else:
            pcm = pcm[:target_samples]
        wave_arr = pcm.reshape(1, -1)
        wave_len = np.array([wave_arr.shape[1]], dtype=np.int64)
        feats, _ = sess.run(None, {"waveforms": wave_arr, "waveforms_lens": wave_len})
        feats = feats.astype(np.float32)
        if feats.shape[2] < T_FIXED:
            feats = np.pad(feats, ((0, 0), (0, 0), (0, T_FIXED - feats.shape[2])))
        else:
            feats = feats[:, :, :T_FIXED]
        feats_list.append(np.ascontiguousarray(feats))
        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{len(paths)} ({time.time()-t0:.1f}s elapsed)")
    print(f"  done in {time.time()-t0:.1f}s")
    return feats_list


def pick_device(hub):
    """Pick the Snapdragon X Elite entry from AI Hub's device list."""
    devs = hub.get_devices()
    elite = [d for d in devs if "X Elite" in d.name]
    if not elite:
        names = sorted({d.name for d in devs})
        for n in names: print(f"  - {n}")
        raise SystemExit("no 'X Elite' device on AI Hub — check the list above")
    # Prefer the plainest name (without trailing suffix) if multiple matches exist
    elite.sort(key=lambda d: (len(d.name), d.name))
    dev = elite[0]
    print(f"target device: {dev.name}")
    return dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32-encoder", required=True,
                    help="encoder-frozen.onnx from build_npu_encoder.py (static-shape FP32)")
    ap.add_argument("--preprocessor", required=True,
                    help="nemo128.onnx (same preprocessor the Rust runtime uses)")
    ap.add_argument("--calib-glob", required=True,
                    help='glob for WAVs, e.g. "C:/.../calibration/fleurs/*/*.wav"')
    ap.add_argument("--max-calib", type=int, default=64,
                    help="cap calibration samples (default 64, AI Hub recommends 100–500)")
    ap.add_argument("--out", required=True,
                    help="output path for the QNN context binary (.bin)")
    ap.add_argument("--skip-quantize", action="store_true",
                    help="skip the quantize step and compile the FP32 ONNX directly "
                         "(useful for sanity-checking compile, will be slow at inference time)")
    ap.add_argument("--reuse-quantize-job", default="",
                    help="reuse an already-completed quantize job by its ID (skips re-upload + re-quantize)")
    ap.add_argument("--seconds", type=int, default=28,
                    help="static audio window length in seconds — MUST match the value used "
                         "in build_npu_encoder.py when generating the input encoder")
    ap.add_argument("--qairt-version", default="2.45",
                    help="AI Hub QAIRT major.minor line (default: 2.45, matching the shipped runtime)")
    args = ap.parse_args()

    if not re.fullmatch(r"\d+\.\d+", args.qairt_version):
        sys.exit("--qairt-version must use major.minor syntax, for example 2.45")

    global AUDIO_SECONDS, T_FIXED
    AUDIO_SECONDS = args.seconds
    T_FIXED = AUDIO_SECONDS * 100 + 1
    print(f"window: {AUDIO_SECONDS} s → {T_FIXED} mel frames")

    enc = Path(args.fp32_encoder)
    if not enc.exists(): sys.exit(f"missing: {enc}")
    data_file = enc.parent / (enc.name + ".data")
    if not data_file.exists():
        sys.exit(f"missing external data file: {data_file}")
    # AI Hub requires "ONNX model directory format" for models with external
    # weights — pass the parent directory, not the .onnx path itself. Parent
    # must contain ONLY the .onnx and its .data (and nothing else), otherwise
    # the upload includes unrelated files. We expect the caller to provide a
    # clean staging directory.
    other_files = [p for p in enc.parent.iterdir() if p.name not in (enc.name, data_file.name)]
    if other_files:
        sys.exit(
            f"directory containing {enc.name} must hold only the .onnx + .onnx.data; "
            f"found extra: {[p.name for p in other_files]}. "
            f"Stage a clean dir first."
        )
    encoder_arg = str(enc.parent)

    import qai_hub as hub

    device = pick_device(hub)

    if args.reuse_quantize_job:
        print(f"reusing quantize job {args.reuse_quantize_job}")
        prior_job = hub.get_job(args.reuse_quantize_job)
        quantized_model = prior_job.get_target_model()
        if quantized_model is None:
            sys.exit(f"prior quantize job not in RESULTS_READY: status={prior_job.get_status()}")
        model_for_compile = quantized_model
        print(f"  got quantized model: {quantized_model.model_id}")
    elif args.skip_quantize:
        model_for_compile = encoder_arg
        print("skipping quantize — submitting FP32 directly")
    else:
        feats = load_mel_calibration(args.preprocessor, args.calib_glob, args.max_calib)
        lengths = [np.array([T_FIXED], dtype=np.int64) for _ in feats]
        calibration_data = {"audio_signal": feats, "length": lengths}

        print(f"\nsubmitting quantize job ({len(feats)} samples)...")
        t0 = time.time()
        quantize_job = hub.submit_quantize_job(
            model=encoder_arg,
            calibration_data=calibration_data,
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT16,
            name="parakeet-encoder-htp-quantize",
        )
        print(f"  url: {quantize_job.url}")
        print("  waiting for completion (this can take 5–30 minutes for a 600M param encoder)...")
        quantized_model = quantize_job.get_target_model()
        if quantized_model is None:
            sys.exit(f"quantize failed: status={quantize_job.get_status()}\n  url: {quantize_job.url}")
        print(f"  quantize done in {time.time()-t0:.0f}s")
        model_for_compile = quantized_model

    compile_options = (
        "--target_runtime qnn_context_binary "
        f"--truncate_64bit_io --qairt_version {args.qairt_version}"
    )
    print(
        f"\nsubmitting compile job (target: {device.name}, "
        f"qnn_context_binary, QAIRT {args.qairt_version})..."
    )
    t0 = time.time()
    compile_job = hub.submit_compile_job(
        model=model_for_compile,
        device=device,
        input_specs={
            "audio_signal": ((1, MEL_BINS, T_FIXED), "float32"),
            "length":       ((1,), "int64"),
        },
        options=compile_options,
        name="parakeet-encoder-htp-compile",
    )
    print(f"  url: {compile_job.url}")
    print("  waiting for completion (typically 5–15 minutes)...")
    target_model = compile_job.get_target_model()
    if target_model is None:
        sys.exit(f"compile failed: status={compile_job.get_status()}\n  url: {compile_job.url}")
    print(f"  compile done in {time.time()-t0:.0f}s")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    target_model.download(str(out))
    provenance = {
        "ai_hub_job_url": str(compile_job.url),
        "ai_hub_model_id": getattr(target_model, "model_id", None),
        "device": device.name,
        "options": compile_options,
        "qairt_version": args.qairt_version,
    }
    provenance_path = out.with_name(out.name + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"\nDONE: {out} ({out.stat().st_size/1e6:.1f} MB)")
    print(f"provenance: {provenance_path}")
    print()
    print("Next steps in Rust:")
    print(f'  - in src/asr/parakeet.rs::build_npu_session, replace commit_from_file(path) with')
    print(f'    a session that loads the .bin via QNNExecutionProvider.qnn_context_binary_file')
    print(f'  - download.rs: ship the .bin alongside the other model files in the NPU dir')


if __name__ == "__main__":
    main()
