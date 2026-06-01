#!/usr/bin/env python3
"""
Export NVIDIA Parakeet TDT 0.6B v3 from NeMo to ONNX for Windows-ARM runtime.

Outputs (in --out-dir):
  encoder.onnx      FastConformer encoder
  decoder.onnx      RNNT/TDT predictor (LSTM)
  joint.onnx        Joint network + TDT duration head
  tokenizer.model   SentencePiece tokenizer
  config.json       sample rate, mel bins, durations [0..4], vocab size

Then quantize to FP16 with onnxruntime.quantization.

Run on the macOS dev box or a Linux box with NeMo installed; ship only the
ONNX artifacts to the Windows ARM target.

Usage:
  pip install nemo_toolkit[asr] onnx onnxruntime
  python scripts/export_parakeet_onnx.py \
      --model nvidia/parakeet-tdt-0.6b-v3 \
      --out-dir models/parakeet-tdt-0.6b-v3 \
      --fp16
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path


def export(model_name: str, out_dir: Path, fp16: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] loading {model_name} via NeMo …")
    import nemo.collections.asr as nemo_asr  # type: ignore

    asr = nemo_asr.models.ASRModel.from_pretrained(model_name)
    asr.eval()

    print("[2/4] exporting to ONNX (encoder + decoder + joint) …")
    # NeMo's ASRModel.export() emits multi-file ONNX for transducer models.
    asr.export(str(out_dir / "model.onnx"))

    # The export step writes encoder/decoder/joint as separate files with a
    # common stem; normalize the names so the Rust runtime can find them.
    for variant in ("encoder", "decoder", "joint"):
        candidates = list(out_dir.glob(f"*{variant}*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"NeMo did not produce a {variant} ONNX file")
        target = out_dir / f"{variant}.onnx"
        if candidates[0] != target:
            shutil.move(str(candidates[0]), str(target))
        print(f"   -> {target.name}")

    print("[3/4] writing tokenizer + config …")
    if hasattr(asr, "tokenizer") and hasattr(asr.tokenizer, "tokenizer_model"):
        shutil.copy(asr.tokenizer.tokenizer_model, out_dir / "tokenizer.model")

    cfg = {
        "sample_rate": 16000,
        "mel_bins": 128,
        "tdt_durations": [0, 1, 2, 3, 4],
        "vocab_size": getattr(asr.tokenizer, "vocab_size", None),
        "blank_id": getattr(asr, "blank_id", None),
        "source_model": model_name,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    if fp16:
        print("[4/4] quantizing to FP16 …")
        from onnxruntime.transformers.float16 import convert_float_to_float16  # type: ignore
        import onnx  # type: ignore

        for name in ("encoder.onnx", "decoder.onnx", "joint.onnx"):
            p = out_dir / name
            m = onnx.load(str(p))
            m_fp16 = convert_float_to_float16(m, keep_io_types=True)
            onnx.save(m_fp16, str(p))
            print(f"   -> {name} (fp16)")
    else:
        print("[4/4] skipping fp16 conversion (use --fp16 to enable)")

    print(f"\nDone. Artifacts in {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--out-dir", type=Path, default=Path("models/parakeet-tdt-0.6b-v3"))
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    export(args.model, args.out_dir, args.fp16)


if __name__ == "__main__":
    main()
