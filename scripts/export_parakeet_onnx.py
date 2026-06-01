#!/usr/bin/env python3
"""
Export NVIDIA Parakeet TDT 0.6B v3 from NeMo to ONNX for Windows-ARM runtime.

Produces, in --out-dir:
    encoder.onnx       FastConformer encoder (inputs: audio_signal, length;
                       outputs: outputs, encoded_lengths)
    decoder.onnx       LSTM predictor with streaming state inputs/outputs
                       (inputs: targets, target_length, states.1 (h), states.2 (c);
                        outputs: outputs, states.1, states.2)
    joint.onnx         Joint + duration head
                       (inputs: encoder_outputs, decoder_outputs;
                        output: outputs of shape [B, T, U, V + |durations|])
    tokenizer.model    SentencePiece tokenizer
    config.json        sample_rate, n_mels, tdt_durations, vocab_size,
                       blank_id, predictor_hidden, predictor_layers

Optionally quantizes the three .onnx files to FP16.

Run on a Linux/macOS box with NeMo installed:

    pip install "nemo_toolkit[asr]" onnx onnxruntime onnxruntime-tools
    python scripts/export_parakeet_onnx.py \
        --model nvidia/parakeet-tdt-0.6b-v3 \
        --out-dir models/parakeet-tdt-0.6b-v3 \
        --fp16

Ship only the resulting directory to:
    %LOCALAPPDATA%\\OpenWritr\\models\\parakeet-tdt-0.6b-v3\\
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path


def export(model_name: str, out_dir: Path, fp16: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/5] loading {model_name} via NeMo …")
    import nemo.collections.asr as nemo_asr  # type: ignore

    asr = nemo_asr.models.ASRModel.from_pretrained(model_name)
    asr.eval()
    asr.freeze()

    print("[2/5] exporting encoder + decoder + joint as separate ONNX graphs …")
    # NeMo's ASRModel.export handles RNNT/TDT models by emitting three ONNX
    # files when given a stem with a `.onnx` suffix; the helper names them
    # encoder-<stem>.onnx etc.
    stem = out_dir / "model.onnx"
    asr.export(
        str(stem),
        check_trace=False,
        dynamic_axes=None,        # NeMo picks safe defaults
        onnx_opset_version=17,
    )
    for variant in ("encoder", "decoder", "joint"):
        candidates = sorted(out_dir.glob(f"*{variant}*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"NeMo did not produce a {variant} ONNX file")
        target = out_dir / f"{variant}.onnx"
        if candidates[0] != target:
            if target.exists():
                target.unlink()
            shutil.move(str(candidates[0]), str(target))
        print(f"   -> {target.name}")

    print("[3/5] writing tokenizer …")
    if hasattr(asr, "tokenizer") and hasattr(asr.tokenizer, "tokenizer_model"):
        shutil.copy(asr.tokenizer.tokenizer_model, out_dir / "tokenizer.model")
    elif hasattr(asr, "tokenizer") and hasattr(asr.tokenizer, "model_path"):
        shutil.copy(asr.tokenizer.model_path, out_dir / "tokenizer.model")
    else:
        print("   !! no SentencePiece model on tokenizer — write it manually")

    print("[4/5] writing config.json …")
    pred_cfg = getattr(asr.cfg.decoder, "prednet", {})
    cfg = {
        "sample_rate": int(getattr(asr.cfg.preprocessor, "sample_rate", 16000)),
        "n_mels": int(getattr(asr.cfg.preprocessor, "features", 128)),
        "tdt_durations": list(getattr(asr.cfg.model_defaults, "tdt_durations", [0, 1, 2, 3, 4])),
        "vocab_size": int(asr.tokenizer.vocab_size),
        "blank_id": int(asr.tokenizer.vocab_size),  # TDT convention: blank == vocab_size
        "predictor_hidden": int(getattr(pred_cfg, "pred_hidden", 640)),
        "predictor_layers": int(getattr(pred_cfg, "pred_rnn_layers", 1)),
        "source_model": model_name,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"   {cfg}")

    if fp16:
        print("[5/5] quantizing to FP16 …")
        from onnxruntime.transformers.float16 import convert_float_to_float16  # type: ignore
        import onnx  # type: ignore
        for name in ("encoder.onnx", "decoder.onnx", "joint.onnx"):
            p = out_dir / name
            m = onnx.load(str(p))
            m_fp16 = convert_float_to_float16(m, keep_io_types=True)
            onnx.save(m_fp16, str(p))
            print(f"   -> {name} (fp16)")
    else:
        print("[5/5] skipping fp16 conversion (use --fp16 to enable)")

    print(f"\nDone. Ship {out_dir} to %LOCALAPPDATA%\\OpenWritr\\models\\parakeet-tdt-0.6b-v3\\")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--out-dir", type=Path, default=Path("models/parakeet-tdt-0.6b-v3"))
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    export(args.model, args.out_dir, args.fp16)


if __name__ == "__main__":
    main()
