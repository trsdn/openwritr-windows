"""
Encoder graph surgery for Parakeet TDT v3 → Qualcomm Hexagon HTP.

Problem: the upstream encoder's attention masking builds its mask shape at
runtime from Shape(input) → Gather → Range → Expand. QNN HTP cannot trace
through that and bails out at the Expand node ("invalid expand shape") for
every input length we tested.

Fix: constant-fold the dynamic shape ops by:
  1. Locking the input shape to a fixed value (here: 28 s of audio = 2800
     mel frames at 10 ms hop, plus 1 for the boundary frame).
  2. Replacing the Shape→Gather→Range chain that feeds the mask Expand
     with a single Constant tensor holding the integer length the encoder
     would have produced at runtime.
  3. Re-running shape inference + constant folding to collapse what's left.

The result is a graph with static shapes everywhere, which QNN HTP can
accept end-to-end. We then quantize with `quantize_static` (INT8 QDQ, asym
activations, sym per-channel weights, MinMax calibration) against real
mel features at the fixed length and emit `encoder-model.onnx`.

Usage:
    python scripts/build_npu_encoder.py \
        --fp32-encoder C:/.../parakeet-fp32/encoder-model.onnx \
        --preprocessor C:/.../parakeet-tdt-0.6b-v3-onnx/nemo128.onnx \
        --calib-wav-glob "calib/*.wav" \
        --out-dir C:/.../parakeet-tdt-0.6b-v3-htp-int8
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


SAMPLE_RATE = 16_000
MEL_BINS = 128
# AUDIO_SECONDS and MEL_FRAMES are resolved from --seconds at runtime.
AUDIO_SECONDS = 28  # default, overridden by --seconds
MEL_FRAMES = AUDIO_SECONDS * 100 + 1


def _set_dims(value_info, shape):
    """Robustly set fixed dim values on an input/output. protobuf 'dim' is a
    oneof(dim_value, dim_param); just assigning dim_value can leave dim_param
    set. Clear() first, then assign — and assert rank to catch surprises."""
    dims = value_info.type.tensor_type.shape.dim
    assert len(dims) == len(shape), (
        f"{value_info.name}: rank mismatch (model has {len(dims)}, want {len(shape)})"
    )
    for d, v in zip(dims, shape):
        d.Clear()
        d.dim_value = v


def freeze_input_shapes(model: onnx.ModelProto, T: int) -> onnx.ModelProto:
    """Bake fixed [1, 128, T] / [1] dims into the graph inputs and outputs.
    Must run AFTER any onnx_graphsurgeon roundtrip, which otherwise drops
    the dim_value annotations on its way back through gs.export_onnx."""
    out_T = (T // 8) + 1
    by_name = {i.name: i for i in model.graph.input}
    if "audio_signal" in by_name: _set_dims(by_name["audio_signal"], [1, MEL_BINS, T])
    if "length"       in by_name: _set_dims(by_name["length"],       [1])
    by_name = {o.name: o for o in model.graph.output}
    if "outputs"          in by_name: _set_dims(by_name["outputs"],          [1, 1024, out_T])
    if "encoded_lengths"  in by_name: _set_dims(by_name["encoded_lengths"],  [1])
    return model


def replace_dynamic_mask_with_constant(model: onnx.ModelProto, T: int) -> onnx.ModelProto:
    """Replace the Shape→Gather→Range subgraph that feeds /Expand with a
    constant Range tensor of length ceil(T/8)."""
    graph = gs.import_onnx(model)
    out_T = (T // 8) + 1

    # The mask path: /Shape → /Gather → /Range → fed into /Expand
    # We replace /Range_output_0 with a literal Constant.
    range_const = gs.Constant(
        name="static_range_for_expand",
        values=np.arange(out_T, dtype=np.int64),
    )

    patched = 0
    for node in graph.nodes:
        if node.op == "Expand" and node.name == "/Expand":
            # First input is /Range_output_0 — replace with our constant.
            node.inputs[0] = range_const
            patched += 1
    if patched == 0:
        print("WARN: did not find /Expand node to patch", file=sys.stderr)

    # Also try to fold the static shape Constant for the Where's
    # ConstantOfShape companion.
    # (Best-effort; quant_pre_process will clean up the rest.)
    graph.cleanup().toposort()
    return gs.export_onnx(graph)


def calibration_mels(preproc_path: str, wav_glob: str, T: int):
    """Generate mel features at the fixed length by running the preprocessor
    on user-supplied calibration WAVs (or synthetic noise if none given)."""
    sess = ort.InferenceSession(preproc_path, providers=["CPUExecutionProvider"])
    paths = sorted(glob.glob(wav_glob)) if wav_glob else []
    if not paths:
        print("calibration: no WAVs; using 8 synthetic noise samples")
        rng = np.random.default_rng(42)
        for _ in range(8):
            n = AUDIO_SECONDS * SAMPLE_RATE
            wave = (rng.standard_normal((1, n)) * 0.05).astype(np.float32)
            yield wave
    else:
        import wave as wavmod
        for p in paths[:32]:
            with wavmod.open(p, "rb") as w:
                sr = w.getframerate()
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            pcm = pcm.astype(np.float32) / 32768.0
            if sr != SAMPLE_RATE:
                # naive resample; acceptable for calibration
                idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * SAMPLE_RATE / sr)).astype(np.int64)
                pcm = pcm[idx]
            target = AUDIO_SECONDS * SAMPLE_RATE
            if pcm.size < target:
                pcm = np.pad(pcm, (0, target - pcm.size))
            else:
                pcm = pcm[:target]
            yield pcm.reshape(1, -1)


class MelCalibReader(CalibrationDataReader):
    def __init__(self, preproc_path: str, wav_glob: str, T: int):
        self.sess = ort.InferenceSession(preproc_path, providers=["CPUExecutionProvider"])
        self.T = T
        self._gen = calibration_mels(preproc_path, wav_glob, T)

    def get_next(self):
        try:
            wave = next(self._gen)
        except StopIteration:
            return None
        wave_len = np.array([wave.shape[1]], dtype=np.int64)
        feats, _ = self.sess.run(None, {"waveforms": wave, "waveforms_lens": wave_len})
        feats = feats.astype(np.float32)
        # Pad/truncate mel time axis to T
        if feats.shape[2] < self.T:
            pad = self.T - feats.shape[2]
            feats = np.pad(feats, ((0, 0), (0, 0), (0, pad)))
        else:
            feats = feats[:, :, : self.T]
        length = np.array([self.T], dtype=np.int64)
        return {"audio_signal": feats, "length": length}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32-encoder", required=True)
    ap.add_argument("--preprocessor", required=True)
    ap.add_argument("--calib-wav-glob", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seconds", type=int, default=28,
                    help="static audio window length in seconds (default 28; "
                         "shorter = faster NPU inference but caps push-to-talk length)")
    args = ap.parse_args()

    global AUDIO_SECONDS, MEL_FRAMES
    AUDIO_SECONDS = args.seconds
    MEL_FRAMES = AUDIO_SECONDS * 100 + 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    T = MEL_FRAMES
    print(f"window: {AUDIO_SECONDS} s → {MEL_FRAMES} mel frames")

    print(f"loading FP32 encoder: {args.fp32_encoder}")
    m = onnx.load(args.fp32_encoder, load_external_data=True)

    # Surgery FIRST, freeze LAST. gs.import_onnx → gs.export_onnx silently
    # drops the dim_value annotations on graph inputs/outputs, so anything we
    # set before the roundtrip is lost.
    print("replacing dynamic mask subgraph with constant Range tensor")
    m = replace_dynamic_mask_with_constant(m, T)

    print(f"freezing inputs to [1, {MEL_BINS}, {T}]")
    m = freeze_input_shapes(m, T)

    # Sanity check: the freeze MUST have stuck, otherwise quant_pre_process'
    # symbolic shape inference can't fold the dynamic-mask subgraph and
    # /Expand fails at runtime with "invalid expand shape".
    by_name = {i.name: i for i in m.graph.input}
    a = by_name["audio_signal"].type.tensor_type.shape.dim
    assert [d.dim_value for d in a] == [1, MEL_BINS, T], (
        f"freeze did not stick: audio_signal dims = {[(d.dim_value, d.dim_param) for d in a]}"
    )

    frozen_path = out_dir / "encoder-frozen.onnx"
    onnx.save(
        m, str(frozen_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="encoder-frozen.onnx.data",
    )
    print(f"  wrote {frozen_path}")

    print("running quant_pre_process (shape inference + symbolic folding)")
    pre_path = out_dir / "encoder-pre.onnx"
    quant_pre_process(
        input_model_path=str(frozen_path),
        output_model_path=str(pre_path),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
        auto_merge=True,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        external_data_location="encoder-pre.onnx.data",
        external_data_size_threshold=1024,
    )

    print("running quantize_static (INT8 QDQ for QNN HTP)")
    final_path = out_dir / "encoder-model.onnx"
    reader = MelCalibReader(args.preprocessor, args.calib_wav_glob, T)
    # QNN HTP requires SCALAR (per-tensor) zero-points everywhere. per_channel=True
    # generates multi-dim zero points for self-attn MatMul, which makes both ORT
    # CPU's QLinearMatMul and QNN HTP FinalizeGraphs (error 6020) reject the model.
    # Stick to per-tensor and symmetric weights — the standard QNN-HTP QDQ recipe.
    quantize_static(
        model_input=str(pre_path),
        model_output=str(final_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
        use_external_data_format=True,
        extra_options={
            "WeightSymmetric": True,
            "ActivationSymmetric": False,
        },
    )

    print(f"DONE: {final_path}")
    print()
    print("validate with:")
    print(f"  python scripts/test_npu_encoder.py --encoder {final_path}")


if __name__ == "__main__":
    main()
