"""
Wrap a raw QNN context binary (.bin from AI Hub) into a thin ONNX file
containing a single EPContext node, so ORT's QNN execution provider can
load it via the standard InferenceSession path.

Why this is needed: AI Hub's submit_compile_job(target_runtime=qnn_context_binary)
returns the raw QAIRT context binary. ORT consumes context binaries via the
EPContext-node ONNX wrapper format, not the raw .bin directly. The standard
ORT helper (onnxruntime.tools.qnn.gen_qnn_ctx_onnx_model) needs a
model_net.json that AI Hub does not produce — so we build the wrapper
manually from the AI Hub Model's input_spec/output_spec.

Usage:
    python scripts/wrap_qnn_context_binary.py \\
        --bin C:/.../parakeet-tdt-0.6b-v3-htp-int8/encoder-model.bin \\
        --aihub-model-id mn7w31p8m \\
        --out C:/.../parakeet-tdt-0.6b-v3-htp-int8/encoder-model.onnx

The output is `encoder-model.onnx` sitting next to `encoder-model.bin`. ORT
loads the .onnx via QNN EP and reads the .bin transparently.
"""

import argparse
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

# AI Hub TensorSpec dtype string → onnx TensorProto enum
DTYPE = {
    "float32": TensorProto.FLOAT,
    "float16": TensorProto.FLOAT16,
    "int64":   TensorProto.INT64,
    "int32":   TensorProto.INT32,
    "uint8":   TensorProto.UINT8,
    "int8":    TensorProto.INT8,
    "uint16":  TensorProto.UINT16,
    "int16":   TensorProto.INT16,
}


def make_tensor_value_info(spec):
    return helper.make_tensor_value_info(spec.name, DTYPE[spec.dtype], list(spec.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="QNN context binary (.bin from AI Hub)")
    ap.add_argument("--aihub-model-id", required=True,
                    help="AI Hub model_id (e.g. mn7w31p8m) — used to fetch input/output specs")
    ap.add_argument("--out", required=True, help="output wrapper ONNX path (sits next to the .bin)")
    ap.add_argument("--embed", action="store_true",
                    help="inline the .bin bytes into the ONNX as a node attribute "
                         "(embed_mode=1); produces a big ONNX but avoids any external "
                         "path resolution by the EP")
    args = ap.parse_args()

    bin_path = Path(args.bin)
    out_path = Path(args.out)
    if not bin_path.exists():
        sys.exit(f"missing: {bin_path}")
    if bin_path.parent != out_path.parent:
        sys.exit(f"the wrapper ONNX must live in the same directory as the .bin "
                 f"(EPContext uses a relative path). bin parent: {bin_path.parent}, "
                 f"out parent: {out_path.parent}")

    import qai_hub as hub
    m = hub.get_model(args.aihub_model_id)
    print(f"AI Hub model: {m.name} (type={m.model_type})")
    inputs = m.input_spec[None]
    outputs = m.output_spec[None]
    print(f"inputs : {[(t.name, t.dtype, t.shape) for t in inputs]}")
    print(f"outputs: {[(t.name, t.dtype, t.shape) for t in outputs]}")

    if args.embed:
        bin_bytes = bin_path.read_bytes()
        print(f"embedding {len(bin_bytes)/1e6:.1f} MB of .bin into the ONNX")
        ep_cache_context = bin_bytes
        embed_mode = 1
    else:
        # ORT QNN EP requires `ep_cache_context` to be a *relative* path; it
        # is resolved against CWD at load time. The caller (Rust) is expected
        # to chdir into the wrapper's directory before CreateSession.
        ep_cache_context = bin_path.name
        embed_mode = 0
    ctx_node = helper.make_node(
        "EPContext",
        inputs=[t.name for t in inputs],
        outputs=[t.name for t in outputs],
        domain="com.microsoft",
        embed_mode=embed_mode,
        ep_cache_context=ep_cache_context,
        main_context=1,
        source="Qnn",
        partition_name="parakeet_encoder",
    )

    graph = helper.make_graph(
        nodes=[ctx_node],
        name="parakeet_encoder_qnn_ctx",
        inputs=[make_tensor_value_info(t) for t in inputs],
        outputs=[make_tensor_value_info(t) for t in outputs],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
        producer_name="openwritr-aihub-wrapper",
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    print(f"\nDONE: {out_path} ({out_path.stat().st_size} bytes)")
    print(f"  references: {bin_path.name} ({bin_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
