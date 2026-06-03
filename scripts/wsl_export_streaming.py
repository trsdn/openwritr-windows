"""
Phase 1 gate: reconfigure Parakeet TDT v3's ConformerEncoder for cache-aware
streaming and export it to ONNX with cache support.

Strategy:
  1. Inspect the trained att_context_size / att_context_style. If the model
     was trained full-context ([-1,-1]) it is NOT a true cache-aware model;
     streaming will work mechanically but accuracy degrades. We report this
     honestly rather than silently shipping a bad model.
  2. Set a limited left/right context (chunk=2s ≈ [70, 13] att steps at the
     80ms encoder step for FastConformer... but v3 is Conformer w/ its own
     subsampling — we read ENCODER_STEP from the model).
  3. model.set_export_config({'cache_support': 'True'})
  4. model.encoder.export(onnx_path) and report the resulting I/O.

Run with the venv python inside WSL.
"""
import sys
import json
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import ASRModel

OUT = "/mnt/c/Users/torstenmahr/AppData/Local/OpenWritr/models/parakeet-streaming"
import os
os.makedirs(OUT, exist_ok=True)

nemo_path = hf_hub_download(repo_id="nvidia/parakeet-tdt-0.6b-v3",
                            filename="parakeet-tdt-0.6b-v3.nemo")
model = ASRModel.restore_from(nemo_path, map_location="cpu")
model.eval()

enc = model.encoder
print("=== trained attention config ===")
print("att_context_size:", getattr(enc, "att_context_size", "n/a"))
print("att_context_style:", getattr(enc, "att_context_style", "n/a"))
# The list of context sizes the model was trained to support (cache-aware
# multi-lookahead models carry several).
cfg = model.cfg.encoder
print("cfg.att_context_size:", cfg.get("att_context_size", "n/a"))
print("cfg.att_context_style:", cfg.get("att_context_style", "n/a"))
print("cfg.conv_context_size:", cfg.get("conv_context_size", "n/a"))
print("cfg.self_attention_model:", cfg.get("self_attention_model", "n/a"))
print("cfg.conv_norm_type:", cfg.get("conv_norm_type", "n/a"))

trained_style = str(cfg.get("att_context_style", "")).lower()
is_cache_aware = "chunked_limited" in trained_style or "regular" not in trained_style

print()
print("=== verdict on cache-aware trainability ===")
if "chunked_limited" in trained_style:
    print("GOOD: model trained chunked_limited — true cache-aware streaming supported.")
else:
    print(f"WARNING: att_context_style={trained_style!r} (not chunked_limited).")
    print("This is an OFFLINE model. Streaming export will run but predictions")
    print("in streaming mode will differ from offline and likely degrade.")

# Try the streaming reconfiguration regardless, so we learn whether the
# export mechanics even work for this architecture.
print()
print("=== attempting streaming reconfig + export ===")
try:
    # ~2 s lookahead. Exact att-step mapping depends on subsampling; we use the
    # documented helper if present.
    if hasattr(enc, "set_default_att_context_size"):
        enc.set_default_att_context_size([70, 13])  # [left, right] att steps
        print("set_default_att_context_size([70, 13]) OK")
except Exception as e:
    print("set_default_att_context_size failed:", repr(e))

try:
    model.set_export_config({"cache_support": "True"})
    print("set_export_config cache_support=True OK")
except Exception as e:
    print("set_export_config failed:", repr(e))

onnx_path = os.path.join(OUT, "encoder-streaming.onnx")
try:
    model.encoder.export(onnx_path)
    print("export OK ->", onnx_path)
    # Inspect I/O.
    import onnx
    m = onnx.load(onnx_path, load_external_data=False)
    print("inputs:")
    for i in m.graph.input:
        dims = [d.dim_value if d.dim_value else (d.dim_param or "?")
                for d in i.type.tensor_type.shape.dim]
        print(f"  {i.name}: {dims}")
    print("outputs:")
    for o in m.graph.output:
        dims = [d.dim_value if d.dim_value else (d.dim_param or "?")
                for d in o.type.tensor_type.shape.dim]
        print(f"  {o.name}: {dims}")
except Exception as e:
    import traceback
    print("export FAILED:")
    traceback.print_exc()

print("DONE")
