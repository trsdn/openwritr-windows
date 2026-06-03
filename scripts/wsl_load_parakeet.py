"""
Download + load the Parakeet TDT v3 .nemo checkpoint inside WSL.

Confirms the restored model class and dumps the encoder type so we know
which streaming/cache API surface applies. Caches the .nemo under
/root/.cache/huggingface so re-runs are fast.

Run with the venv python inside WSL.
"""
import sys
from huggingface_hub import hf_hub_download

print("downloading parakeet-tdt-0.6b-v3.nemo from HF…", flush=True)
nemo_path = hf_hub_download(
    repo_id="nvidia/parakeet-tdt-0.6b-v3",
    filename="parakeet-tdt-0.6b-v3.nemo",
)
print("nemo file:", nemo_path, flush=True)

from nemo.collections.asr.models import ASRModel
print("restoring model…", flush=True)
model = ASRModel.restore_from(nemo_path, map_location="cpu")
model.eval()

print("model class:", type(model).__module__ + "." + type(model).__name__)
print("encoder class:", type(model.encoder).__module__ + "." + type(model.encoder).__name__)
print("decoder class:", type(model.decoder).__module__ + "." + type(model.decoder).__name__)

# Does the encoder expose the cache-aware streaming setup hook?
enc = model.encoder
for attr in ("setup_streaming_params", "set_default_att_context_size",
             "streaming_cfg", "att_context_size", "change_attention_model"):
    print(f"  encoder.{attr}:", "yes" if hasattr(enc, attr) else "no")

# Export config hook lives on the model.
print("  model.set_export_config:", "yes" if hasattr(model, "set_export_config") else "no")

print("DONE")
