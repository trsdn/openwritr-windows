#!/usr/bin/env bash
# Bootstrap a NeMo env inside WSL Ubuntu aarch64 for the streaming-export
# spike. Run as root: wsl -d Ubuntu -u root -- bash /mnt/c/src/openwritr-windows/scripts/wsl_setup_nemo.sh
set -euo pipefail

cd /root

if [ ! -d venv-nemo ]; then
    python3 -m venv venv-nemo
fi
source venv-nemo/bin/activate

pip install --upgrade pip -q

echo "=== installing torch + torchaudio (CPU) ==="
pip install torch torchaudio -q
python -c 'import torch, torchaudio; print("torch", torch.__version__, "torchaudio", torchaudio.__version__)'

echo "=== installing nemo_toolkit[asr] + onnx + hf ==="
pip install "nemo_toolkit[asr]" onnx onnxruntime huggingface_hub -q

echo "=== nemo check ==="
python -c 'import nemo; from nemo.collections.asr.models import EncDecRNNTBPEModel; print("nemo", nemo.__version__, "EncDecRNNTBPEModel OK")'

echo "=== DONE ==="
