"""Verify the NeMo env inside WSL imports cleanly. Run with the venv python."""
import torch
import torchaudio
print("torch", torch.__version__, "torchaudio", torchaudio.__version__)

import onnx
print("onnx", onnx.__version__)

import nemo
from nemo.collections.asr.models import EncDecRNNTBPEModel
print("nemo", nemo.__version__, "EncDecRNNTBPEModel OK")
