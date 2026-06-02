# OpenWritr for Windows v0.1.0 — First Public Release

Push-to-talk voice-to-text tray app for **Windows on ARM** (Snapdragon X).
Hold a hotkey, speak, release — the transcript is pasted at your cursor.
Everything runs locally on your machine; nothing leaves the device unless
you turn on optional LLM cleanup.

## Three transcription engines, all running locally

| Engine | Where it runs | Languages | Latency on 11 s English |
|---|---|---|---|
| **Parakeet TDT v3 — CPU INT8** *(default)* | CPU | 25 | ~160 ms (70× realtime) |
| **Parakeet TDT v3 — NPU INT8** | Hexagon NPU | 25 | ~110 ms (105× realtime) |
| **Whisper Large v3 Turbo — NPU** | Hexagon NPU | 99 | ~800 ms / 30 s window |

NPU runs use Qualcomm AI Engine Direct via `onnxruntime-qnn` 2.1, with the
encoder context cached on first use. Statically INT8-QDQ-quantized
Parakeet encoder runs entirely on the NPU; transcripts match the CPU
baseline exactly on validation clips.

## Features

- Native Windows tray app with Fluent-styled WPF settings dialog
- Animated overlay with live audio-level meter (HiDPI-correct, Mica backdrop)
- Configurable hotkey: any combination of Ctrl / Shift / Alt / Win plus a
  trigger key from Space / Tab / Caps Lock / Scroll Lock / Pause / Insert
  / Right Ctrl / F13..F20
- Auto-paste at cursor with clipboard save/restore
- Soft, warm start/stop audio cues
- Optional LLM cleanup pass via GitHub Copilot (`gh auth token`) or any
  OpenAI-compatible endpoint — toggle with Alt held during recording
- Reliable hotkey FSM: stops cleanly when any required modifier is
  released; safety auto-stop after `max_record_seconds` (60 s default)

## Install (from source)

```powershell
git clone https://github.com/trsdn/openwritr-windows.git
cd openwritr-windows
py -3.11-arm64 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r python\requirements.txt

# 640 MB Parakeet INT8 ONNX (default engine)
python python\fetch_model.py

# Optional: 1.6 GB Whisper Large v3 Turbo NPU build from Qualcomm AI Hub
python python\fetch_whisper.py

python python\openwritr.py
```

A blue microphone icon appears in your system tray. Default hotkey is
**Ctrl + Win + Space** — hold, speak, release.

## What's not in this release

- No pre-built `.exe` installer yet; the app runs from a Python venv.
  PyInstaller packaging is planned for the next release.
- No auto-update mechanism.
- Code signing: the source is MIT; no signed binaries to distribute yet.
- The 50-sample FLEURS-calibrated Parakeet NPU model is not used at
  runtime — the 8-sample calibration variant proved more stable for
  long-form audio. See `scripts/quantize_qdq*.py` for the toolchain.

## Acknowledgements

- macOS original: [trsdn/OpenWritr](https://github.com/trsdn/OpenWritr)
- Parakeet TDT 0.6B v3 ONNX export: [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
- Whisper Large v3 Turbo NPU build: [qualcomm/Whisper-Large-V3-Turbo](https://huggingface.co/qualcomm/Whisper-Large-V3-Turbo)
- `onnx-asr` for the Parakeet decoding pipeline
