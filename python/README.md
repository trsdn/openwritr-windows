# OpenWritr — Python build (Windows ARM64)

Working tray app for Windows on ARM. Push-to-talk voice-to-text powered by
NVIDIA Parakeet TDT 0.6B v3 (INT8 ONNX, ~40x realtime on Snapdragon X CPU).

> The Rust + Tauri scaffold under `../src-tauri/` is the long-term target —
> NPU acceleration, smaller bundle, no Python runtime. This Python build is
> what actually runs **today**.

## Install (Windows ARM64)

```powershell
git clone https://github.com/trsdn/openwritr-windows.git
cd openwritr-windows
py -3.11-arm64 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r python\requirements.txt
python python\fetch_model.py        # ~640 MB, one-time
```

## Run

Either:

```powershell
python python\openwritr.py
```

…or double-click `python\start.cmd` (uses `pythonw.exe`, no console window).

A blue microphone icon appears in the system tray. **Hold `Ctrl + Shift + Space`**
to record, release to transcribe and paste at the caret.

| Icon colour | State |
|---|---|
| Blue | idle |
| Red | recording |
| Orange | transcribing |
| Grey | error (check logs) |

Right-click the tray icon for **Auto-paste toggle**, **Open log folder**, **Quit**.

## Performance (Snapdragon X1E80100, CPU)

| Audio | Transcription time | Ratio |
|---|---|---|
| 11.5 s English | 0.27 s | 43x realtime |
| 7.6 s German | 0.18 s | 42x realtime |
| Model load | 2.5 s | one-time |
| Idle RSS | ~700 MB | model resident |

NPU offload via QNN HTP was attempted on the `istupakov` INT8 release and fails
graph-finalize (the model is dynamic-quantized for x86 CPU, not statically
quantized for HTP). Pure CPU is already faster than realtime and feels instant
for push-to-talk; an NPU port would mainly save battery, not latency.

## Settings

`%LOCALAPPDATA%\OpenWritr\settings.json`:

```json
{
  "hotkey_modifiers": ["ctrl", "shift"],
  "auto_paste": true,
  "min_record_seconds": 0.25,
  "max_record_seconds": 60
}
```

Trigger key is always Space — only the modifier combination is configurable.

## Logs

`%LOCALAPPDATA%\OpenWritr\logs\openwritr.log`

## Permissions

First launch will trigger a **Microphone** permission prompt from Windows.

## Packaging (planned)

Single-file `.exe` via PyInstaller, signed MSIX via the existing GitHub Actions
workflow template at `../docs/build.yml.workflow-template`.
