# OpenWritr v0.2.0 — Native Rust Build for Windows on ARM

Single ~22 MB distribution, no Python runtime, models downloaded on first launch from Hugging Face.

## Install

1. Download `openwritr-windows-arm64-v0.2.0-dev.zip` below.
2. Extract anywhere (e.g. `C:\Program Files\OpenWritr\`).
3. Run `openwritr.exe`. First launch downloads the Parakeet TDT v3 model (~670 MB) from Hugging Face into `%LOCALAPPDATA%\OpenWritr\models\`.
4. Hold **Ctrl + Win + Space**, speak, release. Text is pasted at the cursor. Right-click the tray icon for **Settings…**.

## Highlights

- ~6.4 MB `openwritr.exe` (was ~300 MB Python in v0.1)
- ~30 MB cold-start RSS (was ~700 MB)
- 25 languages via NVIDIA Parakeet TDT 0.6B v3 (INT8 CPU, ~140 ms / 11 s)
- egui Settings dialog with hotkey + engine + LLM cleanup configuration
- Warm start/stop tone cues
- Optional cleanup pass via GitHub Copilot (uses `gh auth token`) or any OpenAI-compatible endpoint (hold Alt with hotkey)
- Reliable hotkey FSM with safety auto-stop on modifier release

## Known limitations vs v0.1 (Python)

- **NPU backends fall back to CPU INT8** in the native build. The `ort` Rust crate 2.0-rc.10 does not expose `RegisterExecutionProviderLibrary`, so we cannot load Qualcomm's `onnxruntime_providers_qnn.dll` the way Python ORT does. The Python v0.1 app (`python/` folder) remains the NPU-capable reference.
- **No animated overlay yet** — tray icon colour change is the visual state feedback. Coming in v0.3.

## Licenses

- OpenWritr source: **MIT**
- NVIDIA Parakeet TDT 0.6B v3: **CC-BY-4.0**
- istupakov Parakeet ONNX export: **CC-BY-4.0**
- ONNX Runtime: **MIT**
- Qualcomm QNN runtime DLLs (bundled): **Qualcomm proprietary**
