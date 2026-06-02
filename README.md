# OpenWritr for Windows (ARM64)

[![Windows ARM64](https://img.shields.io/badge/Windows-ARM64-0078D4?logo=windows)](https://github.com/trsdn/openwritr-windows)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Latest release](https://img.shields.io/github/v/release/trsdn/openwritr-windows)](https://github.com/trsdn/openwritr-windows/releases/latest)

Push-to-talk voice-to-text for **Windows on ARM** (Surface Pro / Snapdragon X
Elite). Local transcription via **NVIDIA Parakeet TDT 0.6B v3** (INT8 ONNX,
25 languages). Optional LLM cleanup via **GitHub Copilot** (Claude Haiku 4.5,
GPT-5 Mini, GPT-4.1) or any OpenAI-compatible endpoint.

Windows port of [trsdn/OpenWritr](https://github.com/trsdn/OpenWritr) (macOS).

## Quick start

1. Download the latest `openwritr-windows-arm64-vX.Y.Z.zip` from
   [Releases](https://github.com/trsdn/openwritr-windows/releases/latest).
2. Unzip to `%LOCALAPPDATA%\OpenWritr\app\`.
3. Run `openwritr.exe`. The first launch downloads the Parakeet model (~670 MB)
   to `%LOCALAPPDATA%\OpenWritr\models\` — one-time, ~2 minutes on a fast link.

A microphone icon appears in your system tray.

## Usage

| Combo | Action |
|---|---|
| **Hold Ctrl + Win** | Record. Release to transcribe and paste at the caret. |
| **Hold Ctrl + Shift + Win** | Record + LLM cleanup (Claude Haiku 4.5 by default). |
| **Tray right-click → Settings** | Change hotkey, engine, LLM provider. |

A small dark pill appears at the bottom-center of the primary monitor while
recording, with white bars that breathe with your voice. Settings changes take
effect immediately — no restart required.

## Settings

Tray icon → right-click → **Settings**. All fields:

- **Hotkey**: any combination of Ctrl / Shift / Alt / Win modifiers, plus an
  optional trigger key (Space, Tab, Caps Lock, F13-F20, or `None` for
  modifiers-only). Default: Ctrl+Win, no trigger.
- **Transcription engine**: Parakeet CPU INT8 (default), Parakeet NPU,
  Whisper Large v3 Turbo NPU. The native build currently falls back to CPU
  for both NPU options (see *NPU support* below). The Python build supports
  real NPU.
- **Behaviour**: auto-paste at cursor, show overlay while recording, play
  start/stop sounds.
- **Enhance**: provider (Off / GitHub Copilot / OpenAI-compatible API), model
  dropdown (Claude Haiku 4.5, GPT-5 Mini, GPT-4.1) or free-form custom model
  name, base URL + API key (OpenAI-compatible only).

Settings are stored at `%LOCALAPPDATA%\OpenWritr\settings.json`. The app polls
the file's mtime so external edits also take effect live.

## Architecture

### Native Rust app (what ships in the release zip)

```
openwritr-windows/
├── src/
│   ├── main.rs              entry; dispatches `--settings` subprocess
│   ├── app.rs               winit event loop, tray, hotkey thread, ASR dispatch
│   ├── audio.rs             cpal WASAPI capture, multi-channel downmix
│   ├── hotkey.rs            GlobalHotKeyManager + GetAsyncKeyState polling
│   ├── overlay.rs           custom Win32 layered top-most window, GDI bars
│   ├── settings.rs          serde struct + JSON load/save
│   ├── settings_ui.rs       eframe/egui dialog (subprocess)
│   ├── asr/                 ONNX Runtime pipeline (mel → encoder → TDT decoder)
│   ├── enhance.rs           Copilot / OpenAI cleanup pass
│   ├── sounds.rs            G3/E3 tone synth (start/stop pings)
│   └── bin/package.rs       distributable-zip builder
└── Cargo.toml
```

Key design decisions:
- **Hotkey on its own thread.** Push-to-talk detection runs in a dedicated
  thread with `GetAsyncKeyState`, sending press/release events to the winit
  loop via `EventLoopProxy`. This survives any UI hang and works even if
  Windows reserves the combo (`RegisterHotKey` is best-effort only).
- **Settings UI as subprocess.** The settings dialog is the same exe re-launched
  with `--settings`. Spawning happens from a worker thread because
  `CreateProcessW` on Windows ARM64 with Defender real-time scanning can block
  several seconds — doing it inline would freeze the tray pump.
- **Overlay on its own message loop.** Layered top-most window with color-key
  transparency, painted with double-buffered GDI. Shares only two atomics
  (`recording` + `last_rms_x10000`) with the recorder, so it cannot deadlock
  the main app.
- **Multi-channel downmix in the audio callback.** The Qualcomm Aqstic mic
  array on Surface Pro exposes 4-8 interleaved channels at 48 kHz; we average
  to mono before resampling to 16 kHz.

### Python legacy app (`python/`)

The original Python v0.1 still lives in the repo because it is currently the
**only** way to use the NPU end-to-end. It uses the same Parakeet model and a
companion Whisper NPU implementation. See [`python/`](python/) for details.

The Rust native app does not call into Python at runtime.

### Build-time Python dependency

The Rust package script (`cargo run --release --bin package`) stages Qualcomm
QNN runtime DLLs from `pip install onnxruntime-qnn` into `target/release/`
before zipping. The DLLs are required for the NPU engine option to work at
all (even the Python path needs them). This is the only Python touch point at
build time; the resulting zip is fully self-contained and Python-free.

```powershell
py -3.11-arm64 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install onnxruntime-qnn
```

After that, build + package:

```powershell
.\envup.ps1        # primes vcvars arm64 + LLVM in PATH
cargo build --release --bin openwritr
cargo run --release --bin package
```

The zip lands in `target/dist/`.

## NPU support

**There is no NPU support in either the native Rust or the Python build today.**
Both run Parakeet on the CPU.

Background: Parakeet TDT v3's encoder uses a dynamic-shape attention masking
pattern (`Shape` → `Gather` → `Range` → `Expand`) that the Qualcomm QNN HTP
execution provider in `onnxruntime-qnn 2.1.1` cannot evaluate correctly. An
earlier INT8 QDQ quantization experiment was pushed to HuggingFace under
`trsdn/parakeet-tdt-0.6b-v3-htp-int8` with optimistic performance claims;
that model has since been withdrawn because it fails at inference time on
the NPU. The NPU code path remains in the Rust source (`asr/parakeet.rs::
build_npu_session`) so that when QNN gains support for the required ops, or
when the encoder is re-exported with constants for the dynamic shapes,
flipping the engine setting will activate it without further code changes.

CPU INT8 is fast enough for casual dictation on Snapdragon X — about 150 ms
of inference per 10 s of audio.

## Models

| Model | Provider | Size | License | Auto-downloaded? |
|---|---|---|---|---|
| Parakeet TDT 0.6B v3 (ONNX, INT8) | [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) | ~670 MB | CC-BY-4.0 | Yes, on first run |
| Whisper Large v3 Turbo (QNN context binary) | [qualcomm/Whisper-Large-V3-Turbo](https://huggingface.co/qualcomm/Whisper-Large-V3-Turbo) | ~1.6 GB | Apache 2.0 + BSD-3 | Python build only |

## Licenses

- **OpenWritr code**: MIT — see [`LICENSE`](LICENSE).
- **Parakeet model**: CC-BY-4.0 (NVIDIA). Attribution preserved when the
  model is downloaded.
- **Qualcomm QNN runtime DLLs** (`QnnHtp.dll`, `QnnCpu.dll`, `Genie.dll`, etc.,
  bundled in the release zip): **Qualcomm AI Engine Direct redistributable
  license**. The full text ships inside every release zip under
  `third-party-licenses/Qualcomm_LICENSE.pdf`, alongside Microsoft's
  `ThirdPartyNotices.txt` for the `onnxruntime-qnn` PyPI package the DLLs
  come from. These DLLs are redistributable as part of applications targeting
  Qualcomm Snapdragon hardware, which is what OpenWritr does.
- **ONNX Runtime DLLs** (`onnxruntime.dll`, `onnxruntime_providers_qnn.dll`):
  MIT (Microsoft), bundled under their respective LICENSE files in the release
  zip.

## Repository layout

```
openwritr-windows/
├── src/             Rust native app (what users run)
├── python/          Legacy v0.1 — current NPU fallback
├── .venv/           gitignored; pip install onnxruntime-qnn happens here
└── target/          gitignored; build output
```

`.venv` is build-time only. The shipped `openwritr.exe` does not call into
Python at runtime.

## Development

```powershell
git clone https://github.com/trsdn/openwritr-windows.git
cd openwritr-windows
py -3.11-arm64 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install onnxruntime-qnn
.\envup.ps1
cargo run --release --bin openwritr
```

For the Python NPU build, additionally:

```powershell
pip install -r python\requirements.txt
python python\openwritr.py
```

## Releases

Tagged releases live at
[github.com/trsdn/openwritr-windows/releases](https://github.com/trsdn/openwritr-windows/releases).
Each release ships a single zip containing `openwritr.exe`, the ONNX Runtime
DLLs, the QNN runtime DLLs, this README, the MIT LICENSE, and a
`third-party-licenses/` folder with the Qualcomm and Microsoft licence files
for the bundled DLLs.
