# OpenWritr for Windows (ARM64)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Windows 11](https://img.shields.io/badge/Windows-11_24H2-0078D4?logo=windows)](https://github.com/trsdn/openwritr-windows)
[![Rust](https://img.shields.io/badge/Rust-1.85-DEA584?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![Snapdragon X Elite](https://img.shields.io/badge/Snapdragon-X_Elite-3253DC?logo=qualcomm&logoColor=white)](https://www.qualcomm.com/products/mobile/snapdragon/pcs-and-tablets/snapdragon-x-elite)
[![Hexagon NPU](https://img.shields.io/badge/Hexagon-NPU-FF2A00?logo=qualcomm&logoColor=white)](https://huggingface.co/trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s)
[![Release](https://img.shields.io/github/v/release/trsdn/openwritr-windows)](https://github.com/trsdn/openwritr-windows/releases/latest)

Push-to-talk voice-to-text for **Windows on ARM** (Surface Pro / Snapdragon X
Elite). Local transcription via **NVIDIA Parakeet TDT 0.6B v3** running on
the **Hexagon NPU**. 25 languages, ~67 ms per 8-second window on the NPU.
Optional LLM cleanup via **GitHub Copilot** (Claude Haiku 4.5, GPT-5 Mini,
GPT-4.1) or any OpenAI-compatible endpoint.

**[Download](https://github.com/trsdn/openwritr-windows/releases/latest)** · **[Releases](https://github.com/trsdn/openwritr-windows/releases)** · **[NPU model on HF](https://huggingface.co/trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s)** · **[macOS sibling](https://github.com/trsdn/OpenWritr)**

## Quick start

Pick the build for your CPU, both per-user (no admin / UAC required):

| Your machine | Build | Engine |
|---|---|---|
| **Snapdragon X** (Surface Pro 11, etc.) | `…-arm64-…` | Hexagon NPU + CPU fallback |
| **Intel / AMD** laptop | `…-x64-…` | CPU INT8 |

Not sure? Snapdragon laptops report "ARM-based processor" in Settings →
System → About. Everything else is x64.

**Installer (recommended).** Download
`openwritr-windows-<arch>-vX.Y.Z-setup.exe` from
[Releases](https://github.com/trsdn/openwritr-windows/releases/latest)
and run it. Sets up the Start Menu shortcut, an optional autostart-at-logon
entry, and a proper uninstaller you'll find under Settings → Apps.

**Portable zip.** Download `openwritr-windows-<arch>-vX.Y.Z.zip`, unzip
into `%LOCALAPPDATA%\OpenWritr\app\`, run `openwritr.exe`. Same binaries,
no shortcuts, no autostart.

The x64 build runs Parakeet on the CPU (no Hexagon NPU on Intel/AMD); the
arm64 build adds the NPU engine. Both share the same multilingual model and
UX.

On first launch the Parakeet model is fetched from Hugging Face into
`%LOCALAPPDATA%\OpenWritr\models\` — one-time, ~1.2 GB on the NPU engine
(600 MB CPU INT8 + 632 MB QNN HTP context binary), ~2 minutes on a fast
link. A microphone icon appears in your system tray when the engine is
ready.

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
- **Transcription engine**: Parakeet CPU INT8, **Parakeet NPU** (default for
  v0.3+), Whisper Large v3 Turbo NPU. The NPU engine runs the encoder on
  the Snapdragon X Elite Hexagon HTP via a pre-compiled QAIRT context
  binary; preprocessor and TDT decoder remain on the CPU. Falls back to
  CPU INT8 automatically if the NPU model fails to load.
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
│   ├── hotkey.rs            push-to-talk combo polling against key_hook state
│   ├── key_hook.rs          global WH_KEYBOARD_LL hook → atomic key bitmap
│   ├── overlay.rs           custom Win32 layered top-most window, GDI bars
│   ├── settings.rs          serde struct + JSON load/save
│   ├── settings_ui.rs       eframe/egui dialog (subprocess)
│   ├── asr/                 ONNX Runtime pipeline (mel → encoder → TDT decoder)
│   │   ├── parakeet.rs      Encoder enum {Cpu, Npu} + chunked long-audio pipeline
│   │   └── qnn_ffi.rs       direct C-API FFI for the NPU encoder session
│   ├── enhance.rs           Copilot / OpenAI cleanup pass
│   ├── sounds.rs            G3/E3 tone synth (start/stop pings)
│   └── bin/package.rs       distributable-zip builder
└── Cargo.toml
```

Key design decisions:
- **Global low-level keyboard hook.** Push-to-talk detection reads physical
  key state from `WH_KEYBOARD_LL` instead of `GetAsyncKeyState`. The OS
  synthesises key-ups during focus changes (PowerShell launched mid-recording,
  UAC prompt, system shortcut handler), which the polling API faithfully
  reports — and which would abort the recording. The LL hook sees only
  physical events.
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

### Python toolchain (`scripts/`)

NPU model preparation lives in `scripts/`. `build_npu_encoder.py`,
`aihub_compile_encoder.py`, `wrap_qnn_context_binary.py`, and
`test_npu_encoder.py` are build-time tools used to produce the .bin
hosted on HF. They are NOT invoked by the shipped `openwritr.exe` at
runtime — the native build pulls the pre-compiled binary directly.

The Rust app does not call into Python at runtime.

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
.\scripts\envup.ps1        # primes vcvars arm64 + LLVM in PATH
cargo build --release --bin openwritr
cargo run --release --bin package
```

The zip lands in `target/dist/`.

## NPU pipeline

The encoder runs as a pre-compiled QAIRT context binary on the Hexagon HTP
of the Snapdragon X Elite. Preprocessor (mel features) and TDT decoder
remain on the CPU EP — they are dynamic-shape, lightweight, and not a
bottleneck.

The compiled binary expects a fixed **8-second audio window**. For
push-to-talk utterances ≤ 8 s the encoder is run once on a padded window.
For longer audio the encoder is run in chunks (8 s window with 1 s
overlap), feature streams are stitched at the seam, and the TDT decoder
runs once over the concatenated features. Tested up to ~23 s without
boundary doubling.

### Measured on Snapdragon X Elite (X1E80100)

| Audio length | Decode (preproc + encode + TDT) | × Realtime | Chunks |
|---|---|---|---|
| 3 s | 128 ms | 23× | 1 |
| 5.8 s | 221 ms | 26× | 1 |
| 16.4 s | 375 ms | 44× | 3 |
| 23.0 s | 626 ms | 37× | 4 |

The encoder itself is ~67 ms ± 0 ms per 8-second window, independent of
the actual audio length within the window.

### How the binary was built

1. `scripts/build_npu_encoder.py` constant-folds the encoder's
   dynamic-shape attention mask (`Shape → Gather → Range → Expand`) against
   a frozen `[1, 128, 801]` input — the HTP backend cannot evaluate that
   subgraph as-is.
2. `scripts/aihub_compile_encoder.py` submits the static-shape FP32 ONNX
   plus FLEURS calibration samples to Qualcomm AI Hub. Quantize job uses
   INT8 weights / INT16 activations (the standard HTP recipe for
   transformer encoders); compile job targets `snapdragon_x_elite_crd`
   with `--target_runtime qnn_context_binary --truncate_64bit_io`.
3. `scripts/wrap_qnn_context_binary.py` wraps the resulting `.bin` in a
   408-byte EPContext-node ONNX so ORT's QNN EP can consume it.
4. `src/asr/qnn_ffi.rs` loads the wrapper via direct `ort_sys` C-API calls,
   bypassing `ort` 2.0-rc.12's session builders (which crash inside QnnHtp
   when consuming EPContext-wrapper ONNX).

### Required helper DLLs

The QNN backend loads several sibling DLLs by name at session-create time.
`src/bin/package.rs` stages all of them into the release zip, but if you
hand-assemble a distribution: alongside `onnxruntime_providers_qnn.dll`
you need `QnnHtp.dll`, `QnnHtpPrepare.dll`, `QnnSystem.dll`, the V73/V81
stubs (`QnnHtpV73Stub.dll`, `QnnHtpV81Stub.dll`), the per-arch skeletons
(`libQnnHtpV73Skel.so`, `libQnnHtpV81Skel.so`), and the catalog files
(`libqnnhtpv73.cat`, `libqnnhtpv81.cat`). Without the SKEL + .cat pair, the
stub fails `LoadLibrary` with `ERROR_MOD_NOT_FOUND` (126) and QnnHtp later
aborts session creation with `STATUS_STACK_BUFFER_OVERRUN` (0xC0000409)
without a useful error.

## Models

| Model | Provider | Size | License | Auto-downloaded? |
|---|---|---|---|---|
| Parakeet TDT 0.6B v3 (CPU INT8 ONNX + companion files) | [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) | ~670 MB | CC-BY-4.0 | Yes, on first run |
| Parakeet TDT 0.6B v3 NPU encoder (QAIRT context binary + wrapper) | [trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s](https://huggingface.co/trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s) | ~632 MB | CC-BY-4.0 | Yes, on first NPU launch |
| Whisper Large v3 Turbo (QNN context binary) | [qualcomm/Whisper-Large-V3-Turbo](https://huggingface.co/qualcomm/Whisper-Large-V3-Turbo) | ~1.6 GB | Apache 2.0 + BSD-3 | Python build only |

The NPU encoder model is device-gated to Snapdragon X Elite (Hexagon V73).
It will not run on X Plus or any other Qualcomm chipset without
recompilation via AI Hub.

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
.\scripts\envup.ps1
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
