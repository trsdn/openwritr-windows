# OpenWritr for Windows

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Windows 11](https://img.shields.io/badge/Windows-11_24H2-0078D4?logo=windows)](https://github.com/trsdn/openwritr-windows)
[![Rust](https://img.shields.io/badge/Rust-1.85-DEA584?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![Snapdragon X Elite](https://img.shields.io/badge/Snapdragon-X_Elite-3253DC?logo=qualcomm&logoColor=white)](https://www.qualcomm.com/products/mobile/snapdragon/pcs-and-tablets/snapdragon-x-elite)
[![Hexagon NPU](https://img.shields.io/badge/Hexagon-NPU-FF2A00?logo=qualcomm&logoColor=white)](https://huggingface.co/trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s)
[![Release](https://img.shields.io/github/v/release/trsdn/openwritr-windows)](https://github.com/trsdn/openwritr-windows/releases/latest)

Push-to-talk voice-to-text for Windows. Local transcription uses **NVIDIA
Parakeet TDT 0.6B v3** on CPU or Qualcomm Hexagon NPU, plus native
**Whisper Large v3 Turbo** on the Snapdragon X Elite NPU.
Optional LLM cleanup via **GitHub Copilot** (Claude Haiku 4.5, GPT-5 Mini,
GPT-4.1) or any OpenAI-compatible endpoint.

**[Microsoft Store](https://apps.microsoft.com/detail/9MSQWR701P2Q)** · **[Website](https://trsdn.github.io/openwritr-windows/)** · **[Releases](https://github.com/trsdn/openwritr-windows/releases)** · **[NPU model on HF](https://huggingface.co/trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s)** · **[macOS sibling](https://github.com/trsdn/OpenWritr)**

## Quick start

**Easiest: [get it from the Microsoft Store](https://apps.microsoft.com/detail/9MSQWR701P2Q).**
One click, it installs the right build for your CPU automatically, keeps
itself updated, and there's no SmartScreen warning.

Prefer a direct download? Pick the build for your CPU, both per-user (no
admin / UAC required):

| Your machine | Build | Engine |
|---|---|---|
| **Snapdragon X Elite** (Surface Pro 11, etc.) | `…-arm64-…` | Parakeet CPU/NPU and Whisper NPU |
| **Intel / AMD** laptop | `…-x64-…` | CPU INT8 |

Not sure? Snapdragon laptops report "ARM-based processor" in Settings →
System → About. Everything else is x64.

**Installer.** Download
`openwritr-windows-<arch>-vX.Y.Z-setup.exe` from
[Releases](https://github.com/trsdn/openwritr-windows/releases/latest)
and run it. Sets up the Start Menu shortcut, an optional autostart-at-logon
entry, and a proper uninstaller you'll find under Settings → Apps.

**Portable zip.** Download `openwritr-windows-<arch>-vX.Y.Z.zip` and unzip
it into **any folder you like** (e.g. `C:\Tools\OpenWritr\`), then run
`openwritr.exe`. The app finds its DLLs next to the exe — the install
location doesn't matter. Same binaries as the installer, just no shortcuts
and no autostart.

> **Note:** user data (settings, downloaded models, logs) always lives under
> `%LOCALAPPDATA%\OpenWritr\` — the app creates that folder automatically on
> first run, you never need to create it yourself. `AppData` is a hidden
> folder; if you want to look inside, paste `%LOCALAPPDATA%\OpenWritr` into
> the Explorer address bar.

Logs are under `%LOCALAPPDATA%\OpenWritr\logs`. Tray actions **Open logs**
and **Export diagnostics** provide bounded logs plus redacted runtime/model
status; diagnostics never include audio, transcript text, clipboard contents,
or API keys.

The x64 build runs Parakeet on the CPU (no Hexagon NPU on Intel/AMD); the
arm64 build adds the NPU engine. Both share the same multilingual model and
UX.

> **Windows SmartScreen warning.** The binaries are not code-signed (yet), so
> the first launch shows "Windows protected your PC". Click **More info →
> Run anyway**. To verify your download is authentic, compare its SHA-256
> against `SHA256SUMS.txt` attached to the release:
> `Get-FileHash .\openwritr-windows-<arch>-vX.Y.Z-setup.exe` in PowerShell.

The default engine is Parakeet CPU. Its verified model files are fetched on
first use into `%LOCALAPPDATA%\OpenWritr\models\` (~670 MB). Selecting an NPU
engine downloads its own pinned assets with visible progress and verification:
Parakeet NPU adds ~650 MB; Whisper NPU downloads a ~2.0 GB archive and uses
about 2.2 GB after extraction.

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
- **Transcription engine**: Parakeet CPU INT8 (default), Parakeet NPU, or
  Whisper Large v3 Turbo NPU. NPU engines require the native ARM64 build on
  Snapdragon X Elite. A selected engine failure is shown explicitly;
  OpenWritr never silently substitutes another engine.
- **Behaviour**: auto-paste at cursor, show overlay while recording, play
  start/stop sounds.
- **Enhance**: provider (Off / GitHub Copilot / OpenAI-compatible API), model
  dropdown (Claude Haiku 4.5, GPT-5 Mini, GPT-4.1) or free-form custom model
  name, base URL + API key (OpenAI-compatible only). API keys are stored in
  Windows Credential Manager, not in the settings file.

Settings are stored at `%LOCALAPPDATA%\OpenWritr\settings.json`. The app polls
the file's mtime so external edits also take effect live. Writes use an atomic
replacement, and legacy plaintext API keys are migrated only after the secure
credential can be written and read back successfully.

## Architecture

### Native Rust app (what ships in the release zip)

```
openwritr-windows/
├── src/
│   ├── main.rs              entry; dispatches `--settings` subprocess
│   ├── app.rs               winit event loop, tray, hotkey thread, ASR dispatch
│   ├── audio.rs             cpal WASAPI capture, multi-channel downmix
│   ├── credentials.rs       Windows Credential Manager API-key storage
│   ├── hotkey.rs            push-to-talk combo polling against key_hook state
│   ├── key_hook.rs          global WH_KEYBOARD_LL hook → atomic key bitmap
│   ├── overlay.rs           custom Win32 layered top-most window, GDI bars
│   ├── settings.rs          validation, migration, atomic JSON persistence
│   ├── settings_ui.rs       eframe/egui dialog (subprocess)
│   ├── asr/                 ONNX Runtime ASR pipelines
│   │   ├── parakeet.rs      CPU/NPU Parakeet pipeline
│   │   ├── whisper_npu.rs   native chunked Whisper NPU engine
│   │   ├── whisper_mel.rs   128-bin Whisper log-mel frontend
│   │   ├── whisper_decoder.rs  QNN encoder/decoder and KV-cache loop
│   │   └── qnn_ffi.rs       reusable typed QNN C-API sessions
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
  transparency, painted with double-buffered GDI. It reads the recorder
  atomics and receives enable/status updates through a command channel, so
  settings changes apply without restarting the app.
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

Python is used only at build time to fetch hash-pinned runtime wheels and
validate the canonical release manifest. The resulting packages are
self-contained and Python-free.

```powershell
.\scripts\envup.ps1
cargo build --release --bins
cargo run --release --bin package
```

The package command fetches the exact runtime tuple, validates versions,
SHA-256 hashes, PE architecture, required files, and licenses, then stages
from `release-manifest.json`. The ZIP lands in `target/dist/`.

## NPU pipeline

Parakeet runs its pre-compiled encoder on the Hexagon HTP; mel preprocessing
and the TDT decoder stay on CPU. Whisper runs both its encoder and
autoregressive KV-cache decoder as pre-compiled QNN sessions. It processes
sequential 30-second chunks, detects language once per recording, and reuses
that language for every later chunk.

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
`scripts/fetch_runtime.py` extracts the exact, hash-pinned set recorded in
`runtime-manifest.json`. Alongside `onnxruntime_providers_qnn.dll`, the
X Elite path requires `QnnHtp.dll`, `QnnHtpPrepare.dll`,
`QnnHtpNetRunExtensions.dll`, `QnnSystem.dll`, the V73 stub and skeleton,
and its catalog file. Without the SKEL + `.cat` pair, the stub fails
`LoadLibrary` with `ERROR_MOD_NOT_FOUND` (126) and QnnHtp later aborts
session creation with `STATUS_STACK_BUFFER_OVERRUN` (0xC0000409) without
a useful error.

## Models

| Model | Provider | Size | License | Auto-downloaded? |
|---|---|---|---|---|
| Parakeet TDT 0.6B v3 (CPU INT8 ONNX + companion files) | [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) | ~670 MB | CC-BY-4.0 | Yes, on first run |
| Parakeet TDT 0.6B v3 NPU encoder (QAIRT context binary + wrapper) | [trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s](https://huggingface.co/trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s) | ~632 MB | CC-BY-4.0 | Yes, on first NPU launch |
| Whisper Large v3 Turbo (QNN encoder, decoder, tokenizer) | [qualcomm/Whisper-Large-V3-Turbo](https://huggingface.co/qualcomm/Whisper-Large-V3-Turbo) | ~2.2 GB extracted | Apache 2.0 + BSD-3 | Yes, on first Whisper launch |

The NPU encoder model is device-gated to Snapdragon X Elite (Hexagon V73).
It will not run on X Plus or any other Qualcomm chipset without
recompilation via AI Hub.

## Licenses

- **OpenWritr code**: MIT — see [`LICENSE`](LICENSE).
- **Parakeet model**: CC-BY-4.0 (NVIDIA). Attribution preserved when the
  model is downloaded.
- **Qualcomm QNN runtime DLLs** (`QnnHtp.dll`, QNN provider, V73 stub, etc.,
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
├── src/                   Rust native app
├── scripts/               pinned runtime and release tooling
├── models-manifest.json   pinned model assets and hashes
├── runtime-manifest.json  pinned ORT/QNN wheels and compatibility tuple
├── release-manifest.json  canonical per-architecture package contents
└── target/                gitignored build and staging output
```

The shipped `openwritr.exe` does not call Python at runtime.

## Development

```powershell
git clone https://github.com/trsdn/openwritr-windows.git
cd openwritr-windows
.\scripts\envup.ps1
cargo build --release --bins
python scripts/fetch_runtime.py --arch arm64
python scripts/prepare_release.py --arch arm64
target\stage\arm64\openwritr.exe --self-check
```

For x64 packaging, stage the CPU runtime with:

```powershell
python scripts/fetch_runtime.py --arch x64
```

See [`docs/RUNTIME_COMPATIBILITY.md`](docs/RUNTIME_COMPATIBILITY.md) for
the shared ORT/QNN/QAIRT contract and its Snapdragon hardware gates.

## Releases

Tagged releases live at
[github.com/trsdn/openwritr-windows/releases](https://github.com/trsdn/openwritr-windows/releases).
Each architecture ships a ZIP and installer generated from the same validated
release stage. CI also builds an unsigned Store `.msixbundle` as a workflow
artifact for Partner Center ingestion, but does not attach it to the public
GitHub release. Tagged builds use the repository variables
`MSIX_IDENTITY_NAME` and `MSIX_PUBLISHER` when configured; otherwise the Store
bundle is skipped rather than publishing a placeholder identity. Every package
includes an `artifact-manifest.json`; CI verifies all required file hashes and
confirms that x64 artifacts contain no Qualcomm runtime files.
