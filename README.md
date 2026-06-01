# OpenWritr for Windows (ARM64)

[![Windows ARM64](https://img.shields.io/badge/Windows-ARM64-0078D4?logo=windows)](https://github.com/trsdn/openwritr-windows)
[![Tauri 2](https://img.shields.io/badge/Tauri-2-FFC131?logo=tauri)](https://tauri.app)
[![Rust](https://img.shields.io/badge/Rust-stable-DEA584?logo=rust)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Windows-on-ARM port of [trsdn/OpenWritr](https://github.com/trsdn/OpenWritr) — a push-to-talk voice-to-text tray app.
Local transcription via **NVIDIA Parakeet TDT v3** running on the **Snapdragon X Hexagon NPU** through ONNX Runtime + QNN EP.

> **Status: scaffold** — building blocks in place, model export & NPU runtime in progress. See [plan.md](docs/plan.md).

## Target Performance

| Metric | macOS (ANE) | Windows ARM64 target |
|---|---|---|
| End-to-end latency | < 1 s | < 1 s on NPU, < 2.5 s DirectML, < 4 s CPU |
| Model | Parakeet TDT 0.6B v3 (CoreML) | Parakeet TDT 0.6B v3 (ONNX, FP16) |
| Inference | Apple Neural Engine | Qualcomm Hexagon NPU via QNN EP |
| Runtime memory | ~38 MB | target < 120 MB |
| Bundle | 7.9 MB | target < 25 MB (model downloaded on first launch) |

## Architecture

```
openwritr-windows/
├── src-tauri/                  Rust core
│   ├── src/
│   │   ├── main.rs             tray app, state machine, IPC
│   │   ├── audio/              cpal WASAPI 16 kHz capture
│   │   ├── hotkey/             global-hotkey + FSM
│   │   ├── asr/                ort + QNN/DirectML/CPU EP, TDT greedy decode
│   │   ├── paste/              enigo Ctrl+V + clipboard save/restore
│   │   └── enhance/            Copilot / OpenAI grammar provider
│   └── tauri.conf.json
├── src/                        SvelteKit / vanilla UI for settings + overlay
├── scripts/
│   └── export_parakeet_onnx.py NeMo → ONNX export + FP16 quantization
└── .github/workflows/          ARM64 build + sign + release
```

## Build (Snapdragon X dev machine)

```powershell
# prerequisites: Rust (stable, aarch64-pc-windows-msvc), Node 20+, pnpm
winget install Rustlang.Rustup Microsoft.NodeJS
rustup target add aarch64-pc-windows-msvc
npm install -g pnpm

pnpm install
pnpm tauri dev          # local run
pnpm tauri build --target aarch64-pc-windows-msvc
```

## CI

The GitHub Actions workflow ships as `docs/build.yml.workflow-template`
(the initial commit was pushed with an OAuth token lacking `workflow` scope).
To activate it once:

```powershell
gh auth refresh -h github.com -s workflow
mkdir .github\workflows
git mv docs\build.yml.workflow-template .github\workflows\build.yml
git commit -m "ci: enable Windows ARM64 build workflow"
git push
```

## License

MIT — see [LICENSE](LICENSE).

Upstream macOS app: [trsdn/OpenWritr](https://github.com/trsdn/OpenWritr).
