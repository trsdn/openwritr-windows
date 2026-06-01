# Port Plan

Mirror of the session plan in human-readable form. See repository issues for live status.

## Phases
1. **scaffold** — Tauri v2 + Cargo workspace + CI (this commit)
2. **audio** — cpal WASAPI 16 kHz mono ring buffer
3. **hotkey** — global-hotkey + FSM (Idle / Recording / Transcribing / Enhancing)
4. **model-export** — NeMo → ONNX (encoder + decoder + joint), FP16 quantization
5. **asr-runtime** — `ort` with QNN → DirectML → CPU; TDT greedy decode in Rust
6. **paste** — enigo Ctrl+V with clipboard save/restore
7. **overlay** — transparent click-through Tauri window with recording indicator
8. **settings** — hotkey, language, enhanced-mode provider, keyring → Credential Manager
9. **enhance** — port GrammarEnhancer.swift to Rust (`reqwest`)
10. **packaging** — MSIX + signed `.exe` for ARM64, GitHub Release workflow
11. **bench** — latency NPU vs DirectML vs CPU, update README

## Risks
- QNN EP wheel availability for ARM64 — may require custom build.
- Parakeet TDT ONNX export: encoder is easy; joint+predictor sometimes need
  manual rewiring. Fallback: FastConformer-CTC variant.
- < 1 s end-to-end latency is a *target*, not a guarantee until the model is
  measured on Hexagon NPU with FP16 weights.
