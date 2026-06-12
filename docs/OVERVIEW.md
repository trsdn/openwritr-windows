# OpenWritr — Overview, Architecture & Purpose

## What it is

OpenWritr is a **push-to-talk voice-to-text tool for Windows**. You hold a
hotkey, speak, and release — the recognized text is pasted at your cursor in
whatever application has focus. It is a small system-tray utility, not a
window you switch to: dictation happens *into* the app you're already using
(email, chat, code editor, browser, anything).

Two things make it different from the dictation built into Windows or from
cloud services:

1. **It runs entirely on your device.** Speech recognition happens locally.
   Your audio never leaves the machine — no cloud, no account, no telemetry.
2. **It uses the NPU.** On Snapdragon X (Copilot+ PC) hardware, the speech
   model runs on the Qualcomm Hexagon Neural Processing Unit, which is far
   faster and far more power-efficient than the CPU. On Intel/AMD machines
   it runs on the CPU instead.

## Purpose / why it exists

Voice input on Windows-on-ARM was a gap. The Snapdragon X laptops (Surface
Pro 11, Surface Laptop 7, etc.) ship a powerful NPU that mostly sits idle,
while the best local speech model — NVIDIA's Parakeet — had no turnkey way
to run on it. OpenWritr closes that: a genuinely local, fast, multilingual
dictation tool that puts the Hexagon NPU to work, with a CPU fallback so the
same app also serves Intel/AMD users.

The design goal is **friction-free dictation**: press a key, talk, done. No
window management, no "start/stop recording" UI, no sending your voice to a
server.

## What it does, concretely

- **25 languages**, auto-detected (English, German, French, Spanish, Italian,
  Dutch, Portuguese, Polish, Russian, Ukrainian, Czech, and more European
  languages).
- **Push-to-talk**: hold a configurable hotkey (default **Ctrl+Win**), speak,
  release. Text is pasted at the caret.
- **Fast**: a 5-second utterance transcribes in well under half a second on
  the NPU (~25–44× realtime); long dictation is chunked transparently.
- **Optional AI cleanup ("Enhance")**: hold Shift as well, and after
  transcription the text is sent to GitHub Copilot or any OpenAI-compatible
  API for punctuation/grammar polish — using *your* credentials, *off by
  default*.
- **Unobtrusive**: a small overlay pill with a live level meter appears only
  while recording; otherwise it's just a tray icon.

## Architecture

OpenWritr is a native **Rust** application (no Electron, no web view, no
bundled runtime). A single ~7 MB executable.

```
┌──────────────────────────────────────────────────────────────┐
│  openwritr.exe  (Rust, winit event loop on the main thread)    │
│                                                                │
│  ┌────────────┐   global WH_KEYBOARD_LL hook (own thread)      │
│  │  key_hook  │── physical key state → atomic bitmap           │
│  └────────────┘                                                │
│        │ press / release events                                │
│  ┌────────────┐   WASAPI capture via cpal, rebuilt per         │
│  │   audio    │── recording; multi-channel → mono → 16 kHz     │
│  └────────────┘                                                │
│        │ f32 samples                                           │
│  ┌──────────────────────────────────────────────────┐         │
│  │  asr pipeline                                       │        │
│  │   mel preprocessor (ONNX, CPU)                      │        │
│  │        ↓                                            │        │
│  │   ENCODER  ── NPU (Hexagon HTP)  OR  CPU INT8       │        │
│  │        ↓                                            │        │
│  │   TDT decoder loop (ONNX, CPU) → tokens → text      │        │
│  └──────────────────────────────────────────────────┘         │
│        │ text                                                  │
│  ┌────────────┐   optional LLM cleanup (your API)              │
│  │  enhance   │── then paste at cursor (clipboard + input)     │
│  └────────────┘                                                │
│                                                                │
│  tray icon · overlay window (layered, GDI) · egui settings     │
│  (settings open as a `--settings` subprocess)                  │
└──────────────────────────────────────────────────────────────┘
```

### The speech model

NVIDIA **Parakeet TDT 0.6B v3** — a 600M-parameter Conformer encoder with a
Token-and-Duration Transducer (TDT) decoder. State-of-the-art accuracy for
its size, multilingual, with built-in punctuation and capitalization.

The pipeline has three ONNX stages:
1. **Mel preprocessor** — turns raw audio into mel-spectrogram features (CPU,
   tiny).
2. **Encoder** — the heavy part (the 600M params). This is what runs on the
   NPU.
3. **TDT decoder** — an autoregressive loop that emits tokens (CPU,
   lightweight).

### The two engines

| | arm64 (Snapdragon) | x64 (Intel/AMD) |
|---|---|---|
| Encoder runs on | Hexagon NPU (HTP) | CPU |
| Model format | INT8/INT16 QAIRT context binary | INT8 ONNX |
| Speed | ~25–44× realtime | ~25× realtime |
| Power draw | very low (NPU offload) | moderate (CPU) |

The **NPU path** was the hard engineering. Parakeet's encoder uses a
dynamic-shape attention mask the Hexagon backend can't evaluate, so the
encoder is reshaped to a fixed 8-second window (its attention-mask subgraph
constant-folded), quantized to INT8 weights / INT16 activations, and compiled
to a Qualcomm AI Hub QNN context binary. At runtime that binary is loaded
through ONNX Runtime's QNN execution provider via a thin direct-FFI layer
(the high-level Rust ORT bindings crash on this particular model, so we call
the C API directly). Audio longer than 8 seconds is run in overlapping
chunks and the encoder features are stitched back together before decoding —
so long-form dictation works without re-running the decoder per chunk.

The **CPU path** is the standard ONNX Runtime CPU execution provider on the
INT8 model — simpler, runs anywhere, no Qualcomm dependency.

### Notable design decisions

- **Global low-level keyboard hook**, not polling. Windows fakes key-release
  events during focus changes (a popup, a UAC prompt, a system shortcut),
  which would otherwise abort a recording mid-sentence. The LL hook sees only
  real physical keystrokes, so recording survives focus steals.
- **Capture stream rebuilt per recording.** Holding one WASAPI stream open
  for the process lifetime led to a "dead mic after long idle" bug (Windows
  silently invalidates idle capture streams). Building a fresh stream on each
  press eliminates that and also picks up mic changes automatically.
- **Settings UI as a subprocess.** The egui settings dialog is the same exe
  relaunched with `--settings`, kept separate from the tray event loop so a
  slow dialog spawn can't freeze the tray.
- **Tray + overlay isolation.** The recording overlay runs on its own message
  loop and shares only two atomics with the recorder, so it can't deadlock
  the main app.

## Privacy

- **No telemetry, no analytics, no accounts.**
- Audio is captured only while the hotkey is held, processed locally, and not
  stored.
- The only network access is (1) a one-time model download from Hugging Face
  on first launch, and (2) the optional Enhance feature, which sends *text*
  (never audio) to a provider you configure with your own credentials, and is
  off by default.
- Settings, models, and a local log live under `%LOCALAPPDATA%\OpenWritr\`
  and never leave the machine.

## Tech stack

- **Language:** Rust
- **UI:** winit + tray-icon + egui (settings) + Win32 (overlay)
- **Audio:** cpal (WASAPI)
- **Inference:** ONNX Runtime (CPU EP everywhere; QNN EP on arm64 for the NPU)
- **Model:** NVIDIA Parakeet TDT 0.6B v3 (CC-BY-4.0); arm64 NPU build
  compiled via Qualcomm AI Hub
- **Distribution:** Microsoft Store (MSIX, both arches) + GitHub Releases
  (installer + portable zip); CI builds on GitHub Actions
- **License:** MIT (own code); bundled ONNX Runtime (MIT) and Qualcomm QNN
  runtime (Qualcomm redistributable license) ship with their notices

## Platform support

- **Windows 11** (22621+) on **ARM64** — full NPU acceleration on Snapdragon X.
- **Windows 11** on **x64** — CPU inference on Intel/AMD.
- Companion macOS app: [trsdn/OpenWritr](https://github.com/trsdn/OpenWritr)
  (Apple Neural Engine via CoreML).

## Links

- Source: https://github.com/trsdn/openwritr-windows
- NPU model: https://huggingface.co/trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s
- Base model: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
