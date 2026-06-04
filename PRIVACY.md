# OpenWritr Privacy Policy

*Last updated: June 2026*

OpenWritr is a push-to-talk voice-to-text tool that runs **entirely on your
device**.

## What we collect

**Nothing.** OpenWritr has no telemetry, no analytics, no accounts, and no
servers operated by us.

## How your voice is processed

- Audio is captured from your microphone **only while you hold the hotkey**.
- Transcription runs **locally on your device** (NVIDIA Parakeet model on
  your CPU or NPU). Your audio never leaves your machine.
- The recognized text is pasted at your cursor and is not stored by
  OpenWritr beyond that.

## Network access

OpenWritr connects to the internet only for:

1. **One-time model download** from Hugging Face (`huggingface.co`) on first
   launch — this fetches the speech-recognition model files. No personal
   data is sent; this is a plain file download.
2. **Optional text cleanup ("Enhance")** — *off by default*. If you enable
   it and configure a provider (GitHub Copilot or an OpenAI-compatible API),
   the **recognized text** (not audio) is sent to that provider for
   grammar/punctuation cleanup using **your own API credentials**. The
   provider's own privacy policy applies to that processing. Disable the
   Enhance feature to keep everything 100% local.

## Data stored on your device

Settings, downloaded models, and a local diagnostic log are stored under
`%LOCALAPPDATA%\OpenWritr\` on your machine. They never leave it. Deleting
that folder (or uninstalling) removes them.

## Contact

Questions: open an issue at
https://github.com/trsdn/openwritr-windows/issues
