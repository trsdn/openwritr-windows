# OpenWritr Privacy Policy

*Last updated: July 2026*

OpenWritr is a local-first push-to-talk voice-to-text tool. Audio capture and
speech recognition run **entirely on your device**.

## What we collect

**Nothing.** OpenWritr has no telemetry, no analytics, no accounts, and no
servers operated by us.

## How your voice is processed

- Audio is captured from your microphone **only while you hold the hotkey**.
- Transcription runs **locally on your device** (NVIDIA Parakeet model on
  your CPU or NPU). Your audio never leaves your machine.
- The recognized text is pasted at your cursor or copied to the Windows
  clipboard, depending on your Settings choice. OpenWritr does not keep a
  transcript history.

## Network access

OpenWritr connects to the internet only for:

1. **One-time model download** from Hugging Face (`huggingface.co`) on first
   launch — this fetches the speech-recognition model files. No personal
   data is sent; this is a plain file download.
2. **Optional text cleanup ("Enhance")** — *off by default*. You can run it
   only when Shift is additionally held or for every recording. If enabled,
   the **recognized text** (never audio) is sent to GitHub Copilot or your
   configured OpenAI-compatible API for grammar and punctuation cleanup using
   your own credentials. The provider's privacy policy applies. Set enhancement
   to **Never** to keep all processing local.

## Data stored on your device

Settings, downloaded models, and a local diagnostic log are stored under
`%LOCALAPPDATA%\OpenWritr\` on your machine. They never leave it. Deleting
that folder (or uninstalling) removes them.

## Contact

Questions: open an issue at
https://github.com/trsdn/openwritr-windows/issues
