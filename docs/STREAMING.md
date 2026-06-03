# Streaming Parakeet — design notes

Status: **research / unstarted**. Lives on `experiment/streaming-parakeet`.

This document captures what we know, what's missing, and how a streaming
NPU path would have to slot into the existing pipeline. The goal is to
replace today's chunk-and-stitch (`MAX_NPU_SECONDS = 8.0`, 1 s overlap)
with true cache-aware streaming so that:

- Latency stays sub-100 ms regardless of utterance length.
- Long utterances don't pay the per-chunk encoder cost.
- Long-form dictation feels live: partial transcripts can be surfaced
  before the user releases the hotkey.

## What "streaming" means for Parakeet TDT v3

Parakeet TDT v3 uses a FastConformer encoder + Token-and-Duration
Transducer decoder. Cache-aware streaming means the encoder is built
to accept and emit cache state alongside its features:

```
encoder(
  audio_chunk:      f32[1, 128, T_chunk],
  length:           i32[1],
  cache_last_channel:     f32[L, 1, T_ctx_l, D]   // self-attention KV cache
  cache_last_time:        f32[L, 1, D, T_ctx_t]   // depthwise-conv cache
  cache_last_channel_len: i64[1]                  // valid length within cache_last_channel
) ->
  features,
  feature_lens,
  cache_last_channel_next,
  cache_last_time_next,
  cache_last_channel_len_next
```

The conventional NeMo parameters for v3 are:

| Parameter | Value |
|---|---|
| `chunk_secs` | 2.0 |
| `left_context_secs` | 10.0 |
| `right_context_secs` | 2.0 |
| Effective latency | ~2.5–3 s |

The "right context" is look-ahead: the encoder needs 2 s of audio
*after* the current chunk to emit its predictions for that chunk. So
"streaming" here is really chunked offline with a 2-s look-ahead delay,
not zero-latency.

A smaller `chunk_secs` (e.g., 0.5 s) reduces latency proportionally
but increases per-second compute cost.

## What's actually available today

Confirmed via NVIDIA's HF discussion ([parakeet-tdt-0.6b-v3 #11](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/discussions/11))
and the open sherpa-onnx issue ([#2918](https://github.com/k2-fsa/sherpa-onnx/issues/2918)):

- **No public ONNX export with cache support exists for Parakeet TDT v3.**
  The istupakov export we use today is offline-only.
- NeMo's `model.set_export_config({'cache_support': True})` is the
  documented hook for adding cache I/O during export, but it has not
  been verified by anyone publicly for v3 specifically.
- NVIDIA's own recommendation when asked about streaming is to switch
  to `stt_en_fastconformer_hybrid_large_streaming_multi` — which is
  English-only. Dealbreaker for us; we want the 25-language coverage.
- NVIDIA mentioned an "upgraded performant streaming variant" in
  development. No release date. We could wait — or build our own.

## Plan, if we build it

### Phase 1 — get a working streaming ONNX

1. Set up a Python env with NeMo (heavy; we may want a separate `.venv-nemo`).
2. Download the `.nemo` checkpoint of `nvidia/parakeet-tdt-0.6b-v3`.
3. Reconfigure the encoder for streaming (cache-aware) with the documented
   `(chunk=2, left=10, right=2)` setup.
4. `model.set_export_config({'cache_support': True})` + `model.export(...)`.
5. Validate the resulting ONNX with synthetic inputs end-to-end in
   Python (compare against the offline output for a known utterance).

Risk: NeMo's streaming export is most thoroughly tested for cache-aware
Conformer variants. Parakeet's FastConformer + TDT decoder may need
patching. There's a real chance this just doesn't work without
upstream NeMo changes.

### Phase 2 — HTP compile

1. Same surgery + freeze playbook as `scripts/build_npu_encoder.py`:
   freeze the chunk input shape (e.g. `[1, 128, 201]` for 2 s),
   freeze the cache shapes (depend on `left_context_secs` and number
   of encoder layers).
2. Submit to AI Hub: quantize + compile. The cache I/O will probably
   need `--truncate_64bit_io` plus careful int-vs-float handling on
   the `cache_last_channel_len` input.
3. Wrap as EPContext ONNX.
4. Validate with our existing `scripts/test_npu_encoder.py` extended
   to feed cache inputs.

Risk: HTP cache I/O sizes can be large (depends on `D × L × T_ctx_l`).
We may run into VTCM allocation limits or HTP graph finalization
failures specific to the cache tensors. Same kind of HTP debugging
we already did for the static-shape encoder.

### Phase 3 — Rust integration

1. `src/asr/qnn_ffi.rs`: extend `NpuEncoderFfi::run` to accept and
   return the cache tensors. Add `NpuEncoderFfi::reset()` to zero the
   cache between utterances.
2. `src/asr/parakeet.rs`: replace the chunked-encode path with a
   streaming loop. New control flow:

       on hotkey press:
           encoder.reset()
           audio_buffer = []
           tdt_state = decoder.new_state()
           start audio capture (continues until release)

       every chunk_secs of buffered audio:
           features, cache' = encoder.run(chunk, cache)
           tokens, tdt_state = decoder.run_step(features, tdt_state)
           append tokens to partial transcript

       on hotkey release:
           process trailing chunk
           drain TDT decoder
           emit final transcript

3. Surface partial transcripts to the overlay (extend the existing
   pill UI) and optionally paste-as-you-speak (probably opt-in;
   competes with the "ENTER caret" UX).

Risk: the TDT decoder's internal state management may not be set up
for incremental decoding. If it isn't, we lose the latency win:
we'd have to re-run the decoder over the full feature stream on
release — back to the chunked-but-no-stitch story we have now.

### Phase 4 — UX choices

Open questions for the user:

- **Partial transcript display.** Stream into the overlay pill?
  Show in a separate floating box? Just internal until release?
- **Auto-paste cadence.** Paste-as-you-speak (every committed chunk)
  or batch-on-release like today?
- **Mode switching.** Streaming for long-form, offline for short
  push-to-talk — automatic by buffer length? Manual setting?

## Phase 1 spike — Windows ARM64 verdict

NeMo cannot be installed on Windows ARM64 directly. Confirmed
experimentally on this branch:

| Component | Status |
|---|---|
| `torch` 2.12.0+cpu | ✅ installs from `https://download.pytorch.org/whl/cpu` (`cp311-win_arm64`) |
| `torchaudio` | ❌ **no win_arm64 wheel exists anywhere** (PyPI nor pytorch.org) |
| `nemo_toolkit[asr]` | ❌ `ctc_segmentation` triggers a NumPy compile from source, which dies on "Broken toolchain: cannot link a simple C program" even with the vcvars-primed MSVC ARM64 shell that builds our Rust code happily |

So Phase 1 cannot proceed in the Windows-native build env.

### Pivot — WSL2 Ubuntu ARM64

Ubuntu ARM64 in WSL2 has full PyPI access, including `torchaudio` and
all NeMo transitive deps with prebuilt aarch64 wheels. The streaming
ONNX produced there is fully portable — we'd bring it back to the
Windows host for HTP compile (AI Hub submission works from anywhere
with a token) and Rust integration.

**Requires manual user action**: enable Windows Subsystem for Linux,
which needs admin / UAC. Once available:

```powershell
# (admin shell, one time)
wsl --install -d Ubuntu

# (user shell, after WSL+Ubuntu come up)
wsl
sudo apt update && sudo apt install -y python3.11 python3.11-venv ffmpeg
python3.11 -m venv ~/venv-nemo
source ~/venv-nemo/bin/activate
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install nemo_toolkit[asr] onnx onnxruntime huggingface_hub
```

The NeMo work runs inside WSL. Output `.onnx` files land under
`/mnt/c/...` so they're directly available to AI Hub scripts on the
Windows side.

### WSL spike — what we ran

WSL2 + Ubuntu 26.04 aarch64 installed cleanly. NeMo env came up on
Python 3.14 with prebuilt aarch64 wheels:

| Component | Status |
|---|---|
| `torch` 2.12.0, `torchaudio` 2.11.0 | ✅ aarch64 wheels on PyPI (cp314) |
| `nemo_toolkit[asr]` 2.5.0 | ✅ (after `pip install -U ml_dtypes` to fix an onnx import) |
| `nvidia/parakeet-tdt-0.6b-v3.nemo` | ✅ restores as `EncDecRNNTBPEModel` + `ConformerEncoder` |

So the Windows-ARM64 blocker is fully solved via WSL. Scripts:
`scripts/wsl_setup_nemo.sh`, `scripts/wsl_load_parakeet.py`,
`scripts/wsl_export_streaming.py`.

## Phase 1 verdict — NO-GO for true streaming on v3

Two independent blockers, found by inspecting the restored checkpoint:

### 1. v3 was trained offline (fundamental)

```
cfg.att_context_size:  [-1, -1]      # full bidirectional attention
cfg.att_context_style: regular        # NOT chunked_limited
cfg.self_attention_model: rel_pos
```

Cache-aware streaming in NeMo only produces correct results for models
**trained** with `att_context_style: chunked_limited` and a limited
left/right context. Parakeet TDT v3 was trained full-context. You can
mechanically force `set_default_att_context_size([70, 13])` (the call
succeeds), but the encoder never learned to predict from limited
context — streaming-mode output would diverge from offline and degrade.
NeMo's own guarantee ("predictions identical in offline and streaming
mode") explicitly does **not** hold for a regular-style model.

### 2. The export path is also broken on torch 2.12 (mechanical, moot)

Even ignoring #1, `model.encoder.export()` with `cache_support: True`
fails:

```
RuntimeError: Failed to convert 'dynamic_axes' to 'dynamic_shapes'.
ValueError: treespec.unflatten(leaves): leaves has length 6 but the
            spec refers to a pytree that holds 5 items
```

torch 2.12's `torch.export` requires `dynamic_shapes`, and NeMo 2.5's
cache-aware ONNX path still emits the legacy `dynamic_axes` for the
extra cache tensors. Fixable (provide `dynamic_shapes` by hand, or pin
an older torch), but not worth it given blocker #1.

## The actual trade-off

True cache-aware streaming models **do** exist — but they're all
English-only:

| Model | Streaming | Languages | Notes |
|---|---|---|---|
| `nvidia/parakeet-tdt-0.6b-v3` (ours) | ❌ offline | **25** | what we ship; chunk-and-stitch on the NPU |
| `nvidia/nemotron-speech-streaming-en-0.6b` | ✅ cache-aware | English | 80/160/560/1120 ms latency modes |
| `nvidia/multitalker-parakeet-streaming-0.6b-v1` | ✅ cache-aware | English | `att_context_size` [70,13]=1.12 s default |

No multilingual cache-aware streaming model is published yet. NVIDIA's
HF discussions say cache-aware multilingual variants are "planned," no
date.

So the choice is fundamental, not technical:

- **Keep 25 languages** → stay on v3, offline. Our current chunk-and-stitch
  on the NPU is already the right answer; there's no streaming win to be
  had without a different model.
- **Want true sub-second streaming** → adopt an English-only model
  (nemotron-speech-streaming is the stronger one per its paper). That's a
  second engine option, not a replacement — English users get streaming,
  everyone else stays on multilingual offline.

## Recommendation

**Park true streaming until NVIDIA ships a multilingual cache-aware
model.** Our chunk-and-stitch already keeps long-form latency reasonable
(44× realtime at 16 s) without sacrificing the 25-language coverage that
is OpenWritr's whole point.

If sub-second English streaming is independently desirable, that's a
separate, additive track: wire `nemotron-speech-streaming-en-0.6b` as an
opt-in "English (streaming)" engine. It reuses all of the Phase 2-4
plumbing sketched below — and unlike v3 it would actually benefit from it.

Phases 2-4 below stay valid as the integration plan **for whichever
cache-aware model we adopt** (English now, multilingual when it lands).

## Related work

- [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) — our offline-mode source. Confirmed no cache support.
- [trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s](https://huggingface.co/trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s) — our v0.3 offline 8 s HTP model.
- [sherpa-onnx #2918](https://github.com/k2-fsa/sherpa-onnx/issues/2918) — public attempt with no working answer.
- [parakeet-tdt-0.6b-v3 discussion #11](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/discussions/11) — NVIDIA's recommendation to use the English-only streaming variant.
- [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) — Nemotron streaming variant, English-only.
- [NeMo streaming docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html) — official cache-aware streaming docs (Conformer-centric).
