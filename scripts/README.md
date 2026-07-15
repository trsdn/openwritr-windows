# `scripts/` — NPU model toolchain

Build-time tools used to produce the QNN HTP context binary that the
native app ships against. **None of these are invoked by openwritr.exe
at runtime** — the released app pulls the pre-compiled `.bin` from
[trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s](https://huggingface.co/trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s)
on first launch.

| Script | What it does |
|---|---|
| `build_npu_encoder.py` | Constant-folds the encoder's dynamic-shape attention mask (`Shape → Gather → Range → Expand`) against a frozen window length and saves a static-shape FP32 ONNX. The HTP backend cannot evaluate the dynamic subgraph; this step is what makes it compilable at all. Configurable via `--seconds`. |
| `aihub_compile_encoder.py` | Submits the static-shape FP32 ONNX plus FLEURS calibration samples to Qualcomm AI Hub. Quantize job: INT8 weights / INT16 activations (the standard HTP recipe for transformer encoders). Compile job: target `snapdragon_x_elite_crd`, runtime `qnn_context_binary`, `--truncate_64bit_io`. Output is the `.bin`. Supports `--reuse-quantize-job <id>` to skip re-quantize when iterating on compile options. |
| `wrap_qnn_context_binary.py` | Wraps the AI Hub `.bin` in a 408-byte EPContext-node ONNX so ORT's QNN EP can consume it. Pulls the I/O specs (renamed `output_0` / `output_1`, `length` as int32 after `--truncate_64bit_io`) from the AI Hub model directly via `qai-hub`. |
| `test_npu_encoder.py` | Standalone Python validator: registers the QNN EP, loads the wrapper via `OrtEpDevice`, runs a synthetic mel-shaped tensor through the encoder and times the steady-state. Useful for proving the model is sound independent of any Rust code. |
| `fetch_runtime.py` | Downloads the exact ARM64 or x64 runtime wheels pinned in `runtime-manifest.json`, verifies wheel SHA-256, extracts the required files, and writes `runtime-versions.json`. |
| `prepare_release.py` | Validates runtime versions, hashes, required files, PE architecture, and licenses, then creates the canonical per-architecture stage from `release-manifest.json`. |
| `verify_artifact.py` | Verifies staged directories, ZIP files, and MSIX packages against their embedded `artifact-manifest.json`; also rejects Qualcomm runtime files from x64 artifacts. |
| `build_msix.ps1` | Builds ARM64/x64 MSIX packages and a Store bundle. Local builds may use the TEST identity; release builds pass `-RequireStoreIdentity` with the Partner Center identity. |
| `bench_nexa_parakeet.ps1` | Benchmark scaffold for NexaAI's pre-compiled NPU Parakeet (license-gated, not used in distribution). Kept as a reference for comparing alternative QNN HTP runtimes. |
| `envup.ps1` | Dev-shell setup: primes vcvars arm64 + LLVM in the current PowerShell session so `cargo build` finds the MSVC toolchain. Source it once per shell (`.\scripts\envup.ps1`) before any cargo command. |

## Setup

```powershell
py -3.11-arm64 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install onnxruntime==1.24.4 onnxruntime-qnn==2.1.1 qai-hub onnx-graphsurgeon onnx scipy
qai-hub configure --api_token <token from aihub.qualcomm.com>
```

The shipped DLLs are not copied from this development environment. Use
`python scripts/fetch_runtime.py --arch arm64` so wheel identities and
SHA-256 hashes come from `runtime-manifest.json`.

## Full encoder rebuild

```powershell
# 1) Local: surgery + freeze (output is encoder-frozen.onnx in --out-dir).
python scripts/build_npu_encoder.py `
    --fp32-encoder C:/.../parakeet-tdt-0.6b-v3-fp32/encoder-model.onnx `
    --preprocessor C:/.../parakeet-tdt-0.6b-v3-fp32/nemo128.onnx `
    --out-dir C:/.../parakeet-htp-8s `
    --seconds 8

# 2) AI Hub: quantize + compile (uploads ~2.4 GB; ~30 min wall time).
python scripts/aihub_compile_encoder.py `
    --fp32-encoder C:/.../_aihub_stage_8s/encoder-frozen.onnx `
    --preprocessor C:/.../parakeet-tdt-0.6b-v3-fp32/nemo128.onnx `
    --calib-glob "C:/.../calibration/fleurs/*/*.wav" `
    --max-calib 32 `
    --seconds 8 `
    --qairt-version 2.45 `
    --out C:/.../parakeet-tdt-0.6b-v3-htp-int8-8s/encoder-model.bin

# 3) Local: build EPContext wrapper next to the .bin.
python scripts/wrap_qnn_context_binary.py `
    --bin C:/.../parakeet-tdt-0.6b-v3-htp-int8-8s/encoder-model.bin `
    --aihub-model-id <model_id from step 2> `
    --out C:/.../parakeet-tdt-0.6b-v3-htp-int8-8s/encoder-model.onnx

# 4) Local: validate end-to-end on the NPU.
python scripts/test_npu_encoder.py `
    --encoder C:/.../parakeet-tdt-0.6b-v3-htp-int8-8s/encoder-model.onnx
```

Typical wall time start-to-finish: ~35 min, dominated by the 2.4 GB
upload + quantize step.

For the parameter trade-offs (chunk size, calibration set size, INT16
activation choice) see the v0.3.0 commit message and the `## NPU
pipeline` section in the top-level `README.md`.
