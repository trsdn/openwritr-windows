# Runtime compatibility

OpenWritr loads one process-wide ONNX Runtime DLL. On ARM64, the same
runtime and QNN execution-provider stack serves Parakeet CPU, Parakeet NPU,
and Whisper NPU, so their versions must move together.

The pinned candidate tuple is:

| Component | Version |
|---|---|
| Rust `ort` crate | `2.0.0-rc.12`, API 24 |
| ONNX Runtime | `1.24.4` |
| `onnxruntime-qnn` | `2.1.1` |
| QAIRT runtime | `2.45.41` |
| AI Hub compile line | `2.45` |
| First NPU target | Snapdragon X Elite |

`onnxruntime-qnn` 2.1.1 declares an exact dependency on ONNX Runtime
1.24.4. Do not force it onto ONNX Runtime 1.25 with `--no-deps`: that
would mix an execution-provider plugin with an unsupported host ABI.

Qualcomm's Whisper Large v3 Turbo v0.57.3 model card records QAIRT 2.45
and ONNX Runtime 1.25.0 for its precompiled QNN ONNX export. The QAIRT
line matches, but the host ORT minor differs from the supported QNN wheel
pair. Loading and inference therefore remain a Snapdragon X Elite
hardware acceptance gate. If the precompiled wrapper is incompatible,
use the QAIRT 2.45 context-binary artifact with a wrapper generated for
the pinned runtime rather than mixing unsupported DLL versions.

## Fetching the runtime

The checked-in [`runtime-manifest.json`](../runtime-manifest.json) records
the exact wheel URLs, SHA-256 hashes, archive paths, and staged filenames.
No package resolver or floating PyPI version is involved.

```powershell
python scripts/fetch_runtime.py --arch arm64
python scripts/fetch_runtime.py --arch x64
```

The commands stage files into `target/release` and `vendor/x64`
respectively and write a deterministic `runtime-versions.json` receipt.

## Validation split

CI can verify hashes, archive contents, Rust builds, API-24 initialization,
CPU inference, and package manifests without NPU hardware. The following
must pass on Snapdragon X Elite:

- QNN provider registration and V73 device enumeration.
- Parakeet NPU context loading and inference.
- Whisper encoder and decoder context loading and inference.
- Numerical and latency checks against the pinned reference corpus.

Parakeet NPU artifacts must be compiled with
`--qairt_version 2.45`; the AI Hub job URL and requested options are
stored next to the downloaded context binary as provenance.
