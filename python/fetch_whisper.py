"""Download and extract Qualcomm AI Hub's pre-compiled Whisper Large v3 Turbo
for Snapdragon X Elite into LOCALAPPDATA.

The archive ships QNN ONNX context binaries for the encoder + decoder, plus
the tokenizer. Once extracted, the engine module can load them.
"""
from __future__ import annotations
import os
import sys
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DST = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "OpenWritr" / "models" / "whisper-large-v3-turbo-qnn"
URL = (
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/"
    "qai-hub-models/models/whisper_large_v3_turbo/releases/v0.54.0/"
    "whisper_large_v3_turbo-precompiled_qnn_onnx-float-qualcomm_snapdragon_x_elite.zip"
)


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    if any(DST.glob("*.onnx")) or any(DST.glob("*.bin")):
        print(f"Whisper already extracted in {DST}", flush=True)
        return 0
    tmp = Path(tempfile.gettempdir()) / "whisper-turbo-qnn.zip"
    if not tmp.exists() or tmp.stat().st_size < 100_000:
        print(f"downloading from {URL}", flush=True)
        with urllib.request.urlopen(URL) as r, tmp.open("wb") as f:
            total = int(r.headers.get("Content-Length", "0"))
            done = 0
            chunk = 1 << 20
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    print(f"  {done/1e6:.1f} / {total/1e6:.1f} MB ({100*done/total:.0f}%)",
                          end="\r", flush=True)
            print()
    print(f"extracting to {DST} …", flush=True)
    with zipfile.ZipFile(tmp, "r") as zf:
        zf.extractall(DST)
    # Many QAI Hub archives nest contents in a subfolder; flatten if needed.
    inner = [p for p in DST.iterdir() if p.is_dir()]
    if len(inner) == 1 and not any(DST.glob("*.onnx")):
        for item in inner[0].iterdir():
            shutil.move(str(item), DST / item.name)
        inner[0].rmdir()
    print("done. contents:", flush=True)
    for p in sorted(DST.iterdir()):
        sz = p.stat().st_size if p.is_file() else 0
        print(f"  {p.name:<60}  {sz/1e6:8.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
