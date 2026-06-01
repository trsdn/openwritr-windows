"""Download Parakeet TDT v3 ONNX model from Hugging Face into LOCALAPPDATA."""
from __future__ import annotations
import os
import sys
import urllib.request
from pathlib import Path

REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
FILES = [
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "nemo128.onnx",
    "vocab.txt",
    "config.json",
]
DST = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "OpenWritr" / "models" / "parakeet-tdt-0.6b-v3"


def fetch(url: str, out: Path) -> None:
    print(f"  -> {out.name}", flush=True)
    with urllib.request.urlopen(url) as r, out.open("wb") as f:
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
                pct = 100 * done / total
                print(f"     {done/1e6:.1f} / {total/1e6:.1f} MB ({pct:.0f}%)", end="\r", flush=True)
        print()


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{REPO}/resolve/main"
    for name in FILES:
        out = DST / name
        if out.exists() and out.stat().st_size > 0:
            print(f"  ok {name} ({out.stat().st_size/1e6:.1f} MB)")
            continue
        fetch(f"{base}/{name}", out)
    print(f"\nModel in {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
