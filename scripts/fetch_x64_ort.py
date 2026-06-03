"""
Fetch the x86_64 (Intel/AMD) CPU build of onnxruntime.dll into vendor/x64/.

The arm64 build gets its DLL from `pip install onnxruntime-qnn`, but that
package is Snapdragon-only. For the Intel/AMD build we need the plain
`onnxruntime` CPU wheel (win_amd64) and just its onnxruntime.dll — no QNN,
no execution-provider plugins. Parakeet runs on the CPU EP.

    python scripts/fetch_x64_ort.py [--version 1.26.0]
"""
import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "vendor" / "x64"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.26.0")
    ap.add_argument("--py", default="cp311")
    args = ap.parse_args()

    meta = json.load(urllib.request.urlopen(
        f"https://pypi.org/pypi/onnxruntime/{args.version}/json"))
    url = next(f["url"] for f in meta["urls"]
               if "win_amd64" in f["filename"] and args.py in f["filename"])
    print(f"downloading {url}")
    whl = urllib.request.urlopen(url).read()

    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(whl)) as z:
        with z.open("onnxruntime/capi/onnxruntime.dll") as src:
            data = src.read()
    (OUT / "onnxruntime.dll").write_bytes(data)
    print(f"wrote {OUT / 'onnxruntime.dll'} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
