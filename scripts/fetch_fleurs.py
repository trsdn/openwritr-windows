"""Fetch FLEURS audio samples for ASR calibration.

Downloads `test.tar.gz` for a few languages from google/fleurs on HF and
extracts ~N audio files per language into a local cache directory.
"""
from __future__ import annotations
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

CACHE = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "OpenWritr" / "calibration" / "fleurs"
LANGS = ["en_us", "de_de", "es_419", "fr_fr"]
PER_LANG = 25


def fetch_lang(lang: str) -> Path:
    out_dir = CACHE / lang
    if out_dir.exists() and len(list(out_dir.glob("*.wav"))) >= PER_LANG:
        print(f"  {lang}: already have {len(list(out_dir.glob('*.wav')))} wavs", flush=True)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = CACHE / f"{lang}-test.tar.gz"
    if not tar_path.exists():
        url = f"https://huggingface.co/datasets/google/fleurs/resolve/main/data/{lang}/audio/test.tar.gz"
        print(f"  {lang}: downloading {url}", flush=True)
        urllib.request.urlretrieve(url, tar_path)
    extracted = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".wav"):
                continue
            target = out_dir / Path(member.name).name
            if target.exists():
                extracted += 1
                if extracted >= PER_LANG:
                    break
                continue
            with tf.extractfile(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted += 1
            if extracted >= PER_LANG:
                break
    print(f"  {lang}: {extracted} wavs in {out_dir}", flush=True)
    return out_dir


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        fetch_lang(lang)
    total = sum(1 for _ in CACHE.rglob("*.wav"))
    print(f"\ntotal: {total} wavs in {CACHE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
