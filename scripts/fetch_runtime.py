"""Fetch hash-pinned ONNX Runtime and QNN files from wheel archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "runtime-manifest.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(package: dict, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    wheel = cache_dir / package["wheel"]
    expected = package["sha256"]
    if wheel.exists() and sha256_file(wheel) == expected:
        print(f"cached {wheel.name}")
        return wheel

    wheel.unlink(missing_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=wheel.name, suffix=".tmp", dir=cache_dir)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        print(f"downloading {package['name']} {package['version']}")
        with urllib.request.urlopen(package["url"], timeout=120) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target)
        actual = sha256_file(temporary)
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {wheel.name}: expected {expected}, got {actual}"
            )
        temporary.replace(wheel)
    finally:
        temporary.unlink(missing_ok=True)
    return wheel


def extract_package(package: dict, wheel: Path, output: Path) -> list[dict]:
    staged = []
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for file_spec in package["files"]:
            source_name = file_spec["source"]
            if source_name not in names:
                if file_spec["required"]:
                    raise RuntimeError(f"{wheel.name} does not contain {source_name}")
                print(f"  optional file not present: {source_name}")
                continue

            destination = output / Path(file_spec["target"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.unlink(missing_ok=True)
            try:
                with archive.open(source_name) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if temporary.stat().st_size == 0:
                    raise RuntimeError(f"{source_name} extracted as an empty file")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)

            staged.append(
                {
                    "path": file_spec["target"].replace("\\", "/"),
                    "required": file_spec["required"],
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                    "package": package["name"],
                    "package_version": package["version"],
                }
            )
            print(f"  staged {file_spec['target']}")
    return staged


def stage_runtime(
    architecture: str,
    output: Path | None = None,
    cache_dir: Path | None = None,
) -> Path:
    manifest = load_manifest()
    try:
        architecture_spec = manifest["architectures"][architecture]
    except KeyError as error:
        raise RuntimeError(f"unsupported architecture: {architecture}") from error

    output = output or ROOT / architecture_spec["default_output"]
    cache_dir = cache_dir or ROOT / "target" / "runtime-cache"
    output.mkdir(parents=True, exist_ok=True)

    receipt_path = output / "runtime-versions.json"
    desired_paths = {
        file_spec["target"].replace("\\", "/")
        for package in architecture_spec["packages"]
        for file_spec in package["files"]
    }
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        for previous_file in previous.get("files", []):
            relative = previous_file["path"]
            if relative not in desired_paths:
                stale = output / Path(relative)
                stale.unlink(missing_ok=True)
                print(f"removed stale {relative}")

    staged = []
    packages = []
    for package in architecture_spec["packages"]:
        wheel = download(package, cache_dir)
        staged.extend(extract_package(package, wheel, output))
        packages.append(
            {
                "name": package["name"],
                "version": package["version"],
                "wheel": package["wheel"],
                "wheel_sha256": package["sha256"],
            }
        )

    receipt = {
        "schema_version": manifest["schema_version"],
        "architecture": architecture,
        "rust_ort": manifest["rust_ort"],
        "qnn": manifest["qnn"] if architecture == "arm64" else None,
        "compatibility": manifest["compatibility"] if architecture == "arm64" else None,
        "packages": packages,
        "files": sorted(staged, key=lambda item: item["path"].lower()),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {receipt_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("arm64", "x64"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    stage_runtime(args.arch, args.out, args.cache_dir)


if __name__ == "__main__":
    main()
