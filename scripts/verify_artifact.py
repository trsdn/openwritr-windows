"""Verify staged, ZIP, or MSIX release contents against the canonical manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

from prepare_release import normalize_relative_path, release_spec


ARTIFACT_MANIFEST = "artifact-manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_directory(path: Path) -> dict[str, bytes]:
    files = {}
    for file_path in path.rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(path).as_posix()
            files[normalize_relative_path(relative)] = file_path.read_bytes()
    return files


def read_archive(path: Path) -> dict[str, bytes]:
    files = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative = normalize_relative_path(info.filename)
            if relative in files:
                raise RuntimeError(f"{path} contains duplicate entry {relative}")
            files[relative] = archive.read(info)
    return files


def load_artifact_manifest(files: dict[str, bytes]) -> dict:
    try:
        raw = files[ARTIFACT_MANIFEST]
    except KeyError as error:
        raise RuntimeError(f"artifact does not contain {ARTIFACT_MANIFEST}") from error
    try:
        manifest = json.load(io.BytesIO(raw))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {ARTIFACT_MANIFEST}: {error}") from error
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported artifact manifest schema")
    return manifest


def verify(architecture: str, artifact: Path, package_format: str) -> None:
    _, release = release_spec(architecture)
    files = read_directory(artifact) if artifact.is_dir() else read_archive(artifact)
    manifest = load_artifact_manifest(files)
    if manifest.get("architecture") != architecture:
        raise RuntimeError(
            f"artifact architecture is {manifest.get('architecture')!r}, "
            f"expected {architecture!r}"
        )

    required = {
        normalize_relative_path(file_spec["target"])
        for file_spec in release["files"]
    }
    optional = {
        normalize_relative_path(file_spec["target"])
        for file_spec in release.get("optional_files", [])
    }
    declared = {}
    for entry in manifest.get("files", []):
        relative = normalize_relative_path(entry.get("path", ""))
        if relative in declared:
            raise RuntimeError(f"artifact manifest contains duplicate file {relative}")
        declared[relative] = entry
    if not required.issubset(declared):
        raise RuntimeError(
            f"artifact manifest is missing required files: {sorted(required - set(declared))}"
        )
    unexpected_declared = set(declared) - required - optional
    if unexpected_declared:
        raise RuntimeError(
            f"artifact manifest declares unexpected files: {sorted(unexpected_declared)}"
        )

    for relative, entry in declared.items():
        try:
            data = files[relative]
        except KeyError as error:
            raise RuntimeError(f"artifact is missing declared file {relative}") from error
        if not data:
            raise RuntimeError(f"artifact contains empty required file {relative}")
        if entry.get("bytes") != len(data):
            raise RuntimeError(
                f"artifact size mismatch for {relative}: "
                f"expected {entry.get('bytes')}, got {len(data)}"
            )
        digest = sha256_bytes(data)
        if entry.get("sha256") != digest:
            raise RuntimeError(
                f"artifact SHA-256 mismatch for {relative}: "
                f"expected {entry.get('sha256')}, got {digest}"
            )

    forbidden = {
        normalize_relative_path(path).lower()
        for path in release.get("forbidden_files", [])
    }
    present_forbidden = sorted(path for path in files if path.lower() in forbidden)
    if present_forbidden:
        raise RuntimeError(
            f"{architecture} artifact contains forbidden files: {present_forbidden}"
        )

    if package_format in ("zip", "directory"):
        expected = set(declared) | {ARTIFACT_MANIFEST}
        actual = set(files)
        if actual != expected:
            raise RuntimeError(
                f"artifact file set mismatch; missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )

    print(
        f"verified {architecture} {package_format} artifact {artifact} "
        f"({len(declared)} release files)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("arm64", "x64"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--format", choices=("directory", "zip", "msix"), required=True
    )
    args = parser.parse_args()
    verify(args.arch, args.artifact, args.format)


if __name__ == "__main__":
    main()
