"""Validate and stage one release architecture from canonical manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
import time
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent.parent
RELEASE_MANIFEST_PATH = ROOT / "release-manifest.json"
RUNTIME_MANIFEST_PATH = ROOT / "runtime-manifest.json"
PE_MACHINES = {"arm64": 0xAA64, "x64": 0x8664}


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load {path}: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise RuntimeError(f"unsafe release path: {value!r}")
    return path.as_posix()


def pe_machine(path: Path) -> int:
    with path.open("rb") as source:
        header = source.read(64)
        if len(header) < 64 or header[:2] != b"MZ":
            raise RuntimeError(f"{path} is not a PE file")
        pe_offset = struct.unpack_from("<I", header, 0x3C)[0]
        source.seek(pe_offset)
        pe_header = source.read(6)
    if len(pe_header) != 6 or pe_header[:4] != b"PE\0\0":
        raise RuntimeError(f"{path} has an invalid PE header")
    return struct.unpack_from("<H", pe_header, 4)[0]


def expected_runtime_packages(runtime_manifest: dict, architecture: str) -> list[dict]:
    packages = runtime_manifest["architectures"][architecture]["packages"]
    return [
        {
            "name": package["name"],
            "version": package["version"],
            "wheel": package["wheel"],
            "wheel_sha256": package["sha256"],
        }
        for package in packages
    ]


def expected_runtime_files(runtime_manifest: dict, architecture: str) -> dict[str, dict]:
    expected = {}
    for package in runtime_manifest["architectures"][architecture]["packages"]:
        for file_spec in package["files"]:
            path = normalize_relative_path(file_spec["target"])
            if path in expected:
                raise RuntimeError(f"duplicate runtime target in manifest: {path}")
            expected[path] = {
                "required": file_spec["required"],
                "package": package["name"],
                "package_version": package["version"],
            }
    return expected


def validate_runtime(runtime_root: Path, architecture: str) -> None:
    runtime_manifest = load_json(RUNTIME_MANIFEST_PATH)
    receipt_path = runtime_root / "runtime-versions.json"
    receipt = load_json(receipt_path)

    expected_qnn = runtime_manifest["qnn"] if architecture == "arm64" else None
    expected_compatibility = (
        runtime_manifest["compatibility"] if architecture == "arm64" else None
    )
    comparisons = {
        "schema_version": runtime_manifest["schema_version"],
        "architecture": architecture,
        "rust_ort": runtime_manifest["rust_ort"],
        "qnn": expected_qnn,
        "compatibility": expected_compatibility,
        "packages": expected_runtime_packages(runtime_manifest, architecture),
    }
    for key, expected in comparisons.items():
        if receipt.get(key) != expected:
            raise RuntimeError(
                f"{receipt_path} has unexpected {key}: expected {expected!r}, "
                f"got {receipt.get(key)!r}"
            )

    expected_files = expected_runtime_files(runtime_manifest, architecture)
    receipt_files = {}
    for entry in receipt.get("files", []):
        path = normalize_relative_path(entry.get("path", ""))
        if path in receipt_files:
            raise RuntimeError(f"{receipt_path} contains duplicate file {path}")
        receipt_files[path] = entry
    if set(receipt_files) != set(expected_files):
        missing = sorted(set(expected_files) - set(receipt_files))
        extra = sorted(set(receipt_files) - set(expected_files))
        raise RuntimeError(
            f"{receipt_path} file set mismatch; missing={missing}, unexpected={extra}"
        )

    for relative, expected in expected_files.items():
        entry = receipt_files[relative]
        path = runtime_root / Path(relative)
        if not path.is_file():
            raise RuntimeError(f"required runtime file is missing: {path}")
        size = path.stat().st_size
        if size == 0:
            raise RuntimeError(f"required runtime file is empty: {path}")
        actual_hash = sha256_file(path)
        checks = {
            "required": expected["required"],
            "package": expected["package"],
            "package_version": expected["package_version"],
            "bytes": size,
            "sha256": actual_hash,
        }
        for key, value in checks.items():
            if entry.get(key) != value:
                raise RuntimeError(
                    f"runtime receipt mismatch for {relative} {key}: "
                    f"expected {value!r}, got {entry.get(key)!r}"
                )


def release_spec(architecture: str) -> tuple[dict, dict]:
    manifest = load_json(RELEASE_MANIFEST_PATH)
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported release manifest schema")
    try:
        return manifest, manifest["architectures"][architecture]
    except KeyError as error:
        raise RuntimeError(f"unsupported release architecture: {architecture}") from error


def collect_sources(architecture: str) -> tuple[dict, list[tuple[Path, str]]]:
    _, spec = release_spec(architecture)
    roots = {
        name: (ROOT / value).resolve()
        for name, value in spec["source_roots"].items()
    }
    if "runtime" not in roots:
        raise RuntimeError(f"{architecture} release has no runtime source root")
    validate_runtime(roots["runtime"], architecture)

    required_targets = set()
    sources = []
    for file_spec in spec["files"]:
        target = normalize_relative_path(file_spec["target"])
        if target in required_targets:
            raise RuntimeError(f"duplicate release target: {target}")
        required_targets.add(target)
        try:
            source_root = roots[file_spec["source_root"]]
        except KeyError as error:
            raise RuntimeError(
                f"unknown source root {file_spec['source_root']!r} for {target}"
            ) from error
        source = source_root / Path(normalize_relative_path(file_spec["source"]))
        validate_source(source, file_spec["kind"], architecture)
        sources.append((source, target))

    optional_targets = set()
    for file_spec in spec.get("optional_files", []):
        target = normalize_relative_path(file_spec["target"])
        if target in required_targets or target in optional_targets:
            raise RuntimeError(f"duplicate optional release target: {target}")
        optional_targets.add(target)
        source = roots[file_spec["source_root"]] / Path(
            normalize_relative_path(file_spec["source"])
        )
        if source.exists():
            validate_source(source, file_spec["kind"], architecture)
            sources.append((source, target))

    return spec, sources


def validate_source(source: Path, kind: str, architecture: str) -> None:
    if not source.is_file():
        raise RuntimeError(f"required release file is missing: {source}")
    if source.stat().st_size == 0:
        raise RuntimeError(f"required release file is empty: {source}")
    if kind == "pe":
        actual = pe_machine(source)
        expected = PE_MACHINES[architecture]
        if actual != expected:
            raise RuntimeError(
                f"{source} has PE machine 0x{actual:04x}; "
                f"expected {architecture} (0x{expected:04x})"
            )
    elif kind not in ("file", "runtime_receipt"):
        raise RuntimeError(f"unsupported release file kind {kind!r} for {source}")


def stage_release(architecture: str, output: Path | None = None) -> Path:
    spec, sources = collect_sources(architecture)
    output = output or ROOT / "target" / "stage" / architecture
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
    )
    try:
        artifact_files = []
        for source, relative in sorted(sources, key=lambda item: item[1].lower()):
            destination = temporary / Path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            artifact_files.append(
                {
                    "path": relative,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
            print(f"  staged {relative}")

        forbidden = {
            normalize_relative_path(path).lower()
            for path in spec.get("forbidden_files", [])
        }
        staged_lower = {entry["path"].lower() for entry in artifact_files}
        forbidden_present = sorted(staged_lower & forbidden)
        if forbidden_present:
            raise RuntimeError(
                f"{architecture} stage contains forbidden files: {forbidden_present}"
            )

        artifact_manifest = {
            "schema_version": 1,
            "architecture": architecture,
            "files": artifact_files,
        }
        (temporary / "artifact-manifest.json").write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        publish_directory(temporary, output)
        print(f"prepared {architecture} release at {output}")
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def publish_directory(temporary: Path, output: Path) -> None:
    for attempt in range(20):
        try:
            if output.exists():
                shutil.rmtree(output)
            temporary.replace(output)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=tuple(PE_MACHINES), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    stage_release(args.arch, args.out)


if __name__ == "__main__":
    main()
