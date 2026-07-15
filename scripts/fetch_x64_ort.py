"""Compatibility wrapper for the hash-pinned x64 runtime fetcher."""

import argparse

from fetch_runtime import load_manifest, stage_runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version")
    args = ap.parse_args()

    pinned = load_manifest()["architectures"]["x64"]["packages"][0]["version"]
    if args.version and args.version != pinned:
        ap.error(
            f"runtime version is pinned to {pinned} in runtime-manifest.json; "
            "update the manifest and hashes instead"
        )
    stage_runtime("x64")


if __name__ == "__main__":
    main()
