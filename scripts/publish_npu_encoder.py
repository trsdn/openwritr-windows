"""
End-to-end wrapper that publishes an NPU encoder for a given window length:
local surgery + freeze → AI Hub quantize + compile → EPContext wrapper →
Hugging Face upload.

Designed for the 16 s window today; reusable for future variants by
changing --seconds. The HF repo name is derived from --seconds:
    trsdn/parakeet-tdt-0.6b-v3-htp-int8-{seconds}s

Usage:
    python scripts/publish_npu_encoder.py --seconds 16
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


LOCALAPP = Path.home() / "AppData" / "Local" / "OpenWritr"
MODELS = LOCALAPP / "models"
CALIB_GLOB = str(LOCALAPP / "calibration" / "fleurs" / "*" / "*.wav")


def run(cmd, **kwargs):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, required=True)
    ap.add_argument("--max-calib", type=int, default=32)
    ap.add_argument("--skip-build", action="store_true",
                    help="skip the local surgery+freeze step (use existing encoder-frozen.onnx)")
    ap.add_argument("--skip-aihub", action="store_true",
                    help="skip AI Hub quantize+compile (use existing .bin)")
    ap.add_argument("--skip-hf", action="store_true",
                    help="skip Hugging Face upload")
    args = ap.parse_args()

    secs = args.seconds
    py = sys.executable
    build_dir = MODELS / f"parakeet-htp-{secs}s"
    stage_dir = MODELS / f"_aihub_stage_{secs}s"
    final_dir = MODELS / f"parakeet-tdt-0.6b-v3-htp-int8-{secs}s"
    fp32_dir = MODELS / "parakeet-tdt-0.6b-v3-fp32"
    cpu_int8_dir = MODELS / "parakeet-tdt-0.6b-v3-onnx"
    hf_repo = f"trsdn/parakeet-tdt-0.6b-v3-htp-int8-{secs}s"

    t_overall = time.time()

    # 1) Local surgery + freeze.
    if not args.skip_build:
        t0 = time.time()
        run([
            py, "scripts/build_npu_encoder.py",
            "--fp32-encoder", fp32_dir / "encoder-model.onnx",
            "--preprocessor", fp32_dir / "nemo128.onnx",
            "--out-dir", build_dir,
            "--seconds", str(secs),
        ])
        print(f"\n=== local build done in {time.time()-t0:.0f} s ===\n")

    # 2) Stage clean dir for AI Hub (just .onnx + .onnx.data).
    if not args.skip_aihub:
        stage_dir.mkdir(parents=True, exist_ok=True)
        for f in stage_dir.iterdir():
            f.unlink()
        for name in ("encoder-frozen.onnx", "encoder-frozen.onnx.data"):
            shutil.copy(build_dir / name, stage_dir / name)
        print(f"staged {stage_dir}")

        # 3) AI Hub quantize + compile.
        final_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        run([
            py, "scripts/aihub_compile_encoder.py",
            "--fp32-encoder", stage_dir / "encoder-frozen.onnx",
            "--preprocessor", fp32_dir / "nemo128.onnx",
            "--calib-glob", CALIB_GLOB,
            "--max-calib", str(args.max_calib),
            "--seconds", str(secs),
            "--out", final_dir / "encoder-model.bin",
        ])
        print(f"\n=== AI Hub done in {time.time()-t0:.0f} s ===\n")

    # 4) Locate the compile_job model_id so we can build the EPContext wrapper
    #    with the right input/output specs.
    import qai_hub as hub
    # The compile job model_id is printed by aihub_compile_encoder.py but not
    # captured here. We list the user's recent jobs and grab the most recent
    # successful compile job whose name matches.
    print("locating compile job…")
    jobs = list(hub.get_job_summaries(limit=20, job_type="compile"))
    job = next((j for j in jobs
                if j.status.code == "SUCCESS" and "parakeet-encoder-htp-compile" in j.name),
               None)
    if job is None:
        raise SystemExit("could not find a SUCCESS compile job named 'parakeet-encoder-htp-compile' in your recent jobs")
    full_job = hub.get_job(job.job_id)
    model = full_job.get_target_model()
    model_id = model.model_id
    print(f"using compile job {job.job_id}, model {model_id}")

    # 5) Build EPContext wrapper.
    run([
        py, "scripts/wrap_qnn_context_binary.py",
        "--bin", final_dir / "encoder-model.bin",
        "--aihub-model-id", model_id,
        "--out", final_dir / "encoder-model.onnx",
    ])

    # 6) HF upload. Generate a window-aware README from the 8s template.
    if not args.skip_hf:
        readme = (MODELS / "parakeet-tdt-0.6b-v3-htp-int8-8s" / "README.md").read_text(encoding="utf-8")
        readme_new = readme \
            .replace("htp-int8-8s", f"htp-int8-{secs}s") \
            .replace("8 s window", f"{secs} s window") \
            .replace("8-second", f"{secs}-second") \
            .replace("8 s of audio", f"{secs} s of audio") \
            .replace("1, 128, 801", f"1, 128, {secs * 100 + 1}") \
            .replace("[1, 1024, 101]", f"[1, 1024, {(secs * 100 + 1) // 8 + 1}]")
        (final_dir / "README.md").write_text(readme_new, encoding="utf-8")

        from huggingface_hub import HfApi, create_repo
        print(f"\nuploading {hf_repo}")
        create_repo(hf_repo, exist_ok=True, private=False, repo_type="model")
        HfApi().upload_folder(
            folder_path=str(final_dir),
            repo_id=hf_repo,
            repo_type="model",
            allow_patterns=["encoder-model.bin", "encoder-model.onnx", "README.md"],
            commit_message=f"Initial upload: encoder for Snapdragon X Elite, {secs} s window",
        )
        print(f"\nDONE: https://huggingface.co/{hf_repo}")

    print(f"\n=== total wall time: {time.time()-t_overall:.0f} s ===")


if __name__ == "__main__":
    main()
