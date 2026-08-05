from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-dir", required=True)
    parser.add_argument("--profile", choices=["clean", "patched"], required=True)
    parser.add_argument("--target-arch-list", required=True)
    parser.add_argument("--wheel-path", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = yaml.safe_load((ROOT / "manifest.yaml").read_text(encoding="utf-8"))
    patch_digest = _patch_digest(manifest["profiles"][args.profile]["patches"])
    wheel = Path(args.wheel_path) if args.wheel_path else None
    data = {
        "schema_version": 1,
        "build_id": _build_id(manifest["upstream"]["commit"], args.profile, patch_digest, args.target_arch_list),
        "vllm_commit": _git(args.vllm_dir, ["rev-parse", "HEAD"]),
        "vllm_profile": args.profile,
        "patch_digest": patch_digest,
        "python_version": platform.python_version(),
        "torch_version": _python_value("import torch; print(torch.__version__)"),
        "torch_cuda_version": _python_value("import torch; print(torch.version.cuda)"),
        "cuda_toolkit_version": _nvcc_version(),
        "nvcc_version": _nvcc_version(),
        "target_cuda_arch_list": args.target_arch_list,
        "detected_gpu_names": _nvidia_smi("name"),
        "detected_compute_capabilities": _nvidia_smi("compute_cap"),
        "wheel_path": str(wheel) if wheel else None,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest() if wheel and wheel.exists() else None,
        "build_timestamp_kst": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    Path(args.out).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(args.out)
    return 0


def _patch_digest(patches: list[dict]) -> str:
    h = hashlib.sha256()
    for patch in patches:
        h.update(str(patch["path"]).encode("utf-8"))
        h.update(str(patch["sha256"]).encode("utf-8"))
    return h.hexdigest()


def _build_id(commit: str, profile: str, patch_digest: str, arch: str) -> str:
    raw = f"{commit}:{profile}:{patch_digest}:{arch}:{platform.python_version()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _git(root: str, args: list[str]) -> str:
    return subprocess.check_output(["git", "-C", root, *args], text=True).strip()


def _python_value(code: str) -> str:
    try:
        return subprocess.check_output(["python", "-c", code], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def _nvcc_version() -> str:
    try:
        return subprocess.check_output(["nvcc", "--version"], text=True).strip().splitlines()[-1]
    except Exception:  # noqa: BLE001
        return "unavailable"


def _nvidia_smi(field: str) -> list[str]:
    try:
        out = subprocess.check_output(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"], text=True)
    except Exception:  # noqa: BLE001
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
