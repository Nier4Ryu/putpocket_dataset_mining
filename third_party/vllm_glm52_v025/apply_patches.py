from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-dir", required=True)
    parser.add_argument("--profile", choices=["clean", "patched"], default="patched")
    args = parser.parse_args()
    manifest = yaml.safe_load((ROOT / "manifest.yaml").read_text(encoding="utf-8"))
    expected_commit = manifest["upstream"]["commit"]
    vllm_dir = Path(args.vllm_dir)
    commit = subprocess.check_output(["git", "-C", str(vllm_dir), "rev-parse", "HEAD"], text=True).strip()
    if commit != expected_commit:
        raise SystemExit(f"vLLM commit mismatch: expected {expected_commit}, got {commit}")
    for patch in manifest["profiles"][args.profile]["patches"]:
        patch_path = ROOT / patch["path"]
        actual = _sha256(patch_path)
        if actual != patch["sha256"]:
            raise SystemExit(f"Patch checksum mismatch for {patch_path}: {actual}")
        check = subprocess.run(["git", "-C", str(vllm_dir), "apply", "--check", str(patch_path)], text=True, capture_output=True)
        if check.returncode == 0:
            subprocess.check_call(["git", "-C", str(vllm_dir), "apply", str(patch_path)])
            continue
        reverse = subprocess.run(["git", "-C", str(vllm_dir), "apply", "--reverse", "--check", str(patch_path)], text=True, capture_output=True)
        if reverse.returncode == 0:
            print(f"already applied: {patch_path}")
            continue
        raise SystemExit(f"Patch cannot apply cleanly and is not already applied: {patch_path}\n{check.stderr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
