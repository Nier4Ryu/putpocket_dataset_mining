from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-dir", required=True)
    parser.add_argument("--profile", choices=["clean", "patched"], default="patched")
    args = parser.parse_args()
    manifest = yaml.safe_load((ROOT / "manifest.yaml").read_text(encoding="utf-8"))
    vllm_dir = Path(args.vllm_dir)
    commit = subprocess.check_output(["git", "-C", str(vllm_dir), "rev-parse", "HEAD"], text=True).strip()
    if commit != manifest["upstream"]["commit"]:
        raise SystemExit(f"commit_mismatch expected={manifest['upstream']['commit']} actual={commit}")
    for patch in manifest["profiles"][args.profile]["patches"]:
        patch_path = ROOT / patch["path"]
        actual = hashlib.sha256(patch_path.read_bytes()).hexdigest()
        if actual != patch["sha256"]:
            raise SystemExit(f"patch_sha_mismatch path={patch_path} actual={actual}")
        reverse = subprocess.run(["git", "-C", str(vllm_dir), "apply", "--reverse", "--check", str(patch_path)], text=True, capture_output=True)
        if reverse.returncode != 0:
            raise SystemExit(f"patch_not_applied path={patch_path}")
    print(f"profile={args.profile} commit={commit} status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
