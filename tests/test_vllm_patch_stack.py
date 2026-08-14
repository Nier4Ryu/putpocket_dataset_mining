from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path("third_party/vllm_glm52_v025")


class VllmPatchStackTests(unittest.TestCase):
    def test_manifest_profiles_and_patch_checksums(self) -> None:
        manifest = yaml.safe_load((ROOT / "manifest.yaml").read_text(encoding="utf-8"))
        self.assertEqual(manifest["upstream"]["commit"], "752a3a504485790a2e8491cacbb35c137339ad34")
        self.assertEqual(manifest["profiles"]["clean"]["patches"], [])
        patch = manifest["profiles"]["patched"]["patches"][0]
        self.assertEqual(hashlib.sha256((ROOT / patch["path"]).read_bytes()).hexdigest(), patch["sha256"])

    def test_build_manifest_key_differs_by_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            subprocess.check_call(["git", "init", str(repo)], stdout=subprocess.DEVNULL)
            (repo / "x").write_text("x", encoding="utf-8")
            subprocess.check_call(["git", "-C", str(repo), "add", "x"], stdout=subprocess.DEVNULL)
            subprocess.check_call(["git", "-C", str(repo), "-c", "user.email=a@b.c", "-c", "user.name=t", "commit", "-m", "x"], stdout=subprocess.DEVNULL)
            out1 = Path(tmp) / "sm86.json"
            out2 = Path(tmp) / "sm90.json"
            subprocess.check_call([sys.executable, str(ROOT / "build_manifest.py"), "--vllm-dir", str(repo), "--profile", "clean", "--target-arch-list", "8.6", "--out", str(out1)])
            subprocess.check_call([sys.executable, str(ROOT / "build_manifest.py"), "--vllm-dir", str(repo), "--profile", "clean", "--target-arch-list", "9.0", "--out", str(out2)])
            self.assertNotEqual(json.loads(out1.read_text())["build_id"], json.loads(out2.read_text())["build_id"])


if __name__ == "__main__":
    unittest.main()
