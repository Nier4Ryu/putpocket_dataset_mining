from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if yaml is None:
        import json

        return json.loads(text)
    return yaml.safe_load(text)


class RunpodAgentImageContractTests(unittest.TestCase):
    def test_tool_lock_pins_agent_versions(self) -> None:
        lock = _load_yaml(ROOT / "configs" / "env" / "runpod_dev_tools.lock.yaml")
        self.assertEqual(lock["node"]["version"], "24.19.0")
        self.assertEqual(lock["node"]["sha256"], "14b342e71204f811bde6153be8e04b62aef63c236fef92b55f9c83154b409647")
        self.assertEqual(lock["node"]["bundled_npm_version"], "11.17.0")
        self.assertEqual(lock["zellij"]["version"], "0.44.3")
        self.assertEqual(lock["zellij"]["sha256"], "0f7c346788627f506c0a28296517768633cff24fc822a739f8264b640ecad751")
        self.assertEqual(lock["codex"]["version"], "0.147.0")
        self.assertIn("sha512-", lock["codex"]["package_integrity"])
        self.assertEqual(lock["uv"]["version"], "0.11.31")

    def test_dockerfile_keeps_cuda_devel_and_installs_pinned_tools(self) -> None:
        text = (ROOT / "cloud" / "runpod" / "Dockerfile.dev-base").read_text(encoding="utf-8")
        self.assertIn(
            "FROM --platform=linux/amd64 nvidia/cuda:12.9.1-devel-ubuntu22.04@sha256:bd4e2680a261c212f1e2fea241606f71497dc67a417f73175d794ec8212b5ba8",
            text,
        )
        self.assertNotIn("cudnn-runtime", text)
        self.assertNotIn("apt-get install -y nodejs", text)
        self.assertNotIn("apt-get install -y npm", text)
        self.assertIn("ARG NODE_VERSION=24.19.0", text)
        self.assertIn("ARG ZELLIJ_VERSION=0.44.3", text)
        self.assertIn("ARG CODEX_VERSION=0.147.0", text)
        self.assertIn("npm install -g \"@openai/codex@${CODEX_VERSION}\"", text)
        self.assertIn("COPY cloud/runpod/start-dev-container.sh", text)
        self.assertIn("      bubblewrap \\", text)
        self.assertIn("rm -f /etc/ssh/ssh_host_*_key", text)
        self.assertIn('CMD ["/usr/local/bin/putpocket-runpod-start"]', text)
        self.assertNotIn("pip install torch", text)
        self.assertNotIn("pip install vllm", text)
        self.assertNotIn("pip install lmcache", text)
        self.assertNotIn("COPY .", text)
        self.assertNotIn("ADD .", text)
        self.assertNotIn("ENTRYPOINT", text)

    def test_dockerfile_sets_tool_path_and_codex_home(self) -> None:
        text = (ROOT / "cloud" / "runpod" / "Dockerfile.dev-base").read_text(encoding="utf-8")
        self.assertIn("ENV CODEX_HOME=/workspace/.private/codex", text)
        self.assertIn('ENV NPM_CONFIG_PREFIX=/opt/npm-global', text)
        self.assertIn('ENV PATH="/opt/node/bin:/opt/npm-global/bin:/usr/local/bin:${PATH}"', text)

    def test_startup_script_is_inert_and_secret_safe(self) -> None:
        script = ROOT / "cloud" / "runpod" / "start-dev-container.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn("chmod 700", text)
        self.assertIn('cli_auth_credentials_store = "file"', text)
        self.assertIn('sandbox_mode = "danger-full-access"', text)
        self.assertIn('approval_policy = "on-request"', text)
        self.assertIn("exec sleep infinity", text)
        self.assertNotIn("auth.json", text)
        self.assertNotIn("codex login", text)
        self.assertNotIn("git clone", text)
        self.assertNotIn("bootstrap_sr.sh", text)
        self.assertNotIn("vllm", text.lower())

    def test_startup_reconciles_existing_codex_config_without_credentials(self) -> None:
        script = ROOT / "cloud" / "runpod" / "start-dev-container.sh"
        with tempfile.TemporaryDirectory() as tmp_name:
            codex_home = Path(tmp_name) / "codex"
            codex_home.mkdir()
            config = codex_home / "config.toml"
            original = 'model = "gpt-test"\nsandbox_mode = "workspace-write"\n\n[features]\nweb_search = true\n'
            config.write_text(original, encoding="utf-8")
            env = os.environ | {"CODEX_HOME": str(codex_home), "PUTPOCKET_RUNPOD_CONFIG_ONLY": "1"}
            subprocess.check_call(["bash", str(script)], env=env)
            updated = config.read_text(encoding="utf-8")
            self.assertIn('model = "gpt-test"', updated)
            self.assertIn('cli_auth_credentials_store = "file"', updated)
            self.assertIn('sandbox_mode = "danger-full-access"', updated)
            self.assertIn('approval_policy = "on-request"', updated)
            self.assertIn('[features]\nweb_search = true', updated)
            backup = codex_home / "config.toml.pre-runpod-policy.bak"
            self.assertEqual(backup.read_text(encoding="utf-8"), original)
            self.assertEqual({path.name for path in codex_home.iterdir()}, {"config.toml", backup.name})
            subprocess.check_call(["bash", str(script)], env=env)
            self.assertEqual(config.read_text(encoding="utf-8"), updated)

    def test_startup_script_syntax(self) -> None:
        subprocess.check_call(["bash", "-n", str(ROOT / "cloud" / "runpod" / "start-dev-container.sh")])

    def test_template_uses_codex_home_and_no_plaintext_api_key(self) -> None:
        text = (ROOT / "cloud" / "runpod" / "template.dev-base.example.yaml").read_text(encoding="utf-8")
        self.assertIn("CODEX_HOME: /workspace/.private/codex", text)
        self.assertIn("start_command: /usr/local/bin/putpocket-runpod-start", text)
        self.assertIn("digest: sha256:<set-after-push>", text)
        self.assertIn("OPENAI_API_KEY: \"<runpod-secret-reference-optional>\"", text)
        self.assertNotIn("sk-", text)
        self.assertNotIn("auth.json:", text)
        self.assertNotIn("api_server", text)

    def test_dockerignore_excludes_credentials_and_runtime_payloads(self) -> None:
        text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        for pattern in (
            ".git",
            "Putpocket_env*",
            ".ssh",
            "auth.json",
            ".env.*",
            "*token*",
            "*credential*",
            "*secret*",
            "models",
            "data",
            "logs",
            "externals",
        ):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, text)


if __name__ == "__main__":
    unittest.main()
