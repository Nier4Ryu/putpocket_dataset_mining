from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

from .constants import BUILD_ENV_OVERRIDES, DEFAULT_DOCKERFILE, REPO_ROOT
from .externals import externals_status


def collect_doctor_report() -> dict[str, Any]:
    commands = {
        "docker": shutil.which("docker"),
        "codex": shutil.which("codex"),
        "git": shutil.which("git"),
        "python3": shutil.which("python3"),
    }
    modules = {
        "yaml": importlib.util.find_spec("yaml") is not None,
        "datasets": importlib.util.find_spec("datasets") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "ray": importlib.util.find_spec("ray") is not None,
        "vllm": importlib.util.find_spec("vllm") is not None,
        "lmcache": importlib.util.find_spec("lmcache") is not None,
    }
    required_paths = {
        "repo_root": str(REPO_ROOT),
        "dockerfile": DEFAULT_DOCKERFILE.exists(),
        "single_config": (REPO_ROOT / "configs" / "dataset_mining" / "mbpp_stateful_single.yaml").exists(),
        "multi_config": (REPO_ROOT / "configs" / "dataset_mining" / "mbpp_stateful_multi.yaml").exists(),
    }
    return {
        "commands": commands,
        "python_modules": modules,
        "build_env_overrides": BUILD_ENV_OVERRIDES,
        "required_paths": required_paths,
        "externals": externals_status(),
    }


def format_doctor_report(report: dict[str, Any]) -> str:
    lines = ["Dataset mining doctor report"]
    lines.append("")
    lines.append("Commands:")
    for name, value in report["commands"].items():
        lines.append(f"  {name}: {value or 'missing'}")
    lines.append("")
    lines.append("Python modules:")
    for name, ok in report["python_modules"].items():
        lines.append(f"  {name}: {'ok' if ok else 'missing'}")
    lines.append("")
    lines.append("Build env overrides:")
    for name, value in report["build_env_overrides"].items():
        lines.append(f"  {name}={value}")
    lines.append("")
    lines.append("Required paths:")
    for name, value in report["required_paths"].items():
        lines.append(f"  {name}: {value}")
    lines.append("")
    lines.append("Externals:")
    for row in report["externals"]:
        lines.append(f"  {row['name']}: exists={row['exists']} path={Path(str(row['path']))}")
    return "\n".join(lines)
