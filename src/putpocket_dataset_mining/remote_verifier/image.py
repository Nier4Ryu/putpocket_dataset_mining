from __future__ import annotations

import fcntl
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from putpocket_dataset_mining.errors import InfraError

from .manifest import sha256_file
from .paths import remote_job_root


@dataclass(frozen=True)
class ImageStatus:
    image: str
    image_id: str | None
    dockerfile_sha256: str | None
    built: bool


def ensure_image(image: str, dockerfile: Path, *, build_if_missing: bool = True, timeout_sec: int = 900) -> ImageStatus:
    docker = shutil.which("docker")
    if docker is None:
        raise InfraError("infra_failed: docker executable missing on verifier host")
    dockerfile_sha = sha256_file(dockerfile) if dockerfile.exists() else None
    lock_path = remote_job_root() / "locks" / "docker-image-build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = _image_id(docker, image)
        if existing:
            return ImageStatus(image=image, image_id=existing, dockerfile_sha256=dockerfile_sha, built=False)
        if not build_if_missing:
            raise InfraError(f"infra_failed: docker image missing and build disabled: {image}")
        if not dockerfile.exists():
            raise InfraError(f"infra_failed: Dockerfile missing: {dockerfile}")
        result = subprocess.run(
            [docker, "build", "-t", image, "-f", str(dockerfile), str(dockerfile.parents[1])],
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        if result.returncode != 0:
            raise InfraError(f"infra_failed: docker image build failed: {result.stderr[-4000:]}")
        return ImageStatus(image=image, image_id=_image_id(docker, image), dockerfile_sha256=dockerfile_sha, built=True)


def _image_id(docker: str, image: str) -> str | None:
    result = subprocess.run([docker, "image", "inspect", image, "--format", "{{.Id}}"], text=True, capture_output=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
