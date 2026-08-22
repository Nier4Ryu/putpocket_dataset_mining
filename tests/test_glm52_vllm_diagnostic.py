from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import putpocket_dataset_mining.glm52_vllm_diagnostic as diagnostic
from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.glm52_vllm_diagnostic import (
    BUILD_SOURCE_COMMIT,
    FULL_LAYERS,
    INSTANCE_ID,
    file_sha256,
    build_runtime_jit_manifest,
    load_lock,
    validate_build_manifest,
    validate_capture_records,
    validate_inventory_csv,
    validate_lock,
    validate_model_config,
    validate_source_tree,
    validate_source_provenance,
    validate_trace_equivalence,
)
from putpocket_dataset_mining.glm52_vllm_diagnostic_slurm import (
    _build_wrapper,
    _run_wrapper,
    load_site,
    render_two_stage_submission,
)
from putpocket_dataset_mining.glm52_vllm_diagnostic_cli import main as diagnostic_cli_main


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs/cluster/glm52_vllm_diagnostic.lock.json"
SITE_PATH = ROOT / "configs/cluster/sites/herdr_vllm_diagnostic.json"
PROJECT_COMMIT = "1" * 40


def _site():
    return load_site(SITE_PATH)


def _render(*, runtime_source_commit: str = PROJECT_COMMIT, allow_runtime_source_split: bool = True) -> str:
    return render_two_stage_submission(
        site=_site(),
        project_url="https://github.com/openai/putpocket-dataset-mining.git",
        runtime_source_commit=runtime_source_commit,
        build_source_commit=BUILD_SOURCE_COMMIT,
        allow_runtime_source_split=allow_runtime_source_split,
        lock_path=LOCK_PATH,
    )


def test_lock_is_exact_vllm_diagnostic() -> None:
    report = validate_lock(load_lock())
    assert report["vllm_commit"] == "4a3447d200e5aa428d68d1a00aa00f1a19a1a729"
    assert report["instance_id"] == INSTANCE_ID
    assert len(report["full_layers"]) == 21
    build = load_lock()["build"]
    assert build["run_wheel_check"] is False
    assert build["upstream_release_wheel_limit_mb"] == 500
    assert build["wheel_size_exception_scope"] == "intentional_sm90_cuda13_source_build_only"
    assert build["project_source_commit"] == BUILD_SOURCE_COMMIT
    assert build["immutable_bundle_root"] == "/home2/jslee202403/putpocket-builds/vllm/vllm-4a3447d200e5-sm90-cu1303-py312-torch2130-patch-fc2f3734-image-3869b846"
    assert build["vllm_wheel_sha256"] == "3c408df63c56e2a711116449d4324fcef5f2043de1b5c3dee4d3bf561908af52"


def test_lock_rejects_precompiled_substitution_and_secrets() -> None:
    lock = load_lock()
    lock["build"]["vllm_use_precompiled"] = True
    with pytest.raises(ConfigError, match="vllm_use_precompiled"):
        validate_lock(lock)
    lock = load_lock(); lock["auth_token"] = "forbidden"
    with pytest.raises(ConfigError, match="SECRET_FIELD"):
        validate_lock(lock)


def test_source_patch_and_instrumentation_digests_are_exact() -> None:
    lock = load_lock(); source = lock["vllm"]
    assert file_sha256(ROOT / source["patch_path"]) == source["patch_sha256"]
    assert file_sha256(ROOT / source["instrumentation_source"]) == source["instrumentation_sha256"]
    patch = (ROOT / source["patch_path"]).read_text(encoding="utf-8")
    assert "fp8_fp4_mqa_logits" in patch
    assert "top_k_per_row_prefill" in patch
    assert "fp8_fp4_paged_mqa_logits" in patch
    assert "cooperative_topk_sm90" in patch
    assert "git -C \"$VLLM_ROOT\" apply --unidiff-zero --check" in (ROOT / "scripts/cluster/build_glm52_vllm_sm90.sh").read_text(encoding="utf-8")


def test_source_tree_wrong_digest_fails_before_patch(tmp_path: Path) -> None:
    lock = load_lock(); source_root = tmp_path / "vllm"; source_root.mkdir()
    with pytest.raises(ConfigError, match="SOURCE_CONTEXT_DIGEST"):
        validate_source_tree(ROOT, source_root, lock)


def _bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path, dict]:
    root = tmp_path / "bundle"; root.mkdir()
    paths = {"runtime_image_tar": "runtime-image.tar", "vllm_wheel": "wheels/vllm-test.whl", "source_bundle": "vllm-source-bundle.tar.gz"}
    for name in paths.values():
        target = root / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(name.encode())
    provenance_names = (
        "source_preflight.json", "source_post_patch.json", "build-wheel-image.log",
        "compiled_arches.txt", "wheel_artifact.json", "build_environment.json", "build_nvcc.txt",
        "build-runtime-image.log", "runtime_environment.json", "runtime_nvcc.txt",
    )
    provenance_paths = {name: f"logs/{name}" for name in provenance_names}
    for name in provenance_paths.values():
        target = root / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(name.encode())
    (root / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
    lock = load_lock(); build = lock["build"]
    wheel_entry = {"path": paths["vllm_wheel"], "sha256": hashlib.sha256((root / paths["vllm_wheel"]).read_bytes()).hexdigest(), "bytes": (root / paths["vllm_wheel"]).stat().st_size}
    monkeypatch.setattr(diagnostic, "VLLM_WHEEL_SHA256", wheel_entry["sha256"])
    build["vllm_wheel_sha256"] = wheel_entry["sha256"]
    wheel_policy = {"schema_version": 1, "run_wheel_check": False, "upstream_release_wheel_limit_mb": 500, "exception_scope": "intentional_sm90_cuda13_source_build_only", "wheel_path": wheel_entry["path"], "wheel_bytes": wheel_entry["bytes"], "wheel_sha256": wheel_entry["sha256"]}
    (root / provenance_paths["wheel_artifact.json"]).write_text(json.dumps(wheel_policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "status": "SUCCESS", "project_commit": BUILD_SOURCE_COMMIT,
        "vllm_commit": lock["vllm"]["commit"],
        "bundle_key": build["bundle_key"], "patch_sha256": lock["vllm"]["patch_sha256"],
        "patch_target_post_sha256": lock["vllm"]["patch_target_post_sha256"],
        "build_patch_target_post_sha256": lock["vllm"]["build_patch_target_post_sha256"],
        "instrumentation_sha256": lock["vllm"]["instrumentation_sha256"], "base_image": build["base_image"],
        "compiler_audit_sha256": lock["vllm"]["compiler_audit_sha256"],
        "python": "3.12", "torch": "2.13.0", "cuda": "13.0.3", "torch_cuda_arch_list": "9.0",
        "cmake_cuda_architectures": "90", "vllm_target_device": "cuda", "vllm_use_precompiled": False,
        "wheel_release_policy": wheel_policy,
        "general_h200_compilation_allowed": False, "h200_runtime_jit_scope": "native_first_use_deepgemm_dsa_only", "pinned_source_runtime_jit_required": True, "runtime_jit_cache_reuse": False,
        "prebuilt_vllm_wheel_used": False, "built_from_scratch": True, "compiled_arch_evidence": ["sm_90"],
        "runtime_gate": "ALLOW_NATIVE_FIRST_USE_JIT_WITH_RUN_LOCAL_AUDIT",
        "runtime_image_id": "sha256:" + "2" * 64,
        "build_environment": {"python_major_minor": "3.12", "torch_base": "2.13.0", "torch_cuda": "13.0", "resolved_packages": ["torch==2.13.0"]},
        "runtime_environment": {"python_major_minor": "3.12", "torch_base": "2.13.0", "torch_cuda": "13.0", "transformers": "5.3.0", "vllm": "0.1.dev0", "resolved_packages": ["torch==2.13.0", "transformers==5.3.0", "vllm==0.1.dev0"]},
        "files": {key: {"path": value, "sha256": hashlib.sha256((root / value).read_bytes()).hexdigest(), "bytes": (root / value).stat().st_size} for key, value in paths.items()},
        "provenance_files": {key: {"path": value, "sha256": hashlib.sha256((root / value).read_bytes()).hexdigest(), "bytes": (root / value).stat().st_size} for key, value in provenance_paths.items()},
    }
    checksum_lines = [f"{item['sha256']}  {item['path']}\n" for group in ("files", "provenance_files") for item in manifest[group].values()]
    (root / "SHA256SUMS").write_text("".join(sorted(checksum_lines, key=lambda line: line.split("  ", 1)[1])), encoding="utf-8")
    return manifest, root, lock


def test_build_bundle_validates_all_immutable_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, root, lock = _bundle(tmp_path, monkeypatch)
    report = validate_build_manifest(manifest, root, lock)
    assert report["status"] == "passed"
    assert report["immutable_build_source_commit"] == BUILD_SOURCE_COMMIT
    assert report["vllm_wheel_sha256"] == manifest["files"]["vllm_wheel"]["sha256"]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [("torch_cuda_arch_list", "", "TARGET_MISMATCH"), ("vllm_use_precompiled", True, "TARGET_MISMATCH"), ("prebuilt_vllm_wheel_used", True, "PREBUILT")],
)
def test_build_bundle_rejects_target_or_prebuilt_substitution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object, error: str) -> None:
    manifest, root, lock = _bundle(tmp_path, monkeypatch); manifest[field] = value
    with pytest.raises(ConfigError, match=error):
        validate_build_manifest(manifest, root, lock)


def test_build_bundle_rejects_changed_file_before_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, root, lock = _bundle(tmp_path, monkeypatch); (root / "runtime-image.tar").write_bytes(b"changed")
    with pytest.raises(ConfigError, match="DIGEST_MISMATCH"):
        validate_build_manifest(manifest, root, lock)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_wheel_check", True),
        ("upstream_release_wheel_limit_mb", 501),
        ("wheel_bytes", 1),
        ("wheel_sha256", "0" * 64),
    ],
)
def test_build_bundle_rejects_wheel_policy_or_artifact_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    manifest, root, lock = _bundle(tmp_path, monkeypatch)
    manifest["wheel_release_policy"][field] = value
    with pytest.raises(ConfigError, match="WHEEL_POLICY|WHEEL_ARTIFACT"):
        validate_build_manifest(manifest, root, lock)


def test_source_provenance_accepts_only_explicit_runtime_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, root, lock = _bundle(tmp_path, monkeypatch)
    report = validate_source_provenance(
        manifest,
        root,
        lock,
        expected_build_source_commit=BUILD_SOURCE_COMMIT,
        runtime_source_commit=PROJECT_COMMIT,
        observed_runtime_source_commit=PROJECT_COMMIT,
        wrapper_source_commit=PROJECT_COMMIT,
        allow_runtime_source_split=True,
    )
    assert report["immutable_build_source_commit"] == BUILD_SOURCE_COMMIT
    assert report["runtime_wrapper_source_commit"] == PROJECT_COMMIT
    assert report["source_split_explicitly_authorized"] is True
    assert report["vllm_wheel_sha256"] == manifest["files"]["vllm_wheel"]["sha256"]


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"expected_build_source_commit": "0" * 40}, "IMMUTABLE_BUILD_SOURCE_COMMIT_MISMATCH"),
        ({"observed_runtime_source_commit": "2" * 40}, "RUNTIME_SOURCE_COMMIT_MISMATCH"),
        ({"wrapper_source_commit": "3" * 40}, "WRAPPER_SOURCE_COMMIT_MISMATCH"),
        ({"allow_runtime_source_split": False}, "RUNTIME_BUILD_SOURCE_SPLIT_NOT_EXPLICITLY_AUTHORIZED"),
    ],
)
def test_source_provenance_rejects_either_identity_or_implicit_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    error: str,
) -> None:
    manifest, root, lock = _bundle(tmp_path, monkeypatch)
    values: dict[str, object] = {
        "expected_build_source_commit": BUILD_SOURCE_COMMIT,
        "runtime_source_commit": PROJECT_COMMIT,
        "observed_runtime_source_commit": PROJECT_COMMIT,
        "wrapper_source_commit": PROJECT_COMMIT,
        "allow_runtime_source_split": True,
    }
    values.update(overrides)
    with pytest.raises(ConfigError, match=error):
        validate_source_provenance(manifest, root, lock, **values)  # type: ignore[arg-type]


def test_site_uses_measured_cpu_and_h200_storage_values() -> None:
    site = load_site(SITE_PATH)
    assert (site.cpu.partition, site.cpu.account, site.cpu.qos) == ("cpu-max24", "gsai-account", "nogpu")
    assert (site.cpu.cpus_per_task, site.cpu.memory, site.cpu.wall_time) == (24, "192G", "06:00:00")
    assert site.cpu.local_scratch_root == Path("/local-data/user-data/jslee202403/putpocket-vllm-build-scratch")
    assert site.h200_storage_parent == Path("/local-data/user-data")
    assert site.h200_work_root == Path("/local-data/user-data/jslee202403/putpocket-glm52-vllm-diagnostic")
    assert site.h200_artifact_root == site.h200_work_root / "artifacts"


def test_site_refuses_invalid_runtime_or_artifact_root(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="OFFICIAL_DOCKER_RUNTIME_REQUIRED"):
        load_site(
            SITE_PATH, cpu_partition="cpu", cpu_account="account", cpu_qos="qos",
            cpu_cpus_per_task=8, cpu_memory="32G", cpu_wall_time="01:00:00",
            cpu_local_scratch_root="/scratch", container_executable="/usr/bin/apptainer",
        )
    value = json.loads(SITE_PATH.read_text(encoding="utf-8"))
    value["h200_run"]["artifact_root"] = "/local-data/user-data/jslee202403/other-artifacts"
    invalid = tmp_path / "invalid-site.json"
    invalid.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ConfigError, match="H200_ARTIFACT_ROOT"):
        load_site(invalid)


def test_renderer_is_two_stage_exact_and_has_afterok_dependency() -> None:
    command = _render()
    assert command.count("sbatch --parsable") == 2
    assert "--dependency=afterok:$VALIDATOR_JOB_ID" in command
    assert "--gres=gpu:H200:4" in command
    assert "--partition=cpu-max24" in command
    assert "--account=gsai-account" in command and "--qos=nogpu" in command
    assert "--cpus-per-task=24" in command and "--mem=192G" in command
    assert "--partition=H200" in command
    assert "PUTPOCKET_RUN_ARTIFACT_ROOT=/local-data/user-data/jslee202403/putpocket-glm52-vllm-diagnostic/artifacts" in command
    assert "/local-data/jslee202403" not in command
    assert "VALIDATOR_JOB_ID=%s\\nRUN_JOB_ID=%s" in command
    assert "--job-name=pp-vllm-bundle-validate" in command
    assert "PUTPOCKET_IMMUTABLE_BUNDLE_REUSE_ONLY=1" in command
    assert f"PUTPOCKET_BUILD_SOURCE_COMMIT={BUILD_SOURCE_COMMIT}" in command
    assert f"PUTPOCKET_RUNTIME_SOURCE_COMMIT={PROJECT_COMMIT}" in command
    assert "PUTPOCKET_ALLOW_RUNTIME_SOURCE_SPLIT=1" in command
    assert "sglang" not in command.lower()
    assert "swe_bench_pro_eval" not in command.lower()
    assert "swebench_pro_full" not in command.lower()


def test_rendered_command_and_wrapped_bodies_are_bash_syntax_valid() -> None:
    site = _site()
    url = "https://github.com/openai/putpocket-dataset-mining.git"
    command = _render()
    result = subprocess.run(["bash", "-n"], input=command + "\n", text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    bundle_key = load_lock()["build"]["bundle_key"]
    for body in (
        _build_wrapper(site, url, PROJECT_COMMIT, BUILD_SOURCE_COMMIT, bundle_key, True),
        _run_wrapper(site, url, PROJECT_COMMIT, BUILD_SOURCE_COMMIT, bundle_key, True),
    ):
        result = subprocess.run(["bash", "-n"], input=body + "\n", text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr


def test_h200_wrapper_validates_and_creates_site_root_before_clone() -> None:
    site = _site(); bundle_key = load_lock()["build"]["bundle_key"]
    body = _run_wrapper(site, "https://github.com/openai/putpocket-dataset-mining.git", PROJECT_COMMIT, BUILD_SOURCE_COMMIT, bundle_key, True)
    assert body.index("E_H200_LOCAL_STORAGE_PARENT_UNWRITABLE") < body.index("mkdir -p /local-data/user-data/jslee202403/putpocket-glm52-vllm-diagnostic")
    assert body.index("E_H200_RUN_ROOT_UNWRITABLE") < body.index("git -C")
    assert "PUTPOCKET_H200_STORAGE_PARENT=/local-data/user-data" in body
    assert "PUTPOCKET_H200_WORK_ROOT=/local-data/user-data/jslee202403/putpocket-glm52-vllm-diagnostic" in body


def test_h200_renderer_preserves_real_slurm_env_for_runtime_forwarding() -> None:
    site = _site(); bundle_key = load_lock()["build"]["bundle_key"]
    body = _run_wrapper(site, "https://github.com/openai/putpocket-dataset-mining.git", PROJECT_COMMIT, BUILD_SOURCE_COMMIT, bundle_key, True)
    assert "exec env PUTPOCKET_CONTAINER_EXECUTABLE=" in body
    assert "env -i" not in body
    assert " SLURM_JOB_ID=" not in body
    assert " SLURM_JOB_NODELIST=" not in body
    assert " SLURM_JOB_NUM_NODES=" not in body
    assert " SLURM_JOB_NAME=" not in body
    assert "-n ${SLURM_JOB_NODELIST:-}" in body
    assert "-n ${SLURM_STEP_ID:-} || -n ${SLURM_JOB_NAME:-}" in body
    rejected = subprocess.run(["/bin/bash", "-c", body], env={}, capture_output=True, check=False)
    assert rejected.returncode == 20
    assert rejected.stderr == b"E_H200_SLURM_ALLOCATION_REQUIRED\n"


def test_login_control_shell_only_creates_shared_parents() -> None:
    site = _site()
    command = _render()
    expected = f"mkdir -p {site.slurm_log_root} {site.shared_build_root}"
    assert command.startswith(f"set -euo pipefail && {expected} && VALIDATOR_JOB_ID=$(sbatch ")
    assert " fetch --depth=1 " in command  # quoted inside allocation-only --wrap bodies
    assert command.index(expected) < command.index("VALIDATOR_JOB_ID=$(sbatch")


def test_renderer_requires_explicit_split_authorization_and_handles_zero_and_one() -> None:
    with pytest.raises(ConfigError, match="NOT_EXPLICITLY_AUTHORIZED"):
        _render(allow_runtime_source_split=False)
    unsplit = _render(runtime_source_commit=BUILD_SOURCE_COMMIT, allow_runtime_source_split=False)
    split = _render()
    assert "PUTPOCKET_ALLOW_RUNTIME_SOURCE_SPLIT=0" in unsplit
    assert "PUTPOCKET_ALLOW_RUNTIME_SOURCE_SPLIT=1" not in unsplit
    assert "PUTPOCKET_ALLOW_RUNTIME_SOURCE_SPLIT=1" in split
    assert "PUTPOCKET_ALLOW_RUNTIME_SOURCE_SPLIT=0" not in split


def test_cli_renders_login_safe_base64_wrapper(capsys: pytest.CaptureFixture[str]) -> None:
    result = diagnostic_cli_main(
        [
            "render-wrap",
            "--site", str(SITE_PATH),
            "--project-url", "https://github.com/openai/putpocket-dataset-mining.git",
            "--runtime-source-commit", PROJECT_COMMIT,
            "--build-source-commit", BUILD_SOURCE_COMMIT,
            "--allow-runtime-source-split",
            "--login-safe-base64",
            "--cpu-partition", "cpu-max24",
            "--cpu-account", "gsai-account",
            "--cpu-qos", "nogpu",
            "--cpu-cpus-per-task", "24",
            "--cpu-memory", "192G",
            "--cpu-wall-time", "06:00:00",
            "--cpu-local-scratch-root", "/local-data/user-data/jslee202403/putpocket-vllm-build-scratch",
            "--container-executable", "/usr/bin/docker",
        ]
    )
    assert result == 0
    wrapper = capsys.readouterr().out.strip()
    assert wrapper.startswith("printf '%s' '")
    assert wrapper.endswith("' | base64 --decode | /bin/bash")
    encoded = wrapper.split("'", 3)[3].split("'", 1)[0]
    assert base64.b64decode(encoded).decode("utf-8") == _render() + "\n"
    syntax = subprocess.run(["bash", "-n"], input=wrapper + "\n", text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr


def test_renderer_rejects_credential_url() -> None:
    with pytest.raises(ConfigError, match="CREDENTIAL"):
        render_two_stage_submission(
            site=_site(),
            project_url="https://name:secret@github.com/openai/repo.git",
            runtime_source_commit=PROJECT_COMMIT,
            build_source_commit=BUILD_SOURCE_COMMIT,
            allow_runtime_source_split=True,
            lock_path=LOCK_PATH,
        )


def test_scripts_gate_allocation_and_bundle_before_heavy_actions() -> None:
    build = (ROOT / "scripts/cluster/build_glm52_vllm_sm90.sh").read_text(encoding="utf-8")
    run = (ROOT / "scripts/cluster/run_glm52_vllm_diagnostic.sh").read_text(encoding="utf-8")
    assert build.index("CPU_SLURM_ALLOCATION_REQUIRED") < build.index("vllm-project/vllm.git") < build.index('"$CONTAINER" build')
    assert build.index("IMMUTABLE_BUILD_BUNDLE_MISSING_REBUILD_FORBIDDEN") < build.index("vllm-project/vllm.git")
    assert build.index("REUSED_BUILD_BUNDLE") < build.index('mv "$TARGET"')
    assert "TORCH_CUDA_ARCH_LIST" not in build or "torch_cuda_arch_list=9.0" in build
    assert "--build-arg VLLM_USE_PRECOMPILED" not in build
    assert build.count("--build-arg RUN_WHEEL_CHECK=false") == 1
    assert "--build-arg RUN_WHEEL_CHECK=true" not in build
    assert "wheel_artifact.json" in build and "wheel_release_policy" in build
    assert "PUTPOCKET_IMMUTABLE_BUNDLE_REUSE_ONLY" in build
    patch = (ROOT / load_lock()["vllm"]["patch_path"]).read_text(encoding="utf-8")
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    assert "unset VLLM_USE_PRECOMPILED VLLM_PRECOMPILED_WHEEL_LOCATION" in added
    assert "export VLLM_USE_PRECOMPILED=1" not in added
    assert run.index("ALLOCATION_INVENTORY_MISMATCH") < run.index("validate-build-bundle") < run.index("snapshot_download")
    assert run.index("COMPUTE_LOCAL_STORAGE_PARENT_UNWRITABLE") < run.index("snapshot_download")
    assert run.index('RUN_ROOT="$ARTIFACT_ROOT/') < run.index("snapshot_download")
    assert "PUTPOCKET_RUN_ARTIFACT_ROOT" in run
    assert "IMMUTABLE_BUNDLE_ROOT_MISMATCH" in build and "IMMUTABLE_BUNDLE_ROOT_MISMATCH" in run
    assert "/local-data/jslee202403" not in run
    assert run.index("RUNTIME_JIT_POLICY_MANIFEST_INVALID") < run.index("snapshot_download")
    assert "--jit-monitor-mode error" in run and "runtime_jit_manifest.json" in run
    assert "--tensor-parallel-size 4" in run and "--cpu-offload-gb 0" in run
    assert "--no-enable-prefix-caching" in run and "/reset_prefix_cache" in run
    assert "--disable-log-requests" not in run
    assert "run_swebench" not in run.lower() and "swebench_pro_full" not in run.lower()
    assert "'immutable_build_source_commit':source_provenance['immutable_build_source_commit']" in run
    assert "'runtime_source_commit':source_provenance['runtime_source_commit']" in run
    assert "'source_provenance':source_provenance" in run


def test_shell_gpu_request_preserves_literal_csv_quotes_in_docker_argv() -> None:
    run = (ROOT / "scripts/cluster/run_glm52_vllm_diagnostic.sh").read_text(encoding="utf-8")
    request_assignment = next(line for line in run.splitlines() if line.startswith("GPU_REQUEST="))
    snippet = f"""
set -euo pipefail
GPU_SELECTOR=0,1,2,3
{request_assignment}
set -- docker run --gpus "$GPU_REQUEST" image
printf '%s\\0' "$@"
"""
    result = subprocess.run(
        ["bash", "-c", snippet],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.split(b"\0") == [
        b"docker",
        b"run",
        b"--gpus",
        b'"device=0,1,2,3"',
        b"image",
        b"",
    ]
    assert run.count('--gpus "$GPU_REQUEST"') == 3
    assert '--gpus "device=$GPU_SELECTOR"' not in run


def test_shell_forwards_exact_outer_slurm_provenance_to_runtime_containers() -> None:
    run = (ROOT / "scripts/cluster/run_glm52_vllm_diagnostic.sh").read_text(encoding="utf-8")
    fail_source = next(line for line in run.splitlines() if line.startswith("fail()"))
    function_start = run.index("configure_slurm_container_env() {")
    function_end = run.index("\n}\n", function_start) + len("\n}\n")
    snippet = "\n".join(
        (
            "set -euo pipefail",
            fail_source,
            run[function_start:function_end],
            "configure_slurm_container_env",
            "printf '%s\\0' \"${SLURM_CONTAINER_ENV[@]}\"",
        )
    )
    outer = {
        "SLURM_JOB_ID": "758673",
        "SLURM_JOB_NODELIST": "n88",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_JOB_NAME": "pp-glm52-vllm-dsa",
    }
    result = subprocess.run(["/bin/bash", "-c", snippet], env=outer, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.split(b"\0") == [
        b"--env",
        b"SLURM_JOB_ID=758673",
        b"--env",
        b"SLURM_JOB_NODELIST=n88",
        b"--env",
        b"SLURM_JOB_NUM_NODES=1",
        b"--env",
        b"SLURM_JOB_NAME=pp-glm52-vllm-dsa",
        b"",
    ]
    runtime_runs = [
        line for line in run.splitlines()
        if '"$CONTAINER" run' in line and '"$RUNTIME_IMAGE_ID"' in line
    ]
    assert len(runtime_runs) == 5
    assert all('"${SLURM_CONTAINER_ENV[@]}"' in line or '"${container_env[@]}"' in line for line in runtime_runs)
    assert 'container_env=(\n  "${SLURM_CONTAINER_ENV[@]}"' in run


def test_shell_rejects_slurm_container_provenance_outside_allocation() -> None:
    run = (ROOT / "scripts/cluster/run_glm52_vllm_diagnostic.sh").read_text(encoding="utf-8")
    fail_source = next(line for line in run.splitlines() if line.startswith("fail()"))
    function_start = run.index("configure_slurm_container_env() {")
    function_end = run.index("\n}\n", function_start) + len("\n}\n")
    snippet = "\n".join(("set -euo pipefail", fail_source, run[function_start:function_end], "configure_slurm_container_env"))
    result = subprocess.run(["/bin/bash", "-c", snippet], env={}, capture_output=True, check=False)
    assert result.returncode == 20
    assert result.stderr == b"SLURM_ALLOCATION_REQUIRED\n"


def test_shell_split_boolean_authorizes_only_literal_one() -> None:
    build = (ROOT / "scripts/cluster/build_glm52_vllm_sm90.sh").read_text(encoding="utf-8")
    run = (ROOT / "scripts/cluster/run_glm52_vllm_diagnostic.sh").read_text(encoding="utf-8")
    for script in (build, run):
        assert "${ALLOW_RUNTIME_SOURCE_SPLIT:+--allow-runtime-source-split}" not in script
        assert "if [[ $ALLOW_RUNTIME_SOURCE_SPLIT == 1 ]]" in script
        assert "provenance_args+=(--allow-runtime-source-split)" in script
    assert "source_split_args=()" in run
    assert "\"${source_split_args[@]}\"" in run

    snippet = """
set -euo pipefail
provenance_args=(base)
source_split_args=()
if [[ $ALLOW_RUNTIME_SOURCE_SPLIT == 1 ]]; then
  provenance_args+=(--allow-runtime-source-split)
  source_split_args+=(--allow-runtime-source-split)
fi
printf 'provenance=%s split=%s\n' "${provenance_args[*]}" "${source_split_args[*]}"
"""
    for value, expected in (("0", "provenance=base split=\n"), ("1", "provenance=base --allow-runtime-source-split split=--allow-runtime-source-split\n")):
        result = subprocess.run(
            ["bash", "-c", snippet],
            text=True,
            capture_output=True,
            check=False,
            env={"ALLOW_RUNTIME_SOURCE_SPLIT": value},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == expected


def test_compiled_import_probe_has_pinned_native_steps() -> None:
    run = (ROOT / "scripts/cluster/run_glm52_vllm_diagnostic.sh").read_text(encoding="utf-8")
    required_steps = (
        "runtime_nvcc",
        "import_torch",
        "import_vllm",
        "validate_sm90_device_capability",
        "import_vllm_C_stable_libtorch",
        "import_vllm_moe_C_stable_libtorch",
        "validate_vllm_C_stable_native_symbols",
        "validate_vllm_moe_marlin_symbols",
        "import_vllm_flashmla_C",
        "import_vllm_flashmla_extension_C",
        "validate_flashmla_sparse_native_symbols",
        "validate_flashmla_sparse_support",
        "import_vendored_deep_gemm_C",
        "validate_vendored_deep_gemm_selection",
        "validate_vendored_deep_gemm_C_symbols",
        "validate_vendored_deep_gemm_symbols",
        "import_sparse_attn_indexer",
        "import_modelopt_nvfp4_w4a16",
        "import_glm_moe_dsa_model",
        "import_flashmla_sparse_backend",
        "import_native_dsa_capture",
    )
    for step in required_steps:
        assert f"PROBE_STEP_START={step}" in run or f'"{step}"' in run


def test_compiled_import_probe_uses_pinned_cuda_extension_abi() -> None:
    run = (ROOT / "scripts/cluster/run_glm52_vllm_diagnostic.sh").read_text(encoding="utf-8")
    assert 'importlib.import_module("vllm._C")' not in run
    for module in (
        "vllm._C_stable_libtorch",
        "vllm._moe_C_stable_libtorch",
        "vllm._flashmla_C",
        "vllm._flashmla_extension_C",
        "vllm.third_party.deep_gemm._C",
    ):
        assert f'importlib.import_module("{module}")' in run
    for symbol in (
        "top_k_per_row_prefill",
        "top_k_per_row_decode",
        "cooperative_topk",
        "persistent_topk",
        "gptq_marlin_repack",
        "marlin_gemm",
        "moe_wna16_marlin_gemm",
        "moe_sum",
        "sparse_prefill_fwd",
        "sparse_decode_fwd",
        "fp8_fp4_mqa_logits",
        "fp8_fp4_paged_mqa_logits",
        "get_paged_mqa_logits_metadata",
    ):
        assert f'"{symbol}"' in run
    probe_source = run.split('python3 - <<"PY"\n', 1)[1].split("\nPY' >>", 1)[0]
    assert "'" not in probe_source  # Probe is carried inside the outer bash -lc single quotes.
    compile(probe_source, "compiled_import_probe.py", "exec")


def test_compiled_import_probe_replays_failure_log_before_model_access() -> None:
    run = (ROOT / "scripts/cluster/run_glm52_vllm_diagnostic.sh").read_text(encoding="utf-8")
    assert 'COMPILED_IMPORT_PROBE_LOG="$RUN_ROOT/phase1/compiled_import_probe.log"' in run
    assert 'cat "$COMPILED_IMPORT_PROBE_LOG" >&2' in run
    assert "COMPILED_SM90_IMPORT_PROBE_LOG_BEGIN" in run
    assert "COMPILED_SM90_IMPORT_PROBE_LOG_END" in run
    assert run.index("validate-build-bundle") < run.index("COMPILED_IMPORT_PROBE_LOG=") < run.index("snapshot_download")
    assert run.index("COMPILED_SM90_IMPORT_PROBE_LOG_BEGIN") < run.index("fail COMPILED_SM90_IMPORT_PROBE_FAILED 31")
    assert 'tuple(capability) != (9, 0)' in run
    assert 'grep -Fq \'release 13.0\' "$COMPILED_IMPORT_PROBE_LOG"' in run


def test_weightless_probe_replays_exact_failure_and_blocked_artifacts_to_shared_slurm_stderr(tmp_path: Path) -> None:
    run = (ROOT / "scripts/cluster/run_glm52_vllm_diagnostic.sh").read_text(encoding="utf-8")
    phase1 = (ROOT / "src/putpocket_dataset_mining/glm52_vllm_diagnostic_cli.py").read_text(encoding="utf-8")
    command = _render()
    assert 'WEIGHTLESS_COMPATIBILITY_LOG="$RUN_ROOT/phase1/weightless_probe.log"' in run
    assert 'replay_file_to_stderr WEIGHTLESS_VLLM_COMPATIBILITY_LOG "$WEIGHTLESS_COMPATIBILITY_LOG" "$WEIGHTLESS_COMPATIBILITY_RC"' in run
    assert run.index("WEIGHTLESS_COMPATIBILITY_RC=$?") < run.index("STATUS=BLOCKED") < run.index("fail WEIGHTLESS_VLLM_COMPATIBILITY_FAILED 32")
    assert 'replay_file_to_stderr DIAGNOSTIC_FAILURE_ARTIFACT "$RUN_ROOT/diagnostic_manifest.json" "$rc"' in run
    assert 'replay_file_to_stderr BLOCKED_ARTIFACT "$blocked_artifact" "$rc"' in run
    assert "printf '%s_BEGIN path=%s exit_code=%s\\n'" in run
    assert run.index("replay_file_to_stderr WEIGHTLESS_VLLM_COMPATIBILITY_LOG") < run.index("fail WEIGHTLESS_VLLM_COMPATIBILITY_FAILED 32")
    for step in (
        "model_revision_resolution",
        "weightless_metadata_download",
        "model_config_validation",
        "vllm_capability_validation",
    ):
        assert f'start("{step}")' in phase1
    assert "PHASE1_STEP_FAILED=" in phase1 and "exception_type=" in phase1 and "detail=" in phase1
    probe_log = tmp_path / "weightless_probe.log"
    probe_log.write_text("exact phase failure detail\n", encoding="utf-8")
    helper_start = run.index("replay_file_to_stderr() {")
    helper_end = run.index("\n}\n", helper_start) + len("\n}\n")
    replay = subprocess.run(
        ["bash", "-s", "--", str(probe_log)],
        input=run[helper_start:helper_end] + '\nreplay_file_to_stderr WEIGHTLESS_VLLM_COMPATIBILITY_LOG "$1" 32\n',
        text=True,
        capture_output=True,
        check=False,
    )
    assert replay.returncode == 0
    assert replay.stderr == (
        f"WEIGHTLESS_VLLM_COMPATIBILITY_LOG_BEGIN path={probe_log} exit_code=32\n"
        "exact phase failure detail\n"
        "WEIGHTLESS_VLLM_COMPATIBILITY_LOG_END\n"
    )
    slurm_stdout = f"--output={_site().slurm_log_root}/%x-%j.out"
    slurm_stderr = f"--error={_site().slurm_log_root}/%x-%j.err"
    assert command.count(slurm_stdout) == 2
    assert command.count(slurm_stderr) == 2


def test_weightless_probe_reports_the_exact_failing_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_NUM_NODES", "SLURM_STEP_ID", "SLURM_JOB_NAME"):
        monkeypatch.delenv(name, raising=False)
    result = diagnostic_cli_main(
        [
            "phase1",
            "--lock", str(LOCK_PATH),
            "--metadata-root", str(tmp_path / "metadata"),
            "--artifact-root", str(tmp_path / "artifacts"),
        ]
    )
    assert result == 2
    stderr = capsys.readouterr().err
    assert "PHASE1_STEP_START=slurm_allocation" in stderr
    assert "PHASE1_STEP_FAILED=slurm_allocation exception_type=ConfigError" in stderr
    assert "E_SLURM_ALLOCATION_REQUIRED" in stderr


def test_inventory_requires_four_full_non_mig_h200s() -> None:
    header = "index,uuid,name,memory_total_mib,memory_free_mib,mig_mode,compute_capability\n"
    rows = "".join(f"{i},GPU-{i},NVIDIA H200,143771,140000,Disabled,9.0\n" for i in range(4))
    assert validate_inventory_csv(header + rows, rows)["gpu_count"] == 4
    with pytest.raises(ConfigError, match="MIG_ENABLED"):
        validate_inventory_csv((header + rows).replace("Disabled", "Enabled", 1), rows)


def test_model_config_normalizes_both_official_layout_representations() -> None:
    base = {"architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa", "num_hidden_layers": 78, "index_topk": 2048, "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}
    types = ["full" if layer in FULL_LAYERS else "shared" for layer in range(78)]
    config = {**base, "indexer_types": types}
    assert validate_model_config(config, load_lock())["normalized_from"] == "indexer_types"
    config = {**base, "index_topk_pattern": "".join("F" if layer in FULL_LAYERS else "S" for layer in range(78))}
    assert validate_model_config(config, load_lock())["normalized_from"] == "index_topk_pattern"


def _response(ids: list[int]) -> dict:
    return {"choices": [{"text": "A coherent deterministic answer", "token_ids": ids, "finish_reason": "stop"}]}


def test_trace_requires_exact_token_id_equivalence() -> None:
    report = validate_trace_equivalence(_response([1, 2]), _response([1, 2]), off_ns=10, on_ns=12)
    assert report["instrumentation_overhead_ns"] == 2
    with pytest.raises(ConfigError, match="TOKEN_ID_MISMATCH"):
        validate_trace_equivalence(_response([1]), _response([2]), off_ns=1, on_ns=1)


def test_trace_rejects_obvious_repetition_garbage() -> None:
    response = {"choices": [{"text": "repeat " * 20, "token_ids": list(range(20)), "finish_reason": "length"}]}
    with pytest.raises(ConfigError, match="REPETITION_GARBAGE"):
        validate_trace_equivalence(response, response, off_ns=1, on_ns=1)


def _prefill_records() -> list[dict]:
    raw = [float(2048 - index) for index in range(2048)]
    ids = list(range(2048)); scores = list(raw)
    lock = load_lock()
    common = {"schema_version": 1, "run_id": "test", "instance_id": INSTANCE_ID, "trace_mode": "ON", "phase": "prefill", "sample_point": "prefill_last_query", "query_position": 2047, "decode_step": None, "context_length": 2048, "native_logits_backend": "fp8_fp4_mqa_logits", "native_topk_backend": "top_k_per_row_prefill", "dtype": "torch.float32", "device": "cuda", "shape": [2048], "topk": 2048, "source_token_coordinate_semantics": "zero_based_logical_causal_source_position", "full_indexer_layers": list(FULL_LAYERS), "shared_layer_mapping": lock["model_layout"]["shared_layer_mapping"], "shared_layer_mapping_sha256": lock["model_layout"]["shared_layer_mapping_sha256"], "revisions": {"model": lock["runtime"]["model_revision"], "vllm": lock["vllm"]["commit"], "project": PROJECT_COMMIT, "patch_sha256": lock["vllm"]["patch_sha256"], "build_image": lock["build"]["base_image"], "bundle_key": lock["build"]["bundle_key"], "runtime_image_id": "sha256:" + "2" * 64}, "raw_scores": raw, "selected_ids": ids, "selected_scores": scores}
    records = [{**common, "rank": rank, "layer": layer} for rank in range(4) for layer in FULL_LAYERS]
    for record in records:
        encoded = (json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n").encode()
        record["record_sha256"] = hashlib.sha256(encoded).hexdigest()
    return records


def test_capture_coverage_and_raw_topk_consistency() -> None:
    assert validate_capture_records(_prefill_records(), output_token_count=1, lock=load_lock())["record_count"] == 84


def test_capture_gap_and_raw_nonexposure_are_fail_closed() -> None:
    records = _prefill_records()
    with pytest.raises(ConfigError, match="COVERAGE_INCOMPLETE"):
        validate_capture_records(records[:-1], output_token_count=1, lock=load_lock())
    blocked = {"schema_version": 1, "status": "BLOCKED", "failure_class": "NATIVE_RAW_SCORE_EXPOSURE_UNAVAILABLE"}
    assert blocked["status"] == "BLOCKED" and "RAW_SCORE" in blocked["failure_class"]


def test_capture_record_digest_and_mapping_are_fail_closed() -> None:
    records = _prefill_records()
    records[0]["record_sha256"] = "0" * 64
    with pytest.raises(ConfigError, match="RECORD_DIGEST"):
        validate_capture_records(records, output_token_count=1, lock=load_lock())
    records = _prefill_records()
    records[0]["shared_layer_mapping"] = {}
    payload = dict(records[0]); payload.pop("record_sha256")
    records[0]["record_sha256"] = hashlib.sha256((json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode()).hexdigest()
    with pytest.raises(ConfigError, match="SHARED_LAYER_MAPPING"):
        validate_capture_records(records, output_token_count=1, lock=load_lock())


def test_tracked_shell_scripts_are_syntax_valid() -> None:
    for relative in ("scripts/cluster/build_glm52_vllm_sm90.sh", "scripts/cluster/run_glm52_vllm_diagnostic.sh"):
        result = subprocess.run(["bash", "-n", str(ROOT / relative)], text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr


def test_runtime_jit_manifest_is_run_local_native_and_checksummed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build, bundle, lock = _bundle(tmp_path, monkeypatch)
    build["_bundle_root"] = str(bundle)
    cache = tmp_path / "run" / "cache" / "deep_gemm"; cache.mkdir(parents=True)
    (cache / "native-sm90.cubin").write_bytes(b"sm90")
    audit = tmp_path / "run" / "compiler_audit.jsonl"
    audit.write_text(json.dumps({"schema_version": 1, "timestamp_utc": "2026-08-19T12:00:01+00:00", "pid": 10, "tool": "nvcc", "real_executable": "/usr/local/cuda/bin/nvcc", "argv": ["nvcc", "/run/cache/deep_gemm/kernel.cu", "-arch=sm_90"]}) + "\n", encoding="utf-8")
    report = build_runtime_jit_manifest(
        tmp_path / "run" / "cache", audit,
        started_utc="2026-08-19T12:00:00+00:00", completed_utc="2026-08-19T12:00:02+00:00",
        build_source_commit=BUILD_SOURCE_COMMIT,
        runtime_source_commit=PROJECT_COMMIT,
        observed_runtime_source_commit=PROJECT_COMMIT,
        wrapper_source_commit=PROJECT_COMMIT,
        allow_runtime_source_split=True,
        runtime_image_id="sha256:" + "2" * 64,
        build_manifest=build, lock=lock,
    )
    assert report["components"] == ["deep_gemm_native_dsa"]
    assert report["files"][0]["sha256"] == hashlib.sha256(b"sm90").hexdigest()
    assert report["build_source_commit"] == BUILD_SOURCE_COMMIT
    assert report["runtime_source_commit"] == PROJECT_COMMIT
    assert report["source_provenance"]["source_split_explicitly_authorized"] is True


def test_runtime_jit_manifest_rejects_general_project_compilation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build, bundle, lock = _bundle(tmp_path, monkeypatch); build["_bundle_root"] = str(bundle)
    cache = tmp_path / "cache" / "deep_gemm"; cache.mkdir(parents=True); (cache / "x").write_bytes(b"x")
    audit = tmp_path / "audit.jsonl"
    audit.write_text(json.dumps({"schema_version": 1, "timestamp_utc": "2026-08-19T12:00:01+00:00", "tool": "nvcc", "argv": ["nvcc", "/project/src/setup.py", "/cache/deep_gemm/x.cu"]}) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="GENERAL_PROJECT_COMPILATION"):
        build_runtime_jit_manifest(tmp_path / "cache", audit, started_utc="2026-08-19T12:00:00+00:00", completed_utc="2026-08-19T12:00:02+00:00", build_source_commit=BUILD_SOURCE_COMMIT, runtime_source_commit=PROJECT_COMMIT, observed_runtime_source_commit=PROJECT_COMMIT, wrapper_source_commit=PROJECT_COMMIT, allow_runtime_source_split=True, runtime_image_id="sha256:" + "2" * 64, build_manifest=build, lock=lock)
