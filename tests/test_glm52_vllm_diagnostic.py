from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.glm52_vllm_diagnostic import (
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
    validate_trace_equivalence,
)
from putpocket_dataset_mining.glm52_vllm_diagnostic_slurm import (
    _build_wrapper,
    _run_wrapper,
    load_site,
    render_two_stage_submission,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs/cluster/glm52_vllm_diagnostic.lock.json"
SITE_PATH = ROOT / "configs/cluster/sites/herdr_vllm_diagnostic.json"
PROJECT_COMMIT = "1" * 40


def _site():
    return load_site(
        SITE_PATH,
        cpu_partition="cpu-measured",
        cpu_account="account-measured",
        cpu_qos="qos-measured",
        cpu_cpus_per_task=64,
        cpu_memory="256G",
        cpu_wall_time="04:00:00",
        cpu_local_scratch_root="/cpu-local/measured",
        container_executable="/usr/bin/docker",
    )


def test_lock_is_exact_vllm_diagnostic() -> None:
    report = validate_lock(load_lock())
    assert report["vllm_commit"] == "4a3447d200e5aa428d68d1a00aa00f1a19a1a729"
    assert report["instance_id"] == INSTANCE_ID
    assert len(report["full_layers"]) == 21


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


def _bundle(tmp_path: Path) -> tuple[dict, Path]:
    root = tmp_path / "bundle"; root.mkdir()
    paths = {"runtime_image_tar": "runtime-image.tar", "vllm_wheel": "wheels/vllm-test.whl", "source_bundle": "vllm-source-bundle.tar.gz"}
    for name in paths.values():
        target = root / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(name.encode())
    provenance_names = (
        "source_preflight.json", "source_post_patch.json", "build-wheel-image.log",
        "compiled_arches.txt", "build_environment.json", "build_nvcc.txt",
        "build-runtime-image.log", "runtime_environment.json", "runtime_nvcc.txt",
    )
    provenance_paths = {name: f"logs/{name}" for name in provenance_names}
    for name in provenance_paths.values():
        target = root / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(name.encode())
    (root / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
    lock = load_lock(); build = lock["build"]
    manifest = {
        "schema_version": 1, "status": "SUCCESS", "vllm_commit": lock["vllm"]["commit"],
        "bundle_key": build["bundle_key"], "patch_sha256": lock["vllm"]["patch_sha256"],
        "patch_target_post_sha256": lock["vllm"]["patch_target_post_sha256"],
        "build_patch_target_post_sha256": lock["vllm"]["build_patch_target_post_sha256"],
        "instrumentation_sha256": lock["vllm"]["instrumentation_sha256"], "base_image": build["base_image"],
        "compiler_audit_sha256": lock["vllm"]["compiler_audit_sha256"],
        "python": "3.12", "torch": "2.13.0", "cuda": "13.0.3", "torch_cuda_arch_list": "9.0",
        "cmake_cuda_architectures": "90", "vllm_target_device": "cuda", "vllm_use_precompiled": False,
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
    return manifest, root


def test_build_bundle_validates_all_immutable_targets(tmp_path: Path) -> None:
    manifest, root = _bundle(tmp_path)
    assert validate_build_manifest(manifest, root, load_lock())["status"] == "passed"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [("torch_cuda_arch_list", "", "TARGET_MISMATCH"), ("vllm_use_precompiled", True, "TARGET_MISMATCH"), ("prebuilt_vllm_wheel_used", True, "PREBUILT")],
)
def test_build_bundle_rejects_target_or_prebuilt_substitution(tmp_path: Path, field: str, value: object, error: str) -> None:
    manifest, root = _bundle(tmp_path); manifest[field] = value
    with pytest.raises(ConfigError, match=error):
        validate_build_manifest(manifest, root, load_lock())


def test_build_bundle_rejects_changed_file_before_model(tmp_path: Path) -> None:
    manifest, root = _bundle(tmp_path); (root / "runtime-image.tar").write_bytes(b"changed")
    with pytest.raises(ConfigError, match="DIGEST_MISMATCH"):
        validate_build_manifest(manifest, root, load_lock())


def test_site_refuses_unmeasured_cpu_values() -> None:
    with pytest.raises(ConfigError, match="CPU_BUILD_SITE_FIELDS_UNSET"):
        load_site(SITE_PATH)
    with pytest.raises(ConfigError, match="OFFICIAL_DOCKER_RUNTIME_REQUIRED"):
        load_site(
            SITE_PATH, cpu_partition="cpu", cpu_account="account", cpu_qos="qos",
            cpu_cpus_per_task=8, cpu_memory="32G", cpu_wall_time="01:00:00",
            cpu_local_scratch_root="/scratch", container_executable="/usr/bin/apptainer",
        )


def test_renderer_is_two_stage_exact_and_has_afterok_dependency() -> None:
    command = render_two_stage_submission(site=_site(), project_url="https://github.com/openai/putpocket-dataset-mining.git", project_commit=PROJECT_COMMIT, lock_path=LOCK_PATH)
    assert command.count("sbatch --parsable") == 2
    assert "--dependency=afterok:$BUILD_JOB_ID" in command
    assert "--gres=gpu:H200:4" in command
    assert "--partition=cpu-measured" in command
    assert "--partition=H200" in command
    assert "BUILD_JOB_ID=%s\\nRUN_JOB_ID=%s" in command
    assert "sglang" not in command.lower()
    assert "swe_bench_pro_eval" not in command.lower()
    assert "swebench_pro_full" not in command.lower()


def test_rendered_command_and_wrapped_bodies_are_bash_syntax_valid() -> None:
    site = _site()
    url = "https://github.com/openai/putpocket-dataset-mining.git"
    command = render_two_stage_submission(site=site, project_url=url, project_commit=PROJECT_COMMIT, lock_path=LOCK_PATH)
    result = subprocess.run(["bash", "-n"], input=command + "\n", text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    bundle_key = load_lock()["build"]["bundle_key"]
    for body in (_build_wrapper(site, url, PROJECT_COMMIT, bundle_key), _run_wrapper(site, url, PROJECT_COMMIT, bundle_key)):
        result = subprocess.run(["bash", "-n"], input=body + "\n", text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr


def test_renderer_rejects_credential_url() -> None:
    with pytest.raises(ConfigError, match="CREDENTIAL"):
        render_two_stage_submission(site=_site(), project_url="https://name:secret@github.com/openai/repo.git", project_commit=PROJECT_COMMIT, lock_path=LOCK_PATH)


def test_scripts_gate_allocation_and_bundle_before_heavy_actions() -> None:
    build = (ROOT / "scripts/cluster/build_glm52_vllm_sm90.sh").read_text(encoding="utf-8")
    run = (ROOT / "scripts/cluster/run_glm52_vllm_diagnostic.sh").read_text(encoding="utf-8")
    assert build.index("CPU_SLURM_ALLOCATION_REQUIRED") < build.index("vllm-project/vllm.git") < build.index('"$CONTAINER" build')
    assert "TORCH_CUDA_ARCH_LIST" not in build or "torch_cuda_arch_list=9.0" in build
    assert "--build-arg VLLM_USE_PRECOMPILED" not in build
    patch = (ROOT / load_lock()["vllm"]["patch_path"]).read_text(encoding="utf-8")
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    assert "unset VLLM_USE_PRECOMPILED VLLM_PRECOMPILED_WHEEL_LOCATION" in added
    assert "export VLLM_USE_PRECOMPILED=1" not in added
    assert run.index("ALLOCATION_INVENTORY_MISMATCH") < run.index("validate-build-bundle") < run.index("snapshot_download")
    assert run.index("RUNTIME_JIT_POLICY_MANIFEST_INVALID") < run.index("snapshot_download")
    assert "--jit-monitor-mode error" in run and "runtime_jit_manifest.json" in run
    assert "--tensor-parallel-size 4" in run and "--cpu-offload-gb 0" in run
    assert "--no-enable-prefix-caching" in run and "/reset_prefix_cache" in run
    assert "--disable-log-requests" not in run
    assert "run_swebench" not in run.lower() and "swebench_pro_full" not in run.lower()


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


def test_runtime_jit_manifest_is_run_local_native_and_checksummed(tmp_path: Path) -> None:
    build, bundle = _bundle(tmp_path)
    build["_bundle_root"] = str(bundle)
    cache = tmp_path / "run" / "cache" / "deep_gemm"; cache.mkdir(parents=True)
    (cache / "native-sm90.cubin").write_bytes(b"sm90")
    audit = tmp_path / "run" / "compiler_audit.jsonl"
    audit.write_text(json.dumps({"schema_version": 1, "timestamp_utc": "2026-08-19T12:00:01+00:00", "pid": 10, "tool": "nvcc", "real_executable": "/usr/local/cuda/bin/nvcc", "argv": ["nvcc", "/run/cache/deep_gemm/kernel.cu", "-arch=sm_90"]}) + "\n", encoding="utf-8")
    report = build_runtime_jit_manifest(
        tmp_path / "run" / "cache", audit,
        started_utc="2026-08-19T12:00:00+00:00", completed_utc="2026-08-19T12:00:02+00:00",
        project_commit=PROJECT_COMMIT, runtime_image_id="sha256:" + "2" * 64,
        build_manifest=build, lock=load_lock(),
    )
    assert report["components"] == ["deep_gemm_native_dsa"]
    assert report["files"][0]["sha256"] == hashlib.sha256(b"sm90").hexdigest()


def test_runtime_jit_manifest_rejects_general_project_compilation(tmp_path: Path) -> None:
    build, bundle = _bundle(tmp_path); build["_bundle_root"] = str(bundle)
    cache = tmp_path / "cache" / "deep_gemm"; cache.mkdir(parents=True); (cache / "x").write_bytes(b"x")
    audit = tmp_path / "audit.jsonl"
    audit.write_text(json.dumps({"schema_version": 1, "timestamp_utc": "2026-08-19T12:00:01+00:00", "tool": "nvcc", "argv": ["nvcc", "/project/src/setup.py", "/cache/deep_gemm/x.cu"]}) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="GENERAL_PROJECT_COMPILATION"):
        build_runtime_jit_manifest(tmp_path / "cache", audit, started_utc="2026-08-19T12:00:00+00:00", completed_utc="2026-08-19T12:00:02+00:00", project_commit=PROJECT_COMMIT, runtime_image_id="sha256:" + "2" * 64, build_manifest=build, lock=load_lock())
