from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.glm52_attention_indexer import (
    analyze_capture,
    analyze_query_sum_capture,
    canonical_json_bytes,
    compare_aligned_scores,
    compare_query_sums,
    js_divergence,
)
from putpocket_dataset_mining.glm52_runpod import (
    DOCTOR_SCHEMA,
    PACKAGE_LOCK,
    REPORT_SCHEMA,
    SCHEDULE,
    capture_probe,
    load_package_lock,
    load_successful_doctor,
    validate_project_artifacts,
    validate_schedule,
    validate_schema,
)
from putpocket_dataset_mining.glm52_runpod_cli import main as runpod_cli_main


ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = ROOT / "patches/vllm/4a3447d200e5aa428d68d1a00aa00f1a19a1a729"
INSTANCE_ID = (
    "instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-"
    "vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _doctor_report(*, status: str = "passed") -> dict[str, object]:
    checks = [
        {"check_id": f"synthetic-{index:02d}", "status": "passed", "evidence": {"ok": True}}
        for index in range(10)
    ]
    payload = {
        "schema_version": 1,
        "test_id": "glm52-runpod-doctor-v1",
        "status": status,
        "read_only": True,
        "project_commit": "1" * 40,
        "vllm_commit": "4a3447d200e5aa428d68d1a00aa00f1a19a1a729",
        "model_revision": "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa",
        "checks": checks,
    }
    return {
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical_json_bytes(payload, newline=False)).hexdigest(),
    }


def _capture_record(**values: object) -> dict[str, object]:
    record = {
        "schema_version": 1,
        "diagnostic_id": "synthetic-diagnostic",
        "instance_id": INSTANCE_ID,
        "scenario_id": "glm52-sys-policy-equal-replacement-v1",
        "probe_kind": "ordinary_target_prefill_q1_boundary_no_q2",
        "q2_in_probe": False,
        **values,
    }
    record["record_sha256"] = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
    return record


def test_package_lock_pins_order_apply_arguments_provenance_and_artifacts() -> None:
    lock = load_package_lock()
    chain = lock["vllm"]["patch_chain"]
    assert [item["role"] for item in chain] == [
        "required_legacy_packaging_base",
        "true_partial_overlay",
        "score_diagnostic_overlay",
    ]
    assert [item["apply_tool"] for item in chain] == ["patch", "git apply", "git apply"]
    assert [item["apply_args"] for item in chain] == [
        ["-p1", "--forward", "--batch"], ["--unidiff-zero"], ["--unidiff-zero"],
    ]
    assert lock["benchmark_provenance"]["dataset"] == "ScaleAI/SWE-bench_Pro"
    assert lock["benchmark_provenance"]["dataset_revision"] == "7ab5114912baf22bb098818e604c02fe7ad2c11f"
    assert lock["benchmark_provenance"]["instance_id"] == INSTANCE_ID
    assert lock["benchmark_provenance"]["mini_swe_submodule_commit"] == "d74716a3c8104a113f77cc9ab94cf407ecdcf1e9"
    assert lock["benchmark_provenance"]["mini_swe_scaffold"].startswith("mini-swe-agent/")
    assert lock["environment"]["python_distributions"]["torch"] == "2.13.0+cu129"
    assert lock["capture"]["scenario_id"] == "glm52-sys-policy-equal-replacement-v1"
    assert lock["probe_boundary"]["q2_present"] is False
    assert lock["analysis"]["dissimilarity_failure"] is False
    assert validate_project_artifacts(ROOT, lock)["status"] == "passed"


def test_bootstrap_is_fail_fast_and_uses_exact_patch_modes() -> None:
    script = (ROOT / "scripts/runpod/bootstrap_glm52_attention_indexer.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in script
    assert 'patch -d "$VLLM_ROOT" -p1 --dry-run --forward --batch' in script
    assert 'patch -d "$VLLM_ROOT" -p1 --forward --batch' in script
    for name in ("putpocket_true_partial_prefill.patch", "glm52_attention_indexer_score_diagnostic.patch"):
        assert f'git -C "$VLLM_ROOT" apply --check --unidiff-zero "$PATCH_ROOT/{name}"' in script
        assert f'git -C "$VLLM_ROOT" apply --unidiff-zero "$PATCH_ROOT/{name}"' in script
    assert "|| true" not in script
    for path in PATCH_ROOT.glob("*.patch"):
        assert not any(line.endswith(" ") or line.endswith("\t") for line in path.read_text(encoding="utf-8").splitlines())
    wrapper = (ROOT / "scripts/runpod/run_glm52_attention_indexer_score_test.sh").read_text(encoding="utf-8")
    assert "prepare-final-probe" in wrapper
    assert "capture-query-sum" in wrapper and "analyze-query-sum" in wrapper
    assert "capture-matrix" in wrapper and "--max-level 6" in wrapper


def test_score_overlay_captures_full_reference_and_native_pre_topk_only() -> None:
    patch = (PATCH_ROOT / "glm52_attention_indexer_score_diagnostic.patch").read_text(encoding="utf-8")
    hook = (ROOT / "instrumentation/vllm/glm52_attention_indexer_scores.py").read_text(encoding="utf-8")
    assert "maybe_capture_main_attention_reference" in patch
    assert "maybe_capture_indexer_native_logits" in patch
    assert "Diagnostic capture is deliberately before top_k_per_row_prefill" in patch
    assert 'score_origin": "reference_recomputed_from_exact_post_rope_q_and_model_projected_k"' in hook
    assert 'score_origin": "kernel_native_fp8_fp4_mqa_logits_before_top_k_per_row_prefill"' in hook
    assert "candidate_positions" in hook and "candidate_token_ids" in hook
    assert 'QUERY_SUM_MODE = "query_range_attention_indexer_comparison"' in hook
    assert '"strict_causal_candidate_position_lt_query_position"' in hook
    assert lock_query_sum_boundary()["older_sampled_mode_is_final"] is False


def lock_query_sum_boundary() -> dict[str, object]:
    return load_package_lock()["query_sum_capture"]


def test_raw_rank_and_normalized_distribution_metrics_are_distinct() -> None:
    main = [[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]]
    result = compare_aligned_scores(main, [0.0, 3.0, 6.0], k_values=[1, 2, 8])
    assert result["pearson_raw"] == pytest.approx(1.0)
    assert result["spearman_raw"] == pytest.approx(1.0)
    assert result["cosine_zscore"] == pytest.approx(1.0)
    assert result["vectors"]["main_raw_mean"] != result["vectors"]["main_probability"]
    assert result["vectors"]["indexer_raw"] != result["vectors"]["indexer_probability"]
    assert [item["k"] for item in result["topk"]] == [1, 2, 3]
    assert js_divergence(result["vectors"]["main_probability"], result["vectors"]["main_probability"]) == pytest.approx(0.0)
    reversed_result = compare_aligned_scores(main, [6.0, 3.0, 0.0], k_values=[1])
    assert reversed_result["spearman_raw"] == pytest.approx(-1.0)
    with pytest.raises(ConfigError, match="ALIGNED_SCORE_LENGTH_MISMATCH"):
        compare_aligned_scores([[1.0, 2.0], [1.0]], [1.0, 2.0], k_values=[1])


def test_query_sum_metrics_preserve_signed_raw_sums() -> None:
    result = compare_query_sums([-3.0, 1.0, 5.0], [-6.0, 2.0, 10.0], k_values=[1, 2])
    assert result["pearson_raw"] == pytest.approx(1.0)
    assert result["spearman_raw"] == pytest.approx(1.0)
    assert result["vectors"]["main_raw_query_sum"][0] == -3.0
    assert result["vectors"]["indexer_raw_query_sum"][0] == -6.0
    assert sum(result["vectors"]["main_probability"]) == pytest.approx(1.0)


def test_all_q1_q2_rows_are_aligned_and_summed_per_layer(tmp_path: Path) -> None:
    lock = copy.deepcopy(load_package_lock())
    lock["query_sum_capture"]["layers"] = [0]
    lock["analysis"]["topk_values"] = [1, 2]
    prompt_ids = [10, 11, 12, 13, 14, 15]
    probe = {
        "prompt": {"token_ids": prompt_ids, "token_ids_sha256": "b" * 64},
        "segments": {"q1_ranges": [[2, 4]], "q2_ranges": [[5, 6]]},
        "propagation_window": [0, 6],
        "claim_boundary": {"real_a1_generated_or_executed": False},
    }
    capture = tmp_path / "capture"
    capture.mkdir()
    for rank in range(4):
        records = []
        for query in (2, 3, 5):
            positions = list(range(query))
            common = {
                "probe_kind": "benchmark_derived_two_query_q1_q2_score_probe",
                "q2_in_probe": True,
                "rank": rank,
                "tensor_parallel_size": 4,
                "layer": 0,
                "query_position": query,
                "query_token_id": prompt_ids[query],
                "candidate_positions": positions,
                "candidate_token_ids": prompt_ids[:query],
            }
            main = [[float(query + position) for position in positions] for _ in range(16)]
            indexer = [float(2 * (query + position)) for position in positions]
            records.extend([
                _capture_record(**common, record_kind="main_attention_reference", score_origin="reference", raw_logits_by_local_head=main),
                _capture_record(**common, record_kind="indexer_native_pre_topk", score_origin="native", raw_logits=indexer),
            ])
        (capture / f"capture-rank-{rank:02d}.jsonl").write_bytes(
            b"".join(canonical_json_bytes(record) for record in records)
        )
    report = analyze_query_sum_capture(
        capture, tmp_path / "report", lock, probe,
        doctor_payload_sha256="a" * 64,
    )
    schema = ROOT / lock["query_sum_capture"]["report_schema"]
    validate_schema(report, schema)
    payload = report["payload"]
    assert payload["query_token_count"] == 3
    assert payload["layers"][0]["pearson_raw"] == pytest.approx(1.0)
    rows = [json.loads(line) for line in (tmp_path / "report/query-summed-token-scores.jsonl").read_text().splitlines()]
    assert rows[0]["receiving_query_row_count"] == 3
    assert next(row for row in rows if row["candidate_position"] == 2)["seed_membership"] == "q1"
    per_query = json.loads((tmp_path / "report/per-query-comparisons.json").read_text())
    assert next(row for row in per_query if row["query_position"] == 5)["segment_membership"] == "q2"


def test_synthetic_tp_capture_analyzes_and_validates_schema(tmp_path: Path) -> None:
    lock = copy.deepcopy(load_package_lock())
    lock["capture"]["layers"] = [0]
    lock["capture"]["query_positions"] = [2]
    lock["analysis"]["topk_values"] = [1, 2]
    capture = tmp_path / "capture"
    capture.mkdir()
    positions = [0, 1, 2]
    token_ids = [10, 11, 12]
    for rank in range(4):
        local_heads = [
            [float(index + head / 1000) for index in positions]
            for head in range(16)
        ]
        common = {
            "rank": rank,
            "tensor_parallel_size": 4,
            "layer": 0,
            "query_position": 2,
            "query_token_id": 12,
            "candidate_positions": positions,
            "candidate_token_ids": token_ids,
            "dtype": "float32",
        }
        records = [
            _capture_record(
                **common,
                record_kind="main_attention_reference",
                score_origin="reference_recomputed",
                raw_logits_by_local_head=local_heads,
            ),
            _capture_record(
                **common,
                record_kind="indexer_native_pre_topk",
                score_origin="kernel_native_pre_topk",
                raw_logits=[0.0, 1.0, 2.0],
            ),
        ]
        path = capture / f"capture-rank-{rank:02d}.jsonl"
        path.write_bytes(b"".join(canonical_json_bytes(record) for record in records))
    output = tmp_path / "report"
    report = analyze_capture(capture, output, lock, doctor_payload_sha256="a" * 64)
    validate_schema(report, REPORT_SCHEMA)
    payload = report["payload"]
    assert payload["comparison_count"] == 1
    assert payload["comparisons"][0]["main_head_count"] == 64
    assert payload["comparisons"][0]["valid_token_count"] == 3
    assert payload["token_level_artifact"]["record_count"] == 3
    assert payload["scientific_result_policy"] == "report_only_no_similarity_threshold_failure"


def test_synthetic_capture_rejects_token_alignment_error(tmp_path: Path) -> None:
    lock = copy.deepcopy(load_package_lock())
    lock["capture"]["layers"] = [0]
    lock["capture"]["query_positions"] = [1]
    capture = tmp_path / "capture"
    capture.mkdir()
    for rank in range(4):
        main = _capture_record(
            record_kind="main_attention_reference", rank=rank, layer=0, query_position=1,
            candidate_positions=[0, 1], candidate_token_ids=[10, 11], query_token_id=11,
            score_origin="reference", raw_logits_by_local_head=[[0.0, 1.0]] * 16,
        )
        indexer = _capture_record(
            record_kind="indexer_native_pre_topk", rank=rank, layer=0, query_position=1,
            candidate_positions=[0, 1], candidate_token_ids=[10, 99], query_token_id=11,
            score_origin="native", raw_logits=[0.0, 1.0],
        )
        (capture / f"capture-rank-{rank:02d}.jsonl").write_bytes(
            canonical_json_bytes(main) + canonical_json_bytes(indexer)
        )
    with pytest.raises(ConfigError, match="CAPTURE_TOKEN_ALIGNMENT_MISMATCH"):
        analyze_capture(capture, tmp_path / "report", lock, doctor_payload_sha256="a" * 64)


def test_doctor_digest_and_capture_dry_run_dependency(tmp_path: Path) -> None:
    doctor_path = tmp_path / "doctor.json"
    report = _doctor_report()
    _write_json(doctor_path, report)
    validate_schema(report, DOCTOR_SCHEMA)
    _, digest = load_successful_doctor(doctor_path)
    probe = {
        "schema_version": 1,
        "probe_id": "synthetic",
        "probe_kind": "ordinary_target_prefill_q1_boundary_no_q2",
        "instance_id": INSTANCE_ID,
        "scenario_id": "glm52-sys-policy-equal-replacement-v1",
        "prompt_token_ids": [1] * 2071,
        "prompt_token_count": 2071,
        "prompt_token_ids_sha256": "b" * 64,
    }
    probe_path = tmp_path / "probe.json"
    _write_json(probe_path, probe)
    plan = capture_probe(
        doctor_report=doctor_path,
        probe_path=probe_path,
        model_root=tmp_path / "model",
        output_root=tmp_path / "capture",
        dry_run=True,
    )
    assert plan["status"] == "dry_run"
    assert plan["doctor_payload_sha256"] == digest
    assert plan["engine"]["sparse_mla_force_mqa"] is True
    changed = copy.deepcopy(report)
    changed["payload"]["project_commit"] = "2" * 40
    _write_json(doctor_path, changed)
    with pytest.raises(ConfigError, match="DOCTOR_PAYLOAD_DIGEST_MISMATCH"):
        load_successful_doctor(doctor_path)


def test_schedule_schema_order_and_cli_proofs(capsys: pytest.CaptureFixture[str]) -> None:
    schedule = validate_schedule(SCHEDULE)
    assert schedule["tests"][1]["depends_on"] == ["glm52-runpod-doctor-v1"]
    assert schedule["tests"][1]["dependency_evidence"] == "doctor.payload_sha256"
    assert runpod_cli_main(["validate-schedule", "--schedule", str(SCHEDULE)]) == 0
    cli_schedule = json.loads(capsys.readouterr().out)
    assert cli_schedule["schedule_id"] == "glm52-indexer-distillation-runpod-v1"
    assert runpod_cli_main([
        "validate-package", "--lock", str(PACKAGE_LOCK),
        "--project-root", str(ROOT), "--phase", "project_artifacts",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
