from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from putpocket_dataset_mining.stateful_edit_scenarios import (
    DEFAULT_CATALOG_PATH,
    ScenarioCatalogError,
    filter_scenarios,
    list_scenarios,
    load_scenario,
    validate_catalog,
    validate_scenarios,
)
from putpocket_dataset_mining.stateful_edit_scenarios_cli import main


ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = DEFAULT_CATALOG_PATH.parent
INSTANCE_ID = (
    "instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-"
    "vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
)
EXPECTED_IDS = [
    "glm52-cache-block-boundary-replacement-v1",
    "glm52-delete-shifted-rope-diagnostic-v1",
    "glm52-insert-shifted-rope-diagnostic-v1",
    "glm52-multi-disjoint-equal-replacement-v1",
    "glm52-sys-interior-equal-replacement-smoke-v1",
    "glm52-sys-policy-equal-replacement-v1",
]


def _copy_catalog(tmp_path: Path) -> Path:
    destination = tmp_path / "catalog"
    shutil.copytree(CATALOG_ROOT, destination)
    return destination / "catalog.json"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_catalog_validates_and_lists_deterministically() -> None:
    catalog = validate_catalog()
    summaries = catalog.list()
    assert catalog.catalog_id == "glm52-stateful-edit-next-experiments-v1"
    assert [item.scenario_id for item in summaries] == EXPECTED_IDS
    assert all(item.artifact_kind == "authoring_template" for item in summaries)
    assert all(item.directly_runnable is False for item in summaries)
    assert [item.scenario_id for item in validate_scenarios()] == EXPECTED_IDS


def test_exact_benchmark_and_project_authorship_provenance_is_on_every_scenario() -> None:
    for document in filter_scenarios():
        benchmark = document["provenance"]["benchmark"]
        instance = document["provenance"]["instance"]
        authorship = document["provenance"]["authorship"]
        assert benchmark == {
            "family": "swe_bench_pro",
            "dataset_id": "ScaleAI/SWE-bench_Pro",
            "dataset_config": None,
            "dataset_config_status": "not_explicitly_named_by_repository_loader",
            "dataset_revision": "7ab5114912baf22bb098818e604c02fe7ad2c11f",
            "dataset_revision_status": "pinned",
            "split": "test",
            "dataset_source_url": "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro",
            "selection_id": "swebench_pro_smoke_one",
            "selection_rule": "instance_id_ascending_limit_1",
            "harness_repository": "https://github.com/scaleapi/SWE-bench_Pro-os.git",
            "harness_commit": "ca10a60a5fcae51e6948ffe1485d4153d421e6c5",
            "official_evaluator": "swe_bench_pro_eval.py",
        }
        assert instance["instance_id"] == INSTANCE_ID
        assert instance["repository_identifier"] == "ansible/ansible"
        assert instance["repository_source_url"] is None
        assert authorship["scenario_definition"] == "project_authored"
        assert authorship["stateful_episode_transformation"] == "project_authored"
        assert authorship["benchmark_native_edit_trajectory"] is False
        assert document["provenance"]["native_benchmark_components"] == [
            "problem_statement_and_row_metadata",
            "repository_base_state_and_dataset_container_mapping",
            "official_swe_bench_pro_evaluator",
        ]


def test_statuses_operations_and_first_gpu_smoke_are_actionable() -> None:
    smoke = load_scenario("glm52-sys-interior-equal-replacement-smoke-v1")
    policy = load_scenario("glm52-sys-policy-equal-replacement-v1")
    boundary = load_scenario("glm52-cache-block-boundary-replacement-v1")
    multiple = load_scenario("glm52-multi-disjoint-equal-replacement-v1")
    insertion = load_scenario("glm52-insert-shifted-rope-diagnostic-v1")
    deletion = load_scenario("glm52-delete-shifted-rope-diagnostic-v1")

    assert smoke["status"] == policy["status"] == "recommended"
    assert smoke["recommendation_order"] == 0
    assert smoke["edit_template"]["tokenization_evidence"]["edit_positions"] == [114]
    assert smoke["selection_contract"]["budget_mode"] == "mandatory_only"
    assert boundary["status"] == "requires_freeze"
    assert boundary["selection_contract"]["budget_mode"] == "cache_block_boundary_probe"
    assert multiple["status"] == "blocked"
    assert multiple["server_constraints"]["client_schema_compatibility"] == "blocked_requires_extension"
    assert insertion["status"] == deletion["status"] == "backend_diagnostic"
    assert insertion["edit_template"]["operation_kind"] == "insertion"
    assert deletion["edit_template"]["operation_kind"] == "deletion"
    assert deletion["selection_contract"]["minimum_selected_count"] == 1
    for diagnostic in (insertion, deletion):
        assert diagnostic["server_constraints"]["shifted_reuse_policy"] == (
            "preserve_donor_bytes_stale_rope_evidence_false"
        )
        assert diagnostic["claim_boundary"]["rope_correct_claim_allowed"] is False
        assert diagnostic["claim_boundary"]["scientific_quality_claim_allowed"] is False


def test_loader_filter_and_copy_isolation() -> None:
    diagnostics = list_scenarios(status="backend_diagnostic")
    assert [item.operation_kind for item in diagnostics] == ["deletion", "insertion"]
    replacements = filter_scenarios(operation_kind="replacement")
    assert [item["scenario_id"] for item in replacements] == [
        "glm52-cache-block-boundary-replacement-v1",
        "glm52-sys-interior-equal-replacement-smoke-v1",
        "glm52-sys-policy-equal-replacement-v1",
    ]
    smoke = load_scenario("glm52-sys-interior-equal-replacement-smoke-v1")
    smoke["status"] = "mutated-local-copy"
    assert load_scenario("glm52-sys-interior-equal-replacement-smoke-v1")["status"] == "recommended"
    with pytest.raises(ScenarioCatalogError, match="UNKNOWN_SCENARIO_ID"):
        load_scenario("does-not-exist")


def test_duplicate_ids_digest_drift_and_unindexed_files_fail_closed(tmp_path: Path) -> None:
    catalog_path = _copy_catalog(tmp_path / "duplicate")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["entries"].append(dict(catalog["entries"][0]))
    _write_json(catalog_path, catalog)
    with pytest.raises(ScenarioCatalogError, match="DUPLICATE_SCENARIO_ID"):
        validate_catalog(catalog_path)

    catalog_path = _copy_catalog(tmp_path / "digest")
    scenario_path = catalog_path.parent / "scenarios" / "glm52-sys-policy-equal-replacement-v1.json"
    scenario_path.write_text(scenario_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ScenarioCatalogError, match="SCENARIO_DIGEST_MISMATCH"):
        validate_catalog(catalog_path)

    catalog_path = _copy_catalog(tmp_path / "unindexed")
    source = catalog_path.parent / "scenarios" / "glm52-sys-policy-equal-replacement-v1.json"
    shutil.copyfile(source, source.with_name("unindexed.json"))
    with pytest.raises(ScenarioCatalogError, match="SCENARIO_FILE_SET_MISMATCH"):
        validate_catalog(catalog_path)


def test_cross_file_metadata_and_claim_boundary_fail_closed(tmp_path: Path) -> None:
    catalog_path = _copy_catalog(tmp_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in catalog["entries"]
        if item["scenario_id"] == "glm52-insert-shifted-rope-diagnostic-v1"
    )
    scenario_path = catalog_path.parent / entry["path"]
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenario["claim_boundary"]["rope_correct_claim_allowed"] = True
    _write_json(scenario_path, scenario)
    entry["sha256"] = _sha256(scenario_path)
    _write_json(catalog_path, catalog)
    with pytest.raises(ScenarioCatalogError, match="DIAGNOSTIC_CLAIM_BOUNDARY_INVALID"):
        validate_catalog(catalog_path)


def test_cli_list_show_and_validate(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["status"] == "passed"
    assert validated["scenario_count"] == 6

    assert main(["list", "--status", "recommended"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert [item["scenario_id"] for item in listing["scenarios"]] == [
        "glm52-sys-interior-equal-replacement-smoke-v1",
        "glm52-sys-policy-equal-replacement-v1",
    ]

    assert main(["show", "glm52-delete-shifted-rope-diagnostic-v1"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["artifact_kind"] == "authoring_template"
    assert shown["readiness"]["directly_runnable"] is False


def test_frozen_artifact_schemas_remain_separate_from_catalog_templates() -> None:
    catalog = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    assert catalog["artifact_kind"] == "authoring_template_catalog"
    assert catalog["frozen_artifact_schemas"] == {
        "frozen_episode": "configs/cluster/schemas/stateful_mid_trajectory_edit_scenario.schema.json",
        "server_manifest": "configs/cluster/schemas/vllm_true_partial_prefill_server_manifest.schema.json",
    }
    for raw in catalog["frozen_artifact_schemas"].values():
        assert (ROOT / raw).is_file()
