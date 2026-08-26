"""Safe loader for project-authored stateful edit scenario templates.

The catalog contains authoring templates, never frozen episodes or server
manifests.  A future experiment driver must resolve and freeze a template
before sending any request to the model server.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .errors import ConfigError


DEFAULT_CATALOG_PATH = (
    REPO_ROOT / "configs" / "cluster" / "stateful_edit_scenarios" / "catalog.json"
)


class ScenarioCatalogError(ConfigError):
    """Raised when catalog, schema, provenance, or cross-file state is invalid."""


@dataclass(frozen=True)
class ScenarioSummary:
    scenario_id: str
    title: str
    status: str
    artifact_kind: str
    operation_kind: str
    recommendation_order: int | None
    tags: tuple[str, ...]
    directly_runnable: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tags"] = list(self.tags)
        return value


@dataclass(frozen=True)
class ScenarioCatalog:
    catalog_path: Path
    catalog_id: str
    documents: Mapping[str, Mapping[str, Any]]

    def list(
        self,
        *,
        status: str | None = None,
        operation_kind: str | None = None,
        tag: str | None = None,
    ) -> tuple[ScenarioSummary, ...]:
        summaries = tuple(_summary(self.documents[key]) for key in sorted(self.documents))
        return tuple(
            item
            for item in summaries
            if (status is None or item.status == status)
            and (operation_kind is None or item.operation_kind == operation_kind)
            and (tag is None or tag in item.tags)
        )

    def load(self, scenario_id: str) -> dict[str, Any]:
        try:
            document = self.documents[scenario_id]
        except KeyError as exc:
            raise ScenarioCatalogError(f"UNKNOWN_SCENARIO_ID: {scenario_id}") from exc
        return json.loads(json.dumps(document))

    def filter(
        self,
        *,
        status: str | None = None,
        operation_kind: str | None = None,
        tag: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        return tuple(
            self.load(item.scenario_id)
            for item in self.list(status=status, operation_kind=operation_kind, tag=tag)
        )


def _load_json_object(path: Path, reason: str) -> dict[str, Any]:
    if not path.is_file():
        raise ScenarioCatalogError(f"{reason}_MISSING: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioCatalogError(f"{reason}_INVALID_JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScenarioCatalogError(f"{reason}_ROOT_NOT_OBJECT: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_file(root: Path, raw: Any, reason: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ScenarioCatalogError(f"{reason}_PATH_INVALID: {raw!r}")
    root = root.resolve()
    resolved = (root / raw).resolve()
    if resolved == root or root not in resolved.parents:
        raise ScenarioCatalogError(f"{reason}_PATH_ESCAPES_ROOT: {raw}")
    if not resolved.is_file():
        raise ScenarioCatalogError(f"{reason}_FILE_MISSING: {resolved}")
    return resolved


def _relative_directory(root: Path, raw: Any, reason: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ScenarioCatalogError(f"{reason}_PATH_INVALID: {raw!r}")
    root = root.resolve()
    resolved = (root / raw).resolve()
    if resolved == root or root not in resolved.parents:
        raise ScenarioCatalogError(f"{reason}_PATH_ESCAPES_ROOT: {raw}")
    if not resolved.is_dir():
        raise ScenarioCatalogError(f"{reason}_DIRECTORY_MISSING: {resolved}")
    return resolved


def _repository_file(raw: Any, reason: str) -> Path:
    return _relative_file(REPO_ROOT, raw, reason)


def _validate_document(
    validator: Draft202012Validator, document: Mapping[str, Any], path: Path
) -> None:
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise ScenarioCatalogError(
            f"SCHEMA_VALIDATION_FAILED: {path}: {location}: {first.message}"
        )


def _summary(document: Mapping[str, Any]) -> ScenarioSummary:
    return ScenarioSummary(
        scenario_id=str(document["scenario_id"]),
        title=str(document["title"]),
        status=str(document["status"]),
        artifact_kind=str(document["artifact_kind"]),
        operation_kind=str(document["edit_template"]["operation_kind"]),
        recommendation_order=document["recommendation_order"],
        tags=tuple(document["tags"]),
        directly_runnable=bool(document["readiness"]["directly_runnable"]),
    )


def _validate_scenario_cross_fields(document: Mapping[str, Any], path: Path) -> None:
    scenario_id = document["scenario_id"]
    provenance = document["provenance"]
    edit = document["edit_template"]
    readiness = document["readiness"]
    server = document["server_constraints"]
    claims = document["claim_boundary"]

    if document["artifact_kind"] != "authoring_template" or readiness["directly_runnable"]:
        raise ScenarioCatalogError(f"AUTHORING_TEMPLATE_MISLABELED: {scenario_id}")
    if provenance["authorship"]["benchmark_native_edit_trajectory"]:
        raise ScenarioCatalogError(f"BENCHMARK_TRAJECTORY_AUTHORSHIP_INVALID: {scenario_id}")
    if provenance["authorship"]["stateful_episode_transformation"] != "project_authored":
        raise ScenarioCatalogError(f"STATEFUL_TRANSFORMATION_AUTHORSHIP_INVALID: {scenario_id}")
    if provenance["leakage_policy"]["outcome_independent"] is not True:
        raise ScenarioCatalogError(f"OUTCOME_INDEPENDENCE_REQUIRED: {scenario_id}")
    if edit["message_path"] != "sys" or server["main_authoring_contract"] != "sys_only":
        raise ScenarioCatalogError(f"SYS_ONLY_CONTRACT_DIVERGED: {scenario_id}")
    if server["target_request_requires_p_gt_h"] is not True:
        raise ScenarioCatalogError(f"Q2_CONTINUATION_REQUIRED: {scenario_id}")
    if not server["reject_zero_recompute"] or not server["reject_full_history_recompute"]:
        raise ScenarioCatalogError(f"SPARSE_RECOMPUTE_BOUNDS_MISSING: {scenario_id}")
    if server["q2_selector_disposition"] != "excluded_and_computed_normally_after_patch":
        raise ScenarioCatalogError(f"Q2_DISPOSITION_INVALID: {scenario_id}")
    if server["ordinary_full_prefill_control_required"] is not True:
        raise ScenarioCatalogError(f"FULL_PREFILL_CONTROL_REQUIRED: {scenario_id}")
    if document["status"] == "blocked" and not readiness["requires_client_schema_extension"]:
        raise ScenarioCatalogError(f"BLOCKED_SCENARIO_WITHOUT_SCHEMA_BLOCKER: {scenario_id}")
    if document["status"] == "backend_diagnostic":
        if claims["scientific_quality_claim_allowed"] or claims["rope_correct_claim_allowed"]:
            raise ScenarioCatalogError(f"DIAGNOSTIC_CLAIM_BOUNDARY_INVALID: {scenario_id}")
    if edit["operation_kind"] in {"insertion", "deletion"}:
        if server["shifted_reuse_policy"] != "preserve_donor_bytes_stale_rope_evidence_false":
            raise ScenarioCatalogError(f"SHIFTED_ROPE_POLICY_INVALID: {scenario_id}")
        if not readiness["requires_client_schema_extension"]:
            raise ScenarioCatalogError(f"LENGTH_CHANGE_SCHEMA_BLOCKER_MISSING: {scenario_id}")
    if edit["text_resolution"]["equal_token_count_proven"]:
        evidence = edit["tokenization_evidence"]
        if evidence is None or len(evidence["old_token_ids"]) != len(evidence["new_token_ids"]):
            raise ScenarioCatalogError(f"EQUAL_TOKEN_EVIDENCE_INVALID: {scenario_id}")
    for key, raw in document["freeze_contract"]["schema_paths"].items():
        _repository_file(raw, f"FREEZE_SCHEMA_{scenario_id}_{key}")
    for index, raw in enumerate(provenance["evidence_paths"]):
        _repository_file(raw, f"PROVENANCE_EVIDENCE_{scenario_id}_{index}")


def validate_catalog(path: str | Path = DEFAULT_CATALOG_PATH) -> ScenarioCatalog:
    catalog_path = Path(path).resolve()
    catalog = _load_json_object(catalog_path, "CATALOG")
    schema_record = catalog.get("scenario_schema")
    if not isinstance(schema_record, dict):
        raise ScenarioCatalogError("CATALOG_SCHEMA_RECORD_INVALID")
    schema_path = _repository_file(schema_record.get("path"), "CATALOG_SCHEMA")
    if _sha256(schema_path) != schema_record.get("sha256"):
        raise ScenarioCatalogError("CATALOG_SCHEMA_DIGEST_MISMATCH")
    schema = _load_json_object(schema_path, "CATALOG_SCHEMA")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ScenarioCatalogError(f"CATALOG_SCHEMA_INVALID: {exc}") from exc
    validator = Draft202012Validator(schema)
    _validate_document(validator, catalog, catalog_path)

    entries = catalog["entries"]
    ids = [entry["scenario_id"] for entry in entries]
    paths = [entry["path"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ScenarioCatalogError("CATALOG_DUPLICATE_SCENARIO_ID")
    if len(paths) != len(set(paths)):
        raise ScenarioCatalogError("CATALOG_DUPLICATE_SCENARIO_PATH")
    if ids != sorted(ids):
        raise ScenarioCatalogError("CATALOG_ORDER_NOT_SCENARIO_ID_ASCENDING")

    documents: dict[str, Mapping[str, Any]] = {}
    resolved_paths: set[Path] = set()
    for entry in entries:
        scenario_path = _relative_file(catalog_path.parent, entry["path"], "SCENARIO")
        resolved_paths.add(scenario_path)
        if _sha256(scenario_path) != entry["sha256"]:
            raise ScenarioCatalogError(f"SCENARIO_DIGEST_MISMATCH: {entry['scenario_id']}")
        document = _load_json_object(scenario_path, "SCENARIO")
        _validate_document(validator, document, scenario_path)
        for key in ("scenario_id", "title", "status", "artifact_kind"):
            if document[key] != entry[key]:
                raise ScenarioCatalogError(
                    f"CATALOG_SCENARIO_FIELD_MISMATCH: {entry['scenario_id']}: {key}"
                )
        if document["provenance"]["benchmark"] != catalog["benchmark_provenance"]:
            raise ScenarioCatalogError(
                f"CATALOG_BENCHMARK_PROVENANCE_MISMATCH: {entry['scenario_id']}"
            )
        if document["scenario_id"] in documents:
            raise ScenarioCatalogError(f"SCENARIO_DUPLICATE_ID: {document['scenario_id']}")
        _validate_scenario_cross_fields(document, scenario_path)
        documents[document["scenario_id"]] = document

    scenario_root = _relative_directory(
        catalog_path.parent, catalog["scenario_directory"], "SCENARIO_DIRECTORY"
    )
    discovered = {item.resolve() for item in scenario_root.glob("*.json")}
    if discovered != resolved_paths:
        extras = sorted(str(item) for item in discovered - resolved_paths)
        missing = sorted(str(item) for item in resolved_paths - discovered)
        raise ScenarioCatalogError(
            f"CATALOG_SCENARIO_FILE_SET_MISMATCH: extras={extras} missing={missing}"
        )
    return ScenarioCatalog(
        catalog_path=catalog_path,
        catalog_id=str(catalog["catalog_id"]),
        documents=documents,
    )


def list_scenarios(
    *,
    status: str | None = None,
    operation_kind: str | None = None,
    tag: str | None = None,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
) -> tuple[ScenarioSummary, ...]:
    return validate_catalog(catalog_path).list(
        status=status, operation_kind=operation_kind, tag=tag
    )


def load_scenario(
    scenario_id: str, *, catalog_path: str | Path = DEFAULT_CATALOG_PATH
) -> dict[str, Any]:
    return validate_catalog(catalog_path).load(scenario_id)


def filter_scenarios(
    *,
    status: str | None = None,
    operation_kind: str | None = None,
    tag: str | None = None,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
) -> tuple[dict[str, Any], ...]:
    return validate_catalog(catalog_path).filter(
        status=status, operation_kind=operation_kind, tag=tag
    )


def validate_scenarios(
    scenario_ids: Iterable[str] | None = None,
    *,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
) -> tuple[ScenarioSummary, ...]:
    catalog = validate_catalog(catalog_path)
    if scenario_ids is None:
        return catalog.list()
    requested = tuple(scenario_ids)
    if len(requested) != len(set(requested)):
        raise ScenarioCatalogError("VALIDATION_REQUEST_DUPLICATE_SCENARIO_ID")
    return tuple(_summary(catalog.load(scenario_id)) for scenario_id in requested)
