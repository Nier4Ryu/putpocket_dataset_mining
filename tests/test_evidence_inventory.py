from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.evidence_inventory import (
    build_inventory,
    load_spec,
    verify_inventory,
    write_inventory,
)


def _spec() -> dict[str, object]:
    return {
        "schema_version": 1,
        "inventory_id": "completion-audit-v1",
        "source_remote_root_id": "runpod-task-root-id-not-an-address",
        "groups": [
            {
                "group_id": "accepted-raw",
                "content_class": "accepted_gpu_evidence",
                "local_path": "accepted",
                "source_remote_relative": "artifacts/test2/accepted",
            },
            {
                "group_id": "plots",
                "content_class": "montblanc_derived_plot",
                "local_path": "plots/summary.json",
                "source_remote_relative": "montblanc-derived/plots/summary.json",
            },
        ],
        "exclusions": [
            {
                "exclusion_id": "model-weights",
                "reason": "Reproducible infrastructure; deliberately not transferred.",
            }
        ],
    }


def _root(tmp_path: Path) -> Path:
    (tmp_path / "accepted").mkdir()
    (tmp_path / "accepted/a.json").write_text('{"a":1}\n', encoding="utf-8")
    (tmp_path / "accepted/b.bin").write_bytes(b"raw\x00scores")
    (tmp_path / "plots").mkdir()
    (tmp_path / "plots/summary.json").write_text("{}\n", encoding="utf-8")
    return tmp_path


def test_inventory_build_write_verify_is_deterministic_and_remote_relative(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    document = build_inventory(root, _spec())
    repeated = build_inventory(root, _spec())
    assert document == repeated
    payload = document["payload"]
    assert payload["entry_count"] == 3
    assert [entry["local_relative_path"] for entry in payload["entries"]] == [
        "accepted/a.json",
        "accepted/b.bin",
        "plots/summary.json",
    ]
    assert payload["entries"][0]["source_remote_relative_path"] == (
        "artifacts/test2/accepted/a.json"
    )
    assert payload["entries"][2]["source_remote_relative_path"] == (
        "montblanc-derived/plots/summary.json"
    )

    inventory = root / "inventory.json"
    checksums = root / "SHA256SUMS"
    write_inventory(
        root,
        document,
        output_path=inventory,
        checksum_path=checksums,
    )
    loaded = json.loads(inventory.read_text(encoding="utf-8"))
    report = verify_inventory(root, loaded)
    assert report == {
        "status": "passed",
        "entry_count": 3,
        "total_bytes": payload["total_bytes"],
        "payload_sha256": document["payload_sha256"],
    }
    assert checksums.read_text(encoding="utf-8").splitlines()[0].endswith(
        "  accepted/a.json"
    )


def test_inventory_fails_closed_on_tamper_duplicate_and_symlink(tmp_path: Path) -> None:
    root = _root(tmp_path)
    document = build_inventory(root, _spec())
    (root / "accepted/a.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="EVIDENCE_ENTRY_(SIZE|DIGEST)_MISMATCH"):
        verify_inventory(root, document)

    duplicate = _spec()
    duplicate["groups"] = [duplicate["groups"][0], duplicate["groups"][0]]  # type: ignore[index]
    with pytest.raises(ConfigError, match="EVIDENCE_GROUP_ID_INVALID_OR_DUPLICATE"):
        build_inventory(root, duplicate)

    clean = tmp_path / "symlink"
    clean.mkdir()
    (clean / "target").write_text("target\n", encoding="utf-8")
    (clean / "linked").symlink_to(clean / "target")
    symlink_spec = copy.deepcopy(_spec())
    symlink_spec["groups"] = [
        {
            "group_id": "linked",
            "content_class": "invalid",
            "local_path": "linked",
            "source_remote_relative": "evidence/linked",
        }
    ]
    with pytest.raises(ConfigError, match="MISSING_OR_SYMLINK"):
        build_inventory(clean, symlink_spec)


def test_spec_loader_rejects_missing_exclusion_contract(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    value = _spec()
    value["exclusions"] = []
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ConfigError, match="EVIDENCE_SPEC_EXCLUSIONS_INVALID"):
        load_spec(path)
