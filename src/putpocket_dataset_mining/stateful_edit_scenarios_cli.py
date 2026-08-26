"""Inspect and validate project-authored stateful edit scenario templates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import ConfigError
from .stateful_edit_scenarios import (
    DEFAULT_CATALOG_PATH,
    list_scenarios,
    load_scenario,
    validate_catalog,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="putpocket-stateful-edit-scenarios")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="List templates in deterministic ID order")
    listing.add_argument("--status")
    listing.add_argument("--operation-kind")
    listing.add_argument("--tag")

    show = commands.add_parser("show", help="Show one validated authoring template")
    show.add_argument("scenario_id")

    commands.add_parser("validate", help="Validate schema, digests, and cross-file contracts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "list":
            scenarios = list_scenarios(
                status=args.status,
                operation_kind=args.operation_kind,
                tag=args.tag,
                catalog_path=args.catalog,
            )
            payload = {
                "schema_version": 1,
                "count": len(scenarios),
                "scenarios": [item.to_dict() for item in scenarios],
            }
        elif args.command == "show":
            payload = load_scenario(args.scenario_id, catalog_path=args.catalog)
        else:
            catalog = validate_catalog(args.catalog)
            payload = {
                "schema_version": 1,
                "status": "passed",
                "catalog_id": catalog.catalog_id,
                "scenario_count": len(catalog.documents),
                "scenario_ids": sorted(catalog.documents),
            }
    except (ConfigError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"schema_version": 1, "status": "failed", "error": str(exc)},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
