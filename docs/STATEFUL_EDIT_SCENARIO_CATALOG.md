# GLM-5.2 stateful edit scenario catalog

The project-internal catalog at
`configs/cluster/stateful_edit_scenarios/catalog.json` is the machine-loadable
input vocabulary for the next experiment driver. Every entry is an
`authoring_template`, not a frozen episode and not a vLLM server manifest. The
driver must resolve and freeze a template before model execution.

## Provenance boundary

The existing GLM episode uses exactly:

- dataset: `ScaleAI/SWE-bench_Pro`;
- dataset config: none explicitly named by the repository loader;
- revision: `7ab5114912baf22bb098818e604c02fe7ad2c11f`;
- split: `test`;
- selection: `swebench_pro_smoke_one`, `instance_id` ascending, limit 1;
- instance:
  `instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5`;
- repository identifier: `ansible/ansible`;
- harness: `https://github.com/scaleapi/SWE-bench_Pro-os.git` at
  `ca10a60a5fcae51e6948ffe1485d4153d421e6c5`;
- evaluator: `swe_bench_pro_eval.py`.

The source lock provides the dataset URL, revision, official harness, container
mapping, and evaluator. No repository source URL for `ansible/ansible` is
recorded in the committed evidence, so the catalog retains the identifier and
uses `null` for that URL rather than guessing it.

SWE-bench Pro supplies the problem row, repository/container mapping, and
official evaluator. It does **not** supply a native mid-trajectory edit episode.
PutPocket authors `SYS_old`/`SYS_new`, generates A1 once, executes its tool
action, freezes Q2 and repository state, replays A1/Q2 across budgets, and
constructs the token alignment, selector, donor manifest, and target manifest.
Every catalog entry records this authorship and the no-outcome-leakage policy.

## Catalog entries

| Scenario ID | Status | Intended use |
|---|---|---|
| `glm52-sys-interior-equal-replacement-smoke-v1` | `recommended` | First GPU smoke: evidenced token-114 replacement, mandatory-only sparse patch, then normal Q2. |
| `glm52-sys-policy-equal-replacement-v1` | `recommended` | First scientific experiment: the evidenced equal-position policy edit with new importance-for-recompute budgets. |
| `glm52-cache-block-boundary-replacement-v1` | `requires_freeze` | Same edit with selected rows spanning cache pages and Q2 allocating or crossing a block boundary. |
| `glm52-multi-disjoint-equal-replacement-v1` | `blocked` | Server operation list supports it; the current single-edit client schema does not preserve multiple spans. |
| `glm52-insert-shifted-rope-diagnostic-v1` | `backend_diagnostic` | Explicit insertion alignment and stale shifted-RoPE evidence; no quality or RoPE-correct claim. |
| `glm52-delete-shifted-rope-diagnostic-v1` | `backend_diagnostic` | Explicit deletion/compaction with nonzero surviving recompute rows and stale shifted-RoPE evidence. |

No entry is directly runnable. Even the evidenced equal-token pair must recreate
and attest the complete frozen A1/Q2 history, `H`, alignment, mandatory closure,
selector, target request with `P > H`, and live-donor server manifests.

## Python API

```python
from putpocket_dataset_mining.stateful_edit_scenarios import (
    filter_scenarios,
    list_scenarios,
    load_scenario,
    validate_catalog,
)

catalog = validate_catalog()
summaries = list_scenarios(status="recommended")
smoke = load_scenario("glm52-sys-interior-equal-replacement-smoke-v1")
diagnostics = filter_scenarios(status="backend_diagnostic")
```

Results are ordered by `scenario_id`. Loading validates the composite Draft
2020-12 schema, schema and scenario SHA-256 values, index/file completeness,
duplicate IDs and paths, entry/document metadata, shared benchmark provenance,
referenced repository files, authorship, readiness, and scenario claim
boundaries. Returned documents are independent copies.

## CLI proof

```bash
putpocket-stateful-edit-scenarios validate
putpocket-stateful-edit-scenarios list --status recommended
putpocket-stateful-edit-scenarios list --operation-kind insertion
putpocket-stateful-edit-scenarios show glm52-sys-interior-equal-replacement-smoke-v1
```

The equivalent source-tree form is
`python -m putpocket_dataset_mining.stateful_edit_scenarios_cli ...`.
Only JSON is loaded; the API does not construct YAML or Python objects from
catalog input.

## Freeze and server boundary

The catalog schema is
`configs/cluster/schemas/stateful_edit_scenario_catalog.schema.json`. A resolved
episode remains a distinct artifact validated by
`stateful_mid_trajectory_edit_scenario.schema.json`; donor and target requests
remain distinct server artifacts validated by
`vllm_true_partial_prefill_server_manifest.schema.json`.

Every template machine-encodes these server constraints:

- the main scientific authoring contract edits SYS only;
- A1 and Q2 replay exactly from fresh state;
- Q2 is excluded from frozen-history selection and runs normally after patching;
- target request length satisfies `P > H`;
- edit targets plus dependency/implementation closure are mandatory;
- eligible selector rows are aligned downstream history before `H`;
- zero and full-history recompute sets are rejected;
- ordinary full target prefill is a required control, never a silent fallback;
- shifted insertion/deletion reuse preserves donor bytes and is explicitly not
  RoPE-correct in the current backend slice.

The catalog does not implement tokenizer resolution, episode freezing,
importance capture, manifest generation, HTTP `vllm_xargs`, server launch, GPU
evidence collection, agent continuation, or official evaluation. Those remain
the next experiment-script plumbing.
