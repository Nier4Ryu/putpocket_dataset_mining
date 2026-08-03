# ClassEval Mining

ClassEval mining reuses the MBPP-stateful two-turn Cline runtime, but changes
the dataset adapter, initial workspace, hidden verifier materialization, and
Docker image.

## Dataset

- Hugging Face dataset: `FudanSELab/ClassEval`
- Split: `test`
- Adapter: `classeval_huggingface`

The adapter stores the original row fields in `source_task.json`. Agent-visible
content is limited to `import_statement` and `skeleton`. The reference
`solution_code`, `test`, and `methods_info[*].test_code` are hidden.

## Workspace Contract

The initial agent workspace contains:

```text
/workspace/solution.py = import_statement + blank line + skeleton
```

If `import_statement` is already present in the skeleton, it is not duplicated.
No tests are written to the agent workspace.

## Hidden Verifier

The verifier writes `tests/test_solution.py` only inside a fresh verifier
container workspace. It combines:

- `row["test"]`
- every non-empty `methods_info[*]["test_code"]`

Exact duplicate test blocks are deduplicated. The verifier prepends
`from solution import *` so ClassEval tests can instantiate the target class.

## Dependency Scan And Docker

`configs/dataset_mining/classeval_dependencies.lock.yaml` records imports from:

- `import_statement`
- `skeleton`
- `test`
- `methods_info[*].test_code`
- `methods_info[*].dependencies`

The ClassEval Docker image is separate from the MBPP image:

```text
docker/classeval_python/Dockerfile
putpocket-classeval-python:ubuntu22.04-py313-v1
```

Runtime dependency installation remains disabled.

## Query-1

The default Query-1 asks the agent to complete the class implementation in
`solution.py`, preserve the class name and public method signatures, implement
all pass/TODO methods, avoid adding tests, and keep the implementation
self-contained.

Full `methods_info` is not included in Query-1 by default because it can contain
hidden tests and reference solution material.

## Finalized Working Dataset

`classeval_stateful_working_v0` is complete and finalized at 18 accepted
samples. The former 20-sample target was abandoned after the full-server run
reached 18 accepted samples and a failed-infra retry produced no additional
accepted samples.

The canonical dataset is exactly:

```text
data/dataset_mining/datasets/classeval_stateful_working_v0/accepted.jsonl
```

The lock manifest is:

```text
configs/dataset_mining/classeval_stateful_working_v0.lock.yaml
```

No additional mining should be performed for this dataset version. Debug and
first-parallel accepted samples are not part of the working dataset. Evaluation
and analysis must consume the frozen 18-row `accepted.jsonl`; they must not
fall back to the original 100-row ClassEval split, diagnostic datasets, rejected
attempts, failed-infrastructure attempts, or all accepted attempts across
dataset versions.

Further ClassEval mining requires a new explicit dataset version, such as
`classeval_stateful_working_v1`.

## Query-2

The initial ClassEval path reuses the existing MBPP Query-2 policy deltas:

- `type_hints_required_v1`
- `google_docstring_required_v1`
- `forbidden_api_v1`

Docstring deltas may be weaker on ClassEval because many skeletons already
contain docstrings.

## Profiles

```bash
CUDA_VISIBLE_DEVICES=0 putpocket-dataset-mining multi \
  --config configs/dataset_mining/classeval_stateful_multi.yaml \
  --profile debug

CUDA_VISIBLE_DEVICES=0,1 putpocket-dataset-mining multi \
  --config configs/dataset_mining/classeval_stateful_multi.yaml \
  --profile first_parallel

CUDA_VISIBLE_DEVICES=0,1,2 putpocket-dataset-mining multi \
  --config configs/dataset_mining/classeval_stateful_multi.yaml \
  --profile full_server
```
