# RunPod Patch Equivalence Review

Original branch: codex/runpod-cuda129-uv-editable-runtime-20260812_173847
Original tip: 2992043a90322d39918d6f26bff0959f28088a9e
Integrated cherry-pick: dec9f62a2916b3adcc985d8ac469e2fcb7c1bafc
Conflict-resolution follow-up: cb84ebd024fb3e16ebe90556e1f2bf2db5d1e040
Doctor-only fix after integration: 3139d8ee69d17993a58d31f2be36a94e7bc30a9e

## RunPod-specific file comparisons against master
- cloud/runpod/Dockerfile.dev-base: identical
- cloud/runpod/README.md: identical
- cloud/runpod/template.dev-base.example.yaml: identical
- configs/env/runpod_base_image.lock.yaml: identical
- configs/env/torch/torch_2_10_cu129.lock.yaml: identical
- scripts/env/bootstrap_sr.sh: identical
- src/putpocket_dataset_mining/runpod_runtime.py: identical
- tests/test_runpod_runtime_contract.py: identical

## Known intentional differences
- scripts/env/env_activate.sh also contains Agent context/canonical runtime guards from master.
- src/putpocket_dataset_mining/bootstrap_sr.py contains doctor-only read-only fix from master consolidation.
- master contains Agent control-plane files that do not exist on the older RunPod branch.

Decision: the RunPod implementation content is represented in master; the original branch may be deleted after final bundle verification.
