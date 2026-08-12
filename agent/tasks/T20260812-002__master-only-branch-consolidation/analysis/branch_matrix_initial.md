# Initial Branch Integration Matrix

| Branch | Tip | Ancestor of master | Master ancestor of tip | Unique commits | Files changed | Initial classification |
|---|---|---:|---:|---:|---|---|
| `agent/T20260812-001__agent-worktree-control-plane` | `46edffb4ff2e` | no | yes | 1 | agent/tasks/T20260812-001__agent-worktree-control-plane/TASK.md, agent/tasks/T20260812-001__agent-worktree-control-plane/handoffs/TO_GPT_20260812-181840.md, configs/env/server2_blackwell.lock.yaml | DIRECT_DESCENDANT_FAST_FORWARD_CANDIDATE |
| `agent/T20260812-002__master-only-branch-consolidation` | `897edfdbb77e` | yes | yes | 0 | - | ANCESTOR_ALREADY_IN_MASTER |
| `blackwell` | `988e8b4aaee5` | yes | no | 19 | - | ANCESTOR_ALREADY_IN_MASTER |
| `codex/bootstrap-entrypoint-audit-20260812_051218` | `b2f78f2decdb` | yes | no | 3 | - | ANCESTOR_ALREADY_IN_MASTER |
| `codex/canonicalize-remote-verifier-20260805_225753` | `a8de7caa0973` | yes | no | 10 | - | ANCESTOR_ALREADY_IN_MASTER |
| `codex/cloud-mainline-consolidation-20260812_031733` | `b2f78f2decdb` | yes | no | 3 | - | ANCESTOR_ALREADY_IN_MASTER |
| `codex/distributed-execution-modes-20260811_205028` | `32b0b3700059` | yes | no | 4 | - | ANCESTOR_ALREADY_IN_MASTER |
| `codex/e2e-two-turn-remote-verifier-20260811_183255` | `042452b50254` | yes | no | 6 | - | ANCESTOR_ALREADY_IN_MASTER |
| `codex/finish-manual-pipeline-20260811_225947` | `b2f78f2decdb` | yes | no | 3 | - | ANCESTOR_ALREADY_IN_MASTER |
| `codex/fix-proxyjump-live-verifier-20260810_065259` | `33a00845908a` | no | no | 8 | src/putpocket_dataset_mining/ssh_transport.py, tests/test_remote_fixture_cli.py, tests/test_remote_transport.py | SUBSTANTIVE_UNIQUE_COMMITS_REQUIRE_INTEGRATION |
| `codex/fresh-e2e-timing-20260811_194006` | `7b9dc4e0f1aa` | yes | no | 5 | - | ANCESTOR_ALREADY_IN_MASTER |
| `codex/multihost-bootstrap-remote-verifier-20260805_151808` | `7da7ba5da661` | yes | no | 13 | - | ANCESTOR_ALREADY_IN_MASTER |
| `codex/runpod-cuda129-uv-editable-runtime-20260812_173847` | `2992043a9032` | no | no | 3 | cloud/runpod/Dockerfile.dev-base, cloud/runpod/README.md, cloud/runpod/template.dev-base.example.yaml, configs/env/runpod_base_image.lock.yaml, configs/env/torch/torch_2_10_cu129.lock.yaml, scripts/env/bootstrap_sr.sh, scripts/env/env_activate.sh, src/putpocket_dataset_mining/bootstrap_sr.py, ... | SUBSTANTIVE_UNIQUE_COMMITS_REQUIRE_INTEGRATION |
| `codex/sr-multihost-cloud-20260804_151155` | `03b24b43c5c3` | yes | no | 17 | - | ANCESTOR_ALREADY_IN_MASTER |
| `codex/unify-server2-bootstrap-env-20260812_142601` | `120353c988c0` | yes | no | 2 | - | ANCESTOR_ALREADY_IN_MASTER |
| `origin/agent/T20260812-001__agent-worktree-control-plane` | `897edfdbb77e` | yes | yes | 0 | - | ANCESTOR_ALREADY_IN_MASTER |
| `origin/blackwell` | `a8de7caa0973` | yes | no | 10 | - | ANCESTOR_ALREADY_IN_MASTER |
| `origin/codex/canonicalize-remote-verifier-20260805_225753` | `a8de7caa0973` | yes | no | 10 | - | ANCESTOR_ALREADY_IN_MASTER |
| `origin/codex/cloud-mainline-consolidation-20260812_031733` | `b2f78f2decdb` | yes | no | 3 | - | ANCESTOR_ALREADY_IN_MASTER |
| `origin/codex/e2e-two-turn-remote-verifier-20260811_183255` | `042452b50254` | yes | no | 6 | - | ANCESTOR_ALREADY_IN_MASTER |
| `origin/codex/fix-proxyjump-live-verifier-20260810_065259` | `33a00845908a` | no | no | 8 | src/putpocket_dataset_mining/ssh_transport.py, tests/test_remote_fixture_cli.py, tests/test_remote_transport.py | REMOTE_ONLY_UNKNOWN |
| `origin/codex/fresh-e2e-timing-20260811_194006` | `7b9dc4e0f1aa` | yes | no | 5 | - | ANCESTOR_ALREADY_IN_MASTER |
| `origin/codex/multihost-bootstrap-remote-verifier-20260805_151808` | `7da7ba5da661` | yes | no | 13 | - | ANCESTOR_ALREADY_IN_MASTER |
| `origin/codex/runpod-cuda129-uv-editable-runtime-20260812_173847` | `2992043a9032` | no | no | 3 | cloud/runpod/Dockerfile.dev-base, cloud/runpod/README.md, cloud/runpod/template.dev-base.example.yaml, configs/env/runpod_base_image.lock.yaml, configs/env/torch/torch_2_10_cu129.lock.yaml, scripts/env/bootstrap_sr.sh, scripts/env/env_activate.sh, src/putpocket_dataset_mining/bootstrap_sr.py, ... | SUBSTANTIVE_UNIQUE_COMMITS_REQUIRE_INTEGRATION |
| `origin/codex/server1-verifier-bootstrap-fix-20260805_220221` | `f42d8e201305` | yes | no | 12 | - | ANCESTOR_ALREADY_IN_MASTER |
| `origin/codex/sr-multihost-cloud-20260804_151155` | `03b24b43c5c3` | yes | no | 17 | - | ANCESTOR_ALREADY_IN_MASTER |
| `origin/codex/unify-server2-bootstrap-env-20260812_142601` | `120353c988c0` | yes | no | 2 | - | ANCESTOR_ALREADY_IN_MASTER |
