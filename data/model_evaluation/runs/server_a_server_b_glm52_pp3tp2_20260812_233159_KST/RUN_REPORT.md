# Server-A / Server-B GLM-5.2 PP3/TP2 run report

## Executive summary

The canonical Server-A / Server-B architecture was implemented, integrated, synchronized to RunPod and Server-1, and live-validated. RunPod required no Docker. Server-1 successfully provided the dedicated isolated workspace container, persistence across reconnect, snapshot creation, verifier Docker, and Codex Judge preflight.

The real GLM-5.2 attempt then initialized the six-rank PP=3/TP=2 topology and selected `GlmMoeDsaForCausalLM`, compressed-tensors, and Marlin. Model readiness was blocked before checkpoint payload loading because every vLLM worker raised `Sparse Attention Indexer CUDA op requires DeepGEMM to be installed.` No alternate topology, model, checkpoint, or unsafe fallback was attempted.

## Classification

`GLM52_PP3_TP2_MODEL_LOAD_BLOCKED`

The distributed architecture itself is `SERVER_A_SERVER_B_ARCHITECTURE_IMPLEMENTED`.

## Source and deployment

- Final master/origin: `a768075e9bf595ca6d167a5acd1d1d772b4daf74`
- RunPod: same SHA
- Server-1: same SHA
- Server-2: not reachable from the approved RunPod transport; synchronization remains pending
- Architecture task: `T20260812-001__remote-server-b-workspace`
- Judge preflight task: `T20260812-001__server-b-judge-preflight`
- Mirror fixes: `T20260812-001__remote-workspace-mirror-sync`, `T20260812-001__remote-workspace-session-path`

## Live workspace evidence

The approved Proxy-A -> Proxy-C -> Server-1 route and all three pinned ED25519 fingerprints passed strict checking. A disposable session created `remote_workspace_smoke.txt` inside the Server-B container, read it through the controller mirror, disconnected and reconnected, observed the same H1 state, created snapshot SHA-256 `4277922e0b2da96da4ea0648d45ffc6ca736b4d21b45df0e28fb3dbaffc2d7da`, and cleaned up. Model-generated commands were never executed on either host shell.

## Verifier evidence

Server-1 verifier wrapper, Docker daemon, verifier image, rsync, remote root, Codex CLI `0.146.0`, and Codex auth presence passed. V1/V2 jobs were not submitted because Query-1 never ran. Local hidden-verifier fallback stayed disabled.

## Model attempt

- Model path: immutable local revision `956f4d228e36c95f66a80058fe64b69af67360ca`
- Checkpoint: previously structurally proven 83/83; no redownload
- GPUs: 6 x RTX PRO 6000 Blackwell Server Edition
- PP/TP: 3/2; world size 6
- Context: 8192; max sequences 1
- Prefix caching: enabled
- Execution: eager
- Quantization: compressed-tensors, Marlin WNA16
- Engine constructor interval: approximately 118 seconds from first vLLM argument log to terminal engine failure
- Weight-load timing: not separately observable; failure occurred during model construction before payload load

## Failure evidence

All ranks PP0/TP0, PP0/TP1, PP1/TP0, PP1/TP1, PP2/TP0, and PP2/TP1 independently reported the same missing DeepGEMM requirement from `vllm/model_executor/layers/sparse_attn_indexer.py`. Full traces are retained in `logs/workflow.log`.

## Integrity and cleanup

The frozen dataset remained at SHA-256 `6031d368ee8359c9dfc3c7b785d5c30e4db9ae5b2969bfba3a7e09512a46b30d`. No samples were attempted, no remote jobs remain, no engine/GPU process remains, and all disposable workspaces were removed.

## Next step

Establish a reproducible DeepGEMM dependency/build contract compatible with the pinned editable vLLM, CUDA 12.9, Python 3.13, and SM120. Implement it through the canonical bootstrap/build workflow with source tests and synchronization, then start a new run ID and repeat this exact PP3/TP2 configuration.
