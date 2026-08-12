# GLM-5.2 AWQ mined-sample run report

## 1. Executive summary

Phase A stopped at the mandatory RunPod hardware gate. The live Pod has six NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, while this baseline explicitly requires one node with eight NVIDIA H200 GPUs. No model or checkpoint was loaded, no dataset was accessed, and Server-1 was not contacted.

## 2. Final classification

`GLM52_AWQ_PREFLIGHT_BLOCKED`

Exact blocker: `RUNPOD_GLM52_HARDWARE_INSUFFICIENT`.

## 3. RunPod hardware/runtime

- Hostname: `a0921d8a8df4`
- Observed GPUs: 6 × NVIDIA RTX PRO 6000 Blackwell Server Edition
- HBM/VRAM: 97,887 MiB per GPU; 587,322 MiB aggregate
- Required GPUs: 8 × NVIDIA H200
- Driver: 580.126.09
- Driver CUDA runtime: 13.0
- CUDA toolkit: 12.9.86
- Topology: PCIe across two NUMA nodes; no NVLink links reported

## 4. Git/vLLM/LMCache revisions

- RunPod `master`: `86cbd1964c78c1441f0fc080c46db360f71c17ea`
- `origin/master`: identical at preflight
- vLLM: editable at `/workspace/putpocket_dataset_mining/externals/vllm`, SHA `b65d39ddbab966bb72110056a481d17e4726892b`
- LMCache: editable at `/workspace/putpocket_dataset_mining/externals/lmcache`, SHA `72eb0e375bcf0739a45046433f46ee32be361656`
- Agent doctor passed canonical ownership with no editable worktree leakage.

## 5. Model identity and checkpoint verification

- Requested model: `cyankiwi/GLM-5.2-AWQ-INT4`
- Requested revision: `956f4d228e36c95f66a80058fe64b69af67360ca`
- Checkpoint verification: not run because the preceding hardware gate failed.
- Model initialization: not attempted.

## 6. Dataset identity

Not accessed. The canonical dataset was not modified.

## 7. Local workspace preflight

Not run because the hardware gate failed first.

## 8. Server-1 SSH/verifier preflight

Not run; Server-1 was not contacted and no remote jobs were created.

## 9. Model-load result

Not run.

## 10. Smoke-sample result

Phase B did not start.

## 11. Full-sample results

Phase C did not start. Zero samples were attempted.

## 12. Verification semantics

The integrated profile semantically supports RunPod `local_vllm`, Server-1 `ssh_rsync`, `history1_pytest_only`, `history2_pytest_then_judge`, and disables local hidden-verifier fallback. No verification was submitted.

## 13. Timing

Only bounded Phase A source/runtime/hardware inspection ran. Model-load and inference timings are unavailable.

## 14. GPU utilization

All six GPUs were idle at inspection, with zero MiB process allocations and approximately 29–30 °C temperatures.

## 15. Remote job IDs

None.

## 16. Failures

`RUNPOD_GLM52_HARDWARE_INSUFFICIENT`: observed 6 × RTX PRO 6000 Blackwell, required 8 × H200.

## 17. Dataset integrity

Dataset was not accessed or modified. Before/after hashes were therefore not computed.

## 18. Source cleanliness

Tracked source is clean and `master` equals `origin/master`. Pre-existing untracked Hugging Face runtime credential/state files remain under `models/hf`; they were not read, printed, removed, or changed.

## 19. Safe-to-stop status

Safe to stop. No vLLM engine, evaluation process, or remote verifier job was started.

## 20. Recommended next pipeline/staged test

Recreate or select the prescribed single-node 8 × H200 Pod, synchronize `master` ff-only, and restart Phase A from the hardware gate. Pipeline and Staged Forward remain out of scope until the sequential baseline completes.
