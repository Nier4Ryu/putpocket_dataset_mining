# GLM-5.2 → GLM-5.3 stateful-edit-v3 source matrix

The exhaustive, hash-bound 97-path ledger is
`GLM52_ACTIVE_PATH_INVENTORY.json`. This matrix records the design-level
decisions. The source tree and every GLM-5.2 artifact remain read-only.

| Concern | GLM-5.2 source | GLM-5.3 target | Disposition / verified difference |
|---|---|---|---|
| Deployment authority | `configs/cluster/glm52_forced_reuse_ablation.lock.json` | `configs/models/glm53_flash_nvfp4_montblanc.lock.json`; experiment lock | Rebound from `nvidia/GLM-5.2-NVFP4`/vLLM `4a3447d`/TP4 H200 to exact `RedHatAI/GLM-5.3-Flash-NVFP4` revision `36c184c`, vLLM `878631b`, FlashInfer SM120, TP1+DP3+EP3. |
| Episode and ratio CLI | `glm52_stateful_cli.py` | `glm53_stateful_edit.py` | Same single SWE-bench Pro instance, frozen A1/Q2 and ratios 0..100. Serialized edit position is derived with the pinned GLM-5.3 tokenizer; token 114 is not imported. |
| Base selector | GLM-5.2 21-layer DSA stability selector | GLM-5.3 native kpool capture + offline stability freeze | Separate default-OFF capture records the last-query native raw pre-top-k pool row on all 11 indexer layers. Pool scores expand to four tokens; incomplete tail tokens are marked unscored and sorted last. GLM-5.2 selectors are rejected. |
| Proxy | `glm52_stateful_proxy.py` | `glm53_stateful_proxy.py` | Same freeze/replay/edit order and outcome isolation. Every forwarded model request adds `X-data-parallel-rank: 0` for donor-cache ownership. |
| Cache mutation hook | `glm52_forced_edit_reuse.py` | `glm53_stateful_edit_accuracy_ablation.py` | Still default OFF and full-prefill-first. GLM-5.3 copies 656-byte `fp8_ds_mla` rows only on 11 MLA layers. |
| Indexer cache | packed per-token GLM-5.2 index rows on 21 layers | 132-byte FP8 compressed-pool rows on 11 MLA layers | Only complete aligned four-token pools wholly selected inside frozen history are copied. Partial/boundary pools and token-granular kpool tail remain target-computed. |
| Non-MLA layers | all 78 GLM-5.2 main rows addressable | 34 GLM-5.3 KDA layers | KDA exposes terminal recurrent/conv state, not token-addressable KV rows. It is never donor-transplanted in this slice. |
| vLLM patch | vLLM `4a3447d` GLM-5.2 model/MLA/indexer patch | vLLM `878631b` GLM5Next model/MLA/kpool patch | Applied after the exact required GLM-5.3 deployment patch chain. Native writes precede overwrite; ordinary behavior is unchanged when the hook is absent. |
| Source packaging | cluster archive packager | `scripts/glm53/prepare_stateful_edit_v3_overlay.sh` | CPU-only, fresh exact source, fail-fast apply/check and exact postimage hashes. No model import, Docker, CUDA, or device access. |
| Experiment runner | cluster Slurm H200 runner/submission | `run_stateful_edit_v3_harness.sh` plus deployment runbook | Harness is endpoint-oriented and preserves official evaluator sequencing. Slurm submission is deliberately not ported. |
| Scenario contract/catalog | later model-neutral documents/catalog | one GLM-5.3 constrained authoring template | Reviewed as model-neutral reference; this task permits only the existing single equal-length SYS edit instance. Multi-span/insert/delete scenarios are not activated. |
| True-partial server path | later GLM-5.2 true-partial overlay | none | Explicitly out of scope. This port computes every target prompt row, then overwrites donor rows; it cannot support compute-saving, latency, or skipped-FLOP claims. |
| Score diagnostics and multihop plots | later GLM-5.2 RunPod captures/reports | none | Kept as historical GLM-5.2 evidence. They are not model/cache compatible and are never relabeled as GLM-5.3 validation. |

## Shared pieces retained

- `swebench_pro_cli` remains the model-neutral preparation/gather layer.
- The official ScaleAI harness commit and exact one-instance selection remain
  unchanged; SWE-bench Pro supplies the problem/repository/evaluator, while
  PutPocket authors the A1-execute-Q2 episode and system edit.
- Each ratio uses exact replay, a fresh evaluator state, a frozen selector,
  and a separate result. Ratio 0 is ordinary target full prefill; ratios above
  zero use the explicit unsafe accuracy-ablation control.

## Static architecture evidence

The pinned model config has 45 layers. Layers `3,7,…,43` are
`deepseek_sparse_attention`; the other 34 are `linear_attention`. It declares
64 attention heads, 288 routed experts, 8 experts/token, indexer 32×128,
`index_kpool=4`, `index_topk=2048`, `qk_rope_head_dim=0`, and
`kv_lora_rank=512`. The exact tokenizer and chat-template digests are recorded
in the experiment lock. Runtime claims remain pending a separately authorized
GPU freeze and smoke.
