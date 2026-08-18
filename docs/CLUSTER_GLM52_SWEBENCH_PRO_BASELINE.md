# GLM-5.2 SWE-bench Pro one-instance baseline smoke

This bounded stacked phase-2 package runs one deterministic public instance with the unmodified
`nvidia/GLM-5.2-NVFP4` baseline on exactly one Slurm node and four H200 GPUs.
It uses TP1 + PCP4 + EP first. TP2 + PCP2 + EP is attempted only when the
primary vLLM log contains a classified parallel-startup compatibility error;
the selected profile and any fallback reason are persisted in the run root.
There is no DSA/MLA cache change, pruning, custom quantization, training, or
model-behavior patch.

## Pinned public contract

- Source lock: `configs/cluster/swebench_pro_sources.lock.yaml`
- Official harness: `scaleapi/SWE-bench_Pro-os` at
  `ca10a60a5fcae51e6948ffe1485d4153d421e6c5`
- SWE-agent Gitlink: `402a7b8fdac8193f3f255bb53859ba274234f596`
- mini-swe-agent Gitlink: `d74716a3c8104a113f77cc9ab94cf407ecdcf1e9`
- Dataset: `ScaleAI/SWE-bench_Pro`, `test`, revision
  `7ab5114912baf22bb098818e604c02fe7ad2c11f` (731 rows)
- Images: `jefzda/sweap-images:<row.dockerhub_tag>`
- Scorer: the pinned, unchanged `swe_bench_pro_eval.py`

The active adapter path produces the deterministic lexicographically first
one-instance smoke selection. The full manifest remains a dormant extension
point and is never loaded or referenced by the smoke renderer. The smoke proves that
the pinned harness image helper resolves to each row's exact `dockerhub_tag`
before inference. It turns mini-swe-agent `preds.json` into `.pred` inputs and
then calls the official `gather_patches.py`; it does not implement scoring.

Smoke results always have `score_percent: null`, `acceptance_pass: null`, and
the explicit `NON_SCORE_ELIGIBLE_SMOKE_ONLY` claim boundary. This task does not
run the full manifest or calculate the 40% acceptance threshold.

## Safe Montblanc render

The tracked `configs/cluster/sites/herdr_h200_smoke.yaml` contains only values
observed for this bounded run: H200, `gsai-account`, `hpgpu`, four typed GPUs,
32 CPUs, 512G, six hours, shared log storage, and allocated-node local storage.
The compute path loads `cuda/12.9` and rejects a different `nvcc` version.

```bash
putpocket-swebench-pro validate --smoke-only
putpocket-swebench-pro render \
  --site configs/cluster/sites/herdr_h200_smoke.yaml \
  --project-url https://github.com/Nier4Ryu/putpocket_dataset_mining.git \
  --project-commit <EXACT_PUSHED_COMMIT> \
  --smoke-only > /absolute/path/glm52-swepro-smoke.sbatch
bash -n /absolute/path/glm52-swepro-smoke.sbatch
```

Rendering never invokes Slurm, a GPU tool, Docker, a package installer, or a
network client. The rendered job requests one node and exactly four typed H200
GPUs. `#SBATCH --export=NONE` prevents ambient Login credentials from being
copied into the job.

## Compact Login and compute sequence

On Montblanc, render one pasteable command after replacing the commit with the
exact pushed source commit:

```bash
putpocket-swebench-pro render-wrap \
  --site configs/cluster/sites/herdr_h200_smoke.yaml \
  --project-url https://github.com/Nier4Ryu/putpocket_dataset_mining.git \
  --project-commit <EXACT_PUSHED_COMMIT>
```

Paste that single `mkdir ... && sbatch --parsable ... --wrap=...` command into
the already authenticated Herdr Login pane. No Login checkout or source file is
created. The wrapped compute command validates explicit Slurm fields, then checks whether Docker is both
present and usable. This happens before project/harness clone, environment
creation, dependency install, checkpoint download, model load, or benchmark.
If unchanged official evaluation cannot use Docker, the job records
`OFFICIAL_EVALUATION_DOCKER_REQUIRED` and exits 42. Apptainer/Singularity may
run the pinned mini-swe-agent scaffold, but it is not silently substituted for
the official Docker-based evaluator.

After preflight, the compute job:

1. initializes a compute-local checkout and fetches only the exact pushed
   project commit;
2. bootstraps pinned uv 0.11.31 under allocated-node storage, initializes
   Environment Modules, loads `cuda/12.9`, and verifies `nvcc` is 12.9;
3. executes the phase-1 allocation-guarded H200/SM90 vLLM 0.26 bootstrap;
4. checks out the pinned official harness and exact submodule Gitlinks;
5. installs benchmark dependencies in the allocation;
6. resolves the requested public model ref to an exact Hugging Face SHA and stages it, or validates
   the configured local checkpoint;
7. captures phase-1 GPU/topology/driver/CUDA/NCCL/package provenance;
8. starts the node-local OpenAI-compatible vLLM endpoint, records the parallel
   profile, and runs a one-shot health/generation request;
9. runs one smoke prepare/inference/gather/unchanged official evaluation/finalize path;
10. verifies the report is complete but non-score-eligible, writes allowlisted provenance, then stops the
   model server through a cleanup trap.

Completed stage markers include an input fingerprint and are skipped on
resume. Failed stages produce a `.failed.json` marker and never a completion
marker. Runtime outputs remain below the configured external artifact root and
are excluded from Git. The Cluster worker does not edit code and never accesses
Vikunja or AFFiNE.

## Claim boundary and next phase

This smoke package makes no quality claim and cannot emit a threshold pass.
Any future 731-row run requires a separate task and explicit authorization. Phase 3 remains the separate
`vllm-026-sm90-trace-port`; no trace or model patch is included here.
