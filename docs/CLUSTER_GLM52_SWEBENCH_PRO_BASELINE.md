# GLM-5.2 SWE-bench Pro public baseline job

This stacked phase-2 package runs the unmodified
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

The adapter produces a deterministic lexicographically first one-instance
smoke selection and the complete 731-instance public selection. It proves that
the pinned harness image helper resolves to each row's exact `dockerhub_tag`
before inference. It turns mini-swe-agent `preds.json` into `.pred` inputs and
then calls the official `gather_patches.py`; it does not implement scoring.

Smoke results always have `score_percent: null` and
`acceptance_pass: null`. A score and the explicit 40% pass/fail are emitted
only when the unchanged official result mapping covers all 731 selected IDs.

## Safe Montblanc render

Copy `configs/cluster/swebench_pro_site.example.yaml` outside the checkout and
fill it only with values observed from Cluster inventory: partition, optional
account/qos, exact four-H200 directive, wall time, memory, CPUs, absolute tool
paths, storage/cache/artifact/log roots, and exact model revision/path. Do not
put tokens or credentials in the site file.

```bash
putpocket-swebench-pro validate
putpocket-swebench-pro render \
  --site /absolute/path/cluster-swepro-site.yaml \
  --project-url https://github.com/Nier4Ryu/putpocket_dataset_mining.git \
  --project-commit <EXACT_PUSHED_COMMIT> \
  --output /absolute/path/glm52-swepro-baseline.sbatch
bash -n /absolute/path/glm52-swepro-baseline.sbatch
```

Rendering never invokes Slurm, a GPU tool, Docker, a package installer, or a
network client. The rendered job requests one node and exactly four typed H200
GPUs. `#SBATCH --export=NONE` prevents ambient Login credentials from being
copied into the job.

## Exact Login and compute sequence

Login performs only inventory, creates the configured Slurm-log directory,
and streams the already-rendered script to `sbatch`:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 Login-1 'mkdir -p <SLURM_LOG_ROOT>'
job_id=$(ssh -o BatchMode=yes -o ConnectTimeout=8 Login-1 'sbatch --parsable' \
  < /absolute/path/glm52-swepro-baseline.sbatch)
ssh -o BatchMode=yes -o ConnectTimeout=8 Login-1 \
  "squeue -j ${job_id} -o '%i|%T|%P|%N|%R'"
```

No Login checkout is created. Inside the allocated compute job only, the
script validates explicit Slurm fields, then checks whether Docker is both
present and usable. This happens before project/harness clone, environment
creation, dependency install, checkpoint download, model load, or benchmark.
If unchanged official evaluation cannot use Docker, the job records
`OFFICIAL_EVALUATION_DOCKER_REQUIRED` and exits 42. Apptainer/Singularity may
run the pinned mini-swe-agent scaffold, but it is not silently substituted for
the official Docker-based evaluator.

After preflight, the compute job:

1. initializes a compute-local checkout and fetches only the exact pushed
   project commit;
2. executes the phase-1 allocation-guarded H200/SM90 vLLM 0.26 bootstrap;
3. checks out the pinned official harness and exact submodule Gitlinks;
4. installs benchmark dependencies in the allocation;
5. stages the exact model revision to the configured storage path, or validates
   the configured local checkpoint;
6. captures phase-1 GPU/topology/driver/CUDA/NCCL/package provenance;
7. starts the node-local OpenAI-compatible vLLM endpoint, records the parallel
   profile, and runs a one-shot health/generation request;
8. runs smoke prepare/inference/gather/official evaluation/finalize, then the
   same restartable stages for the complete public test split;
9. writes the full acceptance report and allowlisted provenance, then stops the
   model server through a cleanup trap.

Completed stage markers include an input fingerprint and are skipped on
resume. Failed stages produce a `.failed.json` marker and never a completion
marker. Runtime outputs remain below the configured external artifact root and
are excluded from Git. The Cluster worker does not edit code and never accesses
Vikunja or AFFiNE.

## Claim boundary and next phase

This package makes no quality claim until a complete Cluster artifact contains
the 731-row official result mapping. Phase 3 remains the separate
`vllm-026-sm90-trace-port`; no trace or model patch is included here.
