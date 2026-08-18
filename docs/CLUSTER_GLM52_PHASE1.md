# GLM-5.2 Cluster Center phase-1 package

This package prepares exact-commit Cluster Center execution for full GLM-5.2
models. Phase 1 delivers configuration, Slurm rendering, allocation guards,
runtime provenance, checkpoint metadata validation, and readiness handoffs. It
does not claim a model load, generated token, SWE-bench result, quality score,
or vLLM trace.

## Package contents

- Environment lock: `configs/env/cluster_h200_sm90_vllm026.lock.yaml`
- Package manifest: `configs/cluster/package_manifest.yaml`
- Site template: `configs/cluster/site.example.yaml`
- Schemas: `configs/cluster/schemas/profile.schema.json` and
  `configs/cluster/schemas/site.schema.json`
- Profiles:
  - `glm52_nvfp4_tp1_pcp4_ep`: `nvidia/GLM-5.2-NVFP4`, four H200s,
    TP1 + PCP4 + EP
  - `glm52_nvfp4_tp2_pcp2_ep`: `nvidia/GLM-5.2-NVFP4`, four H200s,
    TP2 + PCP2 + EP
  - `glm52_fp8_tp8_reference`: `zai-org/GLM-5.2-FP8`, eight H200s,
    TP8 fallback/reference

The environment lock targets H200/SM90 (`TORCH_CUDA_ARCH_LIST=9.0`) and the
clean upstream vLLM 0.26 source commit
`568afb3a13806beb53bb2e6bd518269357b237c0`. The lock does not reuse the
historical vLLM 0.25 checkout or the Server-2 SM120 profile.

## Site values and render-only workflow

Copy the site template outside the Git checkout and have the Cluster operator
fill it with the actual partition, optional account/constraint, wall time,
CPU count, absolute executable paths, repository/environment/source paths,
cache/checkpoint/artifact/log roots, and exact model revisions. Do not place
credentials or tokens in this file. No provider-specific value is committed.

Rendering is safe on a CPU host and never calls `sbatch`, `salloc`, `srun`,
`nvidia-smi`, or a package installer:

```bash
export SITE_CONFIG=/path/supplied/by/orchestrator/cluster-site.yaml
export JOB_PACKAGE=/path/supplied/by/orchestrator/rendered-jobs

putpocket-cluster profiles validate
putpocket-cluster render \
  --profile glm52_nvfp4_tp1_pcp4_ep \
  --site "$SITE_CONFIG" \
  --job environment \
  --output "$JOB_PACKAGE/environment.sbatch"

for profile in \
  glm52_nvfp4_tp1_pcp4_ep \
  glm52_nvfp4_tp2_pcp2_ep \
  glm52_fp8_tp8_reference
do
  putpocket-cluster render --profile "$profile" --site "$SITE_CONFIG" \
    --job readiness --output "$JOB_PACKAGE/${profile}.readiness.sbatch"
  putpocket-cluster render --profile "$profile" --site "$SITE_CONFIG" \
    --job generation-handoff --output "$JOB_PACKAGE/${profile}.generation-handoff.sbatch"
done
```

The orchestrator or site operator must create the configured Slurm log root
before submission because Slurm opens stdout/stderr before the batch body runs.
Checkpoint staging is a separate guarded site operation; no login-node or
Montblanc checkpoint download is part of phase 1.

## Exact Cluster worker sequence

The Cluster worker receives the phase-1 commit SHA and the already-rendered job
package from the orchestrator. The worker does not edit source or configurations
and does not access Vikunja or AFFiNE.

```bash
cd /path/to/putpocket_dataset_mining
git fetch origin <PHASE1_COMMIT_FROM_HANDOFF>
git switch --detach <PHASE1_COMMIT_FROM_HANDOFF>
test "$(git rev-parse HEAD)" = "<PHASE1_COMMIT_FROM_HANDOFF>"
test -z "$(git status --porcelain --untracked-files=no)"

export JOB_PACKAGE=/path/supplied/by/orchestrator/rendered-jobs
environment_job=$(sbatch --parsable "$JOB_PACKAGE/environment.sbatch")

for profile in \
  glm52_nvfp4_tp1_pcp4_ep \
  glm52_nvfp4_tp2_pcp2_ep \
  glm52_fp8_tp8_reference
do
  readiness_job=$(sbatch --parsable --dependency="afterok:${environment_job}" \
    "$JOB_PACKAGE/${profile}.readiness.sbatch")
  sbatch --parsable --dependency="afterok:${readiness_job}" \
    "$JOB_PACKAGE/${profile}.generation-handoff.sbatch"
done
```

Those are lightweight login-node Git inspection and Slurm submission actions.
The batch bodies validate `SLURM_JOB_ID`, `SLURM_JOB_NODELIST`,
`SLURM_JOB_NUM_NODES`, and batch/step context before any heavy action. A
hostname match is never accepted as allocation evidence.

## Readiness stages and evidence

The readiness library/CLI covers:

1. profile and environment-lock validation;
2. Slurm allocation validation;
3. exact GPU count, H200 name, and compute capability 9.0;
4. imports and required symbols for torch, vLLM, FlashInfer, the vLLM
   DeepGEMM compatibility wrapper, and vendored DeepGEMM;
5. checkpoint config/tokenizer/safetensors layout using filenames and sizes,
   without hashing checkpoint tensors;
6. quantization marker compatibility for NVFP4 or FP8;
7. a model-load command marked `ready_not_executed`;
8. a guarded one-shot generation handoff marked `ready_not_executed`.

Each generated run manifest records the exact Git SHA, configured model
revision, Slurm job ID and nodelist, GPU inventory/topology, driver/CUDA/NCCL
and runtime package versions when available, the secret-free exact command,
and artifact root. It records only allowlisted environment fields and rejects
secret-bearing command arguments or private credential paths.

The `cluster_phase1_handoff` artifact-sync profile copies only manifests,
readiness reports, generation handoffs, environment plans, and rendered jobs.
It excludes checkpoint tensors, caches, Slurm streams, raw traces, benchmark
outputs, and secret-like files before any checksum is computed.

## Heavy-action boundary

`putpocket-cluster run-guarded` is the shared boundary for environment builds,
dependency installs, checkpoint staging, GPU smoke, model load, benchmark, and
one-shot generation. Every action refuses before mutation or probing when the
explicit Slurm allocation contract is absent. `env-bootstrap --execute` has an
independent copy of the same guard; its dry-run plan is safe outside Slurm.

## Deferred phases

- Phase 2: add the SWE-bench Pro adapter, scoring/report schemas, and the >=40
  quality evaluation. No quality claim exists in phase 1.
- Phase 3: port vLLM 0.26 SM90 tracing and indexer evidence from the separately
  reviewed trace work. No trace patch is imported in phase 1.
