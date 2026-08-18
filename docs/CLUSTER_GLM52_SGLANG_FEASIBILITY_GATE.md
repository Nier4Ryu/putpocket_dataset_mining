# GLM-5.2 SGLang four-H200 feasibility gate

This package is the first runtime experiment for the unmodified
`nvidia/GLM-5.2-NVFP4` checkpoint. It is not a benchmark and produces no
SWE-bench selection, patch, evaluation, score, or acceptance claim. A later
benchmark job is authorized only after this gate writes a `PASS` manifest.

## Immutable runtime contract

`configs/cluster/glm52_sglang_gate_sources.lock.json` records the official
SGLang source commit used to establish the supported controls, the official
runtime image's human tag and linux/amd64 digest, primary-source URLs, and the
model revision policy. The image digest fixes the installed environment. Phase
1 still probes the installed image because the independently recorded SGLang
source commit is evidence, not an unsupported assertion that Docker Hub's image
was built from that exact commit.

The model's mutable `main` reference is resolved inside the allocation to a
full 40-character Hugging Face commit before `config.json` is fetched. Phase 1
downloads only that metadata file. Checkpoint staging begins only after the
allocation, image, installed backend, and exact config-layout gates pass.

## Cluster execution order

The orchestrator renders one compact command from an exact public project
commit:

```bash
PYTHONPATH=src Putpocket_env/bin/python -m putpocket_dataset_mining.glm52_sglang_gate_cli render-wrap \
  --project-url https://github.com/Nier4Ryu/putpocket_dataset_mining.git \
  --project-commit <40-character-public-commit>
```

The command creates only `/home2/jslee202403/putpocket-slurm` on Login and
streams the allocation request through `sbatch --wrap`. Resource flags are
fixed to partition `H200`, account `gsai-account`, qos `hpgpu`, one node, one
task, four typed H200 GPUs, 32 CPUs, 512G memory, six hours, and `--export=NONE`.
The orchestrator—not a repository worker—pastes or streams that one line in an
already authenticated Login session.

After Slurm starts the allocation, the outer wrapper checks the allocation
variables and compute-local storage, fetches only the exact project commit
under `/local-data/jslee202403`, and execs the tracked entrypoint. The
entrypoint then runs these non-skippable phases:

1. Record and validate the allocation, physical GPU identities/memory/MIG
   state, topology/NVLink, driver, Slurm TRES, and the explicitly loaded
   `cuda/12.9` compiler module.
2. Check Docker and pull the immutable official SGLang image. Inside that
   image, validate installed imports/symbols/backend controls and disabled
   defaults; resolve the model commit; fetch and validate config metadata only.
3. Stage the exact checkpoint, sample HBM, and launch one TP4 SGLang server with
   ModelOpt FP4, Marlin, the required DSA backends, context 4096, and concurrency
   one. There is no alternate model, GPU count, backend, quantizer, or offload
   path.
4. Obtain effective server information, run one greedy one-shot request, and
   validate output, runtime logs, model layout, and per-GPU HBM headroom.

Any mismatch writes a classified `FAIL` manifest. Backend ambiguity and silent
fallback are failures. An unavailable official runtime, repack OOM, nonpositive
HBM headroom, or invalid output does not trigger a retry with relaxed settings.

## Artifacts

Slurm stdout and stderr are:

- `/home2/jslee202403/putpocket-slurm/pp-glm52-sglang-gate-<jobid>.out`
- `/home2/jslee202403/putpocket-slurm/pp-glm52-sglang-gate-<jobid>.err`

Run artifacts are rooted at:

`/local-data/jslee202403/putpocket-glm52-sglang-gate/artifacts/<jobid>`

The root `gate_manifest.json` is authoritative. Phase directories retain the
inventory/TRES/topology, immutable identities, package and model-config probes,
server log/effective server info/exact command, raw request/response,
normalized-output SHA-256, and HBM samples/summary. Checkpoint tensors are never
hashed and credentials are neither passed to the container nor printed.

## Claim boundary

`PASS` means only that this exact four-H200 SGLang configuration loaded
all-resident, used unambiguous required paths, produced one coherent
deterministic output, and retained positive measured HBM headroom. It is not an
accuracy, throughput, benchmark, or production-readiness claim.
