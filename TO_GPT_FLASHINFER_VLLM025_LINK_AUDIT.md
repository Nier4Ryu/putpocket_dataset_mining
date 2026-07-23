# FlashInfer + vLLM 0.25 Link Audit Report

## Executive Summary

FlashInfer is already installed in the separate GLM vLLM 0.25 environment. The active env has `flashinfer-python==0.6.13` and `flashinfer-cubin==0.6.13`, matching the vLLM package requirements, and vLLM's `has_flashinfer_sparse_mla_sm120()` returns `True`.

This means vLLM 0.25 can link to the official FlashInfer sparse MLA SM120 API in this env. Installing FlashInfer is not the blocker anymore.

However, normal/base vLLM 0.25 still does not support tiny GLM h192/v128 directly. In the upstream `HEAD` source for `externals/vllm_glm52_v025`, the SM120 FlashInfer sparse MLA backend inherits `get_supported_head_sizes() -> [576]`, uses the packed `fp8_ds_mla` 656-byte V32 layout, and calls `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`. The current local checkout reports `[576, 192]` only because of the previous h192 experiment patch.

## Environment

- Branch: `blackwell`
- Repo HEAD: `16f361283a771b0b051ae244e3d70908005c5e3a`
- Git state at audit start: branch ahead of `origin/blackwell` by 1; unrelated untracked `TO_GPT_USE_SPARSE_AUDIT.md` left untouched
- Env path: `Putpocket_env_glm52_v025`
- Activation: `source scripts/env/env_activate_glm52_v025.sh`
- Python: `Python 3.13.14`
- Python executable: `/home/dyryu/putpocket_dataset_mining/Putpocket_env_glm52_v025/bin/python`
- Torch: `2.11.0+cu129`
- Torch CUDA: `12.9`
- CUDA available: `True`
- GPU 0: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Capability: `(12, 0)`
- vLLM: `0.25.2.dev0+g752a3a504.d20260723`
- vLLM path: `/home/dyryu/putpocket_dataset_mining/externals/vllm_glm52_v025/vllm/__init__.py`

Important caveat: `externals/vllm_glm52_v025` currently contains the previous local h192 experiment patch. For source conclusions about unpatched vLLM behavior, I inspected `git -C externals/vllm_glm52_v025 show HEAD:...`.

## Current FlashInfer Status Before Install

Import checks in `Putpocket_env_glm52_v025`:

- `flashinfer`: OK, version `0.6.13`, path `Putpocket_env_glm52_v025/lib/python3.13/site-packages/flashinfer/__init__.py`
- `flashinfer_python`: missing as an import name, expected because the package imports as `flashinfer`
- `flash_mla_sm120`: OK, installed from the previous leavelet build

Package metadata:

- `flashinfer-python==0.6.13`
- `flashinfer-cubin==0.6.13`
- `vllm` requires `flashinfer-python==0.6.13` and `flashinfer-cubin==0.6.13`

vLLM utility checks:

- `has_flashinfer()`: `True`
- `has_flashinfer_sparse_mla_sm120()`: `True`
- `has_flashinfer_b12x_gemm()`: `True`
- `has_flashinfer_b12x_moe()`: `True`

## vLLM FlashInfer Integration Source Findings

Key source files:

- `externals/vllm_glm52_v025/vllm/utils/flashinfer.py`
- `externals/vllm_glm52_v025/vllm/v1/attention/backends/registry.py`
- `externals/vllm_glm52_v025/vllm/platforms/cuda.py`
- `externals/vllm_glm52_v025/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`
- `externals/vllm_glm52_v025/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`

vLLM checks FlashInfer with `importlib.util.find_spec("flashinfer")`. If `flashinfer-cubin` is not installed, vLLM also requires `nvcc` for JIT. In this env `flashinfer-cubin` is installed.

vLLM's sparse MLA SM120 check is:

```python
from flashinfer.autotuner import autotune
from flashinfer.decode import (
    trtllm_batch_decode_sparse_mla_dsv4,
    trtllm_batch_decode_with_kv_cache_mla,
)
```

Then it requires all three objects to be callable. This check returns `True` here.

The official vLLM SM120 sparse backend is registered as:

```text
FLASHINFER_MLA_SPARSE_SM120 =
vllm.v1.attention.backends.mla.flashinfer_mla_sparse.FlashInferMLASparseSM120Backend
```

On CUDA capability major `12`, vLLM's platform priority list for MLA includes:

```text
TRITON_MLA
FLASHINFER_MLA_SPARSE_SM120
```

## vLLM Backend Selection / Config Knobs

User-facing CLI:

- `vllm serve --attention-backend ATTENTION_BACKEND`
- Valid enum includes `FLASHINFER_MLA_SPARSE_SM120`
- `vllm serve --help=AttentionConfig` documents `--attention-backend`
- `vllm serve --help=all` also shows `--kv-cache-dtype ... fp8_ds_mla ...`

I did not find a `VLLM_ATTENTION_BACKEND` env var in this vLLM source.

MLA prefill backend:

- Config object has `AttentionConfig.mla_prefill_backend`
- Enum includes `FLASH_ATTN`, `FLASHINFER`, `TRTLLM_RAGGED`, `TOKENSPEED_MLA`, `ROCM_AITER_FA`, `CUSTOM`
- This build's CLI help did not expose a simple `--mla-prefill-backend` flag
- It appears to be settable through structured/nested config paths rather than the basic CLI flag list

Automatic decode backend selection:

- If backend is not forced, CUDA major 12 tries the platform priority list.
- `FLASHINFER_MLA_SPARSE_SM120` is considered automatically if the model is MLA+sparse and the backend passes validation.
- Base validation includes `get_supported_head_sizes()`, so unpatched vLLM rejects tiny GLM h192 before the FlashInfer function can help.

## FlashInfer Installation Attempt

No new install was performed because FlashInfer is already installed in the GLM v0.25 env at the exact versions required by vLLM:

```text
flashinfer-python==0.6.13
flashinfer-cubin==0.6.13
```

I checked available versions:

- latest `flashinfer-python` visible to pip: `0.6.15.post1`
- latest `flashinfer-cubin` visible to pip: `0.6.13`
- no matching distribution found for `flashinfer-jit-cache`

Because vLLM declares exact requirements on `flashinfer-python==0.6.13` and `flashinfer-cubin==0.6.13`, upgrading FlashInfer would move outside this vLLM build's pinned dependency set. That is not the right next move for this audit.

## FlashInfer API Availability After Install

Since FlashInfer was already installed, these are current API results:

Top-level `flashinfer`:

- `sparse_mla_sm120_paged_attention`: absent
- `BatchSparseMLAPagedAttentionWrapper`: absent
- `BatchMLAPagedAttentionWrapper`: present
- `BatchDecodeMlaWithPagedKVCacheWrapper`: present

`flashinfer.decode`:

- `trtllm_batch_decode_sparse_mla_dsv4`: present
- `trtllm_batch_decode_with_kv_cache_mla`: present
- `BatchDecodeMlaWithPagedKVCacheWrapper`: present

vLLM does not look for `sparse_mla_sm120_paged_attention` or `BatchSparseMLAPagedAttentionWrapper` in this source. It looks for the `flashinfer.decode.trtllm_*` functions above.

## Need For leavelet/sparse_mla_sm120

Normal `flashinfer-python 0.6.13` is enough to satisfy vLLM's official `has_flashinfer_sparse_mla_sm120()` check.

`leavelet/sparse_mla_sm120` is not needed for vLLM's official FlashInfer detection. It installs under its own namespace:

```text
flash_mla_sm120
```

It exposes:

- `sparse_mla_decode_fwd`
- `sparse_mla_prefill_fwd`
- `flash_mla_sparse_fwd`
- `flash_mla_with_kvcache`
- `FlashMLASchedMeta`
- `get_mla_metadata`

It does not expose the vLLM-expected `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`, and it does not register itself under the `flashinfer` namespace. Installing it will not make unpatched vLLM's `has_flashinfer_sparse_mla_sm120()` return `True`; normal FlashInfer already does that.

Also, the checked-out leavelet source is not a native tiny GLM h192/v128 implementation. Its source/tests are V32/MODEL1 oriented and reject `d_qk=192`.

## Can B-Style External Import Work?

Yes, B-style external import is technically feasible, but it requires vLLM adapter code.

The external package can be imported as `flash_mla_sm120` and called from a vLLM backend implementation. That is what the prior local experiment did. It is not automatic and not compatible with vLLM's official FlashInfer sparse MLA detection path by itself.

The bigger issue is not importability. The bigger issue is shape/ABI correctness:

- vLLM official FlashInfer SM120 path expects packed `fp8_ds_mla` V32-style cache layout.
- Base vLLM supports `head_size=576` for this backend.
- Tiny GLM uses `head_size=192`, `kv_lora_rank=128`, `v_head_dim=128`, `index_n_heads=8`, `index_head_dim=64`.
- The leavelet kernel currently rejects h192 directly.

## Does vLLM Need A Patch?

Yes, if the target remains `inference-optimization/GLM-5.2-0.8B-A0.8B` tiny h192/v128.

FlashInfer installation alone is not enough. vLLM 0.25 already links to FlashInfer sparse MLA SM120 successfully, but unpatched vLLM still rejects h192 because the SM120 sparse backend's base supported head sizes are `[576]`.

Even if head-size validation is widened, further shape work is still needed for:

- `kv_lora_rank=128` instead of V32 `512`
- `v_head_dim=128` instead of V32 `512`
- `index_n_heads=8`, where DeepGEMM indexer code had a 16/32/64-head assumption in previous smoke
- cache layout compatibility with `fp8_ds_mla`

## Recommended Next Implementation Step

Recommended: do not spend more time on FlashInfer installation. It is already installed and recognized.

Next implementation step: implement or obtain a true h192/v128 SM120 sparse MLA path, then add a narrow vLLM adapter for tiny GLM. Use normal `flashinfer-python==0.6.13` for the official vLLM dependency, and use `leavelet/sparse_mla_sm120` only as an external kernel provider if it gains real h192/v128 support or if you intentionally maintain a padded experimental adapter.

If the goal is to test base vLLM behavior only, use:

```bash
vllm serve inference-optimization/GLM-5.2-0.8B-A0.8B \
  --attention-backend FLASHINFER_MLA_SPARSE_SM120 \
  --kv-cache-dtype fp8_ds_mla \
  --trust-remote-code
```

Expected base-vLLM result for tiny GLM is still backend rejection at h192, not missing FlashInfer.

## Commands Run

```bash
pwd
git status -sb
git branch --show-current
git rev-parse HEAD
source scripts/env/env_activate_glm52_v025.sh
which python
python -V
python - <<'PY'
import sys
print(sys.executable)
PY
python - <<'PY'
import vllm
print(getattr(vllm, "__version__", "unknown"))
print(vllm.__file__)
PY
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
python - <<'PY'
mods = ["flashinfer", "flashinfer_python", "flash_mla_sm120"]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", "unknown"), getattr(mod, "__file__", None))
    except Exception as e:
        print(m, "MISSING_OR_FAILED", repr(e))
PY
python - <<'PY'
import vllm.utils.flashinfer as fi
for name in ["has_flashinfer", "has_flashinfer_sparse_mla_sm120", "has_flashinfer_b12x_gemm", "has_flashinfer_b12x_moe"]:
    obj = getattr(fi, name, None)
    print(name, obj() if obj else "MISSING")
PY
python -m pip show flashinfer-python flashinfer-cubin
python -m pip index versions flashinfer-python
python -m pip index versions flashinfer-cubin
python -m pip index versions flashinfer-jit-cache
rg -n "has_flashinfer_sparse_mla_sm120|FLASHINFER_MLA_SPARSE_SM120|attention_backend|mla_prefill_backend" externals/vllm_glm52_v025/vllm
git -C externals/vllm_glm52_v025 show HEAD:vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py
git -C externals/vllm_glm52_v025 show HEAD:vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py
vllm --help
vllm serve --help
vllm serve --help=AttentionConfig
vllm serve --help=attention-backend
vllm serve --help=all
find externals/sparse_mla_sm120 -maxdepth 3 -type f | sort | head -200
rg -n "sparse_mla_sm120_paged_attention|BatchSparseMLAPagedAttentionWrapper|192|576|trtllm_batch_decode" externals/sparse_mla_sm120
```

## Logs

No `logs/flashinfer_install/<timestamp>/install.log` was created because no install was necessary or performed.

Relevant prior logs from the h192 experiment:

- Native generation through patched path: `logs/sparse_mla_sm120_build/20260723T030132Z/glm_native_smoke.log`
- Standalone leavelet h192 rejection: `logs/sparse_mla_sm120_build/20260723T020227Z/tiny_glm_shape_test.log`

## Known Blockers

- Base vLLM h192 support blocker: unpatched `FLASHINFER_MLA_SPARSE_SM120` supports head size `[576]`, not tiny GLM `192`.
- Shape blocker after any simple whitelist: tiny GLM h192/v128/indexer dimensions differ from the V32 assumptions in vLLM and leavelet source.
- Git state blocker from prior task remains: local branch is ahead of origin and push credentials are unavailable in this session.
