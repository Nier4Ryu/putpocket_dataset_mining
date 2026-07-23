# GLM h192 SM120 Sparse MLA vLLM Patch

This directory records the local experiment applied to `externals/vllm_glm52_v025`.
The external checkout is intentionally not committed by the top-level repo.

Apply from `externals/vllm_glm52_v025`:

```bash
git apply ../../patches/vllm_glm52_h192_sm120/vllm_glm52_h192_sm120.patch
```

The patch is intentionally scoped to the separate vLLM 0.25 GLM stack. It does
not modify `externals/vllm` or the old Qwen/Putpocket-v0.19.1 path.
