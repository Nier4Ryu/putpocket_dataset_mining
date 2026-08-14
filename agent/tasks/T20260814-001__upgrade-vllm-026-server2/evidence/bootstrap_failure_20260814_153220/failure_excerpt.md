# vLLM 0.26 Server-2 Bootstrap Failure Excerpt

Timestamp: 2026-08-14 15:32:20 KST

Command:

```bash
PUTPOCKET_BUILD_JOBS=8 \
MAX_JOBS=8 \
CMAKE_BUILD_PARALLEL_LEVEL=8 \
CARGO_BUILD_JOBS=8 \
NVCC_THREADS=1 \
CCACHE_NOHASHDIR=true \
./scripts/env/bootstrap_sr.sh --preset server2 --build-jobs 8 --force-vllm-build
```

Observed result:

```text
BOOTSTRAP_RC=1
BOOTSTRAP_WALL_SEC=2778
```

vLLM editable build:

```text
Built vllm @ file:///home/dyryu/putpocket_dataset_mining/externals/vllm
Prepared 1 package in 44m 42s
Installed 1 package in 3ms
vllm==0.26.0+cu129
```

LMCache editable build failed:

```text
RuntimeError: ('The detected CUDA version (%s) mismatches the version
that was used to compilePyTorch (%s). Please make sure to use the same
CUDA versions.', '12.9', '13.0')
```

Partial environment import state:

```text
torch OK 2.11.0+cu130
torch_cuda 13.0
vllm FAIL ImportError libcudart.so.12: cannot open shared object file
lmcache FAIL ModuleNotFoundError No module named 'lmcache'
externals/vllm 568afb3a13806beb53bb2e6bd518269357b237c0
externals/lmcache 72eb0e375bcf0739a45046433f46ee32be361656
```

Conclusion:

The source migration to clean upstream vLLM 0.26 was integrated, but the
canonical Server-2 runtime rebuild is blocked by a CUDA contract mismatch:
the resolved official torch 2.11 wheel for Python 3.13 reports CUDA 13.0,
while Server-2's native CUDA toolchain is 12.9.
