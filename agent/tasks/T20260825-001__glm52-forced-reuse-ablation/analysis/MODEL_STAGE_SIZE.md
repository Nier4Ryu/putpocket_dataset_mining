# Pinned checkpoint staging size audit

Metadata-only Hugging Face inspection on 2026-08-24 resolved
`nvidia/GLM-5.2-NVFP4` revision
`aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa` exactly.

- Weight files: 47 safetensors
- Weight bytes: 464,823,042,096 (432.89 GiB)
- All metadata-sized repository files: 464,874,323,992 bytes
- Largest shard: `model-00019-of-00047.safetensors`, 10,000,552,960 bytes

The initial 220 GiB free-space gate was therefore a false-low gate. The compute
launcher now requires 550 GiB free before an in-allocation download, leaving
about 117 GiB over the weight payload for the largest partial shard, metadata,
runtime caches, logs, and margin. It never stages on the login node.
