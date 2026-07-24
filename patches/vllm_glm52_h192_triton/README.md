# GLM h192 Triton vLLM Patch Artifact

This directory stores the current experimental diff against
`externals/vllm_glm52_v025` without adding the external checkout itself.

The patch makes vLLM select a Putpocket Triton h192/v128 sparse MLA decode path
for the tiny GLM-5.2 model on SM120 and includes the local vLLM h128 cache
packing changes used by the current environment.

Status: experimental. The standalone h192 Triton kernel and vLLM fp8 paged-cache
adapter pass PyTorch reference tests, and vLLM selects the path, but native GLM
generation still produces incoherent text. Do not treat this patch as a validated
working model path yet.
