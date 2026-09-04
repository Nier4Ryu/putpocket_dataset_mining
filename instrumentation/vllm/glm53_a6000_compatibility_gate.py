"""Static capability marker for the PutPocket GLM-5.3 A6000 image.

This module deliberately has no torch or CUDA import.  It is installed in the
pinned vLLM wheel so an image audit can distinguish an intentionally blocked
GLM-5.3 package from a runnable sparse-MLA implementation.
"""

SCHEMA_VERSION = 1
TARGET_COMPUTE_CAPABILITY = "8.6"
VLLM_VERSION = "0.29.0.dev"
VLLM_COMMIT = "9cd956c7e6cf54efa366b803cafa15ec6c2df827"
MODEL_ID = "zai-org/GLM-5.3-Flash-BF16"
MODEL_REVISION = "a5b45eb41df6402735dedc900be14a42e8d5e538"
MODEL_ARCHITECTURE = "Glm5NextForConditionalGeneration"
RUNTIME_SUPPORTED = False
BLOCK_REASON_CODES = (
    "official_sparse_mla_backend_requires_hopper_or_newer",
    "no_official_sm86_glm5next_indexer_kernel",
)
PUTPOCKET_COMPONENTS_PACKAGED = (
    "native_compressed_pool_indexer_capture_source",
    "selector_provenance_and_digest_validation",
    "stateful_edit_v3_proxy_and_harness",
    "full_target_prefill_then_donor_overwrite_accuracy_ablation_source",
)
STATEFUL_EDIT_DEFAULT_ENABLED = False
STATEFUL_EDIT_TRUE_PARTIAL_PREFILL = False
MLA_INDEXER_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43)
KDA_LAYER_COUNT = 34
INDEX_KPOOL = 4


def capability() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "target_compute_capability": TARGET_COMPUTE_CAPABILITY,
        "vllm_version": VLLM_VERSION,
        "vllm_commit": VLLM_COMMIT,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_architecture": MODEL_ARCHITECTURE,
        "runtime_supported": RUNTIME_SUPPORTED,
        "block_reason_codes": list(BLOCK_REASON_CODES),
        "putpocket_components_packaged": list(PUTPOCKET_COMPONENTS_PACKAGED),
        "stateful_edit_default_enabled": STATEFUL_EDIT_DEFAULT_ENABLED,
        "stateful_edit_true_partial_prefill": STATEFUL_EDIT_TRUE_PARTIAL_PREFILL,
        "mla_indexer_layers": list(MLA_INDEXER_LAYERS),
        "kda_layer_count": KDA_LAYER_COUNT,
        "index_kpool": INDEX_KPOOL,
    }
