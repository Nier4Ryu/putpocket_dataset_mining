from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.glm_backend_routing import select_glm_sparse_mla_backend


TINY = {
    "model_type": "glm_moe_dsa",
    "architectures": ["GlmMoeDsaForCausalLM"],
    "kv_lora_rank": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "index_n_heads": 8,
    "index_head_dim": 64,
}

FULL = {
    "model_type": "glm_moe_dsa",
    "architectures": ["GlmMoeDsaForCausalLM"],
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "index_n_heads": 8,
    "index_head_dim": 64,
}


class GlmBackendRoutingTests(unittest.TestCase):
    def test_sm120_exact_tiny_selects_custom(self) -> None:
        trace = select_glm_sparse_mla_backend(TINY, (12, 0))
        self.assertTrue(trace.selected_custom_kernel)
        self.assertEqual(trace.derived_head_size, 192)

    def test_sm120_h576_does_not_select_custom(self) -> None:
        trace = select_glm_sparse_mla_backend(FULL, (12, 0))
        self.assertFalse(trace.selected_custom_kernel)
        self.assertEqual(trace.derived_head_size, 576)

    def test_sm90_h576_selects_hopper_upstream(self) -> None:
        trace = select_glm_sparse_mla_backend(FULL, (9, 0))
        self.assertFalse(trace.selected_custom_kernel)
        self.assertEqual(trace.selected_backend, "upstream_hopper_sparse_mla_h576")

    def test_sm90_h192_does_not_select_sm120_custom(self) -> None:
        trace = select_glm_sparse_mla_backend(TINY, (9, 0))
        self.assertFalse(trace.selected_custom_kernel)

    def test_sm86_never_selects_custom(self) -> None:
        trace = select_glm_sparse_mla_backend(TINY, (8, 6))
        self.assertFalse(trace.selected_custom_kernel)

    def test_near_miss_dimensions_fail_closed(self) -> None:
        cfg = dict(TINY)
        cfg["index_head_dim"] = 32
        trace = select_glm_sparse_mla_backend(cfg, (12, 0))
        self.assertFalse(trace.selected_custom_kernel)

    def test_assert_no_tiny_kernel_raises(self) -> None:
        with patch.dict(os.environ, {"SR_ASSERT_NO_TINY_GLM_KERNEL": "1"}):
            with self.assertRaises(ConfigError):
                select_glm_sparse_mla_backend(TINY, (12, 0))


if __name__ == "__main__":
    unittest.main()
