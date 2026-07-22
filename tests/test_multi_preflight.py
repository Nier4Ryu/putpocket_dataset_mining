from __future__ import annotations

import unittest

from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.multi import validate_gpu_slots


def base_config() -> dict:
    return {
        "profiles": {
            "full_server": {"num_workers": 3},
            "debug": {"num_workers": 1},
        },
        "gpu": {
            "allowed_cuda_devices": [0, 1, 2],
            "full_server_slots": [[0], [1], [2]],
            "debug_slots": [[0]],
        },
    }


class MultiPreflightTests(unittest.TestCase):
    def test_full_server_slots_are_valid(self) -> None:
        self.assertEqual(validate_gpu_slots(base_config(), "full_server"), [[0], [1], [2]])

    def test_rejects_disallowed_gpu(self) -> None:
        config = base_config()
        config["gpu"]["debug_slots"] = [[3]]
        with self.assertRaises(ConfigError):
            validate_gpu_slots(config, "debug")

    def test_rejects_overlapping_gpu_slots(self) -> None:
        config = base_config()
        config["gpu"]["full_server_slots"] = [[0], [1], [1]]
        with self.assertRaises(ConfigError):
            validate_gpu_slots(config, "full_server")


if __name__ == "__main__":
    unittest.main()
