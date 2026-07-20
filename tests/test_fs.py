import unittest
from pathlib import Path

from putpocket_dataset_mining.fs import safe_relative_path


class SafeRelativePathTests(unittest.TestCase):
    def test_workspace_prefix_is_normalized(self) -> None:
        self.assertEqual(safe_relative_path("workspace/solution.py"), Path("solution.py"))

    def test_absolute_workspace_prefix_is_normalized(self) -> None:
        self.assertEqual(safe_relative_path("/workspace/solution.py"), Path("solution.py"))

    def test_rejects_parent_traversal_after_normalization(self) -> None:
        with self.assertRaises(ValueError):
            safe_relative_path("/workspace/../solution.py")


if __name__ == "__main__":
    unittest.main()
