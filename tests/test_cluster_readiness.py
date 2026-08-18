from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from putpocket_dataset_mining.cluster_config import load_cluster_profile
from putpocket_dataset_mining.cluster_readiness import FailureClass, ReadinessProbe, run_readiness


ROOT = Path(__file__).resolve().parents[1]
ALLOCATION = {
    "SLURM_JOB_ID": "1234",
    "SLURM_JOB_NODELIST": "gpu01",
    "SLURM_JOB_NUM_NODES": "1",
    "SLURM_JOB_NAME": "synthetic-test",
}


class FakeProbe(ReadinessProbe):
    def __init__(self, *, gpu_rows: str, imports: dict | None = None) -> None:
        self.gpu_rows = gpu_rows
        self.imports = imports
        self.gpu_calls = 0
        self.import_calls = 0

    def gpu_inventory(self, nvidia_smi_executable, allocated_devices=None):  # type: ignore[no-untyped-def]
        self.gpu_calls += 1
        return 0, self.gpu_rows, ""

    def import_expectations(self, expected):  # type: ignore[no-untyped-def]
        self.import_calls += 1
        if self.imports is not None:
            return self.imports
        return {name: {"imported": True, "missing": []} for name in expected}


def h200_rows(count: int) -> str:
    return "".join("NVIDIA H200, 9.0\n" for _ in range(count))


def checkpoint(root: Path, quantization: str = "nvfp4") -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "glm", "quantization_config": {"quant_method": quantization}}),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "model-00001-of-00001.safetensors").write_bytes(b"synthetic-metadata-only")
    return root


class ClusterReadinessTests(unittest.TestCase):
    def test_static_stage_never_queries_gpu_or_imports(self) -> None:
        probe = FakeProbe(gpu_rows="")
        report = run_readiness(
            load_cluster_profile("glm52_nvfp4_tp1_pcp4_ep"),
            stage="static",
            probe=probe,
            repo_root=ROOT,
        )
        self.assertTrue(report.succeeded)
        self.assertEqual(probe.gpu_calls, 0)
        self.assertEqual(probe.import_calls, 0)

    def test_allocation_failure_is_classified_without_gpu_query(self) -> None:
        probe = FakeProbe(gpu_rows="")
        report = run_readiness(
            load_cluster_profile("glm52_nvfp4_tp1_pcp4_ep"),
            stage="gpu",
            env={},
            probe=probe,
            repo_root=ROOT,
        )
        self.assertFalse(report.succeeded)
        self.assertEqual(report.checks[-1].failure_class, FailureClass.NOT_IN_SLURM_ALLOCATION)
        self.assertEqual(probe.gpu_calls, 0)

    def test_gpu_count_and_type_failures_are_distinct(self) -> None:
        profile = load_cluster_profile("glm52_nvfp4_tp1_pcp4_ep")
        count = run_readiness(
            profile,
            stage="gpu",
            env=ALLOCATION,
            probe=FakeProbe(gpu_rows=h200_rows(2)),
            nvidia_smi_executable="/usr/bin/nvidia-smi",
            repo_root=ROOT,
        )
        self.assertEqual(count.checks[-1].failure_class, FailureClass.GPU_COUNT_MISMATCH)
        wrong_type = run_readiness(
            profile,
            stage="gpu",
            env=ALLOCATION,
            probe=FakeProbe(gpu_rows="".join("NVIDIA A100, 8.0\n" for _ in range(4))),
            nvidia_smi_executable="/usr/bin/nvidia-smi",
            repo_root=ROOT,
        )
        self.assertEqual(wrong_type.checks[-1].failure_class, FailureClass.GPU_TYPE_MISMATCH)

    def test_import_and_symbol_failures_are_classified(self) -> None:
        profile = load_cluster_profile("glm52_nvfp4_tp1_pcp4_ep")
        imports = {name: {"imported": True, "missing": []} for name in profile.required_imports}
        imports["flashinfer"] = {"imported": False, "missing": ["__version__"], "error": "not installed"}
        missing_package = run_readiness(
            profile,
            stage="imports",
            env=ALLOCATION,
            probe=FakeProbe(gpu_rows=h200_rows(4), imports=imports),
            nvidia_smi_executable="/usr/bin/nvidia-smi",
            repo_root=ROOT,
        )
        self.assertEqual(missing_package.checks[-1].failure_class, FailureClass.PACKAGE_IMPORT_MISSING)
        imports["flashinfer"] = {"imported": True, "missing": ["__version__"]}
        missing_symbol = run_readiness(
            profile,
            stage="imports",
            env=ALLOCATION,
            probe=FakeProbe(gpu_rows=h200_rows(4), imports=imports),
            nvidia_smi_executable="/usr/bin/nvidia-smi",
            repo_root=ROOT,
        )
        self.assertEqual(missing_symbol.checks[-1].failure_class, FailureClass.PACKAGE_SYMBOL_MISSING)

    def test_checkpoint_layout_and_quantization_failures_are_classified(self) -> None:
        profile = load_cluster_profile("glm52_nvfp4_tp1_pcp4_ep")
        with tempfile.TemporaryDirectory() as tmp_name:
            invalid = Path(tmp_name) / "missing"
            report = run_readiness(
                profile,
                stage="checkpoint",
                model_path=invalid,
                env=ALLOCATION,
                probe=FakeProbe(gpu_rows=h200_rows(4)),
                nvidia_smi_executable="/usr/bin/nvidia-smi",
                repo_root=ROOT,
            )
            self.assertEqual(report.checks[-2].failure_class, FailureClass.CHECKPOINT_NOT_FOUND)
        with tempfile.TemporaryDirectory() as tmp_name:
            path = checkpoint(Path(tmp_name) / "checkpoint", "int8")
            report = run_readiness(
                profile,
                stage="checkpoint",
                model_path=path,
                env=ALLOCATION,
                probe=FakeProbe(gpu_rows=h200_rows(4)),
                nvidia_smi_executable="/usr/bin/nvidia-smi",
                repo_root=ROOT,
            )
            self.assertEqual(report.checks[-1].failure_class, FailureClass.QUANTIZATION_BACKEND_INCOMPATIBLE)

    def test_synthetic_success_reaches_generation_handoff_without_loading_model(self) -> None:
        profile = load_cluster_profile("glm52_nvfp4_tp1_pcp4_ep")
        with tempfile.TemporaryDirectory() as tmp_name:
            path = checkpoint(Path(tmp_name) / "checkpoint")
            report = run_readiness(
                profile,
                stage="generation-handoff",
                model_path=path,
                model_revision="exact-revision",
                env=ALLOCATION,
                probe=FakeProbe(gpu_rows=h200_rows(4)),
                nvidia_smi_executable="/usr/bin/nvidia-smi",
                repo_root=ROOT,
            )
        self.assertTrue(report.succeeded)
        self.assertEqual(report.status, "handoff_ready")
        self.assertEqual(report.checks[-1].name, "one_shot_generation_handoff")
        self.assertEqual(report.handoff["model_load"]["status"], "ready_not_executed")  # type: ignore[index]
        model_command = report.handoff["model_load"]["command"]  # type: ignore[index]
        self.assertIn("--prefill-context-parallel-size", model_command)
        self.assertIn("--enable-expert-parallel", model_command)
        self.assertNotIn("generate", " ".join(model_command))


if __name__ == "__main__":
    unittest.main()
