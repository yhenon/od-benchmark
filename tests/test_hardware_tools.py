from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from odbench_outer.hardware import HardwareProfile, HardwareTarget
from odbench_outer.tools import ToolInvocationError, ToolRuntime


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeSandbox:
    max_command_seconds = 120
    container = "sandbox-test"

    def __init__(self, model: bytes = b"onnx") -> None:
        self.model = model
        self.copied_path: str | None = None

    def copy_file(self, relative: str, destination: Path, *, max_bytes: int) -> None:
        self.copied_path = relative
        if len(self.model) > max_bytes:
            raise AssertionError("test model exceeds configured limit")
        destination.write_bytes(self.model)


class FakeVerifier:
    def __init__(self, *, passed: bool = True, duration: float = 0.09) -> None:
        self.passed = passed
        self.duration = duration
        self.calls: list[dict[str, object]] = []

    def verify(
        self,
        model: Path,
        *,
        allowed_runtime_seconds: float,
        acceptance_mode: str,
        report_directory: Path,
    ) -> dict[str, object]:
        report_directory.mkdir(parents=True, exist_ok=False)
        self.calls.append(
            {
                "model": model.read_bytes(),
                "allowed": allowed_runtime_seconds,
                "mode": acceptance_mode,
            }
        )
        report = {
            "schema_version": 1,
            "type": "hardware_verification",
            "passed": self.passed,
            "stage": "complete",
            "duration_seconds": self.duration,
            "allowed_runtime_seconds": allowed_runtime_seconds,
            "generation_report": "large compiler transcript",
            "profile_report": "large profiler transcript",
        }
        (report_directory / "report.json").write_text(json.dumps(report))
        return report

    def analyze(
        self,
        model: Path,
        *,
        report_directory: Path,
    ) -> dict[str, object]:
        report_directory.mkdir(parents=True, exist_ok=False)
        self.calls.append({"model": model.read_bytes(), "mode": "analysis"})
        report = {
            "schema_version": 1,
            "type": "hardware_analysis",
            "compiled": True,
            "stage": "complete",
            "accelerator_mapping": {"accelerator_epoch_percent": 100.0},
            "generation_report": "large compiler transcript",
        }
        (report_directory / "report.json").write_text(json.dumps(report))
        return report


def make_runtime(root: Path, verifier: FakeVerifier) -> tuple[ToolRuntime, FakeSandbox]:
    labels = root / "labels.json"
    labels.write_text("{}", encoding="utf-8")
    sandbox = FakeSandbox()
    runtime = ToolRuntime(
        repo_root=root,
        sandbox=sandbox,
        labels=labels,
        run_id="run-test",
        run_directory=root / "run",
        dataset="cifar10",
        max_evaluations=2,
        max_train_starts=1,
        max_train_job_seconds=1,
        max_total_train_seconds=1,
        max_onnx_bytes=16 * 1024 * 1024,
        trainer_image="trainer:test",
        evaluator_image="evaluator:test",
        training_hardware=HardwareProfile.load(
            REPO_ROOT / "hardware" / "training" / "local-cpu.json"
        ),
        target_hardware=HardwareTarget.load(
            REPO_ROOT / "hardware" / "targets" / "nucleo-n657x0-q.json"
        ),
        max_inference_runtime_seconds=0.05,
        hardware_verifier=verifier,
        objective_metric="top1_accuracy",
        objective_mode="maximize",
    )
    return runtime, sandbox


def make_submission(root: Path) -> Path:
    root.mkdir()
    (root / "model.onnx").write_bytes(b"onnx")
    (root / "submission.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact": {"format": "onnx", "path": "model.onnx"},
            }
        ),
        encoding="utf-8",
    )
    return root


class HardwareToolTests(unittest.TestCase):
    def test_agent_context_exposes_task_runtime_not_profile_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime, _ = make_runtime(Path(temporary), FakeVerifier())
            target = runtime.agent_context()["target_hardware"]
        self.assertEqual(target["runtime_seconds"], 0.05)
        self.assertEqual(target["submission_runtime_seconds"], 0.0525)
        self.assertEqual(target["profile_default_runtime_seconds"], 0.005)

    def test_analyze_for_hw_compiles_without_flashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            verifier = FakeVerifier()
            runtime, sandbox = make_runtime(root, verifier)
            result = runtime.analyze_for_hw({"model_path": "candidate/model.onnx"})
            full_report = json.loads(
                (root / "run" / result["report_path"]).read_text(encoding="utf-8")
            )
        self.assertTrue(result["compiled"])
        self.assertEqual(sandbox.copied_path, "candidate/model.onnx")
        self.assertEqual(verifier.calls[0]["mode"], "analysis")
        self.assertEqual(result["analysis_call"], 1)
        self.assertNotIn("generation_report", result)
        self.assertEqual(full_report["generation_report"], "large compiler transcript")

    def test_verify_on_hw_uses_strict_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            verifier = FakeVerifier()
            runtime, sandbox = make_runtime(root, verifier)
            result = runtime.verify_on_hw({"model_path": "candidate/model.onnx"})
            full_report = json.loads(
                (root / "run" / result["report_path"]).read_text(encoding="utf-8")
            )
        self.assertTrue(result["passed"])
        self.assertEqual(sandbox.copied_path, "candidate/model.onnx")
        self.assertEqual(verifier.calls[0]["allowed"], 0.05)
        self.assertEqual(verifier.calls[0]["mode"], "strict")
        self.assertNotIn("generation_report", result)
        self.assertNotIn("profile_report", result)
        self.assertEqual(full_report["profile_report"], "large profiler transcript")

    def test_submission_rejects_failed_hardware_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            verifier = FakeVerifier(passed=False, duration=0.106)
            runtime, _ = make_runtime(root, verifier)
            staged = make_submission(root / "staged")
            with (
                patch.object(runtime, "_evaluate_directory") as evaluate,
                self.assertRaisesRegex(ToolInvocationError, "hardware verification"),
            ):
                runtime._accept_staged_submission(
                    staged, automatic=False, reason="test"
                )
            self.assertFalse(evaluate.called)
            self.assertFalse(runtime.submitted)
            self.assertEqual(runtime.evaluation_count, 0)
        self.assertEqual(verifier.calls[0]["allowed"], 0.0525)
        self.assertEqual(verifier.calls[0]["mode"], "submission")

    def test_submission_records_passing_hardware_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            verifier = FakeVerifier(passed=True, duration=0.101)
            runtime, _ = make_runtime(root, verifier)
            staged = make_submission(root / "staged")
            with patch.object(
                runtime,
                "_evaluate_directory",
                return_value={"metrics": {"top1_accuracy": 0.8}},
            ):
                result = runtime._accept_staged_submission(
                    staged, automatic=False, reason="test"
                )
            self.assertTrue(result["hardware_verification"]["passed"])
            self.assertNotIn("generation_report", result["hardware_verification"])
            self.assertIn("report_path", result["hardware_verification"])
            self.assertTrue(runtime.submitted)
            self.assertTrue((root / "run" / "submission" / "model.onnx").is_file())
            self.assertTrue((root / "run" / "submission-result.json").is_file())


if __name__ == "__main__":
    unittest.main()
