from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from odbench_outer.hardware import HardwareProfile
from odbench_outer.tools import ToolRuntime, ToolRuntimeError


REPO_ROOT = Path(__file__).resolve().parents[1]


class EvaluationBudgetTests(unittest.TestCase):
    def runtime(self, root: Path, limit: int) -> ToolRuntime:
        labels = root / "labels.json"
        labels.write_text("{}", encoding="utf-8")
        return ToolRuntime(
            repo_root=root,
            sandbox=object(),
            labels=labels,
            run_id="run-test",
            run_directory=root / "run",
            dataset="cifar10",
            max_evaluations=limit,
            max_train_starts=1,
            max_train_job_seconds=1,
            max_total_train_seconds=1,
            max_onnx_bytes=16 * 1024 * 1024,
            trainer_image="trainer:test",
            evaluator_image="evaluator:test",
            training_hardware=HardwareProfile.load(
                REPO_ROOT / "hardware" / "training" / "local-cpu.json"
            ),
            objective_metric="top1_accuracy",
            objective_mode="maximize",
        )

    def test_nonfinal_calls_cannot_consume_submit_slot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self.runtime(Path(temporary), 2)
            self.assertEqual(runtime._evaluation_budget()["adaptive_remaining"], 1)
            runtime._reserve_evaluation(final=False)
            budget = runtime._evaluation_budget()
            self.assertEqual(budget["used"], 1)
            self.assertEqual(budget["remaining"], 1)
            with self.assertRaisesRegex(ToolRuntimeError, "reserved for submit"):
                runtime._reserve_evaluation(final=False)
            runtime._reserve_evaluation(final=True)
            self.assertEqual(runtime._evaluation_budget()["remaining"], 0)

    def test_terminal_training_wait_can_release_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self.runtime(Path(temporary), 3)
            runtime._reserve_evaluation(final=False)
            runtime._release_evaluation_reservation()
            self.assertEqual(runtime._evaluation_budget()["used"], 0)

    def test_one_slot_budget_allows_only_submit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self.runtime(Path(temporary), 1)
            with self.assertRaisesRegex(ToolRuntimeError, "reserved for submit"):
                runtime._reserve_evaluation(final=False)
            runtime._reserve_evaluation(final=True)


if __name__ == "__main__":
    unittest.main()
