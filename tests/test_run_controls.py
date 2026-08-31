from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from odbench_outer.hardware import HardwareProfile
from odbench_outer.tools import ToolRuntime, ToolRuntimeError


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_runtime(root: Path) -> ToolRuntime:
    labels = root / "labels.json"
    labels.write_text("{}", encoding="utf-8")
    return ToolRuntime(
        repo_root=root,
        sandbox=SimpleNamespace(max_command_seconds=120, container="sandbox-test"),
        labels=labels,
        run_id="run-test",
        run_directory=root / "run",
        dataset="cifar10",
        max_evaluations=5,
        max_train_starts=2,
        max_train_job_seconds=60,
        max_total_train_seconds=100,
        max_onnx_bytes=16 * 1024 * 1024,
        trainer_image="trainer:test",
        evaluator_image="evaluator:test",
        training_hardware=HardwareProfile.load(
            REPO_ROOT / "hardware" / "training" / "local-cpu.json"
        ),
        objective_metric="top1_accuracy",
        objective_mode="maximize",
    )


def make_submission(root: Path, marker: bytes) -> Path:
    root.mkdir()
    (root / "model.onnx").write_bytes(marker)
    (root / "preprocess.py").write_text("def preprocess(image): return image\n")
    (root / "postprocess.py").write_text("def postprocess(outputs): return 0\n")
    (root / "submission.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact": {"format": "onnx", "path": "model.onnx"},
                "preprocess": "preprocess.py",
                "postprocess": "postprocess.py",
            }
        )
    )
    return root


class RunControlTests(unittest.TestCase):
    def test_run_owns_training_and_submission_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = make_runtime(root)
            self.assertEqual(runtime.jobs_root, (root / "run" / "training-jobs").resolve())
            self.assertEqual(
                runtime.submission_directory,
                (root / "run" / "submission").resolve(),
            )

    def test_agent_context_exposes_workspace_command_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = make_runtime(Path(temporary))
            workspace = runtime.agent_context()["workspace"]
        self.assertEqual(workspace["default_command_seconds"], 60.0)
        self.assertEqual(workspace["max_command_seconds"], 120)
        self.assertTrue(workspace["training_budget_independent"])

    def test_epoch_result_makes_paused_lifecycle_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = make_runtime(Path(temporary))
            runtime.active_jobs.add("job-a")
            runtime.pending_events["job-a"] = "epoch-000001"
            result = runtime._with_run_state(
                {
                    "type": "train_epoch_complete",
                    "job_id": "job-a",
                    "event_id": "epoch-000001",
                }
            )
        self.assertEqual(result["job_status"], "paused")
        self.assertEqual(result["required_next_action"], "train_continue_or_train_stop")
        self.assertFalse(result["new_train_start_allowed"])
        self.assertEqual(result["training_budget"]["active_jobs"], ["job-a"])

    def test_new_start_error_names_paused_job_and_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = make_runtime(Path(temporary))
            runtime.active_jobs.add("job-a")
            runtime.pending_events["job-a"] = "epoch-000007"
            with self.assertRaisesRegex(
                ToolRuntimeError,
                "job-a is paused at epoch-000007; call train_continue or train_stop",
            ):
                runtime.train_start({"entrypoint": "train.py"})

    def test_train_start_stages_a_published_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = make_runtime(root)
            checkpoint = root / "published-checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            workspace_path = ".odbench/training/job-a/epoch-000001/checkpoint.pt"
            runtime.published_checkpoints[workspace_path] = checkpoint
            with (
                patch.object(
                    runtime,
                    "_run_json",
                    return_value={"job_id": "job-b"},
                ) as run_json,
                patch.object(
                    runtime,
                    "_await_training",
                    return_value={
                        "type": "train_job_finished",
                        "job_id": "job-b",
                        "status": "completed",
                        "exit_code": 0,
                    },
                ),
            ):
                result = runtime.train_start(
                    {
                        "entrypoint": "train.py",
                        "checkpoint_path": workspace_path,
                    }
                )
        command = run_json.call_args.args[0]
        checkpoint_index = command.index("--resume-checkpoint")
        self.assertEqual(command[checkpoint_index + 1], str(checkpoint))
        self.assertEqual(result["job_status"], "completed")
        self.assertTrue(result["new_train_start_allowed"])

    def test_empty_checkpoint_path_starts_from_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = make_runtime(Path(temporary))
            with (
                patch.object(
                    runtime,
                    "_run_json",
                    return_value={"job_id": "job-a"},
                ) as run_json,
                patch.object(
                    runtime,
                    "_await_training",
                    return_value={
                        "type": "train_job_finished",
                        "job_id": "job-a",
                        "status": "completed",
                        "exit_code": 0,
                    },
                ),
            ):
                result = runtime.train_start(
                    {
                        "entrypoint": "train.py",
                        "checkpoint_path": "",
                    }
                )
        command = run_json.call_args.args[0]
        self.assertNotIn("--resume-checkpoint", command)
        self.assertEqual(result["job_status"], "completed")

    def test_train_start_rejects_unpublished_checkpoint_without_using_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = make_runtime(Path(temporary))
            with self.assertRaisesRegex(ToolRuntimeError, "returned by a prior epoch"):
                runtime.train_start(
                    {
                        "entrypoint": "train.py",
                        "checkpoint_path": "checkpoint.pt",
                    }
                )
            self.assertEqual(runtime.train_start_count, 0)

    def test_training_meter_counts_job_deltas_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = make_runtime(Path(temporary))
            runtime._account_training(
                {"job_id": "job-a", "metering": {"active_wall_seconds": 30}}
            )
            runtime._account_training(
                {"job_id": "job-a", "metering": {"active_wall_seconds": 50}}
            )
            runtime._account_training(
                {"job_id": "job-b", "metering": {"active_wall_seconds": 20}}
            )
            budget = runtime._training_budget()
        self.assertEqual(budget["active_seconds_used"], 70)
        self.assertEqual(budget["active_seconds_remaining"], 30)

    def test_training_start_limits_fail_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = make_runtime(Path(temporary))
            runtime.train_start_count = 2
            with self.assertRaisesRegex(ToolRuntimeError, "start budget exhausted"):
                runtime.train_start({"entrypoint": "train.py"})
            runtime.train_start_count = 0
            runtime.training_active_seconds = 100
            with self.assertRaisesRegex(ToolRuntimeError, "training-time budget exhausted"):
                runtime.train_start({"entrypoint": "train.py"})

    def test_best_candidate_is_copied_and_only_improves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = make_runtime(root)
            first = make_submission(root / "first", b"first")
            worse = make_submission(root / "worse", b"worse")
            best = make_submission(root / "best", b"best")
            runtime._consider_candidate(
                first, {"metrics": {"top1_accuracy": 0.7}}, source={"tool": "test"}
            )
            runtime._consider_candidate(
                worse, {"metrics": {"top1_accuracy": 0.6}}, source={"tool": "test"}
            )
            runtime._consider_candidate(
                best, {"metrics": {"top1_accuracy": 0.8}}, source={"tool": "test"}
            )
            recorded = json.loads((root / "run" / "best_candidate.json").read_text())
            copied = runtime.best_submission_path
        self.assertEqual(recorded["score"], 0.8)
        self.assertIsNotNone(copied)
        self.assertEqual(copied.name, "submission")


if __name__ == "__main__":
    unittest.main()
