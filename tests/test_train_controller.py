from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from docker.train.controller import (
    command_stop,
    container_log_tails,
    metric_context,
    stage_resume_checkpoint,
)


class TrainControllerTests(unittest.TestCase):
    def test_stage_resume_checkpoint_uses_fixed_read_only_input_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pt"
            source.write_bytes(b"checkpoint data")
            input_root = root / "input"
            input_root.mkdir()
            metadata = stage_resume_checkpoint(source, input_root)
            staged = input_root / ".odbench_resume" / "checkpoint.pt"

            self.assertEqual(staged.read_bytes(), b"checkpoint data")
            self.assertEqual(metadata["input_path"], "/job/input/.odbench_resume/checkpoint.pt")
            self.assertEqual(metadata["bytes"], len(b"checkpoint data"))
            self.assertEqual(staged.stat().st_mode & 0o777, 0o444)

    def test_stage_resume_checkpoint_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pt"
            source.write_bytes(b"checkpoint")
            link = root / "link.pt"
            link.symlink_to(source)
            input_root = root / "input"
            input_root.mkdir()
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                stage_resume_checkpoint(link, input_root)

    def test_failed_job_log_tails_are_bounded(self) -> None:
        completed = SimpleNamespace(stdout="x" * 5000, stderr="traceback")
        with patch("docker.train.controller.subprocess.run", return_value=completed):
            tails = container_log_tails("job-container", limit=100)
        self.assertEqual(tails["stdout_tail"], "x" * 100)
        self.assertEqual(tails["stderr_tail"], "traceback")

    def test_metric_context_distinguishes_agent_diagnostics_from_hidden_evaluation(self) -> None:
        context = metric_context({"dataset": "cifar10", "split": "test"})

        self.assertFalse(
            context["train_metrics"]["directly_comparable_to_evaluation"]
        )
        self.assertEqual(
            context["evaluation.metrics"]["model_stage"],
            "published_onnx_submission",
        )
        self.assertEqual(context["evaluation.metrics"]["split"], "test")

    def test_stop_records_pending_epoch_as_last_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs = Path(temporary) / "jobs"
            job_id = "job-20260825T000000-1234abcd"
            root = jobs / job_id
            (root / "decisions").mkdir(parents=True)
            (root / "state.json").write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "container": "train-container",
                        "last_epoch": 3,
                        "pending": {
                            "event": {"event_id": "epoch-000020", "epoch": 20},
                            "notification": {},
                        },
                        "active_started_at": None,
                        "active_seconds": 10.0,
                        "budget_seconds": 30.0,
                        "status": "awaiting_decision",
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.dict(
                    "docker.train.controller.os.environ",
                    {"ODBENCH_JOBS_ROOT": str(jobs)},
                ),
                patch("docker.train.controller.run"),
                patch("docker.train.controller.time.sleep"),
                patch("builtins.print"),
            ):
                command_stop(SimpleNamespace(job_id=job_id))

            state = json.loads((root / "state.json").read_text(encoding="utf-8"))
            decision = json.loads(
                (root / "decisions" / "epoch-000020.json").read_text(encoding="utf-8")
            )

        self.assertEqual(state["last_epoch"], 20)
        self.assertEqual(state["status"], "stopped")
        self.assertIsNone(state["pending"])
        self.assertEqual(decision["action"], "stop")


if __name__ == "__main__":
    unittest.main()
