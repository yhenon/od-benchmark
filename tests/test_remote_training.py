from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docker.train.controller import record_event_evaluation
from docker.train.remote_client import ssh_command
from docker.train.remote_protocol import (
    RemoteProtocolError,
    extract_transfer,
    write_start_payload,
)


class RemoteTrainingTests(unittest.TestCase):
    def test_start_payload_round_trip_preserves_workspace_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "train.py").write_text("print('gpu')\n", encoding="utf-8")
            hooks = workspace / "hooks"
            hooks.mkdir()
            (hooks / "preprocess.py").write_bytes(b"hook")
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            payload = io.BytesIO()

            write_start_payload(
                payload,
                {"entrypoint": "train.py", "gpus": "all"},
                workspace,
                checkpoint,
            )
            payload.seek(0)
            extracted = root / "extracted"
            extracted.mkdir()
            extract_transfer(payload, extracted)

            request = json.loads((extracted / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(request["gpus"], "all")
            self.assertEqual(
                (extracted / "workspace" / "train.py").read_text(encoding="utf-8"),
                "print('gpu')\n",
            )
            self.assertEqual(
                (extracted / "workspace" / "hooks" / "preprocess.py").read_bytes(),
                b"hook",
            )
            self.assertEqual((extracted / "resume-checkpoint.pt").read_bytes(), b"checkpoint")

    def test_start_payload_rejects_workspace_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            target = root / "target"
            target.write_text("secret", encoding="utf-8")
            (workspace / "link").symlink_to(target)
            with self.assertRaisesRegex(RemoteProtocolError, "regular file"):
                write_start_payload(io.BytesIO(), {}, workspace, None)

    def test_remote_evaluation_is_recorded_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "results").mkdir()
            event = {"event_id": "epoch-000001", "epoch": 1}
            notification = {
                "type": "train_epoch_staged",
                "event_id": "epoch-000001",
                "evaluation": None,
                "evaluation_error": None,
            }
            state = {
                "pending": {"event": event, "notification": notification},
            }

            completed = record_event_evaluation(
                root,
                state,
                "epoch-000001",
                evaluation={"dataset": "cifar10", "metrics": {"top1_accuracy": 0.5}},
                evaluation_error=None,
            )
            repeated = record_event_evaluation(
                root,
                state,
                "epoch-000001",
                evaluation={"dataset": "cifar10", "metrics": {"top1_accuracy": 0.1}},
                evaluation_error=None,
            )

            self.assertEqual(completed["type"], "train_epoch_complete")
            self.assertEqual(repeated["evaluation"]["metrics"]["top1_accuracy"], 0.5)

    def test_ssh_worker_command_uses_configured_trusted_paths(self) -> None:
        with patch.dict(
            "docker.train.remote_client.os.environ",
            {
                "ODBENCH_TRAIN_HOST": "odbench@192.168.1.106",
                "ODBENCH_REMOTE_REPO": "/home/odbench/od-benchmark",
                "ODBENCH_REMOTE_JOBS_ROOT": "/var/lib/odbench/jobs",
                "ODBENCH_TRAIN_SSH_OPTIONS": "-o BatchMode=yes",
            },
            clear=True,
        ):
            command = ssh_command("status")

        self.assertEqual(command[:4], ["ssh", "-o", "BatchMode=yes", "odbench@192.168.1.106"])
        self.assertIn("remote_worker.py status", command[-1])
        self.assertIn("ODBENCH_JOBS_ROOT=/var/lib/odbench/jobs", command[-1])


if __name__ == "__main__":
    unittest.main()
