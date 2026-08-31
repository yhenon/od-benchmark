from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from odbench_outer.sandbox import Sandbox


class SandboxTests(unittest.TestCase):
    def test_exec_reports_default_and_effective_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = Sandbox(
                repo_root=Path(temporary),
                container="test-container",
                dataset="cifar10",
                image="agent:test",
                max_command_seconds=120,
            )
            with patch.object(
                sandbox,
                "_run_tool",
                return_value={"exit_code": 0, "timed_out": False},
            ) as run_tool:
                result = sandbox.exec("pwd")

        self.assertEqual(result["requested_timeout_seconds"], 60.0)
        self.assertEqual(result["effective_timeout_seconds"], 60.0)
        self.assertEqual(run_tool.call_args.args[1]["timeout_seconds"], 60.0)

    def test_exec_reports_task_clamp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = Sandbox(
                repo_root=Path(temporary),
                container="test-container",
                dataset="cifar10",
                image="agent:test",
                max_command_seconds=120,
            )
            with patch.object(
                sandbox,
                "_run_tool",
                return_value={"exit_code": -9, "timed_out": True},
            ) as run_tool:
                result = sandbox.exec("sleep 999", 600)

        self.assertEqual(result["requested_timeout_seconds"], 600.0)
        self.assertEqual(result["effective_timeout_seconds"], 120)
        self.assertEqual(run_tool.call_args.args[1]["timeout_seconds"], 120)


if __name__ == "__main__":
    unittest.main()
