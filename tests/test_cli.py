from __future__ import annotations

import contextlib
import io
import unittest

from pathlib import Path

from odbench_outer.cli import RUNS_ROOT, parser


class CliTests(unittest.TestCase):
    def test_runs_are_written_to_the_gitignored_repository_directory(self) -> None:
        self.assertEqual(RUNS_ROOT, Path(__file__).resolve().parents[1] / "runs")

    def test_run_interface_only_accepts_model_and_task(self) -> None:
        arguments = parser().parse_args(
            ["--model", "provider/model", "--task", "cifar10"]
        )
        self.assertEqual(vars(arguments), {"model": "provider/model", "task": "cifar10"})
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser().parse_args(
                    [
                        "--model",
                        "provider/model",
                        "--task",
                        "cifar10",
                        "--max-cost",
                        "1",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
