from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from odbench_outer.plot import generate_plots, load_run


def write_events(path: Path, events: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )


def write_submission_result(
    path: Path,
    accuracy: float,
    *,
    metric: str = "top1_accuracy",
    runtime_seconds: float = 0.002,
    inferences_per_second: float = 500.0,
) -> None:
    path.write_text(
        json.dumps(
            {
                "evaluation": {"metrics": {metric: accuracy}},
                "hardware_verification": {
                    "duration_seconds": runtime_seconds,
                    "inferences_per_second": inferences_per_second,
                },
            }
        ),
        encoding="utf-8",
    )


class PlotTests(unittest.TestCase):
    def test_load_run_extracts_accepted_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runs" / "run-one" / "events.jsonl"
            write_events(
                path,
                [
                    {
                        "type": "run_started",
                        "run_id": "run-one",
                        "task_id": "task/a",
                        "requested_model": "provider/model",
                        "run_context": {
                            "objective": {"metric": "accuracy", "mode": "maximize"}
                        },
                    },
                    {
                        "type": "run_finished",
                        "turns": 10,
                        "total_tokens": 4_000,
                        "total_cost": 0.04,
                    },
                ],
            )
            write_submission_result(
                path.with_name("submission-result.json"), 0.7, metric="accuracy"
            )
            run = load_run(path)

        self.assertIsNotNone(run)
        assert run is not None
        self.assertEqual(run.task, "task/a")
        self.assertEqual(run.model, "provider/model")
        self.assertEqual(run.final_tokens, 4_000)
        self.assertEqual(run.final_cost, 0.04)
        self.assertEqual(run.submitted_score, 0.7)
        self.assertEqual(run.submitted_throughput, 500.0)

    def test_generate_plots_groups_runs_by_task_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            plots = root / "plots"
            icons = root / "icons"
            icons.mkdir()
            (icons / "gemini.png").write_bytes(b"test-png")
            plots.mkdir()
            for legacy_name in (
                "cifar10.svg",
                "cifar10-vs-tokens.svg",
                "cifar10-vs-cost.svg",
            ):
                (plots / legacy_name).write_text("stale", encoding="utf-8")
            common = {
                "type": "run_started",
                "task_id": "cifar10",
                "requested_model": "google/gemini-test",
                "run_context": {
                    "objective": {"metric": "top1_accuracy", "mode": "maximize"}
                },
            }
            write_events(
                runs / "run-one" / "events.jsonl",
                [
                    {**common, "run_id": "run-one"},
                    {
                        "type": "run_finished",
                        "turns": 6,
                        "total_tokens": 5_000,
                        "total_cost": 0.05,
                    },
                ],
            )
            write_submission_result(
                runs / "run-one" / "submission-result.json",
                0.79,
                runtime_seconds=0.001,
                inferences_per_second=1_000.0,
            )
            write_events(
                runs / "run-two" / "events.jsonl",
                [
                    {**common, "run_id": "run-two"},
                    {
                        "type": "run_finished",
                        "turns": 7,
                        "total_tokens": 6_000,
                        "total_cost": 0.06,
                    },
                ],
            )
            write_submission_result(
                runs / "run-two" / "submission-result.json",
                0.69,
                runtime_seconds=0.003,
                inferences_per_second=500.0,
            )

            written = generate_plots(runs, plots, icon_dir=icons)
            submitted_cost_svg = written[0].read_text(encoding="utf-8")
            submitted_token_svg = written[1].read_text(encoding="utf-8")
            model_results_svg = written[2].read_text(encoding="utf-8")
            remaining = {path.name for path in plots.iterdir()}

        self.assertEqual(
            [path.name for path in written],
            [
                "cifar10-submitted-vs-cost.svg",
                "cifar10-submitted-vs-tokens.svg",
                "cifar10-results-by-model.svg",
            ],
        )
        self.assertEqual(
            remaining,
            {
                "cifar10-submitted-vs-cost.svg",
                "cifar10-submitted-vs-tokens.svg",
                "cifar10-results-by-model.svg",
            },
        )
        self.assertIn("Submitted solution accuracy by cost", submitted_cost_svg)
        self.assertIn("Completed submissions: 2", submitted_cost_svg)
        self.assertIn("run-one — cost $0.050: 0.79", submitted_cost_svg)
        self.assertIn("run-two — cost $0.060: 0.69", submitted_cost_svg)
        self.assertEqual(submitted_cost_svg.count('<circle class="icon-frame"'), 2)
        self.assertEqual(submitted_cost_svg.count("<image "), 3)
        self.assertIn("Submitted solution accuracy by tokens", submitted_token_svg)
        self.assertIn("run-one — tokens 5,000: 0.79", submitted_token_svg)
        self.assertIn("run-two — tokens 6,000: 0.69", submitted_token_svg)
        self.assertEqual(submitted_token_svg.count('<circle class="icon-frame"'), 2)
        self.assertEqual(submitted_token_svg.count("<image "), 3)
        self.assertIn("Accepted Results by Model", model_results_svg)
        self.assertIn("Accuracy ↑", model_results_svg)
        self.assertIn("Throughput ↑ (relative to fastest)", model_results_svg)
        self.assertIn("0.74 acc · 750.00 inf/s", model_results_svg)
        self.assertIn("throughput 100.0% of fastest", model_results_svg)
        self.assertIn("11k tok · $0.110", model_results_svg)
        self.assertIn("2 accepted runs", model_results_svg)
        self.assertEqual(model_results_svg.count("<image "), 1)


if __name__ == "__main__":
    unittest.main()
