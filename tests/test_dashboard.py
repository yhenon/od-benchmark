from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from odbench_outer.dashboard import RunStore, build_run_view


def write_events(path: Path, events: list[dict[str, object]], *, partial: bool = False) -> None:
    path.parent.mkdir(parents=True)
    text = "".join(json.dumps(event) + "\n" for event in events)
    if partial:
        text += '{"type":"model_response"'
    path.write_text(text, encoding="utf-8")


class DashboardTests(unittest.TestCase):
    def test_build_run_view_aggregates_live_budgets_scores_and_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runs" / "run-live" / "events.jsonl"
            write_events(
                path,
                [
                    {
                        "type": "run_started",
                        "recorded_at": 100.0,
                        "run_id": "run-live",
                        "task_id": "cifar10",
                        "requested_model": "provider/model",
                        "limits": {"max_turns": 20, "max_total_tokens": 10_000},
                        "run_context": {
                            "objective": {"metric": "accuracy", "mode": "maximize"}
                        },
                    },
                    {
                        "type": "model_response",
                        "recorded_at": 101.0,
                        "turn": 1,
                        "usage": {"total_tokens": 120, "cost": 0.01},
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "train_start",
                                        "arguments": json.dumps(
                                            {"entrypoint": "train.py", "budget_seconds": 30}
                                        ),
                                    },
                                }
                            ],
                        },
                    },
                    {
                        "type": "tool_result",
                        "recorded_at": 110.0,
                        "turn": 1,
                        "tool_call_id": "call-1",
                        "name": "train_start",
                        "result": {
                            "ok": True,
                            "result": {
                                "type": "train_epoch_complete",
                                "epoch": 1,
                                "evaluation": {"metrics": {"accuracy": 0.72}},
                                "best_candidate": {
                                    "candidate_id": "candidate-0001",
                                    "metric": "accuracy",
                                    "score": 0.72,
                                },
                                "training_budget": {
                                    "active_seconds_used": 9.5,
                                    "active_seconds_limit": 60.0,
                                    "starts_used": 1,
                                    "starts_limit": 2,
                                    "active_jobs": ["job-one"],
                                },
                                "evaluation_budget": {"used": 1, "limit": 5, "remaining": 4},
                                "training_hardware": {
                                    "id": "gpu-one",
                                    "accelerator": "cuda",
                                },
                                "agent_budget": {
                                    "turns_limit": 20,
                                    "tokens_limit": 10_000,
                                },
                            },
                        },
                    },
                ],
                partial=True,
            )

            view = build_run_view(path)

        self.assertIsNotNone(view)
        assert view is not None
        self.assertTrue(view["is_running"])
        self.assertEqual(view["usage"]["tokens"], 120)
        self.assertEqual(view["usage"]["tool_calls"], 1)
        self.assertEqual(view["training"]["seconds_used"], 9.5)
        self.assertEqual(view["evaluation"]["used"], 1)
        self.assertEqual(view["score_points"][0]["score"], 0.72)
        self.assertEqual(view["timeline"][0]["arguments"]["entrypoint"], "train.py")
        self.assertEqual(view["best_candidate"]["candidate_id"], "candidate-0001")

    def test_completed_run_uses_final_totals_and_submission_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            path = root / "run-done" / "events.jsonl"
            write_events(
                path,
                [
                    {
                        "type": "run_started",
                        "recorded_at": 100.0,
                        "run_id": "run-done",
                        "task_id": "task",
                        "requested_model": "model",
                        "limits": {"max_turns": 10},
                        "run_context": {
                            "objective": {"metric": "AP", "mode": "maximize"}
                        },
                    },
                    {
                        "type": "run_finished",
                        "recorded_at": 160.0,
                        "status": "submitted",
                        "turns": 4,
                        "total_tokens": 900,
                        "total_cost": 0.2,
                    },
                ],
            )
            (path.parent / "submission-result.json").write_text(
                json.dumps({"evaluation": {"metrics": {"AP": 0.31}}}),
                encoding="utf-8",
            )

            view = build_run_view(path)
            listed = RunStore(root).list()

        self.assertIsNotNone(view)
        assert view is not None
        self.assertFalse(view["is_running"])
        self.assertEqual(view["status"], "submitted")
        self.assertEqual(view["elapsed_seconds"], 60.0)
        self.assertEqual(view["usage"]["tokens"], 900)
        self.assertEqual(view["submitted_score"], {"metric": "AP", "score": 0.31})
        self.assertEqual(listed[0]["run_id"], "run-done")

    def test_store_rejects_paths_outside_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore(Path(temporary) / "runs")
            self.assertIsNone(store.get("../secret"))

    def test_nonzero_workspace_exit_is_counted_as_a_failed_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runs" / "run-failed" / "events.jsonl"
            write_events(
                path,
                [
                    {
                        "type": "run_started",
                        "recorded_at": 100.0,
                        "run_id": "run-failed",
                        "run_context": {},
                    },
                    {
                        "type": "tool_result",
                        "recorded_at": 101.0,
                        "turn": 1,
                        "name": "workspace_exec",
                        "result": {
                            "ok": True,
                            "result": {"exit_code": 2, "duration_ms": 5},
                        },
                    },
                ],
            )

            view = build_run_view(path)

        self.assertIsNotNone(view)
        assert view is not None
        self.assertEqual(view["usage"]["tool_errors"], 1)
        self.assertFalse(view["timeline"][0]["ok"])
        self.assertIn("exit 2", view["timeline"][0]["summary"])


if __name__ == "__main__":
    unittest.main()
