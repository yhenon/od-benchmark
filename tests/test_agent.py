from __future__ import annotations

import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from typing import Any

from odbench_outer.agent import AgentLoop
from odbench_outer.openrouter import Completion, ModelAPIError
from odbench_outer.tools import TOOL_DEFINITIONS


def tool_completion(
    name: str,
    arguments: dict[str, Any],
    call_id: str,
    *,
    reasoning: bool = False,
) -> Completion:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }
    if reasoning:
        message["reasoning_details"] = [
            {"type": "reasoning.summary", "format": "unknown", "summary": "inspect"}
        ]
    return Completion(
        message=message,
        model="test/model",
        finish_reason="tool_calls",
        usage={"total_tokens": 10, "cost": 0.001},
        response_id=call_id,
        raw={},
    )


class FakeClient:
    def __init__(self, responses: list[Completion | BaseException]) -> None:
        self.responses = deque(responses)
        self.requests: list[list[dict[str, Any]]] = []

    def complete(self, **arguments: Any) -> Completion:
        self.requests.append(json.loads(json.dumps(arguments["messages"])))
        response = self.responses.popleft()
        if isinstance(response, BaseException):
            raise response
        return response


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.submitted = False
        self.should_stop = False
        self.closed = False
        self.best_candidate = None

    def agent_context(self) -> dict[str, Any]:
        return {}

    def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        if name == "submit":
            self.submitted = True
            self.should_stop = True
        return {"name": name}

    def close(self) -> None:
        self.closed = True

    def submit_best_candidate(self, reason: str) -> None:
        return None


class AutoSubmitRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.best_candidate = {"score": 0.75}
        self.auto_reason: str | None = None

    def submit_best_candidate(self, reason: str) -> dict[str, Any]:
        self.auto_reason = reason
        self.submitted = True
        self.should_stop = True
        return {"type": "submission_accepted", "automatic": True}


class InterruptClient:
    def complete(self, **arguments: Any) -> Completion:
        raise KeyboardInterrupt


class BrokenRuntime(FakeRuntime):
    def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("broken local invariant")


class AgentLoopTests(unittest.TestCase):
    def test_encrypted_reasoning_data_is_replayed_but_not_logged(self) -> None:
        first = tool_completion(
            "workspace_exec", {"command": "pwd"}, "call-1", reasoning=True
        )
        first.message["reasoning_details"].append(
            {
                "type": "reasoning.encrypted",
                "format": "openai-responses-v1",
                "id": "reasoning-1",
                "data": "opaque-ciphertext",
            }
        )
        client = FakeClient(
            [first, tool_completion("submit", {"submission_dir": "submission"}, "submit")]
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary) / "run"
            AgentLoop(
                client=client,
                tool_runtime=FakeRuntime(),
                run_id="run-test",
                run_directory=run_directory,
                system_prompt="system",
                task="task",
            ).run()
            events = [
                json.loads(line)
                for line in (run_directory / "events.jsonl").read_text().splitlines()
            ]

        replayed_details = client.requests[1][2]["reasoning_details"]
        self.assertEqual(replayed_details[1]["data"], "opaque-ciphertext")
        logged_response = next(event for event in events if event["type"] == "model_response")
        logged_details = logged_response["message"]["reasoning_details"]
        self.assertNotIn("data", logged_details[1])
        self.assertEqual(logged_details[1]["id"], "reasoning-1")

    def test_tool_loop_preserves_reasoning_and_stops_on_submit(self) -> None:
        client = FakeClient(
            [
                tool_completion("workspace_exec", {"command": "pwd"}, "call-1", reasoning=True),
                tool_completion(
                    "workspace_apply_patch",
                    {"patch": "diff --git a/a b/a\n"},
                    "call-2",
                ),
                tool_completion("submit", {"submission_dir": "submission"}, "call-3"),
            ]
        )
        runtime = FakeRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            loop = AgentLoop(
                client=client,
                tool_runtime=runtime,
                run_id="run-test",
                run_directory=Path(temporary) / "run",
                system_prompt="system",
                task="task",
                max_turns=10,
            )
            result = loop.run()
            summary = json.loads((Path(temporary) / "run" / "summary.json").read_text())

        self.assertEqual(result.status, "submitted")
        self.assertTrue(result.submitted)
        self.assertEqual(result.turns, 3)
        self.assertEqual(result.total_tokens, 30)
        self.assertAlmostEqual(result.total_cost, 0.003)
        self.assertFalse(runtime.closed)
        self.assertEqual([name for name, _ in runtime.calls], [
            "workspace_exec",
            "workspace_apply_patch",
            "submit",
        ])
        self.assertIn("reasoning_details", client.requests[1][2])
        self.assertEqual(client.requests[1][3]["role"], "tool")
        self.assertTrue(summary["submitted"])

    def test_invalid_tool_json_is_returned_to_model(self) -> None:
        bad = Completion(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "bad",
                        "type": "function",
                        "function": {"name": "workspace_exec", "arguments": "{"},
                    }
                ],
            },
            model="test/model",
            finish_reason="tool_calls",
            usage={},
            response_id="bad",
            raw={},
        )
        done = Completion(
            message={"role": "assistant", "content": "cannot continue"},
            model="test/model",
            finish_reason="stop",
            usage={},
            response_id="done",
            raw={},
        )
        client = FakeClient([bad, done])
        runtime = FakeRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            result = AgentLoop(
                client=client,
                tool_runtime=runtime,
                run_id="run-test",
                run_directory=Path(temporary) / "run",
                system_prompt="system",
                task="task",
            ).run()
        tool_result = json.loads(client.requests[1][-1]["content"])
        self.assertFalse(tool_result["ok"])
        self.assertEqual(tool_result["error"], "invalid_tool_arguments")
        self.assertEqual(result.status, "finished_without_submission")

    def test_blank_response_is_corrected_and_retried(self) -> None:
        blank = Completion(
            message={"role": "assistant", "content": "  "},
            model="test/model",
            finish_reason="stop",
            usage={"total_tokens": 2},
            response_id="blank",
            raw={},
        )
        client = FakeClient(
            [blank, tool_completion("submit", {"submission_dir": "submission"}, "submit")]
        )
        runtime = FakeRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            result = AgentLoop(
                client=client,
                tool_runtime=runtime,
                run_id="run-test",
                run_directory=Path(temporary) / "run",
                system_prompt="system",
                task="task",
            ).run()
        self.assertEqual(result.status, "submitted")
        self.assertIn("blank", client.requests[1][-1]["content"])

    def test_remote_failure_is_retried_with_context(self) -> None:
        client = FakeClient(
            [
                ModelAPIError("temporary outage"),
                tool_completion("submit", {"submission_dir": "submission"}, "submit"),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = AgentLoop(
                client=client,
                tool_runtime=FakeRuntime(),
                run_id="run-test",
                run_directory=Path(temporary) / "run",
                system_prompt="system",
                task="task",
            ).run()
        self.assertEqual(result.status, "submitted")
        self.assertIn("failed", client.requests[1][-1]["content"])

    def test_parallel_tool_calls_are_rejected_before_execution(self) -> None:
        parallel = Completion(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "one",
                        "type": "function",
                        "function": {"name": "workspace_exec", "arguments": "{}"},
                    },
                    {
                        "id": "two",
                        "type": "function",
                        "function": {"name": "workspace_exec", "arguments": "{}"},
                    },
                ],
            },
            model="test/model",
            finish_reason="tool_calls",
            usage={},
            response_id="parallel",
            raw={},
        )
        client = FakeClient(
            [parallel, tool_completion("submit", {"submission_dir": "submission"}, "submit")]
        )
        runtime = FakeRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            result = AgentLoop(
                client=client,
                tool_runtime=runtime,
                run_id="run-test",
                run_directory=Path(temporary) / "run",
                system_prompt="system",
                task="task",
            ).run()
        self.assertEqual(result.status, "submitted")
        self.assertEqual([name for name, _ in runtime.calls], ["submit"])
        retry_messages = client.requests[1][-2:]
        self.assertTrue(all(message["role"] == "tool" for message in retry_messages))
        self.assertTrue(
            all("parallel_tool_calls_not_supported" in message["content"] for message in retry_messages)
        )

    def test_unexpected_local_failure_propagates(self) -> None:
        client = FakeClient(
            [tool_completion("workspace_exec", {"command": "pwd"}, "call-1")]
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "broken local invariant"):
                AgentLoop(
                    client=client,
                    tool_runtime=BrokenRuntime(),
                    run_id="run-test",
                    run_directory=Path(temporary) / "run",
                    system_prompt="system",
                    task="task",
                ).run()

    def test_tool_names_are_provider_portable(self) -> None:
        names = [tool["function"]["name"] for tool in TOOL_DEFINITIONS]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(name.replace("_", "").isalnum() for name in names))

    def test_turn_exhaustion_auto_submits_best_candidate(self) -> None:
        client = FakeClient(
            [tool_completion("workspace_exec", {"command": "pwd"}, "call-1")]
        )
        runtime = AutoSubmitRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary) / "run"
            result = AgentLoop(
                client=client,
                tool_runtime=runtime,
                run_id="run-test",
                run_directory=run_directory,
                system_prompt="system",
                task="task",
                max_turns=1,
            ).run()
            events = [
                json.loads(line)
                for line in (run_directory / "events.jsonl").read_text().splitlines()
            ]
        self.assertEqual(result.status, "auto_submitted_best")
        self.assertTrue(result.automatic_submission)
        self.assertEqual(runtime.auto_reason, "max_turns_exhausted")
        self.assertIn("auto_submission", [event["type"] for event in events])

    def test_keyboard_interrupt_writes_terminal_summary(self) -> None:
        runtime = FakeRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary) / "run"
            result = AgentLoop(
                client=InterruptClient(),
                tool_runtime=runtime,
                run_id="run-test",
                run_directory=run_directory,
                system_prompt="system",
                task="task",
            ).run()
            summary = json.loads((run_directory / "summary.json").read_text())
            events = [
                json.loads(line)
                for line in (run_directory / "events.jsonl").read_text().splitlines()
            ]
        self.assertEqual(result.status, "interrupted")
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(events[-1]["type"], "run_finished")
        self.assertFalse(runtime.closed)


if __name__ == "__main__":
    unittest.main()
