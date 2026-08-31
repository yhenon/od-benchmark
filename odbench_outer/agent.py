"""Durable OpenRouter tool-calling loop."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .openrouter import Completion, ModelAPIError
from .sandbox import SandboxInvocationError
from .tools import TOOL_DEFINITIONS, ToolInvocationError


class InvalidModelResponse(RuntimeError):
    """A recoverable response-shape failure produced by the model."""


class CompletionClient(Protocol):
    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        session_id: str,
        max_output_tokens: int,
        reasoning_effort: str | None = None,
    ) -> Completion: ...


@dataclass(frozen=True)
class AgentResult:
    run_id: str
    status: str
    turns: int
    total_tokens: int
    total_cost: float
    final_message: str | None
    submitted: bool
    automatic_submission: bool
    best_candidate: dict[str, Any] | None


class RunLog:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.events_path = directory / "events.jsonl"
        if self.events_path.exists():
            raise FileExistsError(f"run log already exists: {self.events_path}")

    def append(self, event: dict[str, Any]) -> None:
        value = {"recorded_at": time.time(), **event}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def write_summary(self, result: AgentResult) -> None:
        path = self.directory / "summary.json"
        temporary = self.directory / ".summary.json.tmp"
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(asdict(result), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def assistant_message_for_history(message: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    for field in ("tool_calls", "reasoning", "reasoning_details"):
        if field in message and message[field] is not None:
            result[field] = message[field]
    return result


def without_encrypted_reasoning_data(value: Any) -> Any:
    """Copy a model payload while omitting opaque encrypted reasoning blobs."""
    if isinstance(value, list):
        return [without_encrypted_reasoning_data(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: without_encrypted_reasoning_data(item)
        for key, item in value.items()
        if not (value.get("type") == "reasoning.encrypted" and key == "data")
    }


def validated_assistant_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise InvalidModelResponse("response does not contain an assistant message")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise InvalidModelResponse("assistant content must be text or null")
    tool_calls = message.get("tool_calls")
    if tool_calls is None:
        if not isinstance(content, str) or not content.strip():
            raise InvalidModelResponse("assistant response is blank")
        return assistant_message_for_history(message)
    if not isinstance(tool_calls, list):
        raise InvalidModelResponse("tool_calls must be an array")
    if not tool_calls:
        if not isinstance(content, str) or not content.strip():
            raise InvalidModelResponse("assistant response is blank")
        return assistant_message_for_history(message)

    call_ids: set[str] = set()
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise InvalidModelResponse("each tool call must be an object")
        call_id = tool_call.get("id")
        function = tool_call.get("function")
        if not isinstance(call_id, str) or not call_id:
            raise InvalidModelResponse("each tool call must have a non-empty id")
        if call_id in call_ids:
            raise InvalidModelResponse("tool call ids must be unique")
        call_ids.add(call_id)
        if (
            not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not function["name"]
        ):
            raise InvalidModelResponse("each tool call must name a function")
        if tool_call.get("type", "function") != "function":
            raise InvalidModelResponse("tool calls must have type function")
        if not isinstance(function.get("arguments"), str):
            raise InvalidModelResponse("tool call arguments must be encoded as JSON text")
    return assistant_message_for_history(message)


class AgentLoop:
    def __init__(
        self,
        *,
        client: CompletionClient,
        tool_runtime: Any,
        run_id: str,
        run_directory: Path,
        system_prompt: str,
        task: str,
        requested_model: str | None = None,
        task_id: str | None = None,
        max_turns: int = 200,
        max_output_tokens: int = 16_384,
        max_total_tokens: int | None = None,
        max_cost: float | None = None,
        reasoning_effort: str | None = None,
        max_model_retries: int = 3,
    ) -> None:
        if (
            isinstance(max_model_retries, bool)
            or not isinstance(max_model_retries, int)
            or max_model_retries < 0
        ):
            raise ValueError("max_model_retries must be a non-negative integer")
        self.client = client
        self.tool_runtime = tool_runtime
        self.run_id = run_id
        self.system_prompt = system_prompt
        self.task = task
        self.requested_model = requested_model
        self.task_id = task_id
        self.max_turns = max_turns
        self.max_output_tokens = max_output_tokens
        self.max_total_tokens = max_total_tokens
        self.max_cost = max_cost
        self.reasoning_effort = reasoning_effort
        self.max_model_retries = max_model_retries
        self.log = RunLog(run_directory)

    def _agent_budget(self, turn: int, total_tokens: int, total_cost: float) -> dict[str, Any]:
        return {
            "turns_used": turn,
            "turns_limit": self.max_turns,
            "turns_remaining": max(0, self.max_turns - turn),
            "tokens_used": total_tokens,
            "tokens_limit": self.max_total_tokens,
            "tokens_remaining": (
                None
                if self.max_total_tokens is None
                else max(0, self.max_total_tokens - total_tokens)
            ),
            "cost_used": total_cost,
            "cost_limit": self.max_cost,
            "cost_remaining": (
                None if self.max_cost is None else max(0.0, self.max_cost - total_cost)
            ),
        }

    def _tool_result(self, name: str, raw_arguments: Any) -> dict[str, Any]:
        try:
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("arguments must decode to an object")
        except Exception as error:
            return {
                "ok": False,
                "error": "invalid_tool_arguments",
                "message": str(error),
            }
        try:
            value = self.tool_runtime.invoke(name, arguments)
            return {"ok": True, "result": value}
        except (SandboxInvocationError, ToolInvocationError) as error:
            return {
                "ok": False,
                "error": type(error).__name__,
                "message": str(error),
            }

    def run(self) -> AgentResult:
        runtime_context = self.tool_runtime.agent_context()
        effective_context = {
            **runtime_context,
            "agent_limits": {
                "max_turns": self.max_turns,
                "max_total_tokens": self.max_total_tokens,
                "max_cost": self.max_cost,
            },
        }
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    self.system_prompt
                    + "\n\n## Effective run configuration\n\n"
                    + json.dumps(effective_context, indent=2, sort_keys=True)
                ),
            },
            {"role": "user", "content": self.task},
        ]
        total_tokens = 0
        total_cost = 0.0
        final_message: str | None = None
        failure_message: str | None = None
        status = "max_turns_exhausted"
        turns = 0
        consecutive_model_failures = 0
        automatic_submission = False

        def retry_model(message: str, correction: str | None) -> bool:
            nonlocal consecutive_model_failures, failure_message, status
            consecutive_model_failures += 1
            failure_message = message
            will_retry = consecutive_model_failures <= self.max_model_retries
            self.log.append(
                {
                    "type": "model_retry",
                    "turn": turns,
                    "failure": message,
                    "attempt": consecutive_model_failures,
                    "retry_limit": self.max_model_retries,
                    "will_retry": will_retry,
                }
            )
            if not will_retry:
                status = "model_interaction_failed"
                return False
            if correction is not None:
                messages.append({"role": "user", "content": correction})
            return True

        self.log.append(
            {
                "type": "run_started",
                "run_id": self.run_id,
                "requested_model": self.requested_model,
                "task_id": self.task_id,
                "system_prompt": self.system_prompt,
                "task": self.task,
                "limits": {
                    "max_turns": self.max_turns,
                    "max_output_tokens": self.max_output_tokens,
                    "max_total_tokens": self.max_total_tokens,
                    "max_cost": self.max_cost,
                    "max_model_retries": self.max_model_retries,
                },
                "run_context": effective_context,
            }
        )

        try:
            while turns < self.max_turns:
                try:
                    completion = self.client.complete(
                        messages=messages,
                        tools=TOOL_DEFINITIONS,
                        session_id=self.run_id,
                        max_output_tokens=self.max_output_tokens,
                        reasoning_effort=self.reasoning_effort,
                    )
                except ModelAPIError as error:
                    message = f"{type(error).__name__}: {str(error)[:4000]}"
                    if not retry_model(
                        message,
                        "The previous model request failed before producing a usable "
                        "response. Continue from the current state and try again.",
                    ):
                        break
                    continue

                turns += 1
                usage_tokens = completion.usage.get("total_tokens", 0)
                if isinstance(usage_tokens, int) and not isinstance(usage_tokens, bool):
                    total_tokens += usage_tokens
                usage_cost = completion.usage.get("cost", 0.0)
                if isinstance(usage_cost, (int, float)) and not isinstance(usage_cost, bool):
                    total_cost += float(usage_cost)
                failure_message = None
                budget_status = None
                if self.max_total_tokens is not None and total_tokens > self.max_total_tokens:
                    budget_status = "token_budget_exhausted"
                elif self.max_cost is not None and total_cost > self.max_cost:
                    budget_status = "cost_budget_exhausted"

                try:
                    assistant = validated_assistant_message(completion.message)
                except InvalidModelResponse as error:
                    self.log.append(
                        {
                            "type": "invalid_model_response",
                            "turn": turns,
                            "response_id": completion.response_id,
                            "model": completion.model,
                            "finish_reason": completion.finish_reason,
                            "usage": completion.usage,
                            "message": without_encrypted_reasoning_data(
                                completion.message
                            ),
                            "failure": str(error),
                        }
                    )
                    if budget_status is not None:
                        status = budget_status
                        break
                    if not retry_model(
                        f"InvalidModelResponse: {error}",
                        "Your previous response could not be processed: "
                        f"{error}. Reply with either one non-empty final message or "
                        "exactly one well-formed tool call.",
                    ):
                        break
                    continue

                messages.append(assistant)
                self.log.append(
                    {
                        "type": "model_response",
                        "turn": turns,
                        "response_id": completion.response_id,
                        "model": completion.model,
                        "finish_reason": completion.finish_reason,
                        "usage": completion.usage,
                        "message": without_encrypted_reasoning_data(assistant),
                    }
                )
                if budget_status is not None:
                    status = budget_status
                    break
                tool_calls = assistant.get("tool_calls")
                if not tool_calls:
                    consecutive_model_failures = 0
                    final_message = assistant["content"]
                    status = (
                        "completed"
                        if self.tool_runtime.submitted
                        else "finished_without_submission"
                    )
                    break

                if len(tool_calls) > 1:
                    parallel_error = {
                        "ok": False,
                        "error": "parallel_tool_calls_not_supported",
                        "message": "Only one tool call is allowed per response; retry serially.",
                    }
                    for tool_call in tool_calls:
                        function = tool_call["function"]
                        self.log.append(
                            {
                                "type": "tool_result",
                                "turn": turns,
                                "tool_call_id": tool_call["id"],
                                "name": function["name"],
                                "result": parallel_error,
                            }
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "name": function["name"],
                                "content": json.dumps(
                                    parallel_error, separators=(",", ":"), sort_keys=True
                                ),
                            }
                        )
                    if not retry_model(
                        "InvalidModelResponse: parallel tool calls are not supported",
                        None,
                    ):
                        break
                    continue

                consecutive_model_failures = 0
                tool_call = tool_calls[0]
                function = tool_call["function"]
                name = function["name"]
                tool_result = self._tool_result(name, function["arguments"])
                value = tool_result.get("result")
                if tool_result.get("ok") is True and isinstance(value, dict):
                    value["agent_budget"] = self._agent_budget(
                        turns, total_tokens, total_cost
                    )
                print(f"turn {turns}: {name} {tool_result}", file=sys.stderr, flush=True)
                self.log.append(
                    {
                        "type": "tool_result",
                        "turn": turns,
                        "tool_call_id": tool_call["id"],
                        "name": name,
                        "result": tool_result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": name,
                        "content": json.dumps(
                            tool_result, separators=(",", ":"), sort_keys=True
                        ),
                    }
                )
                if self.tool_runtime.should_stop:
                    status = "submitted"
                    break
        except KeyboardInterrupt:
            status = "interrupted"
            failure_message = "KeyboardInterrupt: run interrupted"
            self.log.append({"type": "run_interrupted", "message": failure_message})
        except Exception as error:
            self.log.append(
                {
                    "type": "local_error",
                    "message": f"{type(error).__name__}: {error}",
                }
            )
            raise

        if (
            not self.tool_runtime.submitted
            and status
            in {
                "max_turns_exhausted",
                "token_budget_exhausted",
                "cost_budget_exhausted",
                "finished_without_submission",
                "model_interaction_failed",
            }
        ):
            automatic_result = self.tool_runtime.submit_best_candidate(status)
            if automatic_result is not None:
                automatic_submission = True
                status = "auto_submitted_best"
                self.log.append({"type": "auto_submission", "result": automatic_result})

        result = AgentResult(
            run_id=self.run_id,
            status=status,
            turns=turns,
            total_tokens=total_tokens,
            total_cost=total_cost,
            final_message=final_message or failure_message,
            submitted=bool(self.tool_runtime.submitted),
            automatic_submission=automatic_submission,
            best_candidate=self.tool_runtime.best_candidate,
        )
        self.log.append({"type": "run_finished", **asdict(result)})
        self.log.write_summary(result)
        return result
