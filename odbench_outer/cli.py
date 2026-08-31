"""Run one prepared benchmark task with an OpenRouter model."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import uuid
from pathlib import Path

from .agent import AgentLoop
from .openrouter import OpenRouterClient
from .sandbox import Sandbox
from .task import PreparedTask, PreparedTaskError
from .tools import ToolRuntime


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = REPO_ROOT / ".odbench" / "prepared-tasks"
RUNS_ROOT = REPO_ROOT / "runs"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True, help="OpenRouter model identifier.")
    result.add_argument("--task", required=True, help="Prepared task identifier.")
    return result


def main() -> None:
    arguments = parser().parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    try:
        prepared = PreparedTask.load(TASKS_ROOT, arguments.task)
    except PreparedTaskError as error:
        raise SystemExit(str(error)) from error

    now = dt.datetime.now(dt.timezone.utc)
    run_id = f"run-{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_directory = RUNS_ROOT / run_id
    sandbox = Sandbox(
        repo_root=REPO_ROOT,
        container=f"odbench-agent-{run_id}",
        dataset=prepared.dataset,
        image=prepared.agent_image,
        max_command_seconds=prepared.max_command_seconds,
    )
    runtime = None
    try:
        sandbox.start()
        runtime = ToolRuntime(
            repo_root=REPO_ROOT,
            sandbox=sandbox,
            labels=prepared.labels,
            run_id=run_id,
            run_directory=run_directory,
            dataset=prepared.dataset,
            max_evaluations=prepared.max_evaluations,
            max_train_starts=prepared.max_train_starts,
            max_train_job_seconds=prepared.max_train_job_seconds,
            max_total_train_seconds=prepared.max_total_train_seconds,
            max_onnx_bytes=prepared.max_onnx_bytes,
            trainer_image=prepared.trainer_image,
            evaluator_image=prepared.evaluator_image,
            training_hardware=prepared.training_hardware,
            target_hardware=prepared.target_hardware,
            max_inference_runtime_seconds=prepared.max_inference_runtime_seconds,
            objective_metric=prepared.objective_metric,
            objective_mode=prepared.objective_mode,
        )
        result = AgentLoop(
            client=OpenRouterClient(
                api_key=api_key,
                model=arguments.model,
                request_timeout=prepared.model_request_timeout,
                max_retries=prepared.max_transport_retries,
            ),
            tool_runtime=runtime,
            run_id=run_id,
            run_directory=run_directory,
            system_prompt=prepared.system_prompt,
            task=prepared.task_prompt,
            requested_model=arguments.model,
            task_id=arguments.task,
            max_turns=prepared.max_agent_turns,
            max_output_tokens=prepared.max_output_tokens,
            max_total_tokens=prepared.max_total_tokens,
            max_cost=prepared.max_cost,
            reasoning_effort=prepared.reasoning_effort,
            max_model_retries=prepared.max_response_retries,
        ).run()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    finally:
        if runtime is not None:
            runtime.close()
        sandbox.stop()

    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    if not result.submitted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
