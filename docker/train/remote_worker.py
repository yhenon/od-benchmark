#!/usr/bin/env python3
"""SSH-facing trusted worker for a persistent remote Docker GPU host."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from . import controller
    from .remote_protocol import (
        CHECKPOINT_NAME,
        REQUEST_NAME,
        WORKSPACE_PREFIX,
        RemoteProtocolError,
        extract_transfer,
        write_directory,
    )
except ImportError:  # Direct execution on the SSH worker.
    import controller  # type: ignore[no-redef]
    from remote_protocol import (  # type: ignore[no-redef]
        CHECKPOINT_NAME,
        REQUEST_NAME,
        WORKSPACE_PREFIX,
        RemoteProtocolError,
        extract_transfer,
        write_directory,
    )


ALLOWED_OPERATIONS = {"start", "await", "continue", "stop", "status", "record", "fetch"}


def read_request(limit: int = 1024 * 1024) -> dict[str, Any]:
    encoded = sys.stdin.buffer.read(limit + 1)
    if len(encoded) > limit:
        raise RemoteProtocolError("remote request exceeds 1 MiB")
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise RemoteProtocolError("remote request is not valid JSON") from error
    if not isinstance(value, dict):
        raise RemoteProtocolError("remote request must be an object")
    return value


def request_string(request: dict[str, Any], name: str) -> str:
    value = request.get(name)
    if not isinstance(value, str) or not value:
        raise RemoteProtocolError(f"remote request field {name} must be a nonempty string")
    return value


def run_controller(arguments: list[str]) -> dict[str, Any]:
    parsed = controller.parser().parse_args(arguments)
    controller.validate_arguments(parsed)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        parsed.function(parsed)
    try:
        value = json.loads(output.getvalue())
    except json.JSONDecodeError as error:
        raise RemoteProtocolError("training controller returned invalid JSON") from error
    if not isinstance(value, dict):
        raise RemoteProtocolError("training controller returned a non-object")
    return value


def start() -> dict[str, Any]:
    root = controller.jobs_root()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".incoming-", dir=root.parent) as temporary:
        incoming = Path(temporary)
        extract_transfer(sys.stdin.buffer, incoming)
        request_path = incoming / REQUEST_NAME
        workspace = incoming.joinpath(*WORKSPACE_PREFIX.parts)
        if not request_path.is_file() or not workspace.is_dir():
            raise RemoteProtocolError("remote start payload is incomplete")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise RemoteProtocolError("remote start request must be an object")

        arguments = [
            "start",
            "--workspace",
            str(workspace),
            "--entrypoint",
            request_string(request, "entrypoint"),
            "--dataset",
            request_string(request, "dataset"),
            "--image",
            request_string(request, "image"),
            "--budget-seconds",
            str(request.get("budget_seconds")),
            "--max-files",
            str(request.get("max_files")),
            "--max-bytes",
            str(request.get("max_bytes")),
            "--max-onnx-bytes",
            str(request.get("max_onnx_bytes")),
            "--memory",
            request_string(request, "memory"),
            "--shm-size",
            request_string(request, "shm_size"),
            "--cpus",
            str(request.get("cpus")),
            "--pids",
            str(request.get("pids")),
        ]
        gpus = request.get("gpus")
        if gpus is not None:
            if not isinstance(gpus, str):
                raise RemoteProtocolError("remote GPU request must be a string or null")
            arguments.extend(["--gpus", gpus])
        environment = request.get("environment", [])
        if not isinstance(environment, list) or not all(
            isinstance(item, str) for item in environment
        ):
            raise RemoteProtocolError("remote environment must be an array of strings")
        for item in environment:
            arguments.extend(["--environment", item])
        checkpoint = incoming.joinpath(*CHECKPOINT_NAME.parts)
        if checkpoint.exists():
            arguments.extend(["--resume-checkpoint", str(checkpoint)])
        training_args = request.get("training_args", [])
        if not isinstance(training_args, list) or not all(
            isinstance(item, str) for item in training_args
        ):
            raise RemoteProtocolError("remote training arguments must be an array of strings")
        if training_args:
            arguments.extend(["--", *training_args])
        return run_controller(arguments)


def dispatch(operation: str) -> dict[str, Any] | None:
    if operation == "start":
        return start()
    request = read_request()
    job_id = request_string(request, "job_id")
    if operation == "await":
        return run_controller(["await", job_id, "--stage-only"])
    if operation == "continue":
        return run_controller(
            ["continue", job_id, request_string(request, "event_id")]
        )
    if operation == "stop":
        return run_controller(["stop", job_id])
    if operation == "status":
        return run_controller(["status", job_id])
    if operation == "record":
        event_id = request_string(request, "event_id")
        root = controller.job_root(job_id)
        return controller.record_event_evaluation(
            root,
            controller.read_state(root),
            event_id,
            evaluation=request.get("evaluation"),
            evaluation_error=request.get("evaluation_error"),
        )
    if operation == "fetch":
        event_id = request_string(request, "event_id")
        if not controller.EVENT_ID_PATTERN.fullmatch(event_id):
            raise RemoteProtocolError("invalid remote event id")
        stage = controller.job_root(job_id) / "staged" / event_id
        if not stage.is_dir():
            raise RemoteProtocolError("remote staged result does not exist")
        write_directory(sys.stdout.buffer, stage)
        return None
    raise RemoteProtocolError("unsupported remote operation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=sorted(ALLOWED_OPERATIONS))
    arguments = parser.parse_args()
    try:
        result = dispatch(arguments.operation)
    except (OSError, ValueError, RemoteProtocolError) as error:
        raise SystemExit(str(error)) from error
    if result is not None:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
