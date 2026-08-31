#!/usr/bin/env python3
"""Local SSH transport for the trusted remote GPU training worker."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

try:
    from . import controller
    from .remote_protocol import RemoteProtocolError, extract_transfer, write_start_payload
    from .workspace_archive import export_from_container
except ImportError:  # Direct execution via docker/trainer.
    import controller  # type: ignore[no-redef]
    from remote_protocol import (  # type: ignore[no-redef]
        RemoteProtocolError,
        extract_transfer,
        write_start_payload,
    )
    from workspace_archive import export_from_container  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]


class RemoteTrainingError(RuntimeError):
    pass


def remote_host() -> str:
    value = os.environ.get("ODBENCH_TRAIN_HOST", "")
    if not value or any(character.isspace() for character in value) or value.startswith("-"):
        raise RemoteTrainingError("ODBENCH_TRAIN_HOST is invalid")
    return value


def ssh_command(operation: str) -> list[str]:
    options = shlex.split(
        os.environ.get(
            "ODBENCH_TRAIN_SSH_OPTIONS",
            "-o BatchMode=yes -o ConnectTimeout=10",
        )
    )
    remote_repo = os.environ.get("ODBENCH_REMOTE_REPO", "/home/odbench/od-benchmark")
    remote_jobs = os.environ.get("ODBENCH_REMOTE_JOBS_ROOT", "/var/lib/odbench/jobs")
    worker = str(Path(remote_repo) / "docker" / "train" / "remote_worker.py")
    remote = (
        f"ODBENCH_JOBS_ROOT={shlex.quote(remote_jobs)} "
        f"python3 {shlex.quote(worker)} {shlex.quote(operation)}"
    )
    return ["ssh", *options, remote_host(), remote]


def run_remote_json(
    operation: str,
    request: dict[str, Any] | None = None,
    *,
    input_stream: BinaryIO | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    if input_stream is not None and request is not None:
        raise ValueError("provide request or input_stream, not both")
    encoded = None if request is None else json.dumps(request).encode("utf-8")
    result = subprocess.run(
        ssh_command(operation),
        input=encoded if input_stream is None else None,
        stdin=input_stream,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace")[-4000:]
        raise RemoteTrainingError(detail.strip() or f"remote {operation} failed")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        detail = result.stdout.decode("utf-8", errors="replace")[-4000:]
        raise RemoteTrainingError(f"remote {operation} returned invalid JSON: {detail}") from error
    if not isinstance(value, dict):
        raise RemoteTrainingError(f"remote {operation} returned a non-object")
    return value


def sanitized_workspace(arguments: Any, root: Path) -> Path:
    raw = root / "raw"
    workspace = root / "workspace"
    workspace.mkdir()
    if arguments.workspace is not None:
        source = arguments.workspace.resolve()
    else:
        raw.mkdir()
        export_from_container(
            arguments.agent_container,
            ".",
            raw,
            exclude_agent_state=True,
            max_files=arguments.max_files,
            max_bytes=arguments.max_bytes,
        )
        source = raw
    controller.snapshot_workspace(
        source,
        workspace,
        max_files=arguments.max_files,
        max_bytes=arguments.max_bytes,
    )
    return workspace


def command_start(arguments: Any) -> dict[str, Any]:
    image = arguments.image or f"od-benchmark-trainer:{arguments.dataset}-dev"
    training_args = list(arguments.training_args)
    if training_args[:1] == ["--"]:
        training_args = training_args[1:]
    request = {
        "entrypoint": arguments.entrypoint,
        "dataset": arguments.dataset,
        "image": image,
        "budget_seconds": arguments.budget_seconds,
        "max_files": arguments.max_files,
        "max_bytes": arguments.max_bytes,
        "max_onnx_bytes": arguments.max_onnx_bytes,
        "memory": arguments.memory,
        "shm_size": arguments.shm_size,
        "cpus": arguments.cpus,
        "pids": arguments.pids,
        "gpus": arguments.gpus,
        "environment": arguments.environment,
        "training_args": training_args,
    }
    with tempfile.TemporaryDirectory(prefix="odbench-remote-start-") as temporary:
        root = Path(temporary)
        workspace = sanitized_workspace(arguments, root)
        with tempfile.TemporaryFile() as payload:
            write_start_payload(payload, request, workspace, arguments.resume_checkpoint)
            payload.seek(0)
            return run_remote_json("start", input_stream=payload, timeout=300)


def fetch_stage(job_id: str, event_id: str, destination: Path) -> None:
    if destination.is_dir():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ssh_command("fetch"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(json.dumps({"job_id": job_id, "event_id": event_id}).encode("utf-8"))
    process.stdin.close()
    try:
        with tempfile.TemporaryDirectory(prefix=f".{event_id}.", dir=destination.parent) as temporary:
            extracted = Path(temporary)
            extract_transfer(process.stdout, extracted)
            exit_code = process.wait(timeout=300)
            if exit_code != 0:
                detail = (process.stderr.read() if process.stderr else b"").decode(
                    "utf-8", errors="replace"
                )[-4000:]
                raise RemoteTrainingError(detail.strip() or "remote staged-result fetch failed")
            if destination.exists():
                return
            os.replace(extracted, destination)
    except Exception:
        process.kill()
        process.wait()
        raise


def evaluate_stage(stage: Path, labels: Path) -> tuple[dict[str, Any] | None, str | None]:
    environment = os.environ.copy()
    result = subprocess.run(
        [
            str(REPO_ROOT / "docker" / "evaluator"),
            "evaluate",
            str(stage / "submission"),
            str(labels),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        timeout=1800,
    )
    if result.returncode != 0:
        return None, (result.stderr or result.stdout or "local evaluation failed").strip()[-4000:]
    try:
        evaluation = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        return None, f"local evaluator returned invalid JSON: {error}"
    if not isinstance(evaluation, dict):
        return None, "local evaluator returned a non-object"
    return evaluation, None


def command_await(arguments: Any) -> dict[str, Any]:
    if arguments.labels is None:
        raise RemoteTrainingError("remote await requires --labels for trusted local evaluation")
    notification = run_remote_json("await", {"job_id": arguments.job_id})
    event_id = notification.get("event_id")
    if not isinstance(event_id, str):
        return notification

    stage = controller.jobs_root() / arguments.job_id / "staged" / event_id
    fetch_stage(arguments.job_id, event_id, stage)
    if notification.get("type") != "train_epoch_staged":
        return notification
    evaluation, evaluation_error = evaluate_stage(stage, arguments.labels.resolve())
    return run_remote_json(
        "record",
        {
            "job_id": arguments.job_id,
            "event_id": event_id,
            "evaluation": evaluation,
            "evaluation_error": evaluation_error,
        },
        timeout=30,
    )


def dispatch(arguments: Any) -> dict[str, Any]:
    if arguments.command == "start":
        return command_start(arguments)
    if arguments.command == "await":
        return command_await(arguments)
    if arguments.command == "continue":
        return run_remote_json(
            "continue", {"job_id": arguments.job_id, "event_id": arguments.event_id}, timeout=30
        )
    if arguments.command == "stop":
        return run_remote_json("stop", {"job_id": arguments.job_id}, timeout=30)
    if arguments.command == "status":
        return run_remote_json("status", {"job_id": arguments.job_id}, timeout=30)
    raise RemoteTrainingError(f"remote trainer does not support {arguments.command}")


def main() -> None:
    arguments = controller.parser().parse_args()
    controller.validate_arguments(arguments)
    try:
        result = dispatch(arguments)
    except (OSError, RemoteProtocolError, RemoteTrainingError, subprocess.TimeoutExpired) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
