#!/usr/bin/env python3
"""Trusted outer controller for filesystem-mailbox training jobs."""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOBS_ROOT = REPO_ROOT / ".odbench" / "jobs"
DEFAULT_IGNORES = (
    ".git",
    ".venv",
    ".odbench",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
)
EVENT_ID_PATTERN = re.compile(r"^epoch-[0-9]{6}$")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(command: list[str], *, capture: bool = False, check: bool = True) -> str:
    result = subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def jobs_root() -> Path:
    return Path(os.environ.get("ODBENCH_JOBS_ROOT", DEFAULT_JOBS_ROOT)).resolve()


def job_root(job_id: str) -> Path:
    if not re.fullmatch(r"job-[0-9]{8}T[0-9]{6}-[0-9a-f]{8}", job_id):
        raise SystemExit("invalid job id")
    root = jobs_root() / job_id
    if not root.is_dir():
        raise SystemExit(f"unknown job: {job_id}")
    return root


def read_state(root: Path) -> dict[str, Any]:
    return json.loads((root / "state.json").read_text(encoding="utf-8"))


def write_state(root: Path, state: dict[str, Any]) -> None:
    atomic_json(root / "state.json", state)


def load_ignore_patterns(source: Path) -> tuple[str, ...]:
    patterns = list(DEFAULT_IGNORES)
    ignore_file = source / ".odbenchignore"
    if ignore_file.is_file():
        for line in ignore_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                patterns.append(stripped.rstrip("/"))
    return tuple(patterns)


def ignored(relative: Path, patterns: tuple[str, ...]) -> bool:
    value = relative.as_posix()
    return any(
        fnmatch.fnmatch(value, pattern)
        or fnmatch.fnmatch(relative.name, pattern)
        or value.startswith(pattern + "/")
        for pattern in patterns
    )


def snapshot_workspace(
    source: Path,
    destination: Path,
    *,
    max_files: int,
    max_bytes: int,
) -> tuple[str, int, int]:
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f"workspace is not a directory: {source}")
    patterns = load_ignore_patterns(source)
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for current_root, directory_names, file_names in os.walk(source):
        current = Path(current_root)
        relative_directory = current.relative_to(source)
        kept_directories = []
        for name in sorted(directory_names):
            path = current / name
            relative = relative_directory / name
            if ignored(relative, patterns):
                continue
            if path.is_symlink():
                raise ValueError(f"workspace symlink is not allowed: {relative}")
            kept_directories.append(name)
            (destination / relative).mkdir(parents=True, exist_ok=True)
        directory_names[:] = kept_directories

        for name in sorted(file_names):
            source_file = current / name
            relative = relative_directory / name
            if ignored(relative, patterns):
                continue
            file_stat = source_file.lstat()
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"workspace entry is not a regular file: {relative}")
            file_count += 1
            byte_count += file_stat.st_size
            if file_count > max_files:
                raise ValueError(f"workspace exceeds {max_files} files")
            if byte_count > max_bytes:
                raise ValueError(f"workspace exceeds {max_bytes} bytes")
            destination_file = destination / relative
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            with source_file.open("rb") as input_stream, destination_file.open("xb") as output_stream:
                digest.update(relative.as_posix().encode("utf-8") + b"\0")
                for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                    output_stream.write(chunk)
                    digest.update(chunk)
            shutil.copystat(source_file, destination_file, follow_symlinks=False)
    return digest.hexdigest(), file_count, byte_count


def docker_status(container: str) -> tuple[str, int | None]:
    result = subprocess.run(
        ["docker", "container", "inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}", container],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return "missing", None
    status, exit_code = result.stdout.strip().split()
    return status, int(exit_code)


def safe_relative(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} path must be a string")
    path = PurePosixPath(value)
    if not value or value == "." or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid {name} path")
    return str(path)


def copy_regular(source_root: Path, relative: str, destination: Path, limit: int) -> str:
    source = source_root
    for component in PurePosixPath(relative).parts:
        source = source / component
        component_stat = source.lstat()
        if stat.S_ISLNK(component_stat.st_mode):
            raise ValueError(f"symlinks are not allowed in submitted paths: {relative}")
    source_stat = source.lstat()
    if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_size > limit:
        raise ValueError(f"invalid or oversized artifact: {relative}")
    digest = hashlib.sha256()
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
            output_stream.write(chunk)
            digest.update(chunk)
    destination.chmod(0o444)
    return digest.hexdigest()


def pause_and_meter(root: Path, state: dict[str, Any]) -> None:
    run(["docker", "pause", state["container"]], capture=True)
    now = time.time()
    state["active_seconds"] += now - state["active_started_at"]
    state["active_started_at"] = None
    state["status"] = "paused"
    write_state(root, state)


def process_event(
    root: Path,
    state: dict[str, Any],
    event_path: Path,
    labels: Path,
) -> dict[str, Any]:
    if event_path.stat().st_size > 1024 * 1024:
        raise ValueError("epoch event exceeds 1 MiB")
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event_id = event.get("event_id")
    epoch = event.get("epoch")
    if (
        event.get("schema_version") != 1
        or event.get("type") != "epoch_end"
        or event.get("job_id") != state["job_id"]
        or not isinstance(event_id, str)
        or not EVENT_ID_PATTERN.fullmatch(event_id)
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch <= state["last_epoch"]
    ):
        raise ValueError("invalid or non-monotonic epoch event")
    if event_path.name != f"{event_id}.json":
        raise ValueError("event id does not match its filename")

    pause_and_meter(root, state)
    stage = root / "staged" / event_id
    stage.mkdir(parents=True, exist_ok=False)
    submission = stage / "submission"
    submission.mkdir()
    artifact_path = safe_relative(event.get("artifact"), "artifact")
    preprocess_path = safe_relative(event.get("preprocess"), "preprocess")
    postprocess_path = safe_relative(event.get("postprocess"), "postprocess")
    checkpoint_value = event.get("checkpoint")
    checkpoint_path = (
        safe_relative(checkpoint_value, "checkpoint") if checkpoint_value is not None else None
    )
    artifact_hash = copy_regular(root / "output", artifact_path, submission / "model.onnx", 2 * 1024**3)
    copy_regular(root / "input", preprocess_path, submission / "preprocess.py", 1024**2)
    copy_regular(root / "input", postprocess_path, submission / "postprocess.py", 1024**2)
    checkpoint_hash = None
    if checkpoint_path is not None:
        checkpoint_hash = copy_regular(
            root / "output", checkpoint_path, stage / "checkpoint.pt", 4 * 1024**3
        )
    atomic_json(
        submission / "submission.json",
        {
            "schema_version": 1,
            "artifact": {"format": "onnx", "path": "model.onnx"},
            "preprocess": "preprocess.py",
            "postprocess": "postprocess.py",
        },
    )

    evaluation = None
    evaluation_error = None
    try:
        environment = os.environ.copy()
        environment["ODBENCH_DATASET"] = state["dataset"]
        result = subprocess.run(
            [str(REPO_ROOT / "docker" / "evaluator"), "evaluate", str(submission), str(labels)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        evaluation = json.loads(result.stdout)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or str(error)).strip()
        evaluation_error = detail[-4000:]
    except Exception as error:
        evaluation_error = str(error)

    notification = {
        "schema_version": 1,
        "type": "train_epoch_complete" if evaluation is not None else "train_epoch_failed",
        "job_id": state["job_id"],
        "event_id": event_id,
        "epoch": epoch,
        "train_metrics": event.get("metrics", {}),
        "evaluation": evaluation,
        "evaluation_error": evaluation_error,
        "artifact_sha256": artifact_hash,
        "checkpoint_sha256": checkpoint_hash,
        "metering": {
            "active_wall_seconds": state["active_seconds"],
            "budget_seconds": state["budget_seconds"],
        },
    }
    atomic_json(root / "results" / f"{event_id}.json", notification)
    state["pending"] = {"event": event, "notification": notification}
    state["status"] = "awaiting_decision"
    write_state(root, state)
    return notification


def command_build(arguments: argparse.Namespace) -> None:
    dataset = arguments.dataset
    base_image = arguments.base_image or f"od-benchmark-agent:{dataset}-dev"
    image = arguments.image or f"od-benchmark-trainer:{dataset}-dev"
    run(
        [
            "docker",
            "build",
            "--file",
            str(REPO_ROOT / "Dockerfile.train"),
            "--build-arg",
            f"DATASET={dataset}",
            "--build-arg",
            f"BASE_IMAGE={base_image}",
            "--tag",
            image,
            str(REPO_ROOT),
        ]
    )


def command_start(arguments: argparse.Namespace) -> None:
    dataset = arguments.dataset
    now = dt.datetime.now(dt.timezone.utc)
    job_id = f"job-{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    root = jobs_root() / job_id
    for name in ("input", "output", "events", "decisions", "staged", "results"):
        (root / name).mkdir(parents=True, exist_ok=True)

    imported = None
    try:
        if arguments.workspace is not None:
            source = arguments.workspace.resolve()
        else:
            imported = root / "imported-workspace"
            imported.mkdir()
            run(["docker", "cp", f"{arguments.agent_container}:/workspace/.", str(imported)])
            source = imported
        snapshot_hash, file_count, byte_count = snapshot_workspace(
            source,
            root / "input",
            max_files=arguments.max_files,
            max_bytes=arguments.max_bytes,
        )
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    finally:
        if imported is not None:
            shutil.rmtree(imported, ignore_errors=True)

    entrypoint = safe_relative(arguments.entrypoint, "entrypoint")
    if not (root / "input" / entrypoint).is_file():
        shutil.rmtree(root, ignore_errors=True)
        raise SystemExit("entrypoint is absent from the workspace snapshot")
    image = arguments.image or f"od-benchmark-trainer:{dataset}-dev"
    container = f"odbench-train-{job_id}"
    started_at = time.time()
    training_args = arguments.training_args
    if training_args[:1] == ["--"]:
        training_args = training_args[1:]
    state = {
        "schema_version": 1,
        "job_id": job_id,
        "container": container,
        "dataset": dataset,
        "image": image,
        "entrypoint": entrypoint,
        "args": training_args,
        "snapshot_sha256": snapshot_hash,
        "snapshot_files": file_count,
        "snapshot_bytes": byte_count,
        "created_at": utc_now(),
        "started_at": started_at,
        "active_started_at": started_at,
        "active_seconds": 0.0,
        "budget_seconds": arguments.budget_seconds,
        "last_epoch": -1,
        "pending": None,
        "status": "starting",
    }
    write_state(root, state)
    mount = lambda source, target, readonly=False: (
        f"type=bind,source={source},target={target}" + (",readonly" if readonly else "")
    )
    command = [
        "docker",
        "run",
        "--detach",
        "--name",
        container,
        "--hostname",
        "trainer",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges=true",
        "--pids-limit",
        str(arguments.pids),
        "--memory",
        arguments.memory,
        "--memory-swap",
        arguments.memory,
        "--cpus",
        str(arguments.cpus),
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=1073741824",
        "--mount",
        mount(root / "input", "/job/input", True),
        "--mount",
        mount(root / "output", "/job/output"),
        "--mount",
        mount(root / "events", "/job/events"),
        "--mount",
        mount(root / "decisions", "/job/decisions", True),
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        f"ODBENCH_JOB_ID={job_id}",
        "--env",
        "HOME=/tmp",
        image,
        entrypoint,
        *training_args,
    ]
    try:
        # `docker run --detach` prints the container ID. Keep the tool response
        # as one JSON document by capturing that implementation detail.
        run(command, capture=True)
    except Exception:
        state["status"] = "launch_failed"
        write_state(root, state)
        raise
    state["status"] = "running"
    write_state(root, state)
    print(json.dumps({"job_id": job_id, "snapshot_sha256": snapshot_hash}, sort_keys=True))


def command_await(arguments: argparse.Namespace) -> None:
    root = job_root(arguments.job_id)
    labels = arguments.labels.resolve()
    if not labels.is_file():
        raise SystemExit(f"labels file does not exist: {labels}")
    while True:
        state = read_state(root)
        if state.get("pending") is not None:
            print(json.dumps(state["pending"]["notification"], sort_keys=True))
            return
        status, exit_code = docker_status(state["container"])
        if status in {"exited", "dead", "missing"}:
            if state["active_started_at"] is not None:
                state["active_seconds"] += time.time() - state["active_started_at"]
                state["active_started_at"] = None
            state["status"] = "completed" if exit_code == 0 else "failed"
            write_state(root, state)
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "type": "train_job_finished",
                        "job_id": state["job_id"],
                        "status": state["status"],
                        "exit_code": exit_code,
                        "metering": {"active_wall_seconds": state["active_seconds"]},
                    },
                    sort_keys=True,
                )
            )
            return
        active = state["active_seconds"]
        if state["active_started_at"] is not None:
            active += time.time() - state["active_started_at"]
        if active >= state["budget_seconds"]:
            run(["docker", "rm", "--force", state["container"]], capture=True, check=False)
            state["active_seconds"] = active
            state["active_started_at"] = None
            state["status"] = "budget_exhausted"
            write_state(root, state)
            print(json.dumps({"type": "train_budget_exhausted", "job_id": state["job_id"]}))
            return

        for event_path in sorted((root / "events").glob("epoch-*.json")):
            if not EVENT_ID_PATTERN.fullmatch(event_path.stem):
                continue
            epoch = int(event_path.stem.removeprefix("epoch-"))
            if epoch > state["last_epoch"]:
                notification = process_event(root, state, event_path, labels)
                print(json.dumps(notification, sort_keys=True))
                return
        time.sleep(arguments.poll_seconds)


def command_continue(arguments: argparse.Namespace) -> None:
    root = job_root(arguments.job_id)
    state = read_state(root)
    pending = state.get("pending")
    if pending is None or pending["event"]["event_id"] != arguments.event_id:
        raise SystemExit("job is not awaiting that event decision")
    decision = {
        "schema_version": 1,
        "event_id": arguments.event_id,
        "action": "continue",
        "evaluation": pending["notification"].get("evaluation"),
    }
    atomic_json(root / "decisions" / f"{arguments.event_id}.json", decision)
    run(["docker", "unpause", state["container"]], capture=True)
    state["last_epoch"] = pending["event"]["epoch"]
    state["pending"] = None
    state["status"] = "running"
    state["active_started_at"] = time.time()
    write_state(root, state)
    print(json.dumps({"job_id": arguments.job_id, "action": "continue"}, sort_keys=True))


def command_stop(arguments: argparse.Namespace) -> None:
    root = job_root(arguments.job_id)
    state = read_state(root)
    pending = state.get("pending")
    if pending is not None:
        event_id = pending["event"]["event_id"]
        atomic_json(
            root / "decisions" / f"{event_id}.json",
            {"schema_version": 1, "event_id": event_id, "action": "stop"},
        )
        run(["docker", "unpause", state["container"]], capture=True, check=False)
        time.sleep(0.5)
    run(["docker", "rm", "--force", state["container"]], capture=True, check=False)
    state["pending"] = None
    state["active_started_at"] = None
    state["status"] = "stopped"
    write_state(root, state)
    print(json.dumps({"job_id": arguments.job_id, "action": "stop"}, sort_keys=True))


def command_status(arguments: argparse.Namespace) -> None:
    root = job_root(arguments.job_id)
    state = read_state(root)
    container_status, exit_code = docker_status(state["container"])
    print(json.dumps({**state, "container_status": container_status, "exit_code": exit_code}, sort_keys=True))


def command_logs(arguments: argparse.Namespace) -> None:
    state = read_state(job_root(arguments.job_id))
    command = ["docker", "logs"]
    if arguments.tail is not None:
        command.extend(["--tail", str(arguments.tail)])
    command.append(state["container"])
    run(command, check=False)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--dataset", default="cifar10")
    build.add_argument("--base-image")
    build.add_argument("--image")
    build.set_defaults(function=command_build)

    start = subparsers.add_parser("start")
    source = start.add_mutually_exclusive_group(required=True)
    source.add_argument("--workspace", type=Path)
    source.add_argument("--agent-container")
    start.add_argument("--entrypoint", required=True)
    start.add_argument("--dataset", default="cifar10")
    start.add_argument("--image")
    start.add_argument("--budget-seconds", type=float, default=3600.0)
    start.add_argument("--max-files", type=int, default=10_000)
    start.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    start.add_argument("--memory", default="8g")
    start.add_argument("--cpus", type=float, default=4.0)
    start.add_argument("--pids", type=int, default=256)
    start.add_argument("training_args", nargs=argparse.REMAINDER)
    start.set_defaults(function=command_start)

    await_parser = subparsers.add_parser("await")
    await_parser.add_argument("job_id")
    await_parser.add_argument("--labels", required=True, type=Path)
    await_parser.add_argument("--poll-seconds", type=float, default=0.5)
    await_parser.set_defaults(function=command_await)

    continue_parser = subparsers.add_parser("continue")
    continue_parser.add_argument("job_id")
    continue_parser.add_argument("event_id")
    continue_parser.set_defaults(function=command_continue)

    stop = subparsers.add_parser("stop")
    stop.add_argument("job_id")
    stop.set_defaults(function=command_stop)

    status = subparsers.add_parser("status")
    status.add_argument("job_id")
    status.set_defaults(function=command_status)

    logs = subparsers.add_parser("logs")
    logs.add_argument("job_id")
    logs.add_argument("--tail", type=int)
    logs.set_defaults(function=command_logs)
    return result


def main() -> None:
    arguments = parser().parse_args()
    try:
        arguments.function(arguments)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr, end="")
        raise SystemExit(error.returncode) from error


if __name__ == "__main__":
    main()
