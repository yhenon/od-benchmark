"""Bounded in-container implementations of workspace exec and patch tools."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


WORKSPACE = Path("/workspace")
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_COMMAND_BYTES = 64 * 1024
MAX_PATCH_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 1024 * 1024


def read_request() -> dict[str, Any]:
    encoded = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("tool request exceeds 2 MiB")
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise ValueError("tool request must be a JSON object")
    return value


def owned_processes() -> set[int]:
    """Return processes owned by this sandbox user."""

    result: set[int] = set()
    uid = os.getuid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status_text = (entry / "status").read_text(encoding="utf-8")
            uid_line = next(line for line in status_text.splitlines() if line.startswith("Uid:"))
            real_uid = int(uid_line.split()[1])
        except (FileNotFoundError, PermissionError, StopIteration, ValueError):
            continue
        if real_uid == uid:
            result.add(int(entry.name))
    return result


def kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def clean_new_processes(before: set[int]) -> None:
    """Remove background processes that escaped the command's process group."""

    protected = before | {os.getpid()}
    for _ in range(3):
        candidates = owned_processes() - protected
        if not candidates:
            return
        for pid in candidates:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.02)


def append_bounded(chunks: list[bytes], chunk: bytes, remaining: int) -> tuple[int, bool]:
    if remaining <= 0:
        return 0, bool(chunk)
    kept = chunk[:remaining]
    chunks.append(kept)
    return remaining - len(kept), len(kept) != len(chunk)


def execute_command(request: dict[str, Any]) -> dict[str, Any]:
    command = request.get("command")
    timeout_seconds = request.get("timeout_seconds", 60)
    if not isinstance(command, str) or not command:
        raise ValueError("command must be a non-empty string")
    if len(command.encode("utf-8")) > MAX_COMMAND_BYTES:
        raise ValueError("command exceeds 64 KiB")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 0.1 <= timeout_seconds <= 600
    ):
        raise ValueError("timeout_seconds must be between 0.1 and 600")

    before = owned_processes()
    started = time.monotonic()
    process = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=WORKSPACE,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    output = {"stdout": [], "stderr": []}
    remaining = MAX_OUTPUT_BYTES
    truncated = False
    timed_out = False
    deadline = started + float(timeout_seconds)

    while selector.get_map():
        wait_seconds = deadline - time.monotonic()
        if wait_seconds <= 0 and not timed_out:
            timed_out = True
            kill_process_group(process)
            clean_new_processes(before)
            wait_seconds = 0.1
        events = selector.select(max(0.0, min(wait_seconds, 0.1)))
        for key, _ in events:
            chunk = os.read(key.fd, 64 * 1024)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            remaining, was_truncated = append_bounded(output[key.data], chunk, remaining)
            truncated = truncated or was_truncated
        if process.poll() is not None and not events:
            # Pipes can still hold buffered bytes, so continue until EOF.
            continue

    if process.poll() is None:
        kill_process_group(process)
    exit_code = process.wait()
    clean_new_processes(before)
    duration_ms = round((time.monotonic() - started) * 1000)
    return {
        "exit_code": exit_code,
        "stdout": b"".join(output["stdout"]).decode("utf-8", errors="replace"),
        "stderr": b"".join(output["stderr"]).decode("utf-8", errors="replace"),
        "timed_out": timed_out,
        "duration_ms": duration_ms,
        "truncated": truncated,
    }


def apply_patch(request: dict[str, Any]) -> dict[str, Any]:
    patch = request.get("patch")
    if not isinstance(patch, str) or not patch.strip():
        raise ValueError("patch must be a non-empty string")
    if len(patch.encode("utf-8")) > MAX_PATCH_BYTES:
        raise ValueError("patch exceeds 1 MiB")
    if "diff --git " not in patch:
        raise ValueError("patch must be a git-style unified diff")

    encoded = patch.encode("utf-8")
    base_command = [
        "git",
        "-c",
        "core.attributesFile=/dev/null",
        "-c",
        "core.hooksPath=/dev/null",
        "apply",
        "--recount",
        "--whitespace=nowarn",
    ]
    environment = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1"}
    check = subprocess.run(
        [*base_command, "--check", "-"],
        cwd=WORKSPACE,
        env=environment,
        input=encoded,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    if check.returncode != 0 or check.stderr:
        stderr = check.stderr.decode("utf-8", errors="replace")[-32_768:]
        if check.returncode == 0:
            stderr = (
                "patch rejected because git reported structural warnings; "
                "fix the diff headers and hunk boundaries:\n" + stderr
            )
        return {
            "applied": False,
            "stdout": check.stdout.decode("utf-8", errors="replace")[-32_768:],
            "stderr": stderr,
        }
    applied = subprocess.run(
        [*base_command, "-"],
        cwd=WORKSPACE,
        env=environment,
        input=encoded,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    return {
        "applied": applied.returncode == 0 and not applied.stderr,
        "stdout": applied.stdout.decode("utf-8", errors="replace")[-32_768:],
        "stderr": applied.stderr.decode("utf-8", errors="replace")[-32_768:],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("exec", "apply_patch"))
    return result


def main() -> None:
    arguments = parser().parse_args()
    try:
        request = read_request()
        if arguments.action == "exec":
            result = execute_command(request)
        else:
            result = apply_patch(request)
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    except Exception as error:
        print(
            json.dumps(
                {"error": type(error).__name__, "message": str(error)},
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
