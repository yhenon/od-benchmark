"""Host-side lifecycle and tool access for the isolated agent sandbox."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any


class SandboxError(RuntimeError):
    pass


def relative_workspace_path(value: Any, *, name: str = "path") -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string")
    path = PurePosixPath(value)
    if value == "." or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a relative workspace path")
    if len(value.encode("utf-8")) > 4096:
        raise ValueError(f"{name} is too long")
    return str(path)


class Sandbox:
    def __init__(
        self,
        *,
        repo_root: Path,
        container: str,
        dataset: str = "cifar10",
        image: str | None = None,
        max_command_seconds: float = 120,
    ) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", container):
            raise ValueError("invalid sandbox container name")
        self.repo_root = repo_root.resolve()
        self.container = container
        self.dataset = dataset
        self.image = image or f"od-benchmark-agent:{dataset}-dev"
        self.max_command_seconds = max_command_seconds
        self.started = False

    def _sandbox_environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "ODBENCH_CONTAINER": self.container,
            "ODBENCH_DATASET": self.dataset,
            "ODBENCH_IMAGE": self.image,
        }

    def start(self) -> None:
        result = subprocess.run(
            [str(self.repo_root / "docker" / "sandbox"), "start"],
            cwd=self.repo_root,
            env=self._sandbox_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise SandboxError((result.stderr or result.stdout).strip())
        self.started = True

    def stop(self) -> None:
        if not self.started:
            return
        subprocess.run(
            [str(self.repo_root / "docker" / "sandbox"), "stop"],
            cwd=self.repo_root,
            env=self._sandbox_environment(),
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.started = False

    def _run_tool(self, action: str, request: dict[str, Any], timeout: float) -> dict[str, Any]:
        encoded = json.dumps(request, separators=(",", ":"))
        command = [
            "docker",
            "exec",
            "--interactive",
            "--user",
            "10001:10001",
            "--workdir",
            "/workspace",
            self.container,
            "python",
            "-m",
            "odbench.tool_runner",
            action,
        ]
        try:
            result = subprocess.run(
                command,
                input=encoded,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            # A wedged supervisor is not safe to reuse; killing the disposable
            # sandbox also guarantees no command processes survive.
            subprocess.run(
                ["docker", "kill", self.container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.started = False
            raise SandboxError("workspace tool supervisor timed out; sandbox was killed") from error
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            detail = (result.stderr or result.stdout).strip()[-4000:]
            raise SandboxError(f"invalid workspace tool response: {detail}") from error
        if result.returncode != 0 or "error" in response:
            message = response.get("message") or result.stderr.strip() or "workspace tool failed"
            raise SandboxError(str(message))
        if not isinstance(response, dict):
            raise SandboxError("workspace tool returned a non-object response")
        return response

    def exec(self, command: str, timeout_seconds: Any = 60) -> dict[str, Any]:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        bounded_timeout = min(float(timeout_seconds), self.max_command_seconds)
        return self._run_tool(
            "exec",
            {"command": command, "timeout_seconds": bounded_timeout},
            timeout=bounded_timeout + 15,
        )

    def apply_patch(self, patch: str) -> dict[str, Any]:
        return self._run_tool("apply_patch", {"patch": patch}, timeout=45)

    def copy_directory(self, relative: str, destination: Path) -> None:
        relative = relative_workspace_path(relative, name="submission_dir")
        destination.mkdir(parents=True, exist_ok=False)
        result = subprocess.run(
            ["docker", "cp", f"{self.container}:/workspace/{relative}/.", str(destination)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        if result.returncode != 0:
            raise SandboxError((result.stderr or result.stdout).strip())

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

