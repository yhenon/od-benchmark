"""Host-side lifecycle and tool access for the isolated agent sandbox."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from docker.train.workspace_archive import (
    WorkspaceArchiveError,
    export_file_from_container,
    export_from_container,
    import_into_container,
)


DEFAULT_COMMAND_SECONDS = 60.0


class SandboxError(RuntimeError):
    pass


class SandboxInvocationError(SandboxError):
    """A model-requested sandbox operation that was rejected safely."""


def relative_workspace_path(value: Any, *, name: str = "path") -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SandboxInvocationError(f"{name} must be a non-empty string")
    path = PurePosixPath(value)
    if value == "." or path.is_absolute() or ".." in path.parts:
        raise SandboxInvocationError(f"{name} must be a relative workspace path")
    if len(value.encode("utf-8")) > 4096:
        raise SandboxInvocationError(f"{name} is too long")
    return str(path)


class Sandbox:
    def __init__(
        self,
        *,
        repo_root: Path,
        container: str,
        dataset: str,
        image: str,
        max_command_seconds: float,
    ) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", container):
            raise ValueError("invalid sandbox container name")
        self.repo_root = repo_root.resolve()
        self.container = container
        self.dataset = dataset
        self.image = image
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
            "-P",
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
        if not isinstance(response, dict):
            raise SandboxError("workspace tool returned a non-object response")
        if result.returncode != 0 or "error" in response:
            message = response.get("message") or result.stderr.strip() or "workspace tool failed"
            raise SandboxInvocationError(str(message))
        return response

    def exec(
        self, command: str, timeout_seconds: Any = DEFAULT_COMMAND_SECONDS
    ) -> dict[str, Any]:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise SandboxInvocationError("timeout_seconds must be positive")
        requested_timeout = float(timeout_seconds)
        effective_timeout = min(requested_timeout, self.max_command_seconds)
        result = self._run_tool(
            "exec",
            {"command": command, "timeout_seconds": effective_timeout},
            timeout=effective_timeout + 15,
        )
        return {
            **result,
            "requested_timeout_seconds": requested_timeout,
            "effective_timeout_seconds": effective_timeout,
        }

    def apply_patch(self, patch: str) -> dict[str, Any]:
        result = self._run_tool("apply_patch", {"patch": patch}, timeout=45)
        if result.get("applied") is not True:
            detail = str(result.get("stderr") or result.get("stdout") or "patch did not apply")
            raise SandboxInvocationError(detail.strip()[-4000:])
        return result

    def copy_directory(self, relative: str, destination: Path) -> None:
        relative = relative_workspace_path(relative, name="submission_dir")
        destination.mkdir(parents=True, exist_ok=False)
        try:
            export_from_container(
                self.container,
                relative,
                destination,
                max_files=1000,
                max_bytes=4 * 1024**3,
            )
        except WorkspaceArchiveError as error:
            raise SandboxInvocationError(str(error)) from error

    def copy_file(self, relative: str, destination: Path, *, max_bytes: int) -> None:
        relative = relative_workspace_path(relative, name="model_path")
        try:
            export_file_from_container(
                self.container,
                relative,
                destination,
                max_bytes=max_bytes,
            )
        except WorkspaceArchiveError as error:
            raise SandboxInvocationError(str(error)) from error

    def copy_into_directory(self, source: Path, relative: str) -> None:
        """Copy trusted host output into a new directory in the workspace."""

        relative = relative_workspace_path(relative, name="destination")
        try:
            import_into_container(self.container, source, relative)
        except WorkspaceArchiveError as error:
            raise SandboxError(str(error)) from error

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
