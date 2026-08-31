"""Host-side safe extraction/import for the agent workspace stream."""

from __future__ import annotations

import os
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


class WorkspaceArchiveError(RuntimeError):
    pass


def _export_from_container(
    container: str,
    source: str,
    destination: Path,
    *,
    action: str,
    exclude_agent_state: bool = False,
    max_files: int = 10_000,
    max_bytes: int = 4 * 1024**3,
) -> None:
    command = [
        "docker",
        "exec",
        "--user",
        "10001:10001",
        "--workdir",
        "/workspace",
        container,
        "python",
        "-P",
        "-m",
        "odbench.workspace_transfer",
        action,
        source,
    ]
    if action == "export" and exclude_agent_state:
        command.append("--exclude-agent-state")
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        assert process.stdout is not None
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|*") as archive:
                file_count = 0
                byte_count = 0
                directory_modes: list[tuple[Path, int]] = []
                for member in archive:
                    relative = PurePosixPath(member.name)
                    if (
                        not member.name
                        or relative.is_absolute()
                        or ".." in relative.parts
                        or member.issym()
                        or member.islnk()
                    ):
                        raise WorkspaceArchiveError("workspace archive contains an unsafe path")
                    target = destination.joinpath(*relative.parts)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        directory_modes.append((target, member.mode & 0o777))
                        continue
                    if not member.isfile():
                        raise WorkspaceArchiveError("workspace archive contains a special file")
                    file_count += 1
                    byte_count += member.size
                    if file_count > max_files or byte_count > max_bytes:
                        raise WorkspaceArchiveError("workspace archive exceeds its limits")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source_stream = archive.extractfile(member)
                    if source_stream is None:
                        raise WorkspaceArchiveError("workspace archive member has no data")
                    with target.open("xb") as target_stream:
                        remaining = member.size
                        while remaining:
                            chunk = source_stream.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise WorkspaceArchiveError("workspace archive member ended early")
                            target_stream.write(chunk)
                            remaining -= len(chunk)
                    target.chmod(member.mode & 0o777)
                for directory, mode in reversed(directory_modes):
                    directory.chmod(mode)
            exit_code = process.wait(timeout=30)
        except Exception:
            process.kill()
            process.wait()
            raise
        if exit_code != 0:
            errors.seek(0)
            detail = errors.read(64 * 1024).decode("utf-8", errors="replace")
            raise WorkspaceArchiveError(detail.strip() or "workspace export failed")


def export_from_container(
    container: str,
    source: str,
    destination: Path,
    *,
    exclude_agent_state: bool = False,
    max_files: int = 10_000,
    max_bytes: int = 4 * 1024**3,
) -> None:
    _export_from_container(
        container,
        source,
        destination,
        action="export",
        exclude_agent_state=exclude_agent_state,
        max_files=max_files,
        max_bytes=max_bytes,
    )


def export_file_from_container(
    container: str,
    source: str,
    destination: Path,
    *,
    max_bytes: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise WorkspaceArchiveError("workspace export destination already exists")
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary:
        extracted = Path(temporary)
        _export_from_container(
            container,
            source,
            extracted,
            action="export-file",
            max_files=1,
            max_bytes=max_bytes,
        )
        entries = list(extracted.iterdir())
        if len(entries) != 1 or not entries[0].is_file():
            raise WorkspaceArchiveError("workspace file export returned an invalid archive")
        os.replace(entries[0], destination)


def import_into_container(container: str, source: Path, destination: str) -> None:
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for current_root, directory_names, file_names in os.walk(source):
                current = Path(current_root)
                relative_directory = current.relative_to(source)
                for name in sorted(directory_names):
                    path = current / name
                    if path.is_symlink():
                        raise WorkspaceArchiveError("trusted import unexpectedly contains a symlink")
                    relative = (relative_directory / name).as_posix()
                    info = tarfile.TarInfo(relative)
                    info.type = tarfile.DIRTYPE
                    info.mode = path.lstat().st_mode & 0o777
                    archive.addfile(info)
                for name in sorted(file_names):
                    path = current / name
                    path_stat = path.lstat()
                    if not stat.S_ISREG(path_stat.st_mode):
                        raise WorkspaceArchiveError("trusted import contains a special file")
                    relative = (relative_directory / name).as_posix()
                    info = tarfile.TarInfo(relative)
                    info.size = path_stat.st_size
                    info.mode = path_stat.st_mode & 0o777
                    with path.open("rb") as file_stream:
                        archive.addfile(info, file_stream)
        stream.seek(0)
        result = subprocess.run(
            [
                "docker",
                "exec",
                "--interactive",
                "--user",
                "10001:10001",
                "--workdir",
                "/workspace",
                container,
                "python",
                "-P",
                "-m",
                "odbench.workspace_transfer",
                "import",
                destination,
            ],
            stdin=stream,
            text=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace")[-4000:]
        raise WorkspaceArchiveError(detail.strip() or "workspace import failed")
