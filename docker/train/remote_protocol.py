"""Bounded tar transport shared by the SSH training client and worker."""

from __future__ import annotations

import io
import json
import os
import stat
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


MAX_TRANSFER_FILES = 10_005
MAX_TRANSFER_BYTES = 5 * 1024**3
REQUEST_NAME = "request.json"
WORKSPACE_PREFIX = PurePosixPath("workspace")
CHECKPOINT_NAME = PurePosixPath("resume-checkpoint.pt")


class RemoteProtocolError(RuntimeError):
    pass


def _safe_member_name(name: str) -> PurePosixPath:
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts:
        raise RemoteProtocolError("remote transfer contains an unsafe path")
    return relative


def _add_regular(archive: tarfile.TarFile, source: Path, name: str) -> None:
    source_stat = source.lstat()
    if not stat.S_ISREG(source_stat.st_mode) or source.is_symlink():
        raise RemoteProtocolError(f"remote transfer source is not a regular file: {source}")
    info = tarfile.TarInfo(name)
    info.size = source_stat.st_size
    info.mode = source_stat.st_mode & 0o777
    with source.open("rb") as stream:
        archive.addfile(info, stream)


def write_start_payload(
    stream: BinaryIO,
    request: dict[str, Any],
    workspace: Path,
    resume_checkpoint: Path | None,
) -> None:
    encoded = json.dumps(request, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(encoded) > 1024 * 1024:
        raise RemoteProtocolError("remote start request exceeds 1 MiB")
    with tarfile.open(fileobj=stream, mode="w|") as archive:
        request_info = tarfile.TarInfo(REQUEST_NAME)
        request_info.size = len(encoded)
        request_info.mode = 0o600
        archive.addfile(request_info, io.BytesIO(encoded))

        for current_root, directory_names, file_names in os.walk(workspace):
            current = Path(current_root)
            relative_directory = current.relative_to(workspace)
            for name in sorted(directory_names):
                path = current / name
                if path.is_symlink():
                    raise RemoteProtocolError("workspace transfer may not contain symlinks")
                relative = WORKSPACE_PREFIX / relative_directory / name
                info = tarfile.TarInfo(str(relative))
                info.type = tarfile.DIRTYPE
                info.mode = path.lstat().st_mode & 0o777
                archive.addfile(info)
            for name in sorted(file_names):
                path = current / name
                relative = WORKSPACE_PREFIX / relative_directory / name
                _add_regular(archive, path, str(relative))
        if resume_checkpoint is not None:
            _add_regular(archive, resume_checkpoint, str(CHECKPOINT_NAME))


def extract_transfer(
    stream: BinaryIO,
    destination: Path,
    *,
    max_files: int = MAX_TRANSFER_FILES,
    max_bytes: int = MAX_TRANSFER_BYTES,
) -> None:
    file_count = 0
    byte_count = 0
    directory_modes: list[tuple[Path, int]] = []
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            relative = _safe_member_name(member.name)
            if member.issym() or member.islnk():
                raise RemoteProtocolError("remote transfer may not contain links")
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                directory_modes.append((target, member.mode & 0o777))
                continue
            if not member.isfile():
                raise RemoteProtocolError("remote transfer contains a special file")
            file_count += 1
            byte_count += member.size
            if file_count > max_files or byte_count > max_bytes:
                raise RemoteProtocolError("remote transfer exceeds its limits")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RemoteProtocolError("remote transfer member has no data")
            with target.open("xb") as output:
                remaining = member.size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise RemoteProtocolError("remote transfer member ended early")
                    output.write(chunk)
                    remaining -= len(chunk)
            target.chmod(member.mode & 0o777)
    for directory, mode in reversed(directory_modes):
        directory.chmod(mode)


def write_directory(stream: BinaryIO, source: Path) -> None:
    """Stream a trusted staged result without following links."""

    with tarfile.open(fileobj=stream, mode="w|") as archive:
        for current_root, directory_names, file_names in os.walk(source):
            current = Path(current_root)
            relative_directory = current.relative_to(source)
            for name in sorted(directory_names):
                path = current / name
                if path.is_symlink():
                    raise RemoteProtocolError("staged result unexpectedly contains a symlink")
                relative = relative_directory / name
                info = tarfile.TarInfo(relative.as_posix())
                info.type = tarfile.DIRTYPE
                info.mode = path.lstat().st_mode & 0o777
                archive.addfile(info)
            for name in sorted(file_names):
                path = current / name
                relative = relative_directory / name
                _add_regular(archive, path, relative.as_posix())
