"""Trusted streaming transfer protocol for the tmpfs-backed agent workspace."""

from __future__ import annotations

import argparse
import os
import stat
import sys
import tarfile
from pathlib import Path, PurePosixPath


WORKSPACE = Path("/workspace")
DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    ".odbench",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
MAX_FILES = 10_000
MAX_BYTES = 4 * 1024**3


def safe_source(value: str) -> Path:
    relative = PurePosixPath(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source must be a relative workspace directory")
    source = WORKSPACE if value == "." else WORKSPACE.joinpath(*relative.parts)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("source must be a non-symlink workspace directory")
    resolved = source.resolve()
    if resolved != WORKSPACE and WORKSPACE not in resolved.parents:
        raise ValueError("source escapes the workspace")
    return source


def safe_file_source(value: str) -> Path:
    relative = PurePosixPath(value)
    if not value or value == "." or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source must be a relative workspace file")
    source = WORKSPACE.joinpath(*relative.parts)
    current = WORKSPACE
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("source path may not contain symlinks")
    if not source.is_file() or not stat.S_ISREG(source.lstat().st_mode):
        raise ValueError("source must be a regular workspace file")
    resolved = source.resolve()
    if WORKSPACE not in resolved.parents:
        raise ValueError("source escapes the workspace")
    return source


def export_workspace(source_value: str, exclude_agent_state: bool) -> None:
    source = safe_source(source_value)
    file_count = 0
    byte_count = 0
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        for current_root, directory_names, file_names in os.walk(source):
            current = Path(current_root)
            relative_directory = current.relative_to(source)
            kept_directories = []
            for name in sorted(directory_names):
                if exclude_agent_state and name in DEFAULT_EXCLUDES:
                    continue
                path = current / name
                if path.is_symlink():
                    raise ValueError(f"workspace symlink is not allowed: {path.relative_to(source)}")
                kept_directories.append(name)
                relative = (relative_directory / name).as_posix()
                info = tarfile.TarInfo(relative)
                info.type = tarfile.DIRTYPE
                info.mode = path.stat(follow_symlinks=False).st_mode & 0o777
                archive.addfile(info)
            directory_names[:] = kept_directories
            for name in sorted(file_names):
                if exclude_agent_state and name in DEFAULT_EXCLUDES:
                    continue
                path = current / name
                if path.is_symlink():
                    raise ValueError(
                        f"workspace symlink is not allowed: {path.relative_to(source)}"
                    )
                path_stat = path.lstat()
                if not stat.S_ISREG(path_stat.st_mode):
                    raise ValueError(
                        f"workspace entry is not a regular file: {path.relative_to(source)}"
                    )
                file_count += 1
                byte_count += path_stat.st_size
                if file_count > MAX_FILES or byte_count > MAX_BYTES:
                    raise ValueError("workspace export exceeds its file or byte limit")
                relative = (relative_directory / name).as_posix()
                info = tarfile.TarInfo(relative)
                info.size = path_stat.st_size
                info.mode = path_stat.st_mode & 0o777
                with path.open("rb") as stream:
                    archive.addfile(info, stream)


def export_file(source_value: str) -> None:
    source = safe_file_source(source_value)
    source_stat = source.lstat()
    if source_stat.st_size > MAX_BYTES:
        raise ValueError("workspace export exceeds its byte limit")
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        info = tarfile.TarInfo(source.name)
        info.size = source_stat.st_size
        info.mode = source_stat.st_mode & 0o777
        with source.open("rb") as stream:
            archive.addfile(info, stream)


def safe_destination(value: str) -> Path:
    relative = PurePosixPath(value)
    if not value or value == "." or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("destination must be a relative workspace directory")
    destination = WORKSPACE.joinpath(*relative.parts)
    current = WORKSPACE
    for component in relative.parts[:-1]:
        current = current / component
        if current.exists() and current.is_symlink():
            raise ValueError("destination path contains a symlink")
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination already exists")
    return destination


def import_workspace(destination_value: str) -> None:
    destination = safe_destination(destination_value)
    destination.mkdir(parents=True, exist_ok=False)
    file_count = 0
    byte_count = 0
    directory_modes: list[tuple[Path, int]] = []
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if (
                not member.name
                or relative.is_absolute()
                or ".." in relative.parts
                or member.issym()
                or member.islnk()
            ):
                raise ValueError("archive contains an unsafe path or link")
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                directory_modes.append((target, member.mode & 0o777))
                continue
            if not member.isfile():
                raise ValueError("archive entries must be directories or regular files")
            file_count += 1
            byte_count += member.size
            if file_count > MAX_FILES or byte_count > MAX_BYTES:
                raise ValueError("workspace import exceeds its file or byte limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError("archive member has no content")
            with target.open("xb") as stream:
                remaining = member.size
                while remaining:
                    chunk = extracted.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("archive member ended early")
                    stream.write(chunk)
                    remaining -= len(chunk)
            target.chmod(member.mode & 0o777)
    for directory, mode in reversed(directory_modes):
        directory.chmod(mode)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("source")
    export.add_argument("--exclude-agent-state", action="store_true")
    export_file_parser = subparsers.add_parser("export-file")
    export_file_parser.add_argument("source")
    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("destination")
    return result


def main() -> None:
    arguments = parser().parse_args()
    try:
        if arguments.command == "export":
            export_workspace(arguments.source, arguments.exclude_agent_state)
        elif arguments.command == "export-file":
            export_file(arguments.source)
        else:
            import_workspace(arguments.destination)
    except Exception as error:
        print(f"workspace transfer failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
