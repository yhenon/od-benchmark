"""Atomic filesystem mailbox used to synchronize at epoch boundaries."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


EVENTS_ROOT = Path(os.environ.get("ODBENCH_EVENTS_DIR", "/job/events"))
DECISIONS_ROOT = Path(os.environ.get("ODBENCH_DECISIONS_DIR", "/job/decisions"))
OUTPUT_ROOT = Path(os.environ.get("ODBENCH_OUTPUT_DIR", "/job/output"))
INPUT_ROOT = Path(os.environ.get("ODBENCH_INPUT_DIR", "/job/input"))
JOB_ID = os.environ.get("ODBENCH_JOB_ID", "unknown-job")
_last_epoch = -1


@dataclass(frozen=True)
class Decision:
    action: str
    evaluation: dict[str, Any] | None = None

    @property
    def stop(self) -> bool:
        return self.action == "stop"


def _relative_path(value: str, *, name: str) -> str:
    path = PurePosixPath(value)
    if not value or value == "." or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a relative path without '..'")
    return str(path)


def _sync_regular_file(root: Path, relative_path: str, *, name: str) -> None:
    path = root
    for component in PurePosixPath(relative_path).parts:
        path = path / component
        if path.is_symlink():
            raise ValueError(f"{name} path may not contain symlinks")
    if not path.is_file():
        raise ValueError(f"{name} must reference a regular file")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n"
    if len(encoded.encode("utf-8")) > 1024 * 1024:
        raise ValueError("epoch event exceeds 1 MiB")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def epoch_end(
    *,
    epoch: int,
    artifact: str,
    preprocess: str,
    postprocess: str,
    checkpoint: str | None = None,
    metrics: dict[str, Any] | None = None,
    poll_seconds: float = 0.5,
) -> Decision:
    """Publish an epoch artifact and block until the outer controller decides."""

    global _last_epoch
    if "ODBENCH_JOB_ID" not in os.environ:
        raise RuntimeError(
            "epoch_end() is only available inside a training job; "
            "launch this script with the train_start tool"
        )
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= _last_epoch:
        raise ValueError("epoch must be a monotonically increasing integer")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")

    artifact = _relative_path(artifact, name="artifact")
    preprocess = _relative_path(preprocess, name="preprocess")
    postprocess = _relative_path(postprocess, name="postprocess")
    checkpoint = (
        _relative_path(checkpoint, name="checkpoint") if checkpoint is not None else None
    )
    _sync_regular_file(OUTPUT_ROOT, artifact, name="artifact")
    if checkpoint is not None:
        _sync_regular_file(OUTPUT_ROOT, checkpoint, name="checkpoint")
    _sync_regular_file(INPUT_ROOT, preprocess, name="preprocess")
    _sync_regular_file(INPUT_ROOT, postprocess, name="postprocess")

    event_id = f"epoch-{epoch:06d}"
    event = {
        "schema_version": 1,
        "type": "epoch_end",
        "job_id": JOB_ID,
        "event_id": event_id,
        "epoch": epoch,
        "artifact": artifact,
        "checkpoint": checkpoint,
        "preprocess": preprocess,
        "postprocess": postprocess,
        "metrics": metrics or {},
    }
    # Ensure submitted metrics are JSON-safe before publishing anything.
    json.dumps(event)
    _atomic_json(EVENTS_ROOT / f"{event_id}.json", event)

    decision_path = DECISIONS_ROOT / f"{event_id}.json"
    while True:
        try:
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            break
        except FileNotFoundError:
            time.sleep(poll_seconds)
    if decision.get("event_id") != event_id or decision.get("action") not in {
        "continue",
        "stop",
    }:
        raise RuntimeError("invalid epoch decision")
    _last_epoch = epoch
    return Decision(action=decision["action"], evaluation=decision.get("evaluation"))
