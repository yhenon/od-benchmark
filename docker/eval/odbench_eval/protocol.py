"""Small bounded binary protocol between the trusted provider and hook worker."""

from __future__ import annotations

import os
import select
import struct
import time


MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 64


def _write_all(file_descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(file_descriptor, view)
        view = view[written:]


def send_frame(file_descriptor: int, payload: bytes) -> None:
    _write_all(file_descriptor, struct.pack("!I", len(payload)) + payload)


def _read_exact(file_descriptor: int, size: int, deadline: float | None = None) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        if deadline is not None:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise TimeoutError("worker response timed out")
            readable, _, _ = select.select([file_descriptor], [], [], timeout)
            if not readable:
                raise TimeoutError("worker response timed out")
        chunk = os.read(file_descriptor, remaining)
        if not chunk:
            raise EOFError("worker closed the protocol pipe")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_frame(
    file_descriptor: int,
    *,
    maximum_size: int,
    timeout: float | None = None,
) -> bytes:
    deadline = None if timeout is None else time.monotonic() + timeout
    size = struct.unpack("!I", _read_exact(file_descriptor, 4, deadline))[0]
    if size > maximum_size:
        raise RuntimeError(f"protocol frame exceeds {maximum_size} bytes")
    return _read_exact(file_descriptor, size, deadline)
