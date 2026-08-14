"""Validate and execute a Python entrypoint from the immutable job snapshot."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: training-image ENTRYPOINT [ARG ...]")
    input_root = Path("/job/input").resolve()
    entrypoint = (input_root / sys.argv[1]).resolve()
    if input_root not in entrypoint.parents or not entrypoint.is_file():
        raise SystemExit("training entrypoint must be a file under /job/input")
    os.chdir(input_root)
    os.execv(sys.executable, [sys.executable, str(entrypoint), *sys.argv[2:]])


if __name__ == "__main__":
    main()
