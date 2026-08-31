"""Run ST's N6 loader with reliable ST-LINK GDB-server cleanup."""

from __future__ import annotations

import logging
import runpy
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


GDBSERVER_STOP_SECONDS = 5.0
GDBSERVER_FREQUENCY_KHZ = 200
GDBSERVER_FAILURE_SETTLE_SECONDS = 3.0


def _is_stlink_gdbserver(command: Any) -> bool:
    if not isinstance(command, (list, tuple)) or not command:
        return False
    executable = Path(str(command[0])).name.lower()
    return executable in {"st-link_gdbserver", "st-link_gdbserver.exe"}


def _with_stlink_frequency(command: Any, frequency_khz: int) -> Any:
    if not _is_stlink_gdbserver(command):
        return command
    adjusted = list(command)
    try:
        frequency_index = adjusted.index("--frequency") + 1
    except ValueError:
        return command
    if frequency_index >= len(adjusted):
        return command
    adjusted[frequency_index] = str(frequency_khz)
    return tuple(adjusted) if isinstance(command, tuple) else adjusted


def _stop_process(process: Any, *, timeout: float = GDBSERVER_STOP_SECONDS) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            return
        process.wait(timeout=timeout)


def patch_gcc_compiler(
    gcc_compiler: type[Any],
    vendor_subprocess: Any,
    *,
    frequency_khz: int = GDBSERVER_FREQUENCY_KHZ,
    failure_settle_seconds: float = GDBSERVER_FAILURE_SETTLE_SECONDS,
) -> None:
    """Ensure every GDB server spawned by ``load_and_run`` is reaped."""

    original_load_and_run = gcc_compiler.load_and_run
    if getattr(original_load_and_run, "_odbench_gdbserver_cleanup", False):
        return

    def load_and_run_with_cleanup(self: Any) -> int:
        original_popen = vendor_subprocess.Popen
        gdbservers: list[Any] = []
        return_code: int | None = None

        def tracked_popen(*args: Any, **kwargs: Any) -> Any:
            command = args[0] if args else kwargs.get("args")
            if _is_stlink_gdbserver(command):
                command = _with_stlink_frequency(command, frequency_khz)
                if args:
                    args = (command, *args[1:])
                else:
                    kwargs["args"] = command
                logger = getattr(self, "logger", None)
                if callable(logger):
                    logger(
                        logging.DEBUG,
                        f"od-benchmark forcing ST-LINK gdbserver frequency to "
                        f"{frequency_khz} kHz",
                    )
            process = original_popen(*args, **kwargs)
            if _is_stlink_gdbserver(command):
                gdbservers.append(process)
            return process

        vendor_subprocess.Popen = tracked_popen
        try:
            return_code = original_load_and_run(self)
            return return_code
        finally:
            vendor_subprocess.Popen = original_popen
            for process in gdbservers:
                _stop_process(process)
            logger = getattr(self, "logger", None)
            if gdbservers and callable(logger):
                logger(logging.DEBUG, "ST-LINK gdbserver cleanup complete")
            if gdbservers and return_code != 0 and failure_settle_seconds > 0:
                if callable(logger):
                    logger(
                        logging.DEBUG,
                        f"waiting {failure_settle_seconds:g}s for ST-LINK USB to settle",
                    )
                time.sleep(failure_settle_seconds)

    load_and_run_with_cleanup._odbench_gdbserver_cleanup = True  # type: ignore[attr-defined]
    gcc_compiler.load_and_run = load_and_run_with_cleanup


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: n6_loader_wrapper.py VENDOR_N6_LOADER [ARGS ...]")
    loader = Path(sys.argv[1]).resolve()
    if not loader.is_file():
        raise SystemExit(f"ST N6 loader is missing: {loader}")

    sys.path.insert(0, str(loader.parent))
    from n6_utils_pkg import compilers  # type: ignore[import-not-found]

    patch_gcc_compiler(compilers.GCCCompiler, compilers.subprocess)
    sys.argv = [str(loader), *sys.argv[2:]]
    runpy.run_path(str(loader), run_name="__main__")


if __name__ == "__main__":
    main()
