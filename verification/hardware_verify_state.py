#!/usr/bin/env python3
"""Run the current Nucleo verifier and record the hardware state it leaves behind."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from odbench_outer.hardware import HardwareTarget
from odbench_outer.hardware_verification import NucleoN657Verifier


TARGET_PATH = REPO_ROOT / "hardware" / "targets" / "nucleo-n657x0-q.json"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_tiny_relu_model(path: Path) -> None:
    """Create the parameter-free model that previously reached build_and_load."""

    graph = helper.make_graph(
        [helper.make_node("Relu", ["input"], ["output"])],
        "tiny_relu",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])],
    )
    model = helper.make_model(
        graph,
        producer_name="odbench-hardware-state-diagnostic",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    onnx.save_model(model, path)


def run_command(command: list[str], *, timeout: float = 20) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "argv": command,
            "elapsed_seconds": time.monotonic() - started,
            "error": str(error),
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }
    return {
        "argv": command,
        "elapsed_seconds": time.monotonic() - started,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def process_snapshot() -> dict[str, Any]:
    return {
        "gdb_processes": run_command(
            ["pgrep", "-afil", "ST-LINK_gdbserver|arm-none-eabi-gdb|n6_loader.py"]
        ),
        "gdb_port": run_command(
            ["lsof", "-nP", "-iTCP:61234", "-sTCP:LISTEN"]
        ),
    }


def probe_command(verifier: NucleoN657Verifier) -> list[str]:
    cubeide = verifier._cubeide_path()
    programmer = verifier._programmer(cubeide)
    command = [str(programmer), "-q", "-c", "port=SWD"]
    stlink_serial = verifier.environment.get("ODBENCH_STLINK_SERIAL")
    if stlink_serial:
        command.append(f"sn={stlink_serial}")
    command.extend(["mode=HOTPLUG", "freq=200", "ap=1"])
    return command


def probe(verifier: NucleoN657Verifier) -> dict[str, Any]:
    result = run_command(probe_command(verifier))
    output = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    result["healthy"] = (
        result.get("returncode") == 0
        and "Device ID" in output
        and "DEV_USB_COMM_ERR" not in output
        and "Unable to get core ID" not in output
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        help="ONNX model to verify; defaults to a generated parameter-free ReLU model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New diagnostic output directory (default: runs/hardware-state-<timestamp>)",
    )
    parser.add_argument(
        "--run-if-unhealthy",
        action="store_true",
        help="Run verification even when the preflight HotPlug probe fails",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or REPO_ROOT / "runs" / f"hardware-state-{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)

    model = args.model.resolve() if args.model else output / "tiny-relu.onnx"
    if args.model is None:
        make_tiny_relu_model(model)
    if not model.is_file():
        raise FileNotFoundError(f"model does not exist: {model}")

    target = HardwareTarget.load(TARGET_PATH)
    verifier = NucleoN657Verifier(target, environment=os.environ)

    before = {
        "recorded_at_unix": time.time(),
        "probe": probe(verifier),
        "processes": process_snapshot(),
    }
    write_json(output / "before.json", before)
    if not before["probe"]["healthy"] and not args.run_if_unhealthy:
        summary = {
            "completed": False,
            "reason": "preflight probe was unhealthy; verifier was not run",
            "model": str(model),
            "output": str(output),
        }
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 2

    report = verifier.verify(
        model,
        allowed_runtime_seconds=target.runtime_seconds,
        acceptance_mode="state-diagnostic",
        report_directory=output / "verification",
    )

    immediate = {
        "recorded_at_unix": time.time(),
        "probe": probe(verifier),
        "processes": process_snapshot(),
    }
    write_json(output / "after-immediate.json", immediate)
    time.sleep(2)
    settled = {
        "recorded_at_unix": time.time(),
        "probe": probe(verifier),
        "processes": process_snapshot(),
    }
    write_json(output / "after-2s.json", settled)

    summary = {
        "completed": True,
        "model": str(model),
        "output": str(output),
        "verification_stage": report.get("stage"),
        "verification_passed": report.get("passed"),
        "verification_error": report.get("error"),
        "probe_healthy_before": before["probe"]["healthy"],
        "probe_healthy_immediately_after": immediate["probe"]["healthy"],
        "probe_healthy_after_2s": settled["probe"]["healthy"],
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
