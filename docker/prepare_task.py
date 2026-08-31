#!/usr/bin/env python3
"""Prepare one dataset task and its trusted outer-loop assets."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import uuid
from pathlib import Path

from odbench_outer.task import PreparedTaskError, TaskDefinition, sha256_file
from odbench_outer.tools import MAX_ONNX_BYTES
from odbench_outer.hardware import HardwareProfile, HardwareProfileError, HardwareTarget


REPO_ROOT = Path(__file__).resolve().parents[1]


def atomic_bytes(path: Path, value: bytes, mode: int = 0o600) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(mode)
    os.replace(temporary, path)


def run(command: list[str], environment: dict[str, str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("task")
    result.add_argument(
        "--tasks-root",
        type=Path,
        default=REPO_ROOT / ".odbench" / "prepared-tasks",
    )
    result.add_argument("--skip-build", action="store_true")
    result.add_argument(
        "--skip-trainer-build",
        action="store_true",
        help="Build the local agent and evaluator but provision the trainer separately.",
    )
    result.add_argument(
        "--train-data",
        type=Path,
        help="Local training split root for datasets that are not downloaded at build time.",
    )
    result.add_argument(
        "--eval-data",
        type=Path,
        help="Local validation/evaluation split root for datasets that are not downloaded.",
    )
    return result


def main() -> None:
    os.umask(0o077)
    arguments = parser().parse_args()
    try:
        definition = TaskDefinition.load(REPO_ROOT, arguments.task)
    except PreparedTaskError as error:
        raise SystemExit(str(error)) from error
    dataset = definition.dataset
    max_onnx_bytes = definition.limits["max_onnx_bytes"]
    if not 0 < max_onnx_bytes <= MAX_ONNX_BYTES:
        raise SystemExit(f"max ONNX size must be between 1 byte and {MAX_ONNX_BYTES} bytes")
    try:
        training_hardware = HardwareProfile.load(definition.training_hardware)
        target_hardware = HardwareTarget.load(definition.target_hardware)
    except HardwareProfileError as error:
        raise SystemExit(str(error)) from error

    required = (
        REPO_ROOT / "docker" / "datasets" / dataset / "prepare.py",
        REPO_ROOT / "docker" / "datasets" / dataset / "runtime.py",
        REPO_ROOT / "docker" / "eval" / "datasets" / dataset / "prepare.py",
        REPO_ROOT / "docker" / "eval" / "datasets" / dataset / "runtime.py",
        definition.task_prompt,
        definition.system_prompt,
    )
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"task inputs are missing: {', '.join(missing)}")

    environment = {
        **os.environ,
        "ODBENCH_DATASET": dataset,
        "ODBENCH_IMAGE": definition.agent_image,
        "ODBENCH_EVAL_IMAGE": definition.evaluator_image,
    }
    if dataset in {"visdrone", "widerface"}:
        if dataset == "visdrone":
            default_train_data = REPO_ROOT / "data" / "VisDrone2019-DET-train"
            default_eval_data = REPO_ROOT / "data" / "VisDrone2019-DET-val"
            display_name = "VisDrone"
        else:
            # Both image archives share the official wider_face_split directory.
            default_train_data = REPO_ROOT / "data"
            default_eval_data = REPO_ROOT / "data"
            display_name = "WIDER FACE"
        train_data = arguments.train_data or default_train_data
        eval_data = arguments.eval_data or default_eval_data
        for value, name in ((train_data, "train"), (eval_data, "evaluation")):
            if not value.is_dir():
                raise SystemExit(f"{display_name} {name} data is not a directory: {value}")
        environment["ODBENCH_TRAIN_DATA_SOURCE"] = str(train_data.resolve())
        environment["ODBENCH_EVAL_DATA_SOURCE"] = str(eval_data.resolve())
    if not arguments.skip_build:
        run([str(REPO_ROOT / "docker" / "sandbox"), "build"], environment)
        if not arguments.skip_trainer_build:
            run(
                [
                    str(REPO_ROOT / "docker" / "trainer"),
                    "build",
                    "--dataset",
                    dataset,
                    "--base-image",
                    definition.agent_image,
                    "--image",
                    definition.trainer_image,
                ],
                environment,
            )
        run([str(REPO_ROOT / "docker" / "evaluator"), "build"], environment)

    task_root = arguments.tasks_root.resolve() / definition.task_id
    task_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    task_root.chmod(0o700)
    labels_directory = task_root / "private"
    labels_path = labels_directory / "labels.json"
    if not labels_directory.exists():
        run(
            [str(REPO_ROOT / "docker" / "evaluator"), "export-labels", str(labels_directory)],
            environment,
        )
    if not labels_path.is_file():
        raise SystemExit(f"prepared label file is missing: {labels_path}")
    try:
        labels_document = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"prepared label file is invalid: {labels_path}") from error
    if (
        labels_document.get("schema_version") != 1
        or labels_document.get("dataset") != dataset
        or not isinstance(labels_document.get("num_examples"), int)
    ):
        raise SystemExit("prepared labels do not match the task dataset")

    task_prompt_path = task_root / "task.md"
    system_prompt_path = task_root / "system.md"
    training_hardware_path = task_root / "training-hardware.json"
    target_hardware_path = task_root / "target-hardware.json"
    atomic_bytes(task_prompt_path, definition.task_prompt.read_bytes())
    atomic_bytes(system_prompt_path, definition.system_prompt.read_bytes())
    atomic_bytes(
        training_hardware_path,
        (json.dumps(training_hardware.document, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    atomic_bytes(
        target_hardware_path,
        (json.dumps(target_hardware.document, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    manifest = {
        "schema_version": 5,
        "id": definition.task_id,
        "dataset": dataset,
        "prepared_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "images": {
            "agent": definition.agent_image,
            "trainer": definition.trainer_image,
            "evaluator": definition.evaluator_image,
        },
        "prompts": {
            "task": "task.md",
            "task_sha256": sha256_file(task_prompt_path),
            "system": "system.md",
            "system_sha256": sha256_file(system_prompt_path),
        },
        "private": {
            "labels": "private/labels.json",
            "labels_sha256": sha256_file(labels_path),
            "source": "development-label-export",
        },
        "training_hardware": {
            "id": training_hardware.profile_id,
            "config": "training-hardware.json",
            "config_sha256": sha256_file(training_hardware_path),
        },
        "target_hardware": {
            "id": target_hardware.target_id,
            "config": "target-hardware.json",
            "config_sha256": sha256_file(target_hardware_path),
        },
        "evaluation": {
            "split": labels_document.get("split"),
            "metric": definition.objective_metric,
            "objective": definition.objective_mode,
            "num_examples": labels_document["num_examples"],
        },
        "limits": definition.limits,
        "model": definition.model,
    }
    atomic_bytes(
        task_root / "task.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "task": definition.task_id,
                "manifest": str(task_root / "task.json"),
                "images": manifest["images"],
                "num_examples": manifest["evaluation"]["num_examples"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
