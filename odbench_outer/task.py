"""Validated prepared-task manifests for trusted outer-loop assets."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .hardware import HardwareProfile, HardwareProfileError, HardwareTarget


TASK_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
INTEGER_LIMIT_NAMES = (
    "max_onnx_bytes",
    "max_agent_turns",
    "max_evaluations",
    "max_train_starts",
    "max_train_job_seconds",
    "max_total_train_seconds",
    "max_command_seconds",
)
INFERENCE_RUNTIME_LIMIT = "max_inference_runtime_seconds"


class PreparedTaskError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained_regular_file(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PreparedTaskError(f"{name} must be a relative file path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise PreparedTaskError(f"{name} escapes the prepared task")
    path = root
    for component in relative.parts:
        path = path / component
        try:
            component_stat = path.lstat()
        except FileNotFoundError as error:
            raise PreparedTaskError(f"{name} is missing: {value}") from error
        if stat.S_ISLNK(component_stat.st_mode):
            raise PreparedTaskError(f"{name} may not contain symlinks")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise PreparedTaskError(f"{name} is not a regular file")
    return path


def positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PreparedTaskError(f"{name} must be a positive integer")
    return value


def nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PreparedTaskError(f"{name} must be a non-negative integer")
    return value


def optional_positive_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return positive_integer(value, name)


def optional_positive_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise PreparedTaskError(f"{name} must be a positive finite number or null")
    return float(value)


def positive_number(value: Any, name: str) -> float:
    result = optional_positive_number(value, name)
    if result is None:
        raise PreparedTaskError(f"{name} must be a positive finite number")
    return result


def validated_limits(document: dict[str, Any]) -> dict[str, Any]:
    limits: dict[str, Any] = {
        name: positive_integer(document.get(name), name) for name in INTEGER_LIMIT_NAMES
    }
    runtime_seconds = optional_positive_number(
        document.get(INFERENCE_RUNTIME_LIMIT), INFERENCE_RUNTIME_LIMIT
    )
    if runtime_seconds is not None:
        limits[INFERENCE_RUNTIME_LIMIT] = runtime_seconds
    return limits


def validated_model_settings(document: dict[str, Any]) -> dict[str, Any]:
    reasoning_effort = document.get("reasoning_effort")
    if reasoning_effort is not None and (
        not isinstance(reasoning_effort, str) or not reasoning_effort
    ):
        raise PreparedTaskError("reasoning_effort must be a non-empty string or null")
    return {
        "max_output_tokens": positive_integer(
            document.get("max_output_tokens"), "max_output_tokens"
        ),
        "max_total_tokens": optional_positive_integer(
            document.get("max_total_tokens"), "max_total_tokens"
        ),
        "max_cost": optional_positive_number(document.get("max_cost"), "max_cost"),
        "reasoning_effort": reasoning_effort,
        "request_timeout_seconds": positive_number(
            document.get("request_timeout_seconds"), "request_timeout_seconds"
        ),
        "max_transport_retries": nonnegative_integer(
            document.get("max_transport_retries"), "max_transport_retries"
        ),
        "max_response_retries": nonnegative_integer(
            document.get("max_response_retries"), "max_response_retries"
        ),
    }


def definition_file(repo_root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PreparedTaskError(f"{name} must be a repository-relative file path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise PreparedTaskError(f"{name} escapes the repository")
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as error:
        raise PreparedTaskError(f"{name} escapes the repository") from error
    if not path.is_file():
        raise PreparedTaskError(f"{name} is missing: {value}")
    return path


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    dataset: str
    task_prompt: Path
    system_prompt: Path
    training_hardware: Path
    target_hardware: Path
    agent_image: str
    trainer_image: str
    evaluator_image: str
    objective_metric: str
    objective_mode: str
    limits: dict[str, Any]
    model: dict[str, Any]

    @classmethod
    def load(cls, repo_root: Path, task_id: str) -> "TaskDefinition":
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise PreparedTaskError("invalid task id")
        repo_root = repo_root.resolve()
        path = repo_root / "tasks" / f"{task_id}.json"
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PreparedTaskError(f"invalid task definition: {path}") from error
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise PreparedTaskError("unsupported task definition schema")
        if document.get("id") != task_id:
            raise PreparedTaskError("task definition id does not match its filename")
        dataset = document.get("dataset")
        if not isinstance(dataset, str) or not TASK_ID_PATTERN.fullmatch(dataset):
            raise PreparedTaskError("task definition has an invalid dataset id")
        prompts = document.get("prompts")
        images = document.get("images")
        evaluation = document.get("evaluation")
        limits = document.get("limits")
        model = document.get("model")
        if not all(
            isinstance(value, dict)
            for value in (prompts, images, evaluation, limits, model)
        ):
            raise PreparedTaskError("task definition sections are missing")
        image_values = [images.get(name) for name in ("agent", "trainer", "evaluator")]
        if not all(isinstance(value, str) and value for value in image_values):
            raise PreparedTaskError("task definition image names are invalid")
        objective_metric = evaluation.get("metric")
        objective_mode = evaluation.get("objective")
        if not isinstance(objective_metric, str) or not objective_metric:
            raise PreparedTaskError("task definition evaluation metric is invalid")
        if objective_mode not in {"maximize", "minimize"}:
            raise PreparedTaskError("task definition evaluation objective is invalid")
        limit_settings = validated_limits(limits)
        model_settings = validated_model_settings(model)
        return cls(
            task_id=task_id,
            dataset=dataset,
            task_prompt=definition_file(repo_root, prompts.get("task"), "task prompt"),
            system_prompt=definition_file(
                repo_root, prompts.get("system"), "system prompt"
            ),
            training_hardware=definition_file(
                repo_root, document.get("training_hardware"), "training hardware"
            ),
            target_hardware=definition_file(
                repo_root, document.get("target_hardware"), "target hardware"
            ),
            agent_image=image_values[0],
            trainer_image=image_values[1],
            evaluator_image=image_values[2],
            objective_metric=objective_metric,
            objective_mode=objective_mode,
            limits=limit_settings,
            model=model_settings,
        )


@dataclass(frozen=True)
class PreparedTask:
    task_id: str
    dataset: str
    root: Path
    task_prompt: str
    system_prompt: str
    labels: Path
    agent_image: str
    trainer_image: str
    evaluator_image: str
    training_hardware: HardwareProfile
    target_hardware: HardwareTarget
    objective_metric: str
    objective_mode: str
    max_onnx_bytes: int
    max_agent_turns: int
    max_evaluations: int
    max_train_starts: int
    max_train_job_seconds: int
    max_total_train_seconds: int
    max_command_seconds: int
    max_inference_runtime_seconds: float
    max_output_tokens: int
    max_total_tokens: int | None
    max_cost: float | None
    reasoning_effort: str | None
    model_request_timeout: float
    max_transport_retries: int
    max_response_retries: int
    manifest: dict[str, Any]

    @classmethod
    def load(cls, tasks_root: Path, task_id: str) -> "PreparedTask":
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise PreparedTaskError("invalid task id")
        resolved_tasks_root = tasks_root.resolve()
        requested_root = resolved_tasks_root / task_id
        if requested_root.is_symlink():
            raise PreparedTaskError("prepared task directory may not be a symlink")
        root = requested_root.resolve()
        if root.parent != resolved_tasks_root:
            raise PreparedTaskError("prepared task directory escapes the task store")
        manifest_path = root / "task.json"
        try:
            manifest_stat = manifest_path.lstat()
            if not stat.S_ISREG(manifest_stat.st_mode):
                raise PreparedTaskError("prepared task manifest is not a regular file")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PreparedTaskError(f"invalid prepared task manifest: {manifest_path}") from error
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 5:
            raise PreparedTaskError("unsupported prepared task manifest schema")
        if manifest.get("id") != task_id:
            raise PreparedTaskError("prepared task id does not match its directory")
        dataset = manifest.get("dataset")
        if not isinstance(dataset, str) or not TASK_ID_PATTERN.fullmatch(dataset):
            raise PreparedTaskError("prepared task has an invalid dataset id")

        prompts = manifest.get("prompts")
        private = manifest.get("private")
        images = manifest.get("images")
        training_hardware = manifest.get("training_hardware")
        target_hardware = manifest.get("target_hardware")
        evaluation = manifest.get("evaluation")
        limits = manifest.get("limits")
        model = manifest.get("model")
        if not all(
            isinstance(value, dict)
            for value in (
                prompts,
                private,
                images,
                training_hardware,
                target_hardware,
                evaluation,
                limits,
                model,
            )
        ):
            raise PreparedTaskError("prepared task sections are missing")
        task_prompt_path = contained_regular_file(root, prompts.get("task"), "task prompt")
        system_prompt_path = contained_regular_file(
            root, prompts.get("system"), "system prompt"
        )
        labels_path = contained_regular_file(root, private.get("labels"), "private labels")
        training_hardware_path = contained_regular_file(
            root, training_hardware.get("config"), "training hardware"
        )
        target_hardware_path = contained_regular_file(
            root, target_hardware.get("config"), "target hardware"
        )
        expected_hashes = {
            task_prompt_path: prompts.get("task_sha256"),
            system_prompt_path: prompts.get("system_sha256"),
            labels_path: private.get("labels_sha256"),
            training_hardware_path: training_hardware.get("config_sha256"),
            target_hardware_path: target_hardware.get("config_sha256"),
        }
        for path, expected in expected_hashes.items():
            if not isinstance(expected, str) or sha256_file(path) != expected:
                raise PreparedTaskError(f"prepared task hash mismatch: {path.name}")

        image_values = [images.get(name) for name in ("agent", "trainer", "evaluator")]
        if not all(isinstance(value, str) and value for value in image_values):
            raise PreparedTaskError("prepared task image names are invalid")
        try:
            training = HardwareProfile.load(training_hardware_path)
        except HardwareProfileError as error:
            raise PreparedTaskError(str(error)) from error
        if training_hardware.get("id") != training.profile_id:
            raise PreparedTaskError("prepared training hardware id mismatch")
        try:
            target = HardwareTarget.load(target_hardware_path)
        except HardwareProfileError as error:
            raise PreparedTaskError(str(error)) from error
        if target_hardware.get("id") != target.target_id:
            raise PreparedTaskError("prepared target hardware id mismatch")
        objective_metric = evaluation.get("metric")
        objective_mode = evaluation.get("objective")
        if not isinstance(objective_metric, str) or not objective_metric:
            raise PreparedTaskError("prepared task evaluation metric is invalid")
        if objective_mode not in {"maximize", "minimize"}:
            raise PreparedTaskError("prepared task evaluation objective is invalid")
        limit_settings = validated_limits(limits)
        model_settings = validated_model_settings(model)
        inference_runtime_seconds = limit_settings.get(
            INFERENCE_RUNTIME_LIMIT, target.runtime_seconds
        )
        return cls(
            task_id=task_id,
            dataset=dataset,
            root=root,
            task_prompt=task_prompt_path.read_text(encoding="utf-8"),
            system_prompt=system_prompt_path.read_text(encoding="utf-8"),
            labels=labels_path,
            agent_image=image_values[0],
            trainer_image=image_values[1],
            evaluator_image=image_values[2],
            training_hardware=training,
            target_hardware=target,
            objective_metric=objective_metric,
            objective_mode=objective_mode,
            max_onnx_bytes=limit_settings["max_onnx_bytes"],
            max_agent_turns=limit_settings["max_agent_turns"],
            max_evaluations=limit_settings["max_evaluations"],
            max_train_starts=limit_settings["max_train_starts"],
            max_train_job_seconds=limit_settings["max_train_job_seconds"],
            max_total_train_seconds=limit_settings["max_total_train_seconds"],
            max_command_seconds=limit_settings["max_command_seconds"],
            max_inference_runtime_seconds=inference_runtime_seconds,
            max_output_tokens=model_settings["max_output_tokens"],
            max_total_tokens=model_settings["max_total_tokens"],
            max_cost=model_settings["max_cost"],
            reasoning_effort=model_settings["reasoning_effort"],
            model_request_timeout=model_settings["request_timeout_seconds"],
            max_transport_retries=model_settings["max_transport_retries"],
            max_response_retries=model_settings["max_response_retries"],
            manifest=manifest,
        )
