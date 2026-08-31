"""Validated training-compute profiles and physical target definitions."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROFILE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
ACCELERATOR_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
TARGET_KIND_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
BOARD_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
MEMORY_PATTERN = re.compile(r"^[1-9][0-9]*(?:\.[0-9]+)?[kmgtKMGT]?[bB]?$")


class HardwareProfileError(RuntimeError):
    pass


def _positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise HardwareProfileError(f"{name} must be a positive finite number")
    return float(value)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HardwareProfileError(f"{name} must be a positive integer")
    return value


def _fraction(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= value <= 0.5
    ):
        raise HardwareProfileError(f"{name} must be between 0 and 0.5")
    return float(value)


@dataclass(frozen=True)
class HardwareProfile:
    profile_id: str
    accelerator: str
    description: str
    cpus: float
    memory: str
    shared_memory: str
    pids: int
    gpus: str | None
    environment: dict[str, str]
    document: dict[str, Any]

    @classmethod
    def from_document(cls, document: Any) -> "HardwareProfile":
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise HardwareProfileError("unsupported hardware profile schema")
        profile_id = document.get("id")
        accelerator = document.get("accelerator")
        description = document.get("description")
        memory = document.get("memory")
        shared_memory = document.get("shared_memory", "1g")
        gpus = document.get("gpus")
        environment = document.get("environment")
        if not isinstance(profile_id, str) or not PROFILE_ID_PATTERN.fullmatch(profile_id):
            raise HardwareProfileError("hardware profile id is invalid")
        if not isinstance(accelerator, str) or not ACCELERATOR_PATTERN.fullmatch(accelerator):
            raise HardwareProfileError("hardware accelerator is invalid")
        if not isinstance(description, str) or not description or len(description) > 1000:
            raise HardwareProfileError("hardware description is invalid")
        if not isinstance(memory, str) or not MEMORY_PATTERN.fullmatch(memory):
            raise HardwareProfileError("hardware memory limit is invalid")
        if not isinstance(shared_memory, str) or not MEMORY_PATTERN.fullmatch(shared_memory):
            raise HardwareProfileError("hardware shared memory limit is invalid")
        if gpus is not None and (
            not isinstance(gpus, str) or not gpus or len(gpus.encode("utf-8")) > 256
        ):
            raise HardwareProfileError("hardware GPU request is invalid")
        if accelerator == "cpu" and gpus is not None:
            raise HardwareProfileError("a CPU profile may not request GPUs")
        if not isinstance(environment, dict):
            raise HardwareProfileError("hardware environment must be an object")
        validated_environment: dict[str, str] = {}
        for name, value in environment.items():
            if not isinstance(name, str) or not ENVIRONMENT_NAME_PATTERN.fullmatch(name):
                raise HardwareProfileError(f"invalid hardware environment name: {name!r}")
            if not isinstance(value, str) or len(value.encode("utf-8")) > 4096 or "\x00" in value:
                raise HardwareProfileError(f"invalid hardware environment value: {name}")
            validated_environment[name] = value
        return cls(
            profile_id=profile_id,
            accelerator=accelerator,
            description=description,
            cpus=_positive_number(document.get("cpus"), "hardware cpus"),
            memory=memory,
            shared_memory=shared_memory,
            pids=_positive_integer(document.get("pids"), "hardware pids"),
            gpus=gpus,
            environment=validated_environment,
            document=document,
        )

    @classmethod
    def load(cls, path: Path) -> "HardwareProfile":
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HardwareProfileError(f"invalid hardware profile: {path}") from error
        return cls.from_document(document)

    def public_metadata(self) -> dict[str, Any]:
        return {
            "id": self.profile_id,
            "accelerator": self.accelerator,
            "description": self.description,
            "cpus": self.cpus,
            "memory": self.memory,
            "shared_memory": self.shared_memory,
            "pids": self.pids,
            "gpus": self.gpus,
            "environment": dict(self.environment),
        }


@dataclass(frozen=True)
class HardwareTarget:
    """A physical deployment target and its acceptance threshold."""

    target_id: str
    kind: str
    board: str
    description: str
    model_format: str
    runtime_seconds: float
    submission_tolerance_fraction: float
    benchmark_samples: int
    max_external_image_bytes: int
    external_flash_timeout_seconds: float
    document: dict[str, Any]

    @classmethod
    def from_document(cls, document: Any) -> "HardwareTarget":
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise HardwareProfileError("unsupported hardware target schema")
        target_id = document.get("id")
        kind = document.get("kind")
        board = document.get("board")
        description = document.get("description")
        model_format = document.get("model_format")
        if not isinstance(target_id, str) or not PROFILE_ID_PATTERN.fullmatch(target_id):
            raise HardwareProfileError("hardware target id is invalid")
        if not isinstance(kind, str) or not TARGET_KIND_PATTERN.fullmatch(kind):
            raise HardwareProfileError("hardware target kind is invalid")
        if not isinstance(board, str) or not BOARD_PATTERN.fullmatch(board):
            raise HardwareProfileError("hardware target board is invalid")
        if not isinstance(description, str) or not description or len(description) > 1000:
            raise HardwareProfileError("hardware target description is invalid")
        if model_format != "onnx":
            raise HardwareProfileError("hardware target model format must be onnx")
        return cls(
            target_id=target_id,
            kind=kind,
            board=board,
            description=description,
            model_format=model_format,
            runtime_seconds=_positive_number(
                document.get("runtime_seconds"), "hardware target runtime_seconds"
            ),
            submission_tolerance_fraction=_fraction(
                document.get("submission_tolerance_fraction"),
                "hardware target submission_tolerance_fraction",
            ),
            benchmark_samples=_positive_integer(
                document.get("benchmark_samples"),
                "hardware target benchmark_samples",
            ),
            max_external_image_bytes=_positive_integer(
                document.get("max_external_image_bytes"),
                "hardware target max_external_image_bytes",
            ),
            external_flash_timeout_seconds=_positive_number(
                document.get("external_flash_timeout_seconds"),
                "hardware target external_flash_timeout_seconds",
            ),
            document=document,
        )

    @classmethod
    def load(cls, path: Path) -> "HardwareTarget":
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HardwareProfileError(f"invalid hardware target: {path}") from error
        return cls.from_document(document)

    def allowed_runtime_seconds(
        self,
        *,
        final: bool,
        runtime_seconds: float | None = None,
    ) -> float:
        base_runtime = (
            self.runtime_seconds
            if runtime_seconds is None
            else _positive_number(runtime_seconds, "task inference runtime_seconds")
        )
        tolerance = self.submission_tolerance_fraction if final else 0.0
        return round(base_runtime * (1.0 + tolerance), 12)

    def public_metadata(self, *, runtime_seconds: float | None = None) -> dict[str, Any]:
        configured_runtime = (
            self.runtime_seconds
            if runtime_seconds is None
            else _positive_number(runtime_seconds, "task inference runtime_seconds")
        )
        return {
            "id": self.target_id,
            "kind": self.kind,
            "board": self.board,
            "description": self.description,
            "model_format": self.model_format,
            "runtime_seconds": configured_runtime,
            "profile_default_runtime_seconds": self.runtime_seconds,
            "submission_tolerance_fraction": self.submission_tolerance_fraction,
            "submission_runtime_seconds": self.allowed_runtime_seconds(
                final=True, runtime_seconds=configured_runtime
            ),
            "benchmark_samples": self.benchmark_samples,
            "max_external_image_bytes": self.max_external_image_bytes,
            "external_flash_timeout_seconds": self.external_flash_timeout_seconds,
            "quantization_owner": "agent",
        }
