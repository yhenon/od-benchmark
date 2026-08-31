"""Agent-facing tool schemas and trusted host-side implementations."""

from __future__ import annotations

import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from .hardware import HardwareProfile, HardwareTarget
from .hardware_verification import NucleoN657Verifier
from .sandbox import DEFAULT_COMMAND_SECONDS, Sandbox, relative_workspace_path


MAX_ONNX_BYTES = 16 * 1024 * 1024
HARDWARE_REPORT_DETAIL_FIELDS = frozenset(
    {"generation_report", "profile_report", "raw_error"}
)


def _hardware_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Keep large vendor transcripts in report.json, not in the model context."""
    return {
        key: value
        for key, value in report.items()
        if key not in HARDWARE_REPORT_DETAIL_FIELDS
    }


def _pretrained_context(repo_root: Path) -> dict[str, Any]:
    """Expose only stable, useful registry metadata to the model."""
    path = repo_root / "docker" / "pretrained" / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "available": False,
            "models": [],
            "note": "No offline pretrained registry is installed.",
        }
    models = manifest.get("models")
    if manifest.get("schema_version") != 1 or not isinstance(models, list):
        raise ValueError(f"unsupported pretrained registry manifest: {path}")
    public_fields = (
        "id",
        "kind",
        "architecture",
        "training_dataset",
        "parameters",
        "gflops_224",
        "gflops_320",
        "feature_channels",
        "feature_reductions",
        "input_size",
        "normalization",
    )
    return {
        "available": True,
        "offline": True,
        "api": {
            "catalog": "from odbench import list_pretrained",
            "backbone": "from odbench import load_backbone",
            "detector": "from odbench import load_detector",
        },
        "models": [
            {field: model[field] for field in public_fields if field in model}
            for model in models
        ],
        "note": (
            "Weights are bundled and verified; these APIs never download. "
            "Use num_classes=11 for the VisDrone detector (background plus 10 classes). "
            "GFLOPS values are upstream reference estimates at the input size named in "
            "the field; use them for initial sizing, not as target-runtime predictions."
        ),
    }


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "workspace_exec",
            "description": (
                "Run one short Bash command in the persistent, isolated agent workspace. "
                "The command starts in /workspace with no network or GPU. Its timeout is "
                "independent of all training budgets: it defaults to 60 seconds and is "
                "clamped to the task-specific maximum shown in run_context.workspace. Use "
                "this for reading files, searching, tests, smoke tests, and small fixed-batch "
                "benchmarks. Never run a full training epoch with this tool; use train_start."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash source to execute."},
                    "timeout_seconds": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 600,
                        "default": DEFAULT_COMMAND_SECONDS,
                        "description": (
                            "Requested command timeout in seconds. Defaults to 60 and is "
                            "clamped to run_context.workspace.max_command_seconds."
                        ),
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workspace_apply_patch",
            "description": (
                "Apply a git-style unified diff in /workspace. Include `diff --git`, `---`, "
                "`+++`, and `@@` lines. Use paths prefixed with a/ and b/. This is preferred "
                "for creating and editing source files. Every `diff --git` header must begin "
                "at column 1. Send the raw Git diff only; do not include `*** Begin Patch` or "
                "`*** End Patch` wrapper lines. Structural Git warnings reject the entire "
                "patch before any file is changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {"type": "string", "description": "Git-style unified diff."}
                },
                "required": ["patch"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_start",
            "description": (
                "Snapshot /workspace and start isolated metered training. This call blocks "
                "until the job reaches an epoch boundary, fails, finishes, or exhausts its budget. "
                "The snapshot is read-only at /job/input; training code writes artifacts and "
                "checkpoints under /job/output. In epoch_end, artifact/checkpoint paths are "
                "relative to /job/output, but preprocess/postprocess paths are relative to "
                "/job/input and must exist in /workspace before this call. "
                "An epoch evaluation consumes one shared evaluation-budget slot. Arrange the "
                "first epoch_end early—prefer epoch 1 for an unbenchmarked script, and always "
                "within roughly 10-20% of the job budget—so a slow job still yields an artifact. "
                "In epoch results, train_metrics are diagnostics reported by the training script "
                "for an agent-defined model and split; evaluation.metrics are trusted hidden-set "
                "scores for the published ONNX submission. They are not directly comparable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entrypoint": {
                        "type": "string",
                        "description": "Python entrypoint relative to /workspace.",
                    },
                    "arguments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arguments passed verbatim to the Python entrypoint.",
                    },
                    "budget_seconds": {
                        "type": "number",
                        "minimum": 1,
                        "description": "Requested active training time budget.",
                    },
                    "checkpoint_path": {
                        "type": "string",
                        "description": (
                            "Optional checkpoint_path returned by a prior epoch result. Omit "
                            "this field or pass an empty string when starting from scratch. The "
                            "harness stages it into the new job and sets "
                            "ODBENCH_RESUME_CHECKPOINT to its /job/input path. Training code "
                            "must read that environment variable; do not pass the original "
                            ".odbench path to the entrypoint because agent state is excluded "
                            "from ordinary workspace snapshots."
                        ),
                    },
                },
                "required": ["entrypoint"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_continue",
            "description": (
                "Continue a paused training job after reviewing an epoch evaluation. This "
                "call blocks until the next epoch boundary or terminal event and reserves one "
                "shared evaluation-budget slot for a possible epoch event. Every "
                "train_epoch_complete result means the process remains paused inside "
                "epoch_end; it cannot exit or advance until you call train_continue or "
                "train_stop. Treat train_metrics as agent-reported diagnostics; only "
                "evaluation.metrics scores the published ONNX artifact on the hidden set."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "event_id": {"type": "string"},
                },
                "required": ["job_id", "event_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_stop",
            "description": (
                "Stop a paused or running training job and release its compute allocation. "
                "Use this after a final useful epoch boundary when you do not need the script "
                "to continue or exit naturally; the frozen artifact and checkpoint remain valid."
            ),
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate",
            "description": (
                "Evaluate a submission directory from /workspace on the hidden evaluation "
                "set. Returns trusted aggregate metrics for that submitted ONNX artifact and "
                "consumes one shared evaluation slot. Do not infer quantization loss by comparing "
                "these hidden-set metrics with training-script metrics measured on a public dev "
                "split or a float model. The ONNX file must not exceed the harness size limit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "submission_dir": {
                        "type": "string",
                        "description": "Directory relative to /workspace containing submission.json.",
                    }
                },
                "required": ["submission_dir"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_for_hw",
            "description": (
                "Compile one ONNX model for the physical hardware target without resetting or "
                "flashing the board. Returns compiler validity, accelerator/software epoch "
                "mapping, memory usage, warnings, and the run-relative report_path containing "
                "the full compiler transcript. Use this inexpensive preflight before verify_on_hw, "
                "especially after export or quantization changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to an ONNX file relative to /workspace.",
                    }
                },
                "required": ["model_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_on_hw",
            "description": (
                "Compile, flash, and benchmark one ONNX model on the physical hardware target. "
                "The trusted outer loop copies the model out of the isolated workspace and "
                "returns a compact structured summary plus a run-relative report_path for the full "
                "ST generation/profiling transcript, and pass/fail against the strict task runtime "
                "target. Quantization and ONNX export remain your responsibility. This operation "
                "is slow and reprograms the target, so use it for serious candidates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to an ONNX file relative to /workspace.",
                    }
                },
                "required": ["model_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Submit the final artifact directory. Before consuming the reserved hidden-set "
                "evaluation, the harness compiles, flashes, and benchmarks its ONNX model on "
                "the physical target using the configured submission timing tolerance. A model "
                "that cannot run or misses that limit is rejected without ending the run. A "
                "passing artifact is evaluated, preserved, and permanently ends the run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "submission_dir": {
                        "type": "string",
                        "description": "Directory relative to /workspace containing submission.json.",
                    }
                },
                "required": ["submission_dir"],
                "additionalProperties": False,
            },
        },
    },
]


class ToolRuntimeError(RuntimeError):
    pass


class ToolInvocationError(ToolRuntimeError):
    """A model-requested tool operation that was rejected safely."""


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_copied_tree(root: Path, *, max_files: int = 1000, max_bytes: int = 4 * 1024**3) -> None:
    file_count = 0
    byte_count = 0
    for current_root, directory_names, file_names in os.walk(root):
        current = Path(current_root)
        for name in directory_names:
            path = current / name
            if path.is_symlink():
                raise ToolInvocationError("submission directories may not contain symlinks")
        for name in file_names:
            path = current / name
            file_stat = path.lstat()
            if not stat.S_ISREG(file_stat.st_mode):
                raise ToolInvocationError("submission entries must be regular files")
            file_count += 1
            byte_count += file_stat.st_size
            if file_count > max_files or byte_count > max_bytes:
                raise ToolInvocationError("submission exceeds the file or byte limit")


def validate_submission_artifact(root: Path, max_onnx_bytes: int) -> int:
    manifest_path = root / "submission.json"
    try:
        manifest_stat = manifest_path.lstat()
    except FileNotFoundError as error:
        raise ToolInvocationError("submission.json is missing") from error
    if not stat.S_ISREG(manifest_stat.st_mode) or manifest_stat.st_size > 64 * 1024:
        raise ToolInvocationError(
            "submission.json must be a regular file no larger than 64 KiB"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ToolInvocationError("submission.json is invalid") from error
    artifact = manifest.get("artifact") if isinstance(manifest, dict) else None
    if not isinstance(artifact, dict) or artifact.get("format") != "onnx":
        raise ToolInvocationError("submission artifact format must be onnx")
    relative_value = artifact.get("path")
    if not isinstance(relative_value, str) or not relative_value:
        raise ToolInvocationError("submission ONNX path must be a non-empty string")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ToolInvocationError("submission ONNX path escapes its directory")
    artifact_path = root.joinpath(*relative.parts)
    try:
        artifact_stat = artifact_path.lstat()
    except FileNotFoundError as error:
        raise ToolInvocationError("submission ONNX artifact is missing") from error
    if not stat.S_ISREG(artifact_stat.st_mode):
        raise ToolInvocationError("submission ONNX artifact must be a regular file")
    if artifact_stat.st_size > max_onnx_bytes:
        raise ToolInvocationError(
            f"ONNX artifact is {artifact_stat.st_size} bytes; limit is {max_onnx_bytes} bytes"
        )
    return artifact_stat.st_size


class ToolRuntime:
    def __init__(
        self,
        *,
        repo_root: Path,
        sandbox: Sandbox,
        labels: Path,
        run_id: str,
        run_directory: Path,
        dataset: str,
        max_evaluations: int,
        max_train_starts: int,
        max_train_job_seconds: float,
        max_total_train_seconds: float,
        max_onnx_bytes: int,
        trainer_image: str,
        evaluator_image: str,
        training_hardware: HardwareProfile,
        objective_metric: str,
        objective_mode: str,
        target_hardware: HardwareTarget | None = None,
        max_inference_runtime_seconds: float | None = None,
        hardware_verifier: Any | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.sandbox = sandbox
        self.labels = labels.resolve()
        self.run_id = run_id
        self.run_directory = run_directory.resolve()
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.jobs_root = self.run_directory / "training-jobs"
        self.submission_directory = self.run_directory / "submission"
        self.max_evaluations = max_evaluations
        self.max_train_starts = max_train_starts
        self.max_train_job_seconds = float(max_train_job_seconds)
        self.max_total_train_seconds = float(max_total_train_seconds)
        self.max_onnx_bytes = max_onnx_bytes
        self.trainer_image = trainer_image
        self.evaluator_image = evaluator_image
        self.training_hardware = training_hardware
        self.target_hardware = target_hardware
        if max_inference_runtime_seconds is None:
            self.max_inference_runtime_seconds = (
                self.target_hardware.runtime_seconds
                if self.target_hardware is not None
                else None
            )
        else:
            self.max_inference_runtime_seconds = float(max_inference_runtime_seconds)
            if (
                isinstance(max_inference_runtime_seconds, bool)
                or not math.isfinite(self.max_inference_runtime_seconds)
                or self.max_inference_runtime_seconds <= 0
            ):
                raise ValueError("max_inference_runtime_seconds must be positive and finite")
        self.hardware_verifier = hardware_verifier
        if self.target_hardware is not None and self.hardware_verifier is None:
            self.hardware_verifier = NucleoN657Verifier(self.target_hardware)
        self.objective_metric = objective_metric
        self.objective_mode = objective_mode
        self.evaluation_count = 0
        self.train_start_count = 0
        self.training_active_seconds = 0.0
        self.job_active_seconds: dict[str, float] = {}
        self.active_jobs: set[str] = set()
        self.pending_events: dict[str, str] = {}
        self.published_checkpoints: dict[str, Path] = {}
        self.candidate_count = 0
        self.hardware_analysis_count = 0
        self.hardware_verification_count = 0
        self.best_candidate: dict[str, Any] | None = None
        self.best_submission_path: Path | None = None
        self.submitted = False
        self.should_stop = False
        if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int):
            raise ValueError("max_evaluations must be an integer")
        if max_evaluations < 1:
            raise ValueError("max_evaluations must allow at least the final submission")
        if isinstance(max_train_starts, bool) or not isinstance(max_train_starts, int) or max_train_starts < 1:
            raise ValueError("max_train_starts must be a positive integer")
        for value, name in (
            (self.max_train_job_seconds, "max_train_job_seconds"),
            (self.max_total_train_seconds, "max_total_train_seconds"),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if objective_mode not in {"maximize", "minimize"} or not objective_metric:
            raise ValueError("evaluation objective is invalid")
        if (
            isinstance(max_onnx_bytes, bool)
            or not isinstance(max_onnx_bytes, int)
            or not 0 < max_onnx_bytes <= MAX_ONNX_BYTES
        ):
            raise ValueError(f"max_onnx_bytes must be between 1 and {MAX_ONNX_BYTES}")
        if not self.labels.is_file():
            raise ValueError(f"private label file does not exist: {self.labels}")

    def _evaluation_budget(self) -> dict[str, Any]:
        remaining = self.max_evaluations - self.evaluation_count
        return {
            "used": self.evaluation_count,
            "limit": self.max_evaluations,
            "remaining": remaining,
            "adaptive_remaining": 0 if self.submitted else max(0, remaining - 1),
            "final_submission_reserved": not self.submitted and remaining > 0,
        }

    def _training_budget(self) -> dict[str, Any]:
        return {
            "starts_used": self.train_start_count,
            "starts_limit": self.max_train_starts,
            "starts_remaining": max(0, self.max_train_starts - self.train_start_count),
            "active_seconds_used": self.training_active_seconds,
            "active_seconds_limit": self.max_total_train_seconds,
            "active_seconds_remaining": max(
                0.0, self.max_total_train_seconds - self.training_active_seconds
            ),
            "per_job_seconds_limit": self.max_train_job_seconds,
            "active_jobs": sorted(self.active_jobs),
        }

    def budget_status(self) -> dict[str, Any]:
        return {
            "evaluation": self._evaluation_budget(),
            "training": self._training_budget(),
        }

    def agent_context(self) -> dict[str, Any]:
        return {
            "training_hardware": self.training_hardware.public_metadata(),
            "target_hardware": (
                self.target_hardware.public_metadata(
                    runtime_seconds=self.max_inference_runtime_seconds
                )
                if self.target_hardware is not None
                else None
            ),
            "budgets": self.budget_status(),
            "workspace": {
                "default_command_seconds": DEFAULT_COMMAND_SECONDS,
                "max_command_seconds": self.sandbox.max_command_seconds,
                "training_budget_independent": True,
            },
            "objective": {"metric": self.objective_metric, "mode": self.objective_mode},
            "pretrained_weights": _pretrained_context(self.repo_root),
        }

    def _reserve_evaluation(self, *, final: bool) -> None:
        allowed = self.max_evaluations if final else self.max_evaluations - 1
        if self.evaluation_count >= allowed:
            if final:
                raise ToolInvocationError("total evaluation budget exhausted")
            raise ToolInvocationError(
                "adaptive evaluation budget exhausted; the remaining slot is reserved for submit"
            )
        self.evaluation_count += 1

    def _release_evaluation_reservation(self) -> None:
        if self.evaluation_count <= 0:
            raise RuntimeError("no evaluation reservation to release")
        self.evaluation_count -= 1

    def _with_run_state(self, result: dict[str, Any]) -> dict[str, Any]:
        enriched = {
            **result,
            "evaluation_budget": self._evaluation_budget(),
            "training_budget": self._training_budget(),
            "training_hardware": self.training_hardware.public_metadata(),
            "best_candidate": self.best_candidate,
        }
        result_type = result.get("type")
        if result_type in {"train_epoch_complete", "train_epoch_failed"}:
            enriched.update(
                {
                    "job_status": "paused",
                    "required_next_action": "train_continue_or_train_stop",
                    "new_train_start_allowed": False,
                }
            )
        elif result_type in {"train_job_finished", "train_budget_exhausted"}:
            enriched.update(
                {
                    "job_status": result.get("status", "finished"),
                    "required_next_action": None,
                    "new_train_start_allowed": not self.active_jobs,
                }
            )
        elif result.get("action") == "stop":
            enriched.update(
                {
                    "job_status": "stopped",
                    "required_next_action": None,
                    "new_train_start_allowed": not self.active_jobs,
                }
            )
        return enriched

    def _account_training(self, result: dict[str, Any]) -> None:
        job_id = result.get("job_id")
        metering = result.get("metering")
        if not isinstance(job_id, str) or not isinstance(metering, dict):
            return
        active = metering.get("active_wall_seconds")
        if (
            isinstance(active, bool)
            or not isinstance(active, (int, float))
            or not math.isfinite(float(active))
            or active < 0
        ):
            return
        previous = self.job_active_seconds.get(job_id, 0.0)
        current = max(previous, float(active))
        self.training_active_seconds += current - previous
        self.job_active_seconds[job_id] = current

    def _consider_candidate(
        self,
        submission: Path,
        evaluation: Any,
        *,
        source: dict[str, Any],
    ) -> None:
        metrics = evaluation.get("metrics") if isinstance(evaluation, dict) else None
        score = metrics.get(self.objective_metric) if isinstance(metrics, dict) else None
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            return
        numeric_score = float(score)
        if self.best_candidate is not None:
            previous = float(self.best_candidate["score"])
            better = (
                numeric_score > previous
                if self.objective_mode == "maximize"
                else numeric_score < previous
            )
            if not better:
                return
        self.candidate_count += 1
        candidate_id = f"candidate-{self.candidate_count:04d}"
        candidate_root = self.run_directory / "candidates" / candidate_id
        candidate_root.parent.mkdir(parents=True, exist_ok=True)
        copied_submission = candidate_root / "submission"
        shutil.copytree(submission, copied_submission)
        validate_copied_tree(copied_submission)
        validate_submission_artifact(copied_submission, self.max_onnx_bytes)
        candidate = {
            "candidate_id": candidate_id,
            "metric": self.objective_metric,
            "objective": self.objective_mode,
            "score": numeric_score,
            "source": source,
            "recorded_at_unix": time.time(),
        }
        self.best_candidate = candidate
        self.best_submission_path = copied_submission
        atomic_json(self.run_directory / "best_candidate.json", candidate)

    def _controller_environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "ODBENCH_JOBS_ROOT": str(self.jobs_root),
            "ODBENCH_DATASET": self.dataset,
            "ODBENCH_EVAL_IMAGE": self.evaluator_image,
        }

    def _run_json(
        self,
        command: list[str],
        *,
        timeout: float | None = None,
        check: bool = True,
    ) -> dict[str, Any]:
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_root,
                env=self._controller_environment(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise ToolInvocationError(f"trusted tool timed out: {command[0]}") from error
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-4000:]
            raise ToolInvocationError(detail or f"trusted tool exited {result.returncode}")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            detail = (result.stderr or result.stdout).strip()[-4000:]
            raise ToolRuntimeError(f"trusted tool returned invalid JSON: {detail}") from error
        if not isinstance(value, dict):
            raise ToolRuntimeError("trusted tool returned a non-object response")
        return value

    def _await_training(self, job_id: str) -> dict[str, Any]:
        result = self._run_json(
            [
                str(self.repo_root / "docker" / "trainer"),
                "await",
                job_id,
                "--labels",
                str(self.labels),
            ],
            timeout=self.max_train_job_seconds + 3600,
        )
        self._account_training(result)
        event_id = result.get("event_id")
        if isinstance(event_id, str):
            staged = self.jobs_root / job_id / "staged" / event_id
            if staged.is_dir():
                validate_copied_tree(staged)
                workspace_stage = f".odbench/training/{job_id}/{event_id}"
                self.sandbox.copy_into_directory(staged, workspace_stage)
                result = {
                    **result,
                    "artifact_bytes": (staged / "submission" / "model.onnx").stat().st_size,
                    "max_onnx_bytes": self.max_onnx_bytes,
                    "submission_dir": f"{workspace_stage}/submission",
                    "checkpoint_path": (
                        f"{workspace_stage}/checkpoint.pt"
                        if (staged / "checkpoint.pt").is_file()
                        else None
                    ),
                }
                checkpoint_path = result.get("checkpoint_path")
                if isinstance(checkpoint_path, str):
                    self.published_checkpoints[checkpoint_path] = staged / "checkpoint.pt"
                if result.get("evaluation") is not None:
                    self._consider_candidate(
                        staged / "submission",
                        result["evaluation"],
                        source={
                            "tool": "epoch_end",
                            "job_id": job_id,
                            "event_id": event_id,
                            "epoch": result.get("epoch"),
                            "workspace_submission_dir": result["submission_dir"],
                        },
                    )
        return result

    def _evaluate_directory(self, submission: Path, *, final: bool = False) -> dict[str, Any]:
        self._reserve_evaluation(final=final)
        result = self._run_json(
            [
                str(self.repo_root / "docker" / "evaluator"),
                "evaluate",
                str(submission),
                str(self.labels),
            ],
            timeout=1800,
        )
        return self._with_run_state(result)

    def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ToolInvocationError("tool arguments must be an object")
        if name == "workspace_exec":
            return self.sandbox.exec(
                arguments.get("command"),
                arguments.get("timeout_seconds", DEFAULT_COMMAND_SECONDS),
            )
        if name == "workspace_apply_patch":
            patch = arguments.get("patch")
            if not isinstance(patch, str):
                raise ToolInvocationError("patch must be a string")
            return self.sandbox.apply_patch(patch)
        if name == "train_start":
            return self.train_start(arguments)
        if name == "train_continue":
            return self.train_continue(arguments)
        if name == "train_stop":
            return self.train_stop(arguments)
        if name == "evaluate":
            return self.evaluate(arguments)
        if name == "analyze_for_hw":
            return self.analyze_for_hw(arguments)
        if name == "verify_on_hw":
            return self.verify_on_hw(arguments)
        if name == "submit":
            return self.submit(arguments)
        raise ToolInvocationError(f"unknown tool: {name}")

    def train_start(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.active_jobs:
            job_id = sorted(self.active_jobs)[0]
            event_id = self.pending_events.get(job_id)
            detail = f"active training job {job_id}"
            if event_id is not None:
                detail += f" is paused at {event_id}"
            raise ToolInvocationError(
                f"{detail}; call train_continue or train_stop before train_start"
            )
        if self.train_start_count >= self.max_train_starts:
            raise ToolInvocationError("training start budget exhausted")
        total_remaining = self.max_total_train_seconds - self.training_active_seconds
        if total_remaining < 1:
            raise ToolInvocationError("total active training-time budget exhausted")
        entrypoint = relative_workspace_path(arguments.get("entrypoint"), name="entrypoint")
        training_arguments = arguments.get("arguments", [])
        if not isinstance(training_arguments, list) or not all(
            isinstance(item, str) for item in training_arguments
        ):
            raise ToolInvocationError("arguments must be an array of strings")
        budget = arguments.get("budget_seconds", self.max_train_job_seconds)
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget <= 0:
            raise ToolInvocationError("budget_seconds must be positive")
        budget = min(float(budget), self.max_train_job_seconds, total_remaining)
        checkpoint_value = arguments.get("checkpoint_path")
        if isinstance(checkpoint_value, str) and not checkpoint_value.strip():
            checkpoint_value = None
        checkpoint_source = None
        if checkpoint_value is not None:
            checkpoint_path = relative_workspace_path(
                checkpoint_value, name="checkpoint_path"
            )
            checkpoint_source = self.published_checkpoints.get(checkpoint_path)
            if checkpoint_source is None or not checkpoint_source.is_file():
                raise ToolInvocationError(
                    "checkpoint_path must be an available path returned by a prior epoch result"
                )
        command = [
            str(self.repo_root / "docker" / "trainer"),
            "start",
            "--agent-container",
            self.sandbox.container,
            "--entrypoint",
            entrypoint,
            "--dataset",
            self.dataset,
            "--image",
            self.trainer_image,
            "--budget-seconds",
            str(budget),
            "--max-onnx-bytes",
            str(self.max_onnx_bytes),
            "--memory",
            self.training_hardware.memory,
            "--shm-size",
            self.training_hardware.shared_memory,
            "--cpus",
            str(self.training_hardware.cpus),
            "--pids",
            str(self.training_hardware.pids),
        ]
        if checkpoint_source is not None:
            command.extend(["--resume-checkpoint", str(checkpoint_source)])
        if self.training_hardware.gpus is not None:
            command.extend(["--gpus", self.training_hardware.gpus])
        for name, value in sorted(self.training_hardware.environment.items()):
            command.extend(["--environment", f"{name}={value}"])
        if training_arguments:
            command.extend(["--", *training_arguments])
        self._reserve_evaluation(final=False)
        try:
            started = self._run_json(command, timeout=120)
        except Exception:
            self._release_evaluation_reservation()
            raise
        job_id = started.get("job_id")
        if not isinstance(job_id, str):
            raise ToolRuntimeError("training controller did not return a job id")
        self.train_start_count += 1
        self.active_jobs.add(job_id)
        result = self._await_training(job_id)
        event_id = result.get("event_id")
        if isinstance(event_id, str):
            self.pending_events[job_id] = event_id
        if not isinstance(result.get("event_id"), str):
            self._release_evaluation_reservation()
        if result.get("type") in {"train_job_finished", "train_budget_exhausted"}:
            self.active_jobs.discard(job_id)
            self.pending_events.pop(job_id, None)
        return self._with_run_state(result)

    def train_continue(self, arguments: dict[str, Any]) -> dict[str, Any]:
        job_id = arguments.get("job_id")
        event_id = arguments.get("event_id")
        if not isinstance(job_id, str) or job_id not in self.active_jobs:
            raise ToolInvocationError("unknown training job id")
        if not isinstance(event_id, str):
            raise ToolInvocationError("event_id must be a string")
        pending_event = self.pending_events.get(job_id)
        if pending_event is not None and event_id != pending_event:
            raise ToolInvocationError(
                f"job {job_id} is paused at {pending_event}, not {event_id}"
            )
        if self.max_total_train_seconds - self.training_active_seconds < 1:
            raise ToolInvocationError("total active training-time budget exhausted")
        self._reserve_evaluation(final=False)
        try:
            self._run_json(
                [str(self.repo_root / "docker" / "trainer"), "continue", job_id, event_id],
                timeout=30,
            )
        except Exception:
            self._release_evaluation_reservation()
            raise
        result = self._await_training(job_id)
        self.pending_events.pop(job_id, None)
        next_event_id = result.get("event_id")
        if isinstance(next_event_id, str):
            self.pending_events[job_id] = next_event_id
        if not isinstance(result.get("event_id"), str):
            self._release_evaluation_reservation()
        if result.get("type") in {"train_job_finished", "train_budget_exhausted"}:
            self.active_jobs.discard(job_id)
            self.pending_events.pop(job_id, None)
        return self._with_run_state(result)

    def train_stop(self, arguments: dict[str, Any]) -> dict[str, Any]:
        job_id = arguments.get("job_id")
        if not isinstance(job_id, str) or job_id not in self.active_jobs:
            raise ToolInvocationError("unknown training job id")
        result = self._run_json(
            [str(self.repo_root / "docker" / "trainer"), "stop", job_id], timeout=30
        )
        self.active_jobs.discard(job_id)
        self.pending_events.pop(job_id, None)
        self._account_training(result)
        return self._with_run_state(result)

    def evaluate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = relative_workspace_path(
            arguments.get("submission_dir"), name="submission_dir"
        )
        with tempfile.TemporaryDirectory(prefix="odbench-evaluate-") as temporary:
            submission = Path(temporary) / "submission"
            self.sandbox.copy_directory(relative, submission)
            validate_copied_tree(submission)
            artifact_bytes = validate_submission_artifact(submission, self.max_onnx_bytes)
            result = self._evaluate_directory(submission)
            self._consider_candidate(
                submission,
                result,
                source={"tool": "evaluate", "workspace_submission_dir": relative},
            )
            return {
                **result,
                "artifact_bytes": artifact_bytes,
                "max_onnx_bytes": self.max_onnx_bytes,
                "training_budget": self._training_budget(),
                "best_candidate": self.best_candidate,
            }

    def _verify_hardware_model(self, model: Path, *, final: bool) -> dict[str, Any]:
        if self.target_hardware is None or self.hardware_verifier is None:
            raise ToolInvocationError("this task has no physical hardware target")
        self.hardware_verification_count += 1
        relative_report = Path("hardware") / f"verification-{self.hardware_verification_count:04d}"
        report_directory = self.run_directory / relative_report
        report = self.hardware_verifier.verify(
            model,
            allowed_runtime_seconds=self.target_hardware.allowed_runtime_seconds(
                final=final,
                runtime_seconds=self.max_inference_runtime_seconds,
            ),
            acceptance_mode="submission" if final else "strict",
            report_directory=report_directory,
        )
        if not isinstance(report, dict):
            raise ToolRuntimeError("hardware verifier returned a non-object report")
        report_path = relative_report.as_posix() + "/report.json"
        return {
            **_hardware_report_summary(report),
            "verification_call": self.hardware_verification_count,
            "report_path": report_path,
        }

    def _analyze_hardware_model(self, model: Path) -> dict[str, Any]:
        if self.target_hardware is None or self.hardware_verifier is None:
            raise ToolInvocationError("this task has no physical hardware target")
        self.hardware_analysis_count += 1
        relative_report = Path("hardware") / f"analysis-{self.hardware_analysis_count:04d}"
        report_directory = self.run_directory / relative_report
        report = self.hardware_verifier.analyze(
            model,
            report_directory=report_directory,
        )
        if not isinstance(report, dict):
            raise ToolRuntimeError("hardware verifier returned a non-object analysis")
        report_path = relative_report.as_posix() + "/report.json"
        return {
            **_hardware_report_summary(report),
            "analysis_call": self.hardware_analysis_count,
            "report_path": report_path,
        }

    @staticmethod
    def _validate_hardware_model_path(arguments: dict[str, Any]) -> str:
        relative = relative_workspace_path(arguments.get("model_path"), name="model_path")
        if PurePosixPath(relative).suffix.lower() != ".onnx":
            raise ToolInvocationError("model_path must name an .onnx file")
        return relative

    def _copy_hardware_model(self, relative: str, destination: Path) -> None:
        self.sandbox.copy_file(relative, destination, max_bytes=self.max_onnx_bytes)
        model_stat = destination.lstat()
        if not stat.S_ISREG(model_stat.st_mode):
            raise ToolInvocationError("hardware model must be a regular file")
        if model_stat.st_size > self.max_onnx_bytes:
            raise ToolInvocationError(
                f"ONNX artifact is {model_stat.st_size} bytes; "
                f"limit is {self.max_onnx_bytes} bytes"
            )

    def analyze_for_hw(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = self._validate_hardware_model_path(arguments)
        with tempfile.TemporaryDirectory(prefix="odbench-hardware-analysis-") as temporary:
            model = Path(temporary) / "model.onnx"
            self._copy_hardware_model(relative, model)
            return {
                **self._analyze_hardware_model(model),
                "workspace_model_path": relative,
                "max_onnx_bytes": self.max_onnx_bytes,
            }

    def verify_on_hw(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = self._validate_hardware_model_path(arguments)
        with tempfile.TemporaryDirectory(prefix="odbench-hardware-model-") as temporary:
            model = Path(temporary) / "model.onnx"
            self._copy_hardware_model(relative, model)
            return {
                **self._verify_hardware_model(model, final=False),
                "workspace_model_path": relative,
                "max_onnx_bytes": self.max_onnx_bytes,
            }

    def submit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.submitted:
            raise ToolInvocationError("this run has already submitted")
        relative = relative_workspace_path(
            arguments.get("submission_dir"), name="submission_dir"
        )
        with tempfile.TemporaryDirectory(
            prefix=".submission-", dir=self.run_directory
        ) as temporary:
            staged = Path(temporary) / "submission"
            self.sandbox.copy_directory(relative, staged)
            return self._accept_staged_submission(
                staged,
                automatic=False,
                reason="agent_submit",
            )

    def submit_best_candidate(self, reason: str) -> dict[str, Any] | None:
        if self.submitted or self.best_submission_path is None:
            return None
        with tempfile.TemporaryDirectory(
            prefix=".best-submission-", dir=self.run_directory
        ) as temporary:
            staged = Path(temporary) / "submission"
            shutil.copytree(self.best_submission_path, staged)
            return self._accept_staged_submission(
                staged,
                automatic=True,
                reason=reason,
            )

    def _accept_staged_submission(
        self,
        staged: Path,
        *,
        automatic: bool,
        reason: str,
    ) -> dict[str, Any]:
        destination = self.submission_directory
        if destination.exists():
            raise ToolRuntimeError(f"submission destination already exists: {destination}")
        validate_copied_tree(staged)
        artifact_bytes = validate_submission_artifact(staged, self.max_onnx_bytes)
        manifest = json.loads((staged / "submission.json").read_text(encoding="utf-8"))
        artifact_relative = PurePosixPath(manifest["artifact"]["path"])
        artifact_path = staged.joinpath(*artifact_relative.parts)
        hardware_verification = None
        if self.target_hardware is not None:
            hardware_verification = self._verify_hardware_model(artifact_path, final=True)
            if not hardware_verification.get("passed"):
                runtime = hardware_verification.get("duration_seconds")
                allowed = hardware_verification.get("allowed_runtime_seconds")
                detail = hardware_verification.get("error") or (
                    f"measured {runtime}s; allowed {allowed}s"
                )
                raise ToolInvocationError(
                    "final submission failed physical hardware verification at "
                    f"{hardware_verification.get('stage')}: {detail}"
                )
        evaluation = self._evaluate_directory(staged, final=True)
        os.replace(staged, destination)
        self.submitted = True
        self.should_stop = True
        submitted_budget = self._evaluation_budget()
        evaluation = {
            **evaluation,
            "evaluation_budget": submitted_budget,
            "training_budget": self._training_budget(),
        }
        result = {
            "schema_version": 1,
            "type": "submission_accepted",
            "run_id": self.run_id,
            "evaluation": evaluation,
            "evaluation_calls_used": self.evaluation_count,
            "evaluation_budget": submitted_budget,
            "training_budget": self._training_budget(),
            "best_candidate": self.best_candidate,
            "artifact_bytes": artifact_bytes,
            "max_onnx_bytes": self.max_onnx_bytes,
            "hardware_verification": hardware_verification,
            "submitted_at_unix": time.time(),
            "automatic": automatic,
            "reason": reason,
        }
        atomic_json(self.run_directory / "submission-result.json", result)
        self.stop_active_jobs()
        return result

    def stop_active_jobs(self) -> None:
        first_error: Exception | None = None
        for job_id in tuple(self.active_jobs):
            try:
                result = self._run_json(
                    [str(self.repo_root / "docker" / "trainer"), "stop", job_id],
                    timeout=30,
                    check=False,
                )
                self._account_training(result)
            except Exception as error:
                if first_error is None:
                    first_error = error
            self.active_jobs.discard(job_id)
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        self.stop_active_jobs()
