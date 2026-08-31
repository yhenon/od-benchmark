from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from odbench_outer.task import PreparedTask, PreparedTaskError, sha256_file


def make_task(tasks_root: Path) -> Path:
    root = tasks_root / "cifar10"
    private = root / "private"
    private.mkdir(parents=True)
    task_prompt = root / "task.md"
    system_prompt = root / "system.md"
    labels = private / "labels.json"
    training_hardware = root / "training-hardware.json"
    target_hardware = root / "target-hardware.json"
    task_prompt.write_text("task", encoding="utf-8")
    system_prompt.write_text("system", encoding="utf-8")
    labels.write_text('{"labels": []}', encoding="utf-8")
    training_hardware.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "test-cpu",
                "accelerator": "cpu",
                "description": "Test CPU profile.",
                "cpus": 2,
                "memory": "1g",
                "shared_memory": "512m",
                "pids": 32,
                "gpus": None,
                "environment": {"OMP_NUM_THREADS": "2"},
            }
        ),
        encoding="utf-8",
    )
    target_hardware.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "nucleo-n657x0-q",
                "kind": "stm32n6",
                "board": "NUCLEO-N657X0-Q",
                "description": "Test deployment target.",
                "model_format": "onnx",
                "runtime_seconds": 0.1,
                "submission_tolerance_fraction": 0.05,
                "benchmark_samples": 10,
                "max_external_image_bytes": 4 * 1024 * 1024,
                "external_flash_timeout_seconds": 900,
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 5,
        "id": "cifar10",
        "dataset": "cifar10",
        "images": {
            "agent": "agent:image",
            "trainer": "trainer:image",
            "evaluator": "evaluator:image",
        },
        "prompts": {
            "task": "task.md",
            "task_sha256": sha256_file(task_prompt),
            "system": "system.md",
            "system_sha256": sha256_file(system_prompt),
        },
        "private": {
            "labels": "private/labels.json",
            "labels_sha256": sha256_file(labels),
        },
        "training_hardware": {
            "id": "test-cpu",
            "config": "training-hardware.json",
            "config_sha256": sha256_file(training_hardware),
        },
        "target_hardware": {
            "id": "nucleo-n657x0-q",
            "config": "target-hardware.json",
            "config_sha256": sha256_file(target_hardware),
        },
        "evaluation": {
            "metric": "top1_accuracy",
            "objective": "maximize",
        },
        "limits": {
            "max_onnx_bytes": 16 * 1024 * 1024,
            "max_agent_turns": 100,
            "max_evaluations": 20,
            "max_train_starts": 4,
            "max_train_job_seconds": 1800,
            "max_total_train_seconds": 3600,
            "max_command_seconds": 120,
            "max_inference_runtime_seconds": 0.025,
        },
        "model": {
            "max_output_tokens": 16384,
            "max_total_tokens": None,
            "max_cost": None,
            "reasoning_effort": None,
            "request_timeout_seconds": 300,
            "max_transport_retries": 3,
            "max_response_retries": 3,
        },
    }
    (root / "task.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class PreparedTaskTests(unittest.TestCase):
    def test_loads_and_resolves_trusted_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tasks_root = Path(temporary)
            root = make_task(tasks_root)
            task = PreparedTask.load(tasks_root, "cifar10")
            self.assertEqual(task.task_prompt, "task")
            self.assertEqual(task.system_prompt, "system")
            self.assertEqual(task.labels, (root / "private" / "labels.json").resolve())
            self.assertEqual(task.max_onnx_bytes, 16 * 1024 * 1024)
            self.assertEqual(
                task.training_hardware.environment["OMP_NUM_THREADS"], "2"
            )
            self.assertEqual(task.training_hardware.shared_memory, "512m")
            self.assertEqual(task.target_hardware.runtime_seconds, 0.1)
            self.assertEqual(
                task.target_hardware.allowed_runtime_seconds(final=True), 0.105
            )
            self.assertEqual(task.max_train_starts, 4)
            self.assertEqual(task.max_inference_runtime_seconds, 0.025)
            self.assertEqual(task.max_output_tokens, 16384)
            self.assertEqual(task.max_transport_retries, 3)
            self.assertEqual(task.max_response_retries, 3)

    def test_rejects_modified_private_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tasks_root = Path(temporary)
            root = make_task(tasks_root)
            (root / "private" / "labels.json").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(PreparedTaskError, "hash mismatch"):
                PreparedTask.load(tasks_root, "cifar10")

    def test_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tasks_root = Path(temporary)
            root = make_task(tasks_root)
            manifest_path = root / "task.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["private"]["labels"] = "../labels.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(PreparedTaskError, "escapes"):
                PreparedTask.load(tasks_root, "cifar10")

    def test_rejects_invalid_inference_runtime_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tasks_root = Path(temporary)
            root = make_task(tasks_root)
            manifest_path = root / "task.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["limits"]["max_inference_runtime_seconds"] = 0
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                PreparedTaskError, "max_inference_runtime_seconds"
            ):
                PreparedTask.load(tasks_root, "cifar10")


if __name__ == "__main__":
    unittest.main()
