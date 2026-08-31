"""Exercise train-to-workspace publication and final submission."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from odbench_outer.sandbox import Sandbox
from odbench_outer.task import PreparedTask
from odbench_outer.tools import ToolRuntime


REPO_ROOT = Path(__file__).resolve().parents[1]


class PassingHardwareVerifier:
    def verify(
        self,
        model: Path,
        *,
        allowed_runtime_seconds: float,
        acceptance_mode: str,
        report_directory: Path,
    ) -> dict[str, object]:
        report_directory.mkdir(parents=True, exist_ok=False)
        report = {
            "type": "hardware_verification",
            "passed": True,
            "stage": "complete",
            "duration_seconds": min(0.001, allowed_runtime_seconds),
            "allowed_runtime_seconds": allowed_runtime_seconds,
            "acceptance_mode": acceptance_mode,
        }
        (report_directory / "report.json").write_text(json.dumps(report))
        return report


def add_file_patch(relative: str, content: str) -> str:
    lines = content.splitlines()
    body = "\n".join(f"+{line}" for line in lines)
    if content.endswith("\n"):
        body += "\n"
    return (
        f"diff --git a/{relative} b/{relative}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{relative}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{body}"
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: integration_outer_tools.py TASKS_ROOT RUNS_ROOT")
    tasks_root, runs_root = map(Path, sys.argv[1:])
    prepared = PreparedTask.load(tasks_root, "cifar10")
    run_id = f"run-integration-{uuid.uuid4().hex[:8]}"
    sandbox = Sandbox(
        repo_root=REPO_ROOT,
        container=f"odbench-agent-integration-{uuid.uuid4().hex[:8]}",
        dataset=prepared.dataset,
        image=prepared.agent_image,
        max_command_seconds=10,
    )
    runtime = None
    try:
        sandbox.start()
        source = REPO_ROOT / "examples" / "cifar10-train"
        patch = "".join(
            add_file_patch(name, (source / name).read_text(encoding="utf-8"))
            for name in ("train.py", "preprocess.py", "postprocess.py")
        )
        assert sandbox.apply_patch(patch)["applied"]
        workspace_files = sandbox.exec("find . -maxdepth 2 -type f -print | sort", 5)
        assert workspace_files["exit_code"] == 0, workspace_files
        assert "./train.py\n" in workspace_files["stdout"], workspace_files
        runtime = ToolRuntime(
            repo_root=REPO_ROOT,
            sandbox=sandbox,
            labels=prepared.labels,
            run_id=run_id,
            run_directory=runs_root / run_id,
            dataset=prepared.dataset,
            max_evaluations=prepared.max_evaluations,
            max_train_starts=3,
            max_train_job_seconds=300,
            max_total_train_seconds=300,
            max_onnx_bytes=prepared.max_onnx_bytes,
            trainer_image=prepared.trainer_image,
            evaluator_image=prepared.evaluator_image,
            training_hardware=prepared.training_hardware,
            target_hardware=prepared.target_hardware,
            max_inference_runtime_seconds=prepared.max_inference_runtime_seconds,
            hardware_verifier=PassingHardwareVerifier(),
            objective_metric=prepared.objective_metric,
            objective_mode=prepared.objective_mode,
        )
        epoch = runtime.invoke(
            "train_start",
            {"entrypoint": "train.py", "budget_seconds": 300},
        )
        assert epoch["type"] == "train_epoch_complete", epoch
        assert epoch["event_id"] == "epoch-000000", epoch
        assert epoch["job_status"] == "paused", epoch
        assert epoch["required_next_action"] == "train_continue_or_train_stop", epoch
        assert not epoch["new_train_start_allowed"], epoch
        assert epoch["evaluation_budget"]["used"] == 1, epoch
        assert epoch["training_budget"]["starts_used"] == 1, epoch
        assert epoch["training_budget"]["active_seconds_used"] > 0, epoch
        assert epoch["training_hardware"]["id"] == "local-cpu", epoch
        assert epoch["train_metrics"]["odbench_cpus"] == "4", epoch
        assert epoch["best_candidate"]["score"] == epoch["evaluation"]["metrics"]["top1_accuracy"]
        assert (runtime.run_directory / "best_candidate.json").is_file()
        assert epoch["evaluation_budget"]["final_submission_reserved"], epoch
        assert 0 < epoch["artifact_bytes"] <= epoch["max_onnx_bytes"] == 16 * 1024 * 1024
        assert epoch["submission_dir"].endswith("/submission"), epoch
        visible = sandbox.exec(f"test -f {epoch['submission_dir']}/model.onnx", 5)
        assert visible["exit_code"] == 0, visible

        stopped = runtime.invoke("train_stop", {"job_id": epoch["job_id"]})
        assert stopped["job_status"] == "stopped", stopped
        assert stopped["new_train_start_allowed"], stopped

        resumed = runtime.invoke(
            "train_start",
            {
                "entrypoint": "train.py",
                "budget_seconds": 300,
                "checkpoint_path": epoch["checkpoint_path"],
            },
        )
        assert resumed["type"] == "train_epoch_complete", resumed
        assert resumed["event_id"] == "epoch-000001", resumed
        assert resumed["train_metrics"]["resumed"] is True, resumed
        assert resumed["job_status"] == "paused", resumed
        runtime.invoke("train_stop", {"job_id": resumed["job_id"]})

        failure_patch = add_file_patch(
            "fail.py",
            'import sys\nprint("intentional training failure", file=sys.stderr)\nraise RuntimeError("boom")\n',
        )
        assert sandbox.apply_patch(failure_patch)["applied"]
        failed = runtime.invoke(
            "train_start",
            {"entrypoint": "fail.py", "budget_seconds": 30},
        )
        assert failed["type"] == "train_job_finished", failed
        assert failed["job_status"] == "failed", failed
        assert failed["exit_code"] != 0, failed
        assert "intentional training failure" in failed["stderr_tail"], failed
        assert failed["new_train_start_allowed"], failed

        submitted = runtime.invoke("submit", {"submission_dir": resumed["submission_dir"]})
        assert submitted["type"] == "submission_accepted", submitted
        assert submitted["evaluation"]["num_examples"] == 10_000, submitted
        assert submitted["evaluation_calls_used"] == 3, submitted
        assert submitted["evaluation_budget"]["used"] == 3, submitted
        assert not submitted["evaluation_budget"]["final_submission_reserved"], submitted
        assert submitted["artifact_bytes"] == resumed["artifact_bytes"], submitted
        assert submitted["max_onnx_bytes"] == 16 * 1024 * 1024, submitted
        assert (runtime.run_directory / "submission" / "model.onnx").is_file()
        assert (runtime.run_directory / "submission-result.json").is_file()
        print(
            json.dumps(
                {
                    "epoch_accuracy": epoch["evaluation"]["metrics"]["top1_accuracy"],
                    "evaluation_calls_used": submitted["evaluation_calls_used"],
                    "submission_accuracy": submitted["evaluation"]["metrics"]["top1_accuracy"],
                    "published_submission": epoch["submission_dir"],
                },
                sort_keys=True,
            )
        )
    finally:
        if runtime is not None:
            runtime.close()
        sandbox.stop()


if __name__ == "__main__":
    main()
