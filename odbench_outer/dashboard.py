"""Local browser dashboard for live and completed benchmark runs."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"
STATIC_ROOT = Path(__file__).with_name("dashboard_static")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_events(path: Path) -> list[dict[str, Any]]:
    """Read complete JSONL records, tolerating a partially-written final line."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _compact(value: Any, *, depth: int = 0) -> Any:
    """Bound large event fields before returning them to the browser."""
    if depth >= 5:
        return "…"
    if isinstance(value, str):
        return value if len(value) <= 4_000 else value[:4_000] + "\n… truncated"
    if isinstance(value, list):
        items = [_compact(item, depth=depth + 1) for item in value[:50]]
        if len(value) > 50:
            items.append(f"… {len(value) - 50} more")
        return items
    if isinstance(value, dict):
        return {
            str(key): _compact(item, depth=depth + 1)
            for key, item in list(value.items())[:80]
        }
    return value


def _tool_arguments(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    arguments: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("type") != "model_response":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            raw = function.get("arguments")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                parsed = {"raw": raw}
            if isinstance(parsed, dict):
                arguments[call["id"]] = parsed
    return arguments


def _metric_value(metrics: Any, preferred: str | None) -> tuple[str, float] | None:
    if not isinstance(metrics, dict):
        return None
    if preferred:
        value = _finite_number(metrics.get(preferred))
        if value is not None:
            return preferred, value
    for name, raw_value in metrics.items():
        value = _finite_number(raw_value)
        if value is not None:
            return str(name), value
    return None


def _result_payload(event: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    envelope = event.get("result")
    if not isinstance(envelope, dict):
        return False, {}
    payload = envelope.get("result")
    return envelope.get("ok") is True, payload if isinstance(payload, dict) else {}


def _tool_category(name: str) -> str:
    if name.startswith("train_"):
        return "training"
    if name in {"evaluate", "submit"}:
        return "scoring"
    if name in {"analyze_for_hw", "verify_on_hw"}:
        return "hardware"
    return "workspace"


def _tool_succeeded(name: str, envelope_ok: bool, payload: dict[str, Any]) -> bool:
    if not envelope_ok:
        return False
    if name == "workspace_exec":
        return payload.get("exit_code") == 0 and payload.get("timed_out") is not True
    if name == "workspace_apply_patch":
        return payload.get("applied") is True
    if name == "analyze_for_hw":
        return payload.get("compiled") is True
    if name == "verify_on_hw":
        return payload.get("passed") is True
    if name == "submit":
        result_type = str(payload.get("type", ""))
        return not result_type.endswith("rejected")
    if name.startswith("train_"):
        result_type = str(payload.get("type", ""))
        return not result_type.endswith("failed")
    return True


def _tool_summary(
    name: str, arguments: dict[str, Any], ok: bool, payload: dict[str, Any]
) -> str:
    if not ok:
        return "Tool call failed"
    if name == "workspace_exec":
        command = str(arguments.get("command", "")).strip().splitlines()
        label = command[0][:100] if command else "Shell command"
        exit_code = payload.get("exit_code")
        duration = _finite_number(payload.get("duration_ms"))
        suffix = f" · exit {exit_code}" if exit_code is not None else ""
        if duration is not None:
            suffix += f" · {duration / 1000:.1f}s"
        return label + suffix
    if name == "workspace_apply_patch":
        return "Patch applied" if payload.get("applied") else "Patch was not applied"
    if name.startswith("train_"):
        result_type = str(payload.get("type") or payload.get("job_status") or "updated")
        epoch = payload.get("epoch")
        evaluation = payload.get("evaluation")
        metric = _metric_value(
            evaluation.get("metrics") if isinstance(evaluation, dict) else None,
            None,
        )
        parts = [result_type.replace("_", " ")]
        if epoch is not None:
            parts.append(f"epoch {epoch}")
        if metric:
            parts.append(f"{metric[0]} {metric[1]:.4g}")
        return " · ".join(parts)
    if name == "evaluate":
        metric = _metric_value(payload.get("metrics"), None)
        return f"{metric[0]} {metric[1]:.4g}" if metric else "Evaluation completed"
    if name == "analyze_for_hw":
        mapping = payload.get("accelerator_mapping")
        percent = mapping.get("accelerator_epoch_percent") if isinstance(mapping, dict) else None
        suffix = f" · {percent:.0f}% accelerator" if _finite_number(percent) is not None else ""
        return ("Compiled" if payload.get("compiled") else "Compilation failed") + suffix
    if name == "verify_on_hw":
        duration = _finite_number(payload.get("duration_seconds"))
        suffix = f" · {duration * 1000:.2f} ms" if duration is not None else ""
        return ("Hardware passed" if payload.get("passed") else "Hardware failed") + suffix
    if name == "submit":
        result_type = str(payload.get("type", "submission"))
        return result_type.replace("_", " ")
    return "Completed"


def _score_points(
    tool_events: list[dict[str, Any]], objective_metric: str | None
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for event in tool_events:
        ok, payload = _result_payload(event)
        if not ok:
            continue
        name = str(event.get("name", ""))
        evaluations: list[tuple[str, Any]] = []
        if name.startswith("train_"):
            evaluation = payload.get("evaluation")
            if isinstance(evaluation, dict):
                evaluations.append(("training epoch", evaluation.get("metrics")))
        elif name == "evaluate":
            evaluations.append(("evaluation", payload.get("metrics")))
        elif name == "submit":
            evaluation = payload.get("evaluation")
            if isinstance(evaluation, dict):
                evaluations.append(("submission", evaluation.get("metrics")))
        for source, metrics in evaluations:
            metric = _metric_value(metrics, objective_metric)
            if metric is None:
                continue
            points.append(
                {
                    "at": event.get("recorded_at"),
                    "turn": event.get("turn"),
                    "score": metric[1],
                    "metric": metric[0],
                    "source": source,
                    "epoch": payload.get("epoch"),
                }
            )
    return points


def _status(finished: dict[str, Any] | None, summary: dict[str, Any] | None) -> str:
    value = (finished or summary or {}).get("status")
    return str(value) if value else "running"


def build_run_view(events_path: Path) -> dict[str, Any] | None:
    events = _read_events(events_path)
    started = next((event for event in events if event.get("type") == "run_started"), None)
    if started is None:
        return None

    run_directory = events_path.parent
    summary_file = _read_json(run_directory / "summary.json")
    best_file = _read_json(run_directory / "best_candidate.json")
    submission = _read_json(run_directory / "submission-result.json")
    finished = next(
        (event for event in reversed(events) if event.get("type") == "run_finished"),
        None,
    )
    tool_events = [event for event in events if event.get("type") == "tool_result"]
    arguments_by_id = _tool_arguments(events)
    run_context = started.get("run_context")
    run_context = run_context if isinstance(run_context, dict) else {}
    objective = run_context.get("objective")
    objective = objective if isinstance(objective, dict) else {}
    objective_metric = objective.get("metric")
    objective_metric = str(objective_metric) if objective_metric else None

    tokens = 0
    cost = 0.0
    turns = 0
    for event in events:
        if event.get("type") != "model_response":
            continue
        turns = max(turns, int(event.get("turn", 0) or 0))
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue
        usage_tokens = usage.get("total_tokens")
        if isinstance(usage_tokens, int) and not isinstance(usage_tokens, bool):
            tokens += usage_tokens
        usage_cost = _finite_number(usage.get("cost"))
        if usage_cost is not None:
            cost += usage_cost
    final_values = finished or summary_file or {}
    tokens = int(final_values.get("total_tokens", tokens) or tokens)
    cost = float(final_values.get("total_cost", cost) or cost)
    turns = int(final_values.get("turns", turns) or turns)

    latest_training: dict[str, Any] = {}
    latest_evaluation: dict[str, Any] = {}
    latest_agent: dict[str, Any] = {}
    training_hardware: dict[str, Any] = {}
    timeline: list[dict[str, Any]] = []
    error_count = 0
    for event in tool_events:
        envelope_ok, payload = _result_payload(event)
        succeeded = _tool_succeeded(str(event.get("name", "")), envelope_ok, payload)
        if not succeeded:
            error_count += 1
        training = payload.get("training_budget")
        evaluation = payload.get("evaluation_budget")
        agent = payload.get("agent_budget")
        hardware = payload.get("training_hardware")
        if isinstance(training, dict):
            latest_training = training
        if isinstance(evaluation, dict):
            latest_evaluation = evaluation
        if isinstance(agent, dict):
            latest_agent = agent
        if isinstance(hardware, dict):
            training_hardware = hardware
        call_id = event.get("tool_call_id")
        arguments = arguments_by_id.get(call_id, {}) if isinstance(call_id, str) else {}
        name = str(event.get("name", "unknown"))
        envelope = event.get("result") if isinstance(event.get("result"), dict) else {}
        timeline.append(
            {
                "at": event.get("recorded_at"),
                "turn": event.get("turn"),
                "name": name,
                "ok": succeeded,
                "category": _tool_category(name),
                "summary": _tool_summary(name, arguments, envelope_ok, payload),
                "arguments": _compact(arguments),
                "result": _compact(payload if envelope_ok else envelope),
            }
        )

    limits = started.get("limits")
    limits = limits if isinstance(limits, dict) else {}
    points = _score_points(tool_events, objective_metric)
    best = best_file
    if best is None:
        candidate = next(
            (
                payload.get("best_candidate")
                for event in reversed(tool_events)
                for ok, payload in [_result_payload(event)]
                if ok and isinstance(payload.get("best_candidate"), dict)
            ),
            None,
        )
        best = candidate if isinstance(candidate, dict) else None

    submitted_score = None
    if isinstance(submission, dict):
        evaluation = submission.get("evaluation")
        if isinstance(evaluation, dict):
            submitted_score = _metric_value(evaluation.get("metrics"), objective_metric)

    started_at = _finite_number(started.get("recorded_at")) or events_path.stat().st_mtime
    updated_at = max(
        (_finite_number(event.get("recorded_at")) or started_at for event in events),
        default=started_at,
    )
    ended_at = _finite_number(finished.get("recorded_at")) if finished else None
    current_status = _status(finished, summary_file)

    return {
        "run_id": str(started.get("run_id") or run_directory.name),
        "task": started.get("task_id") or "unknown task",
        "model": started.get("requested_model") or "unknown model",
        "status": current_status,
        "is_running": current_status == "running",
        "started_at": started_at,
        "updated_at": updated_at,
        "ended_at": ended_at,
        "elapsed_seconds": max(0.0, (ended_at or time.time()) - started_at),
        "objective": {
            "metric": objective_metric,
            "mode": objective.get("mode"),
        },
        "usage": {
            "turns": turns,
            "turns_limit": limits.get("max_turns") or latest_agent.get("turns_limit"),
            "tokens": tokens,
            "tokens_limit": limits.get("max_total_tokens") or latest_agent.get("tokens_limit"),
            "cost": cost,
            "cost_limit": limits.get("max_cost") or latest_agent.get("cost_limit"),
            "tool_calls": len(timeline),
            "tool_errors": error_count,
        },
        "training": {
            "seconds_used": latest_training.get("active_seconds_used", 0),
            "seconds_limit": latest_training.get("active_seconds_limit"),
            "starts_used": latest_training.get("starts_used", 0),
            "starts_limit": latest_training.get("starts_limit"),
            "active_jobs": latest_training.get("active_jobs", []),
            "hardware": {
                key: training_hardware.get(key)
                for key in ("id", "description", "accelerator", "gpus", "cpus", "memory")
                if key in training_hardware
            },
        },
        "evaluation": {
            "used": latest_evaluation.get("used", 0),
            "limit": latest_evaluation.get("limit"),
            "remaining": latest_evaluation.get("remaining"),
        },
        "best_candidate": _compact(best),
        "submitted_score": (
            {"metric": submitted_score[0], "score": submitted_score[1]}
            if submitted_score
            else None
        ),
        "score_points": points,
        "timeline": timeline,
    }


class RunStore:
    def __init__(self, runs_root: Path) -> None:
        self.runs_root = runs_root.resolve()
        self._cache: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _event_paths(self) -> list[Path]:
        return sorted(self.runs_root.glob("run-*/events.jsonl"), reverse=True)

    def get(self, run_id: str) -> dict[str, Any] | None:
        path = self.runs_root / run_id / "events.jsonl"
        if path.parent.parent != self.runs_root or not path.is_file():
            return None
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            cached = self._cache.get(path)
            if cached and cached[0] == signature:
                view = cached[1]
                if view.get("is_running"):
                    view = {**view, "elapsed_seconds": max(0.0, time.time() - view["started_at"])}
                return view
        view = build_run_view(path)
        if view is not None:
            with self._lock:
                self._cache[path] = (signature, view)
        return view

    def list(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for path in self._event_paths():
            view = self.get(path.parent.name)
            if view is None:
                continue
            runs.append(
                {
                    key: view[key]
                    for key in (
                        "run_id",
                        "task",
                        "model",
                        "status",
                        "is_running",
                        "started_at",
                        "updated_at",
                    )
                }
            )
        runs.sort(key=lambda item: item["started_at"], reverse=True)
        return runs


def make_handler(store: RunStore) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "ODBenchDashboard/1"

        def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(
                json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            path = unquote(urlparse(self.path).path)
            if path in {"/", "/index.html"}:
                try:
                    body = (STATIC_ROOT / "index.html").read_bytes()
                except OSError:
                    self._json({"error": "dashboard assets are missing"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                self._send(body, "text/html; charset=utf-8")
                return
            if path == "/api/runs":
                self._json({"runs": store.list(), "server_time": time.time()})
                return
            prefix = "/api/runs/"
            if path.startswith(prefix):
                run_id = path[len(prefix) :]
                view = store.get(run_id)
                if view is None:
                    self._json({"error": "run not found"}, HTTPStatus.NOT_FOUND)
                else:
                    self._json(view)
                return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return DashboardHandler


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_ROOT)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8765)
    result.add_argument("--run-id", help="Open this run initially.")
    result.add_argument("--no-open", action="store_true", help="Do not open a browser tab.")
    return result


def main() -> None:
    arguments = parser().parse_args()
    store = RunStore(arguments.runs_dir)
    server = ThreadingHTTPServer((arguments.host, arguments.port), make_handler(store))
    run_fragment = f"#{quote(arguments.run_id)}" if arguments.run_id else ""
    url = f"http://{arguments.host}:{server.server_port}/{run_fragment}"
    print(f"OD Benchmark dashboard: {url}", flush=True)
    if not arguments.no_open:
        threading.Timer(0.2, webbrowser.open_new_tab, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
