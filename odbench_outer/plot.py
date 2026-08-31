"""Generate per-task SVG plots from durable benchmark run logs."""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"
DEFAULT_PLOTS_ROOT = DEFAULT_RUNS_ROOT / "plots"
DEFAULT_ICONS_ROOT = REPO_ROOT / "icons"

PROVIDER_COLORS = {
    "openai": "#111827",
    "minimax": "#e11d48",
    "deepseek": "#5b5ce2",
    "gemini": "#4285f4",
    "gemma": "#34a853",
    "claude": "#d97757",
    "kimi": "#7c3aed",
    "grok": "#1f2937",
    "arcee": "#16a34a",
    "qwen": "#f97316",
    "z-ai": "#0f766e",
    "mistral": "#ff7000",
    "poolside": "#0ea5e9",
    "meta": "#0668e1",
}

PROVIDER_ALIASES = (
    ("deepseek", ("deepseek/", "deepseek")),
    ("gemini", ("google/gemini", "gemini")),
    ("gemma", ("google/gemma", "gemma")),
    ("claude", ("anthropic/", "claude", "anthropic")),
    ("kimi", ("moonshotai/", "kimi")),
    ("grok", ("x-ai/", "grok")),
    ("arcee", ("arcee-ai/", "arcee")),
    ("qwen", ("qwen/", "qwen")),
    ("z-ai", ("z-ai/", "glm")),
    ("minimax", ("minimax/", "minimax")),
    ("mistral", ("mistralai/", "mistral")),
    ("poolside", ("poolside/", "poolside")),
    ("openai", ("openai/", "gpt-", "gpt_", "o1", "o3", "o4")),
    ("meta", ("meta/", "meta-llama/", "llama", "muse-spark")),
)

FALLBACK_COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
    "#be185d",
    "#4d7c0f",
)


@dataclass(frozen=True)
class RunSeries:
    run_id: str
    task: str
    model: str
    metric: str
    objective: str
    final_tokens: int | None
    final_cost: float | None
    submitted_score: float | None
    submitted_throughput: float | None


@dataclass(frozen=True)
class ModelSummary:
    model: str
    run_count: int
    mean_score: float
    mean_throughput: float | None
    total_tokens: int
    total_cost: float


@dataclass(frozen=True)
class Axis:
    key: str
    title: str
    label: str
    tooltip_label: str
    minimum: float
    final_value: Callable[[RunSeries], float | None]
    format_tick: Callable[[float], str]
    format_tooltip: Callable[[float], str]


def _format_tokens(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1_000:.0f}k"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(int(value))


def _format_cost(value: float) -> str:
    if value >= 1:
        return f"${value:.2f}"
    if value >= 0.01:
        return f"${value:.3f}"
    return f"${value:.5f}"


def _format_throughput(value: float) -> str:
    if value >= 1_000:
        return f"{value:,.0f} inf/s"
    return f"{value:.2f} inf/s"


SUBMISSION_AXES = (
    Axis(
        key="cost",
        title="Submitted solution accuracy by cost",
        label="Recorded inference cost (USD)",
        tooltip_label="cost",
        minimum=0.0,
        final_value=lambda run: run.final_cost,
        format_tick=_format_cost,
        format_tooltip=_format_cost,
    ),
    Axis(
        key="tokens",
        title="Submitted solution accuracy by tokens",
        label="Tokens used",
        tooltip_label="tokens",
        minimum=0.0,
        final_value=lambda run: (
            float(run.final_tokens) if run.final_tokens is not None else None
        ),
        format_tick=_format_tokens,
        format_tooltip=lambda value: f"{int(value):,}",
    ),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"Run log directory (default: {DEFAULT_RUNS_ROOT}).",
    )
    result.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_ROOT,
        help=f"SVG output directory (default: {DEFAULT_PLOTS_ROOT}).",
    )
    result.add_argument(
        "--icon-dir",
        type=Path,
        default=DEFAULT_ICONS_ROOT,
        help=f"Provider PNG directory (default: {DEFAULT_ICONS_ROOT}).",
    )
    result.add_argument(
        "--no-icons", action="store_true", help="Do not embed provider icons."
    )
    return result


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(event, dict):
                raise ValueError(f"{path}:{line_number}: event must be an object")
            events.append(event)
    return events


def _finite_number(value: Any) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _token_count(value: Any) -> int | None:
    numeric = _finite_number(value)
    if numeric is None or numeric < 0:
        return None
    return int(numeric)


def _submission_values(
    events_path: Path, metric: str
) -> tuple[float | None, float | None]:
    result_path = events_path.with_name("submission-result.json")
    if not result_path.is_file():
        return None, None
    try:
        document = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{result_path}: invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{result_path}: submission result must be an object")
    evaluation = document.get("evaluation")
    metrics = evaluation.get("metrics") if isinstance(evaluation, dict) else None
    score = _finite_number(metrics.get(metric)) if isinstance(metrics, dict) else None
    hardware = document.get("hardware_verification")
    throughput = (
        _finite_number(hardware.get("inferences_per_second"))
        if isinstance(hardware, dict)
        else None
    )
    if throughput is None and isinstance(hardware, dict):
        runtime = _finite_number(hardware.get("duration_seconds"))
        if runtime is not None and runtime > 0:
            throughput = 1.0 / runtime
    if throughput is None and isinstance(hardware, dict):
        duration_ms = hardware.get("duration_ms")
        mean_ms = (
            _finite_number(duration_ms.get("mean"))
            if isinstance(duration_ms, dict)
            else None
        )
        if mean_ms is not None and mean_ms > 0:
            throughput = 1_000.0 / mean_ms
    return score, throughput


def load_run(events_path: Path) -> RunSeries | None:
    events = _read_events(events_path)
    started = next((event for event in events if event.get("type") == "run_started"), {})
    run_id = started.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        run_id = events_path.parent.name

    task = started.get("task_id")
    if not isinstance(task, str) or not task:
        return None

    model = started.get("requested_model")
    if not isinstance(model, str) or not model:
        response = next(
            (event for event in events if event.get("type") == "model_response"), {}
        )
        model = response.get("model")
    if not isinstance(model, str) or not model:
        return None

    context = started.get("run_context")
    objective = context.get("objective") if isinstance(context, dict) else None
    metric = objective.get("metric") if isinstance(objective, dict) else None
    mode = objective.get("mode") if isinstance(objective, dict) else None
    if not isinstance(metric, str) or not metric:
        metric = "score"
    if mode not in {"maximize", "minimize"}:
        mode = "maximize"

    finished = next(
        (event for event in reversed(events) if event.get("type") == "run_finished"), {}
    )
    submitted_score, submitted_throughput = _submission_values(events_path, metric)
    if submitted_score is None:
        return None
    final_tokens = _token_count(finished.get("total_tokens"))
    final_cost = _finite_number(finished.get("total_cost"))
    return RunSeries(
        run_id=run_id,
        task=task,
        model=model,
        metric=metric,
        objective=mode,
        final_tokens=final_tokens,
        final_cost=final_cost,
        submitted_score=submitted_score,
        submitted_throughput=submitted_throughput,
    )


def load_runs(runs_root: Path) -> list[RunSeries]:
    if not runs_root.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {runs_root}")
    runs: list[RunSeries] = []
    for events_path in sorted(runs_root.glob("*/events.jsonl")):
        run = load_run(events_path)
        if run is not None:
            runs.append(run)
    return runs


def _ticks(low: float, high: float, count: int = 5) -> list[float]:
    if low == high:
        return [low]
    return [low + (high - low) * index / count for index in range(count + 1)]


def _number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "task"


def _provider_for_model(model: str) -> str | None:
    haystack = model.lower()
    for provider, aliases in PROVIDER_ALIASES:
        if any(alias in haystack for alias in aliases):
            return provider
    return None


def _load_provider_icons(icon_dir: Path | None) -> dict[str, str]:
    if icon_dir is None or not icon_dir.is_dir():
        return {}
    icons: dict[str, str] = {}
    for path in sorted(icon_dir.glob("*.png")):
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        icons[path.stem.lower()] = f"data:image/png;base64,{data}"
    return icons


def _series_color(model: str, index: int) -> str:
    provider = _provider_for_model(model)
    return PROVIDER_COLORS.get(provider or "", FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def _legend_layout(models: list[str], available_width: float) -> tuple[list[tuple[float, int]], int]:
    placements: list[tuple[float, int]] = []
    x = 0.0
    row = 0
    for model in models:
        item_width = 58 + len(model) * 7.0
        if x and x + item_width > available_width:
            x = 0.0
            row += 1
        placements.append((x, row))
        x += item_width
    return placements, row + 1


def _model_summaries(runs: list[RunSeries]) -> list[ModelSummary]:
    by_model: dict[str, list[RunSeries]] = {}
    for run in runs:
        by_model.setdefault(run.model, []).append(run)
    summaries: list[ModelSummary] = []
    for model, model_runs in by_model.items():
        scores = [run.submitted_score for run in model_runs if run.submitted_score is not None]
        throughputs = [
            run.submitted_throughput
            for run in model_runs
            if run.submitted_throughput is not None
        ]
        if not scores:
            continue
        summaries.append(
            ModelSummary(
                model=model,
                run_count=len(model_runs),
                mean_score=sum(scores) / len(scores),
                mean_throughput=(
                    sum(throughputs) / len(throughputs) if throughputs else None
                ),
                total_tokens=sum(run.final_tokens or 0 for run in model_runs),
                total_cost=sum(run.final_cost or 0.0 for run in model_runs),
            )
        )
    return summaries


def render_model_results_svg(
    task: str,
    runs: list[RunSeries],
    *,
    icons: dict[str, str] | None = None,
) -> str:
    metrics = {run.metric for run in runs}
    objectives = {run.objective for run in runs}
    if len(metrics) != 1 or len(objectives) != 1:
        raise ValueError(f"task {task!r} mixes metrics or objective directions")
    metric = next(iter(metrics))
    objective = next(iter(objectives))
    summaries = _model_summaries(runs)
    if not summaries or not any(
        summary.mean_throughput is not None for summary in summaries
    ):
        raise ValueError(f"task {task!r} has no submitted throughput data")
    summaries.sort(
        key=lambda summary: summary.mean_score,
        reverse=objective == "maximize",
    )
    fastest_throughput = max(
        summary.mean_throughput or 0.0 for summary in summaries
    )

    width = 1160
    bar_left, bar_width = 260, 500
    result_x, usage_x = 790, 970
    top, row_height = 78, 64
    axis_y = top + len(summaries) * row_height + 8
    height = axis_y + 48
    icons = icons or {}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">Accepted results by model — {html.escape(task)}</title>',
        '<desc id="desc">Accepted submission accuracy and inference throughput by model, with total token usage and cost.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#172033}'
        '.heading{font-size:20px;font-weight:700}.header{font-size:12px;fill:#526078}'
        '.model{font-size:12px;font-weight:600}.value{font-size:11px;fill:#344054}'
        '.detail{font-size:10px;fill:#667085}.tick{font-size:11px;fill:#667085}'
        '.grid{stroke:#e3e7eb;stroke-width:1}</style>',
        '<text class="heading" x="24" y="34">Accepted Results by Model</text>',
        f'<text class="header" x="{bar_left}" y="58">Accuracy ↑</text>',
        f'<text class="header" x="{bar_left+92}" y="58">Throughput ↑ (relative to fastest)</text>',
        f'<text class="header" x="{result_x}" y="58">Accepted result</text>',
        f'<text class="header" x="{usage_x}" y="58">Total tokens / cost</text>',
    ]

    for tick in _ticks(0.0, 1.0, count=4):
        tick_x = bar_left + tick * bar_width
        parts.append(
            f'<line class="grid" x1="{tick_x:.2f}" y1="68" x2="{tick_x:.2f}" y2="{axis_y}"/>'
        )
        parts.append(
            f'<text class="tick" x="{tick_x:.2f}" y="{axis_y+23}" text-anchor="middle">{tick*100:.0f}%</text>'
        )

    for index, summary in enumerate(summaries):
        row_y = top + index * row_height
        color = _series_color(summary.model, index)
        provider = _provider_for_model(summary.model)
        icon = icons.get(provider or "")
        if icon:
            parts.append(
                f'<image x="24" y="{row_y+3}" width="24" height="24" href="{icon}" preserveAspectRatio="xMidYMid meet"/>'
            )
        else:
            parts.append(
                f'<circle cx="36" cy="{row_y+15}" r="9" fill="{color}"/>'
            )
        parts.append(
            f'<text class="model" x="58" y="{row_y+21}">{html.escape(summary.model)}</text>'
        )
        for offset in (0, 19):
            parts.append(
                f'<rect x="{bar_left}" y="{row_y+1+offset}" width="{bar_width}" height="13" rx="2" fill="#eef1f4"/>'
            )
        accuracy_fraction = min(max(summary.mean_score, 0.0), 1.0)
        parts.append(
            f'<rect x="{bar_left}" y="{row_y+1}" width="{accuracy_fraction*bar_width:.2f}" height="13" rx="2" fill="{color}">'
            f'<title>Mean accepted {html.escape(metric.replace("_", " "))}: {_number(summary.mean_score)}</title></rect>'
        )
        if summary.mean_throughput is not None and fastest_throughput > 0:
            throughput_fraction = summary.mean_throughput / fastest_throughput
            parts.append(
                f'<rect x="{bar_left}" y="{row_y+20}" width="{throughput_fraction*bar_width:.2f}" height="13" rx="2" fill="{color}" fill-opacity="0.28">'
                f'<title>Mean throughput: {html.escape(_format_throughput(summary.mean_throughput))} ({throughput_fraction*100:.1f}% of fastest; higher is better)</title></rect>'
            )
        throughput_text = (
            _format_throughput(summary.mean_throughput)
            if summary.mean_throughput is not None
            else "throughput n/a"
        )
        parts.append(
            f'<text class="value" x="{result_x}" y="{row_y+15}">{_number(summary.mean_score)} acc · {html.escape(throughput_text)}</text>'
        )
        throughput_detail = (
            f"throughput {summary.mean_throughput / fastest_throughput * 100:.1f}% of fastest"
            if summary.mean_throughput is not None and fastest_throughput > 0
            else "throughput unavailable"
        )
        run_label = "run" if summary.run_count == 1 else "runs"
        parts.append(
            f'<text class="detail" x="{result_x}" y="{row_y+33}">{summary.run_count} accepted {run_label} · {html.escape(throughput_detail)}</text>'
        )
        parts.append(
            f'<text class="value" x="{usage_x}" y="{row_y+15}">{html.escape(_format_tokens(float(summary.total_tokens)))} tok · {html.escape(_format_cost(summary.total_cost))}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_submissions_svg(
    task: str,
    runs: list[RunSeries],
    *,
    axis: Axis,
    icons: dict[str, str] | None = None,
) -> str:
    if axis.key not in {"tokens", "cost"}:
        raise ValueError("submission scatter plots require the tokens or cost axis")
    metrics = {run.metric for run in runs}
    objectives = {run.objective for run in runs}
    if len(metrics) != 1 or len(objectives) != 1:
        raise ValueError(f"task {task!r} mixes metrics or objective directions")
    metric = next(iter(metrics))
    objective = next(iter(objectives))
    observations = [
        (run, value, run.submitted_score)
        for run in runs
        if run.submitted_score is not None
        and (value := axis.final_value(run)) is not None
    ]
    if not observations:
        raise ValueError(f"task {task!r} has no submitted {axis.key} usage data")

    models = sorted({run.model for run, _, _ in observations})
    model_indexes = {model: index for index, model in enumerate(models)}
    width = 1000
    left, right, top, plot_height = 92, 34, 88, 430
    plot_width = width - left - right
    legend, legend_rows = _legend_layout(models, plot_width)
    legend_top = top + plot_height + 84
    height = int(legend_top + legend_rows * 30 + 18)
    x_low = axis.minimum
    x_high = max(value for _, value, _ in observations)
    if x_high <= x_low:
        x_high = x_low + 1.0
    scores = [score for _, _, score in observations]
    score_low, score_high = min(scores), max(scores)
    padding = max((score_high - score_low) * 0.08, abs(score_high) * 0.01, 0.01)
    y_low, y_high = score_low - padding, score_high + padding

    def x(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * plot_width

    def y(score: float) -> float:
        return top + (y_high - score) / (y_high - y_low) * plot_height

    title = axis.title
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)} — {html.escape(task)}</title>',
        '<desc id="desc">One point per completed run, showing its submitted solution score and total usage.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#172033}'
        '.tick{font-size:12px;fill:#526078}.label{font-size:14px;font-weight:600}.title{font-size:22px;font-weight:700}'
        '.subtitle{font-size:12px;fill:#526078}.grid{stroke:#dfe5ed;stroke-width:1}.frame{stroke:#aab4c3;fill:none}'
        '.point{stroke:#fff;stroke-width:2}.icon-frame{fill:#fff;stroke-width:2}</style>',
        f'<text class="title" x="{left}" y="36">{html.escape(title)}</text>',
        f'<text class="subtitle" x="{left}" y="58">Task: {html.escape(task)} · Objective: {html.escape(objective)} {html.escape(metric)} · Completed submissions: {len(observations)}</text>',
    ]

    for tick in _ticks(y_low, y_high):
        tick_y = y(tick)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{tick_y:.2f}" x2="{width-right}" y2="{tick_y:.2f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left-12}" y="{tick_y+4:.2f}" text-anchor="end">{_number(tick)}</text>'
        )
    for tick in _ticks(x_low, x_high):
        tick_x = x(tick)
        parts.append(
            f'<line class="grid" x1="{tick_x:.2f}" y1="{top}" x2="{tick_x:.2f}" y2="{top+plot_height}"/>'
        )
        parts.append(
            f'<text class="tick" x="{tick_x:.2f}" y="{top+plot_height+24}" text-anchor="middle">{html.escape(axis.format_tick(tick))}</text>'
        )
    parts.append(
        f'<rect class="frame" x="{left}" y="{top}" width="{plot_width}" height="{plot_height}"/>'
    )

    icons = icons or {}
    for run, value, score in observations:
        color = _series_color(run.model, model_indexes[run.model])
        point_x, point_y = x(value), y(score)
        provider = _provider_for_model(run.model)
        icon = icons.get(provider or "")
        tooltip = (
            f"{html.escape(run.model)} · {html.escape(run.run_id)} — "
            f"{html.escape(axis.tooltip_label)} "
            f"{html.escape(axis.format_tooltip(value))}: {_number(score)}"
        )
        parts.append(f"<g><title>{tooltip}</title>")
        if icon:
            parts.extend(
                [
                    f'<circle class="icon-frame" cx="{point_x:.2f}" cy="{point_y:.2f}" r="13" stroke="{color}"/>',
                    f'<image x="{point_x-10:.2f}" y="{point_y-10:.2f}" width="20" height="20" href="{icon}" preserveAspectRatio="xMidYMid meet"/>',
                    f'<circle cx="{point_x:.2f}" cy="{point_y:.2f}" r="13" fill="none" stroke="{color}" stroke-width="2"/>',
                ]
            )
        else:
            parts.append(
                f'<circle class="point" cx="{point_x:.2f}" cy="{point_y:.2f}" r="7" fill="{color}"/>'
            )
        parts.append("</g>")

    for index, model in enumerate(models):
        offset, row = legend[index]
        item_x = left + offset
        item_y = legend_top + row * 30
        color = _series_color(model, index)
        parts.append(
            f'<circle cx="{item_x+12:.1f}" cy="{item_y-12:.1f}" r="7" fill="{color}"/>'
        )
        provider = _provider_for_model(model)
        icon = icons.get(provider or "")
        if icon:
            parts.append(
                f'<image x="{item_x+31:.1f}" y="{item_y-22:.1f}" width="20" height="20" href="{icon}" preserveAspectRatio="xMidYMid meet"/>'
            )
        parts.append(
            f'<text class="tick" x="{item_x+57:.1f}" y="{item_y-6:.1f}">{html.escape(model)}</text>'
        )
    parts.extend(
        [
            f'<text class="label" x="{left+plot_width/2:.2f}" y="{top+plot_height+54}" text-anchor="middle">{html.escape(axis.label)}</text>',
            f'<text class="label" transform="translate(26 {top+plot_height/2:.2f}) rotate(-90)" text-anchor="middle">Submitted {html.escape(metric.replace("_", " "))}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def generate_plots(
    runs_root: Path,
    plots_root: Path,
    *,
    icon_dir: Path | None = DEFAULT_ICONS_ROOT,
) -> list[Path]:
    runs = load_runs(runs_root)
    if not runs:
        raise ValueError(f"no plottable runs found under {runs_root}")
    by_task: dict[str, list[RunSeries]] = {}
    for run in runs:
        by_task.setdefault(run.task, []).append(run)
    icons = _load_provider_icons(icon_dir)
    plots_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for task, task_runs in sorted(by_task.items()):
        slug = _slug(task)
        for legacy_name in (
            f"{slug}.svg",
            f"{slug}-vs-tokens.svg",
            f"{slug}-vs-cost.svg",
        ):
            (plots_root / legacy_name).unlink(missing_ok=True)
        for axis in SUBMISSION_AXES:
            path = plots_root / f"{slug}-submitted-vs-{axis.key}.svg"
            try:
                svg = render_submissions_svg(task, task_runs, axis=axis, icons=icons)
            except ValueError as error:
                if f"no submitted {axis.key} usage data" in str(error):
                    continue
                raise
            path.write_text(svg, encoding="utf-8")
            written.append(path)
        path = plots_root / f"{slug}-results-by-model.svg"
        try:
            svg = render_model_results_svg(task, task_runs, icons=icons)
        except ValueError as error:
            if "no submitted throughput data" in str(error):
                continue
            raise
        path.write_text(svg, encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    arguments = parser().parse_args()
    icon_dir = None if arguments.no_icons else arguments.icon_dir
    for path in generate_plots(
        arguments.runs_dir, arguments.plots_dir, icon_dir=icon_dir
    ):
        print(path)


if __name__ == "__main__":
    main()
