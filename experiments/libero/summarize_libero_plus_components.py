#!/usr/bin/env python3
"""Summarize LIBERO-plus results by official component categories."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


RESULT_RE = re.compile(r"^gpu(?P<gpu>[^_]+)_task(?P<task_id>\d+)_results\.json$")
DEFAULT_COMPONENT_ORDER = [
    "Background Textures",
    "Robot Initial States",
    "Camera Viewpoints",
    "Language Instructions",
    "Sensor Noise",
    "Objects Layout",
    "Light Conditions",
    "Unclassified",
]


def _new_stats() -> dict[str, Any]:
    return {
        "tasks": 0,
        "episodes": 0,
        "successes": 0,
        "duration_sec": 0.0,
        "max_duration_sec": 0.0,
        "psnr_sum": 0.0,
        "psnr_count": 0,
        "chunk_infer_time_sum": 0.0,
        "chunk_infer_time_count": 0,
    }


def _add_result(stats: dict[str, Any], result: dict[str, Any]) -> None:
    duration = float(result.get("duration", 0.0) or 0.0)
    episodes = int(result.get("total_episodes", 0) or 0)
    successes = int(result.get("successes", 0) or 0)

    stats["tasks"] += 1
    stats["episodes"] += episodes
    stats["successes"] += successes
    stats["duration_sec"] += duration
    stats["max_duration_sec"] = max(float(stats["max_duration_sec"]), duration)

    psnr = result.get("future_video_psnr_mean")
    if psnr is not None:
        stats["psnr_sum"] += float(psnr)
        stats["psnr_count"] += 1

    chunk_time = result.get("action_chunk_infer_time_sec_mean")
    if chunk_time is not None:
        stats["chunk_infer_time_sum"] += float(chunk_time)
        stats["chunk_infer_time_count"] += 1


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    episodes = int(stats["episodes"])
    tasks = int(stats["tasks"])
    finalized = dict(stats)
    finalized["success_rate"] = (
        float(stats["successes"]) / episodes if episodes > 0 else None
    )
    finalized["avg_duration_sec"] = (
        float(stats["duration_sec"]) / tasks if tasks > 0 else None
    )
    finalized["avg_future_video_psnr"] = (
        float(stats["psnr_sum"]) / int(stats["psnr_count"])
        if int(stats["psnr_count"]) > 0
        else None
    )
    finalized["avg_action_chunk_infer_time_sec"] = (
        float(stats["chunk_infer_time_sum"]) / int(stats["chunk_infer_time_count"])
        if int(stats["chunk_infer_time_count"]) > 0
        else None
    )
    return finalized


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}"


def _float(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _normalize_difficulty(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return "unknown"
    return str(value)


def _difficulty_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def _find_latest_run(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"LIBERO-plus results root does not exist: {root}")

    candidates: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir() and any(path.glob("*/gpu*_task*_results.json")):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No LIBERO-plus result run found under: {root}")

    return max(candidates, key=lambda p: (p.stat().st_mtime, str(p)))


def _load_classification(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    by_suite: dict[str, dict[int, dict[str, Any]]] = {}
    for suite, rows in raw.items():
        by_suite[suite] = {int(row["id"]) - 1: row for row in rows}
    return by_suite


def _load_results(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for result_path in sorted(suite_dir.glob("gpu*_task*_results.json")):
            match = RESULT_RE.match(result_path.name)
            if match is None:
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            suite = str(result.get("task_suite") or suite_dir.name)
            task_id = int(result.get("task_id", match.group("task_id")))
            rows.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "result_path": str(result_path),
                    "result": result,
                }
            )
    return rows


def _csv_row(group_name: str, stats: dict[str, Any], expected_tasks: int | None = None) -> dict[str, Any]:
    completed_tasks = int(stats["tasks"])
    episodes = int(stats["episodes"])
    successes = int(stats["successes"])
    row = {
        "group": group_name,
        "completed_tasks": completed_tasks,
        "expected_tasks": "" if expected_tasks is None else expected_tasks,
        "completion_rate_percent": (
            "" if expected_tasks in (None, 0) else f"{completed_tasks / expected_tasks * 100:.2f}"
        ),
        "episodes": episodes,
        "successes": successes,
        "success_rate_percent": _pct(stats["success_rate"]),
        "avg_duration_sec": _float(stats["avg_duration_sec"]),
        "max_duration_sec": _float(float(stats["max_duration_sec"])),
        "total_duration_sec": _float(float(stats["duration_sec"])),
        "avg_future_video_psnr": _float(stats["avg_future_video_psnr"], digits=4),
        "avg_action_chunk_infer_time_sec": _float(
            stats["avg_action_chunk_infer_time_sec"], digits=4
        ),
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(run_dir: Path, classification_path: Path) -> dict[str, Any]:
    classification = _load_classification(classification_path)
    result_rows = _load_results(run_dir)

    overall = _new_stats()
    by_component = defaultdict(_new_stats)
    by_suite = defaultdict(_new_stats)
    by_suite_component = defaultdict(_new_stats)
    by_difficulty = defaultdict(_new_stats)
    by_component_difficulty = defaultdict(_new_stats)
    task_rows: list[dict[str, Any]] = []

    expected_by_component = defaultdict(int)
    expected_by_suite = defaultdict(int)
    expected_by_suite_component = defaultdict(int)
    expected_by_difficulty = defaultdict(int)
    expected_by_component_difficulty = defaultdict(int)

    for suite, tasks in classification.items():
        for _, info in tasks.items():
            component = str(info.get("category", "Unclassified"))
            difficulty = _normalize_difficulty(info.get("difficulty_level"))
            expected_by_component[component] += 1
            expected_by_suite[suite] += 1
            expected_by_suite_component[(suite, component)] += 1
            if difficulty:
                expected_by_difficulty[difficulty] += 1
                expected_by_component_difficulty[(component, difficulty)] += 1

    for row in result_rows:
        suite = row["suite"]
        task_id = int(row["task_id"])
        result = row["result"]
        task_info = classification.get(suite, {}).get(task_id)

        if task_info is None:
            component = "Unclassified"
            difficulty = "unknown"
            task_name = ""
        else:
            component = str(task_info.get("category", "Unclassified"))
            difficulty = _normalize_difficulty(task_info.get("difficulty_level"))
            task_name = str(task_info.get("name", ""))

        _add_result(overall, result)
        _add_result(by_component[component], result)
        _add_result(by_suite[suite], result)
        _add_result(by_suite_component[(suite, component)], result)
        _add_result(by_difficulty[difficulty], result)
        _add_result(by_component_difficulty[(component, difficulty)], result)

        episodes = int(result.get("total_episodes", 0) or 0)
        successes = int(result.get("successes", 0) or 0)
        task_rows.append(
            {
                "suite": suite,
                "task_id": task_id,
                "component": component,
                "difficulty_level": difficulty,
                "successes": successes,
                "episodes": episodes,
                "success_rate_percent": _pct(successes / episodes if episodes > 0 else None),
                "duration_sec": _float(float(result.get("duration", 0.0) or 0.0)),
                "task_name": task_name,
                "task_description": result.get("task_description", ""),
                "result_path": row["result_path"],
            }
        )

    component_order = [
        c for c in DEFAULT_COMPONENT_ORDER if c in by_component or c in expected_by_component
    ]
    component_order += sorted(
        c for c in set(by_component) | set(expected_by_component) if c not in component_order
    )

    summary = {
        "run_dir": str(run_dir),
        "classification_path": str(classification_path),
        "overall": _finalize_stats(overall),
        "by_component": {
            component: _finalize_stats(by_component[component])
            for component in component_order
        },
        "by_suite": {
            suite: _finalize_stats(by_suite[suite])
            for suite in sorted(set(by_suite) | set(expected_by_suite))
        },
        "by_suite_component": {
            f"{suite}/{component}": _finalize_stats(by_suite_component[(suite, component)])
            for suite, component in sorted(set(by_suite_component) | set(expected_by_suite_component))
        },
        "by_difficulty": {
            difficulty: _finalize_stats(by_difficulty[difficulty])
            for difficulty in sorted(
                set(by_difficulty) | set(expected_by_difficulty),
                key=_difficulty_sort_key,
            )
        },
        "by_component_difficulty": {
            f"{component}/level_{difficulty}": _finalize_stats(
                by_component_difficulty[(component, difficulty)]
            )
            for component, difficulty in sorted(
                set(by_component_difficulty) | set(expected_by_component_difficulty),
                key=lambda x: (x[0], _difficulty_sort_key(x[1])),
            )
        },
        "task_rows": task_rows,
        "expected": {
            "by_component": dict(expected_by_component),
            "by_suite": dict(expected_by_suite),
            "by_suite_component": {
                f"{suite}/{component}": count
                for (suite, component), count in expected_by_suite_component.items()
            },
            "by_difficulty": dict(expected_by_difficulty),
            "by_component_difficulty": {
                f"{component}/level_{difficulty}": count
                for (component, difficulty), count in expected_by_component_difficulty.items()
            },
        },
    }
    return summary


def write_outputs(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    expected = summary["expected"]
    written: list[Path] = []

    component_rows = [
        _csv_row(
            component,
            stats,
            expected["by_component"].get(component),
        )
        for component, stats in summary["by_component"].items()
    ]
    path = output_dir / "libero_plus_component_summary.csv"
    _write_csv(path, component_rows)
    written.append(path)

    suite_rows = [
        _csv_row(suite, stats, expected["by_suite"].get(suite))
        for suite, stats in summary["by_suite"].items()
    ]
    path = output_dir / "libero_plus_suite_summary.csv"
    _write_csv(path, suite_rows)
    written.append(path)

    suite_component_rows = [
        _csv_row(group, stats, expected["by_suite_component"].get(group))
        for group, stats in summary["by_suite_component"].items()
    ]
    path = output_dir / "libero_plus_suite_component_summary.csv"
    _write_csv(path, suite_component_rows)
    written.append(path)

    difficulty_rows = [
        _csv_row(f"level_{difficulty}", stats, expected["by_difficulty"].get(difficulty))
        for difficulty, stats in summary["by_difficulty"].items()
    ]
    path = output_dir / "libero_plus_difficulty_summary.csv"
    _write_csv(path, difficulty_rows)
    written.append(path)

    component_difficulty_rows = [
        _csv_row(group, stats, expected["by_component_difficulty"].get(group))
        for group, stats in summary["by_component_difficulty"].items()
    ]
    path = output_dir / "libero_plus_component_difficulty_summary.csv"
    _write_csv(path, component_difficulty_rows)
    written.append(path)

    path = output_dir / "libero_plus_task_results_with_components.csv"
    _write_csv(path, summary["task_rows"])
    written.append(path)

    json_summary = dict(summary)
    json_summary.pop("task_rows", None)
    path = output_dir / "libero_plus_component_summary.json"
    path.write_text(json.dumps(json_summary, indent=2), encoding="utf-8")
    written.append(path)

    return written


def print_table(title: str, rows: list[dict[str, Any]], limit: int | None = None) -> None:
    print(f"\n=== {title} ===")
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        print("(empty)")
        return
    headers = ["group", "completed_tasks", "expected_tasks", "episodes", "successes", "success_rate_percent"]
    widths = {h: max(len(h), *(len(str(row.get(h, ""))) for row in rows)) for h in headers}
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize LIBERO-plus evaluation results by Camera/Robot/Language/etc. components."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="A concrete LIBERO-plus run directory. Defaults to latest run under --results-root.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("evaluate_results/libero_plus"),
        help="Root containing LIBERO-plus result runs.",
    )
    parser.add_argument(
        "--classification",
        type=Path,
        default=Path("third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"),
        help="LIBERO-plus official task classification JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write summary files. Defaults to --run-dir.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Only print summaries; do not write CSV/JSON files.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or _find_latest_run(args.results_root)
    output_dir = args.output_dir or run_dir
    summary = summarize(run_dir.resolve(), args.classification.resolve())

    expected = summary["expected"]
    component_rows = [
        _csv_row(component, stats, expected["by_component"].get(component))
        for component, stats in summary["by_component"].items()
    ]
    suite_rows = [
        _csv_row(suite, stats, expected["by_suite"].get(suite))
        for suite, stats in summary["by_suite"].items()
    ]
    overall_row = [_csv_row("overall", summary["overall"], sum(expected["by_suite"].values()))]

    print(f"Run directory: {run_dir}")
    print_table("Overall", overall_row)
    print_table("By Component", component_rows)
    print_table("By Suite", suite_rows)

    if not args.no_write:
        written = write_outputs(summary, output_dir)
        print("\nWrote summary files:")
        for path in written:
            print(f"- {path}")


if __name__ == "__main__":
    main()
