import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
EVAL_STEP_LIMIT_FILE = PROJECT_ROOT / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


_TRAINING_ONLY_KEYS = frozenset({
    "batch_size",
    "global_batch_size",
    "num_workers",
    "num_epochs",
    "max_steps",
    "log_every",
    "save_every",
    "save_state_every",
    "eval_every",
    "gradient_accumulation_steps",
    "output_dir",
    "resume",
    "lr_scheduler_type",
    "learning_rate",
    "weight_decay",
    "mixed_precision",
    "seed",
    "max_grad_norm",
    "eval_num_inference_steps",
})
_TRAINING_ONLY_PREFIXES = ("wandb.", "data.")


def _is_blocked_from_file(raw_override: str) -> bool:
    if _is_blocked_override(raw_override):
        return True
    key = raw_override.split("=", 1)[0].lstrip("+~")
    return key in _TRAINING_ONLY_KEYS or any(
        key.startswith(prefix) for prefix in _TRAINING_ONLY_PREFIXES
    )


def _collect_worker_overrides() -> list[str]:
    return [ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)]


def _format_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    s = str(v)
    if any(c in s for c in " ,{}[]\\"):
        s = f"'{s}'"
    return s


def _flatten_to_overrides(obj: Any, prefix: str, result: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_to_overrides(v, f"{prefix}.{k}", result)
    elif isinstance(obj, list):
        items = ",".join(_format_scalar(i) for i in obj)
        result.append(f"{prefix}=[{items}]")
    else:
        result.append(f"{prefix}={_format_scalar(obj)}")


_BLOCKED_MODEL_KEYS_FROM_YAML = frozenset({
    "load_text_encoder",
    "skip_dit_load_from_pretrain",
    "action_dit_pretrained_path",
})


def _parse_config_yaml(path: Path) -> list[str]:
    raw = OmegaConf.load(path)
    config = OmegaConf.to_container(raw, resolve=False)
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    overrides: list[str] = []
    if isinstance(model_cfg, dict):
        for k, v in model_cfg.items():
            if k in _BLOCKED_MODEL_KEYS_FROM_YAML:
                continue
            _flatten_to_overrides(v, f"model.{k}", overrides)
    return [ov for ov in overrides if not _is_blocked_from_file(ov)]


def _resolve_overrides_file(path_str: str) -> Path:
    path = _resolve_path(path_str, base=PROJECT_ROOT)
    if not path.is_file():
        raise FileNotFoundError(f"MULTIRUN.overrides_file not found: {path}")
    if path.suffix not in (".yaml", ".yml"):
        raise ValueError(
            f"Unsupported MULTIRUN.overrides_file for RoboTwin: {path}. "
            "Expected a training config.yaml file."
        )
    return path


def _resolve_gpu_ids(num_gpus: int) -> list[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible == "":
        return list(range(num_gpus))
    gpu_ids: list[int] = []
    for raw in visible.split(","):
        item = raw.strip()
        if item == "":
            continue
        try:
            gpu_ids.append(int(item))
        except ValueError as exc:
            raise ValueError(
                "RoboTwin eval currently expects numeric CUDA_VISIBLE_DEVICES entries; "
                f"got {visible!r}."
            ) from exc
    if len(gpu_ids) < num_gpus:
        raise ValueError(
            f"MULTIRUN.num_gpus={num_gpus} but CUDA_VISIBLE_DEVICES only exposes "
            f"{len(gpu_ids)} GPU(s): {visible!r}."
        )
    return gpu_ids[:num_gpus]


def _load_all_tasks() -> list[str]:
    if not EVAL_STEP_LIMIT_FILE.exists():
        raise FileNotFoundError(f"Task list file not found: {EVAL_STEP_LIMIT_FILE}")
    with EVAL_STEP_LIMIT_FILE.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {EVAL_STEP_LIMIT_FILE}")
    tasks = list(task_map.keys())
    # Keep original order and remove duplicates.
    seen = set()
    dedup_tasks: list[str] = []
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        dedup_tasks.append(task)
    return dedup_tasks


def _parse_success_rate(result_file: Path) -> float:
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    return last_value


def _phase_result_filename(phase: str) -> str:
    if phase == "clean":
        return "_result_clean.txt"
    if phase == "random":
        return "_result_random.txt"
    raise ValueError(f"Unsupported phase: {phase}")


def _resolve_eval_phases(eval_phase: Any) -> list[str]:
    phase = "both" if eval_phase is None else str(eval_phase).strip().lower()
    aliases = {
        "both": ["clean", "random"],
        "all": ["clean", "random"],
        "clean_random": ["clean", "random"],
        "clean": ["clean"],
        "random": ["random"],
    }
    if phase not in aliases:
        raise ValueError(
            f"Unsupported EVALUATION.eval_phase={eval_phase!r}. "
            "Expected one of: both, clean, random."
        )
    return aliases[phase]


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _to_jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    phase: str  # "clean" | "random"
    process: subprocess.Popen[str]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.exists():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    if num_gpus <= 0:
        raise ValueError("`MULTIRUN.num_gpus` must be > 0.")
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if max_tasks_per_gpu <= 0:
        raise ValueError("`MULTIRUN.max_tasks_per_gpu` must be > 0.")
    gpu_ids = _resolve_gpu_ids(num_gpus)

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    run_output_dir = PROJECT_ROOT / "evaluate_results" / "robotwin" / ckpt_tag / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)

    manager_log = run_output_dir / "manager.log"
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"

    task_names_cfg = cfg.EVALUATION.get("task_names", None)
    task_name_cfg = cfg.EVALUATION.task_name
    if task_names_cfg is not None and len(task_names_cfg) > 0:
        tasks = [str(t) for t in task_names_cfg]
    elif task_name_cfg is not None and str(task_name_cfg).strip() != "":
        tasks = [str(task_name_cfg)]
    else:
        tasks = _load_all_tasks()
    selected_phases = _resolve_eval_phases(cfg.EVALUATION.get("eval_phase", "both"))

    file_overrides: list[str] = []
    overrides_file_cfg = cfg.MULTIRUN.get("overrides_file")
    if overrides_file_cfg:
        overrides_file = _resolve_overrides_file(str(overrides_file_cfg))
        print(f"Loaded training config overrides from: {overrides_file}", flush=True)
        file_overrides = _parse_config_yaml(overrides_file)

    cli_overrides = _collect_worker_overrides()
    cli_keys = {ov.split("=", 1)[0].lstrip("+~") for ov in cli_overrides}
    extra_overrides = [
        ov for ov in file_overrides
        if ov.split("=", 1)[0].lstrip("+~") not in cli_keys
    ] + cli_overrides

    task_rates: dict[str, dict[str, float | None]] = {
        task: {"clean": None, "random": None} for task in tasks
    }
    failed_records: list[dict[str, Any]] = []
    running_states: list[RunningState] = []

    phase_to_task_config = {
        "clean": "demo_clean",
        "random": "demo_randomized",
    }

    def result_file_for(task_name: str, phase: str) -> Path:
        return run_output_dir / task_name / phase_to_task_config[phase] / _phase_result_filename(phase)

    # Reuse existing task/phase results from prior interrupted runs.
    pending_items: list[tuple[str, str]] = []
    for task in tasks:
        for phase in selected_phases:
            result_file = result_file_for(task, phase)
            if result_file.exists():
                try:
                    task_rates[task][phase] = _parse_success_rate(result_file)
                    continue
                except Exception:
                    # Invalid or partial result files are treated as incomplete and rerun.
                    pass
            pending_items.append((task, phase))

    # Each entry is (task_name, phase) — clean and random are independent subtasks.
    pending_subtasks: deque[tuple[str, str]] = deque(pending_items)

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def build_cmd(*, task_name: str, gpu_id: int, phase: str) -> list[str]:
        task_config = phase_to_task_config[phase]
        cmd = [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={str(ckpt_path)}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={task_config}",
            f"EVALUATION.output_dir={str(output_dir)}",
        ]
        cmd.extend(extra_overrides)
        return cmd

    def launch_phase(task_name: str, gpu_id: int, phase: str) -> RunningState:
        cmd = build_cmd(task_name=task_name, gpu_id=gpu_id, phase=phase)
        log(
            f"launch task={task_name} phase={phase} gpu={gpu_id} "
            f"cmd={' '.join(cmd)}"
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
        )
        return RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            phase=phase,
            process=process,
        )

    def terminate_all_running() -> None:
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            log(f"terminating task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
            state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(f"killing task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
                state.process.kill()
                state.process.wait()

    def gpu_running_count(gpu_id: int) -> int:
        count = 0
        for state in running_states:
            if state.gpu_id != gpu_id:
                continue
            if state.process.poll() is None:
                count += 1
        return count

    def try_launch_pending(gpu_id: int) -> None:
        while len(pending_subtasks) > 0 and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            task_name, phase = pending_subtasks.popleft()
            running_states.append(launch_phase(task_name=task_name, gpu_id=gpu_id, phase=phase))

    def write_outputs() -> None:
        clean_mean = _mean_or_none([task_rates[t]["clean"] for t in tasks])
        random_mean = _mean_or_none([task_rates[t]["random"] for t in tasks])

        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
            for task in tasks:
                writer.writerow(
                    [
                        task,
                        task_rates[task]["clean"],
                        task_rates[task]["random"],
                    ]
                )
            writer.writerow(["__overall__", clean_mean, random_mean])

        payload = {
            "per_task": [
                {
                    "task_name": task,
                    "clean_success_rate": _to_jsonable(task_rates[task]["clean"]),
                    "random_success_rate": _to_jsonable(task_rates[task]["random"]),
                }
                for task in tasks
            ],
            "overall": {
                "clean_mean_success_rate": _to_jsonable(clean_mean),
                "random_mean_success_rate": _to_jsonable(random_mean),
            },
            "eval_phase": str(cfg.EVALUATION.get("eval_phase", "both")),
            "selected_phases": list(selected_phases),
        }
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        incomplete_records = [
            {
                "task_name": task,
                "phase": phase,
                "gpu_id": -1,
                "return_code": -1,
                "reason": "missing_result",
            }
            for task in tasks
            for phase in selected_phases
            if task_rates[task][phase] is None
        ]
        with failed_tasks_file.open("w", encoding="utf-8") as f:
            for rec in failed_records + incomplete_records:
                f.write(
                    f"{rec['task_name']},{rec['phase']},gpu={rec['gpu_id']},"
                    f"return_code={rec['return_code']},reason={rec['reason']}\n"
                )

    log(
        f"manager start tasks={len(tasks)} gpu_ids={gpu_ids} "
        f"max_tasks_per_gpu={max_tasks_per_gpu} phases={selected_phases} "
        f"pending_subtasks={len(pending_subtasks)} output_dir={run_output_dir}"
    )

    if len(pending_subtasks) == 0:
        log("all requested task/phase results already exist; regenerating summary only")
        write_outputs()
        log(f"summary saved: {summary_csv} and {summary_json}")
        log("manager finished successfully")
        return

    # Launch initial tasks for each GPU up to capacity.
    for gpu_id in gpu_ids:
        try_launch_pending(gpu_id)

    has_failure = False
    failure_message = ""

    while len(running_states) > 0:
        progressed = False
        for state in list(running_states):
            gpu_id = state.gpu_id
            return_code = state.process.poll()
            if return_code is None:
                continue
            progressed = True
            running_states.remove(state)

            if return_code != 0:
                has_failure = True
                failure_message = (
                    f"worker failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, return_code={return_code}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "process_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            result_file = result_file_for(state.task_name, state.phase)
            try:
                success_rate = _parse_success_rate(result_file)
            except Exception as exc:
                has_failure = True
                failure_message = (
                    f"result parse failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, error={repr(exc)}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "result_parse_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            task_rates[state.task_name][state.phase] = success_rate
            log(
                f"done task={state.task_name} phase={state.phase} gpu={gpu_id} "
                f"success_rate={success_rate:.4f}"
            )

            try_launch_pending(gpu_id)

        if has_failure:
            break
        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)

    # Mark not started subtasks when failure happened.
    if has_failure:
        for task_name, phase in pending_subtasks:
            failed_records.append(
                {
                    "task_name": task_name,
                    "phase": phase,
                    "gpu_id": -1,
                    "return_code": -1,
                    "reason": "aborted_not_started",
                }
            )

    write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")

    if has_failure:
        raise RuntimeError(failure_message)

    log("manager finished successfully")


if __name__ == "__main__":
    main()
