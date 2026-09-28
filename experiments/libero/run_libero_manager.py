import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from experiments.libero.robosuite_compat import disable_robosuite_file_logging

disable_robosuite_file_logging()

from libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf


def create_task_file(output_file: Path, task_suite_names: list[str]) -> Path:
    benchmark_dict = benchmark.get_benchmark_dict()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_tasks = 0
    with output_file.open("w", encoding="utf-8") as f:
        for suite_name in task_suite_names:
            task_suite = benchmark_dict[suite_name]()
            n_tasks = int(task_suite.n_tasks)
            print(f"\n{suite_name}:")
            print(f"- Number of tasks: {n_tasks}")
            for task_id in range(n_tasks):
                f.write(f"{suite_name},{task_id}\n")
                total_tasks += 1

    print(f"\nTask list created: {output_file}")
    print(f"Total tasks: {total_tasks}")
    return output_file


# Keys that belong to the training loop and must not be forwarded to eval workers.
_TRAINING_ONLY_KEYS = frozenset({
    "batch_size", "global_batch_size", "num_workers", "num_epochs",
    "max_steps", "log_every", "save_every", "eval_every",
    "gradient_accumulation_steps", "output_dir", "resume",
    "lr_scheduler_type", "learning_rate", "weight_decay",
    "mixed_precision", "seed", "max_grad_norm",
    "overwrite_video_latents", "video_latent_save_dtype",
    "video_latent_batch_size", "video_latent_num_workers",
    "eval_num_inference_steps",
})
_TRAINING_ONLY_PREFIXES = ("wandb.", "data.")


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked_exact = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
    }
    if key in blocked_exact:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _is_blocked_from_file(raw_override: str) -> bool:
    """Extends _is_blocked_override with training-only keys (used when reading from file)."""
    if _is_blocked_override(raw_override):
        return True
    key = raw_override.split("=", 1)[0].lstrip("+~")
    return key in _TRAINING_ONLY_KEYS or any(key.startswith(p) for p in _TRAINING_ONLY_PREFIXES)


def collect_worker_overrides() -> list[str]:
    hydra_overrides = list(HydraConfig.get().overrides.task)
    return [ov for ov in hydra_overrides if not _is_blocked_override(ov)]


def _resolve_worker_task_choice() -> str:
    task_choice = HydraConfig.get().runtime.choices.get("task")
    if task_choice is None or str(task_choice).strip() == "":
        raise ValueError(
            "Hydra task choice is empty. Please pass task=... (e.g., task=my_config)."
        )
    return str(task_choice)


def _find_overrides_file(ckpt: str) -> Path | None:
    """Walk up from the checkpoint file looking for hydra_overrides.txt (max 6 levels)."""
    p = Path(ckpt).resolve()
    for i, parent in enumerate(p.parents):
        if i > 6:
            break
        candidate = parent / "hydra_overrides.txt"
        if candidate.is_file():
            return candidate
    return None


def _parse_overrides_file(path: Path) -> tuple[str, list[str]]:
    """Parse a hydra_overrides.txt saved during training.

    Returns (task_choice, extra_overrides) where extra_overrides contains only
    non-training-specific overrides safe to forward to eval workers.
    Raises ValueError if no task= line is found.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    overrides = [l.strip() for l in lines if l.strip() and not l.startswith("#")]

    task_choice = None
    extra: list[str] = []
    for ov in overrides:
        key = ov.split("=", 1)[0].lstrip("+~")
        if key == "task":
            task_choice = ov.split("=", 1)[1]
        elif not _is_blocked_from_file(ov):
            extra.append(ov)

    if task_choice is None:
        raise ValueError(
            f"No 'task=' line found in {path}. "
            "Cannot determine which model config to use for evaluation."
        )
    return task_choice, extra


# ---------------------------------------------------------------------------
# YAML config.yaml support
# ---------------------------------------------------------------------------

def _format_scalar(v) -> str:
    """Format a Python scalar for Hydra override syntax."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    s = str(v)
    # Quote strings that contain characters Hydra would misparse.
    if any(c in s for c in " ,{}[]\\"):
        s = f"'{s}'"
    return s


def _flatten_to_overrides(obj, prefix: str, result: list[str]) -> None:
    """Recursively convert a nested config value to Hydra override strings."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_to_overrides(v, f"{prefix}.{k}", result)
    elif isinstance(obj, list):
        items = ",".join(_format_scalar(i) for i in obj)
        result.append(f"{prefix}=[{items}]")
    else:
        result.append(f"{prefix}={_format_scalar(obj)}")


# Model keys that control how the model is *loaded* at eval time.
# These are set by the eval config (sim_libero.yaml) and must not be
# overridden with training-time values from config.yaml.
_BLOCKED_MODEL_KEYS_FROM_YAML = frozenset({
    "load_text_encoder",
    "skip_dit_load_from_pretrain",
    "action_dit_pretrained_path",
})


def _parse_config_yaml(path: Path) -> list[str]:
    """Extract model.* overrides from a full config.yaml saved during training.

    Returns Hydra override strings for architecture-relevant keys under `model`.
    Eval-specific loading flags (load_text_encoder, etc.) are excluded so that
    the eval config's values are preserved.
    Task name is NOT extracted here — it must be supplied via the CLI task= arg
    or auto-detected from the file path by the caller.
    """
    raw = OmegaConf.load(path)
    config = OmegaConf.to_container(raw, resolve=False)
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    overrides: list[str] = []
    if isinstance(model_cfg, dict):
        for k, v in model_cfg.items():
            if k not in _BLOCKED_MODEL_KEYS_FROM_YAML:
                _flatten_to_overrides(v, f"model.{k}", overrides)
    return overrides


def _get_explicit_cli_task() -> str | None:
    """Return the task name if `task=` was explicitly passed on the CLI, else None."""
    for ov in HydraConfig.get().overrides.task:
        key = ov.split("=", 1)[0].lstrip("+~")
        if key == "task":
            return ov.split("=", 1)[1]
    return None


def run_evaluation(
    *,
    task_file: Path,
    task_choice: str,
    ckpt: str,
    num_gpus: int,
    num_trials: int,
    max_tasks_per_gpu: int,
    output_dir: Path,
    extra_overrides: list[str],
    persistent_workers: bool,
) -> None:
    script_name = "run_libero_persistent_workers.sh" if persistent_workers else "run_libero_parallel_test.sh"
    script_path = Path("experiments/libero") / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Evaluation script not found: {script_path}")

    root_dir = os.getcwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_args = shlex.join(extra_overrides) if extra_overrides else ""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    env = os.environ.copy()
    env.update(
        {
            "CONFIG": task_choice,
            "CKPT": ckpt,
            "NUM_GPUS": str(num_gpus),
            "NUM_TRIALS": str(num_trials),
            "MAX_TASKS_PER_GPU": str(max_tasks_per_gpu),
            "ROOT_DIR": root_dir,
            "RUN_ID": run_id,
            "OUTPUT_DIR": str(output_dir),
            "EXTRA_ARGS": extra_args,
            "EXP_NAME": os.environ.get("EXP_NAME", ""),
            "PYTHON_BIN": os.environ.get("PYTHON_BIN", sys.executable),
        }
    )

    print("\nStarting evaluation (Hydra manager)...")
    print(f"task: {task_choice}")
    print(f"Checkpoint: {ckpt}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Trials per task: {num_trials}")
    print(f"Max tasks per GPU: {max_tasks_per_gpu}")
    print(f"Persistent workers: {persistent_workers}")
    print(f"Output directory: {output_dir}")
    if extra_args:
        print(f"Forwarded overrides: {extra_args}")

    try:
        subprocess.run(
            ["bash", str(script_path), str(task_file)],
            env=env,
            check=True,
            text=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as e:
        print(f"Evaluation script failed with return code: {e.returncode}")
        failed_tasks = output_dir / "failed_tasks.txt"
        if failed_tasks.exists() and failed_tasks.stat().st_size > 0:
            print(f"Failed subtask list: {failed_tasks}")
            print(failed_tasks.read_text(encoding='utf-8'))
        raise


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    if cfg.EVALUATION.output_dir is None:
        raise ValueError("EVALUATION.output_dir must not be None.")

    manager = cfg.MULTIRUN

    # --- Resolve task choice and extra overrides ---
    # Priority: explicit MULTIRUN.overrides_file > auto-detect from ckpt path > task= CLI arg.
    overrides_file_cfg = manager.get("overrides_file")
    overrides_file: Path | None = None
    if overrides_file_cfg:
        overrides_file = Path(os.path.expanduser(os.path.expandvars(str(overrides_file_cfg))))
        if not overrides_file.is_file():
            raise FileNotFoundError(f"MULTIRUN.overrides_file not found: {overrides_file}")
    else:
        overrides_file = _find_overrides_file(str(cfg.ckpt))

    if overrides_file is not None:
        print(f"Loaded training overrides from: {overrides_file}")
        if overrides_file.suffix in (".yaml", ".yml"):
            # Full config.yaml: extract model.* overrides; derive task separately.
            file_overrides = _parse_config_yaml(overrides_file)
            explicit_task = _get_explicit_cli_task()
            if explicit_task:
                task_choice = explicit_task
            else:
                # Guess from path structure: runs/{task_name}/{timestamp}/config.yaml
                try:
                    task_choice = overrides_file.resolve().parents[1].name
                    print(f"Auto-detected task from config path: {task_choice}")
                except Exception:
                    raise ValueError(
                        "Cannot determine task choice from the YAML overrides_file path. "
                        "Please also pass task=<config_name> explicitly."
                    )
        else:
            # hydra_overrides.txt: task and extra overrides come from the file.
            file_task, file_overrides = _parse_overrides_file(overrides_file)
            # Explicit CLI task= takes precedence over the one in the file.
            task_choice = _get_explicit_cli_task() or file_task

        # CLI overrides take precedence over file overrides for the same key.
        cli_overrides = collect_worker_overrides()
        cli_keys = {ov.split("=", 1)[0].lstrip("+~") for ov in cli_overrides}
        extra_overrides = [
            ov for ov in file_overrides
            if ov.split("=", 1)[0].lstrip("+~") not in cli_keys
        ] + cli_overrides
    else:
        task_choice = _resolve_worker_task_choice()
        extra_overrides = collect_worker_overrides()

    output_dir = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir))))
    output_dir.mkdir(parents=True, exist_ok=True)

    task_file_cfg = manager.get("task_file")
    if task_file_cfg:
        task_file = Path(os.path.expanduser(os.path.expandvars(str(task_file_cfg))))
    else:
        task_file = output_dir / "tasks.txt"
    task_file = create_task_file(task_file, list(manager.task_suite_names))

    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))

    if bool(manager.get("create_only", False)):
        print("create_only=True, only create the task list and exit.")
        return

    run_evaluation(
        task_file=task_file,
        task_choice=task_choice,
        ckpt=str(cfg.ckpt),
        num_gpus=int(manager.num_gpus),
        num_trials=int(cfg.EVALUATION.num_trials),
        max_tasks_per_gpu=int(manager.max_tasks_per_gpu),
        output_dir=output_dir,
        extra_overrides=extra_overrides,
        persistent_workers=bool(manager.get("persistent_workers", False)),
    )


if __name__ == "__main__":
    main()
