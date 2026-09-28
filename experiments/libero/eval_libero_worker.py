import json
import logging
import time
from pathlib import Path

import hydra
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig

from simplewam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from simplewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from simplewam.utils.pytorch_utils import set_global_seed
from experiments.libero.robosuite_compat import disable_robosuite_file_logging

disable_robosuite_file_logging()

from libero.libero import benchmark

from experiments.libero.eval_libero_single import (
    NumpyEncoder,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    maybe_compile_action_inference,
    _resolve_eval_device,
    _validate_visualize_future_video_cfg,
    ensure_num_initial_states,
    run_single_task,
)


def _read_task_file(task_file: Path) -> list[tuple[str, int]]:
    tasks: list[tuple[str, int]] = []
    for line_no, raw_line in enumerate(task_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid task line {line_no} in {task_file}: {raw_line!r}")
        tasks.append((parts[0], int(parts[1])))
    return tasks


def _resolve_worker_task_file(cfg: DictConfig) -> Path:
    task_file = cfg.MULTIRUN.get("task_file")
    if task_file is None or str(task_file).strip() == "":
        raise ValueError("MULTIRUN.task_file must point to this worker's task shard.")
    path = Path(str(task_file)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Worker task file not found: {path}")
    return path


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_worker_process(cfg: DictConfig):
    worker_start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)
    profile_action_chunk_time = bool(cfg.EVALUATION.get("profile_action_chunk_time", False))

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError("Persistent LIBERO workers currently support only EVALUATION.env_num=1.")

    task_file = _resolve_worker_task_file(cfg)
    assigned_tasks = _read_task_file(task_file)
    if len(assigned_tasks) == 0:
        print(f"No tasks assigned to worker gpu_id={cfg.gpu_id}; exiting.")
        return []

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device="cpu")
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()
    maybe_compile_action_inference(model, cfg)

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_cache = {}
    all_results = []

    print(
        f"Persistent worker started: gpu_id={cfg.gpu_id}, "
        f"tasks={len(assigned_tasks)}, task_file={task_file}"
    )

    for task_index, (suite_name, task_id) in enumerate(assigned_tasks, start=1):
        task_start_time = time.time()
        cfg.EVALUATION.task_suite_name = suite_name
        cfg.EVALUATION.task_id = int(task_id)

        video_dir = local_log_dir / suite_name / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        predicted_video_dir = local_log_dir / suite_name / "predicted_videos"
        if bool(cfg.EVALUATION.get("visualize_future_video", False)):
            predicted_video_dir.mkdir(parents=True, exist_ok=True)
        attention_dir = local_log_dir / suite_name / "attention_maps"
        if bool(cfg.EVALUATION.get("save_attention_maps", False)):
            attention_dir.mkdir(parents=True, exist_ok=True)

        if suite_name not in task_suite_cache:
            task_suite_cache[suite_name] = benchmark_dict[suite_name]()
        task_suite = task_suite_cache[suite_name]
        task = task_suite.get_task(task_id)
        initial_states = ensure_num_initial_states(
            task_suite.get_task_init_states(task_id),
            int(cfg.EVALUATION.num_trials),
        )

        results = {
            "task_suite": suite_name,
            "task_id": int(task_id),
            "task_description": None,
            "successes": 0,
            "total_episodes": int(cfg.EVALUATION.num_trials),
            "gpu_id": int(cfg.gpu_id),
            "success_episodes": [],
            "failure_episodes": [],
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": 0,
        }

        print(f"[{task_index}/{len(assigned_tasks)}] Running {suite_name} task_id={task_id}")
        task_results = run_single_task(
            task=task,
            initial_states=initial_states,
            model=model,
            processor=processor,
            cfg=cfg,
            video_dir=video_dir,
            predicted_video_dir=predicted_video_dir,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            attention_dir=attention_dir,
            profile_action_chunk_time=profile_action_chunk_time,
        )
        results.update(task_results)
        results["duration"] = time.time() - task_start_time

        output_dir = local_log_dir / suite_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"gpu{cfg.gpu_id}_task{task_id}_results.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)

        all_results.append(results)
        print(
            f"Task {suite_name}/{task_id} completed: "
            f"{results['successes']}/{cfg.EVALUATION.num_trials} successes, "
            f"time={results['duration']:.2f}s"
        )

    print(
        f"Persistent worker finished: gpu_id={cfg.gpu_id}, "
        f"tasks={len(assigned_tasks)}, total_time={time.time() - worker_start_time:.2f}s"
    )
    torch.cuda.empty_cache()
    return all_results


if __name__ == "__main__":
    eval_worker_process()
