import logging
import json
import inspect
import os
import re
from datetime import datetime
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler, ResumableWeightedEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir, self._wandb_run_name = self._resolve_run_identity(cfg)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        _save_state_every = cfg.get("save_state_every", None)
        self.save_state_every = int(_save_state_every) if _save_state_every is not None else self.save_every
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.split_video_action_loss_backward = bool(cfg.get("split_video_action_loss_backward", False))
        self.split_video_action_loss_mode = str(
            cfg.get("split_video_action_loss_mode", "alternating_batches")
        ).strip().lower()
        if self.split_video_action_loss_mode not in {"alternating_batches", "same_batch"}:
            raise ValueError(
                "Unsupported split_video_action_loss_mode="
                f"{self.split_video_action_loss_mode!r}. "
                "Expected one of: alternating_batches, same_batch."
            )
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.finetune_mode = str(cfg.model.get("finetune_mode", "full_dit")).strip().lower()
        if self.finetune_mode not in {"full_dit", "action_only"}:
            raise ValueError(
                f"Unsupported model.finetune_mode: {self.finetune_mode}. "
                "Expected one of: ['full_dit', 'action_only']."
            )
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        if (
            self.split_video_action_loss_backward
            and self.split_video_action_loss_mode == "alternating_batches"
            and self.gradient_accumulation_steps % 2 != 0
        ):
            raise ValueError(
                "split_video_action_loss_backward=true with "
                "split_video_action_loss_mode=alternating_batches requires an even "
                f"gradient_accumulation_steps, got {self.gradient_accumulation_steps}. "
                "Use gradient_accumulation_steps=2 for alternating video/action batches."
            )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )

        global_batch_size = cfg.global_batch_size
        if global_batch_size is not None:
            global_batch_size = int(global_batch_size)
            divisor = self.accelerator.num_processes * self.gradient_accumulation_steps
            if global_batch_size % divisor != 0:
                raise ValueError(
                    f"global_batch_size={global_batch_size} is not divisible by "
                    f"num_processes={self.accelerator.num_processes} * "
                    f"gradient_accumulation_steps={self.gradient_accumulation_steps} "
                    f"(divisor={divisor})"
                )
            self.batch_size = global_batch_size // divisor
        else:
            if cfg.batch_size is None:
                raise ValueError(
                    "Either batch_size or global_batch_size must be specified in the config."
                )
            self.batch_size = int(cfg.batch_size)

        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d split_video_action_loss_backward=%s split_video_action_loss_mode=%s grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.split_video_action_loss_backward,
            self.split_video_action_loss_mode,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model, finetune_mode=self.finetune_mode)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found after train-mode setup.")
        self.optimizer = self._build_optimizer()
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self._preloaded_weight_checkpoint = False
        self._preload_weight_checkpoint_before_prepare()

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self._wandb_run_name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self._wandb_run_name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    @staticmethod
    def _resolve_run_identity(cfg) -> tuple:
        """Resolve output_dir and wandb run name from config.

        Cases:
        - Neither set  → both derived from the Hydra task config name.
        - Only wandb.name set  → output_dir = ./runs/{wandb_name}/{timestamp}.
        - Only output_dir set  → wandb_name = basename(output_dir).
        - Both set  → used as-is.
        """
        try:
            from hydra.core.hydra_config import HydraConfig
            task_name = HydraConfig.get().runtime.choices.get("task") or "train"
        except Exception:
            task_name = "train"

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        raw_output_dir = cfg.output_dir
        raw_wandb_name = cfg.wandb.name

        if raw_output_dir is None and raw_wandb_name is None:
            output_dir = f"./runs/{task_name}/{timestamp}"
            wandb_name = task_name
        elif raw_output_dir is None:
            wandb_name = str(raw_wandb_name)
            output_dir = f"./runs/{wandb_name}/{timestamp}"
        elif raw_wandb_name is None:
            output_dir = str(raw_output_dir)
            wandb_name = os.path.basename(output_dir.rstrip("/\\"))
        else:
            output_dir = str(raw_output_dir)
            wandb_name = str(raw_wandb_name)

        return output_dir, wandb_name

    def _build_loader(self, dataset, worker_init_fn=None):
        sampling_cfg = self.cfg.get("sampling", {}) or {}
        weighted_sampling = bool(sampling_cfg.get("weighted", False))
        if weighted_sampling:
            if not hasattr(dataset, "build_sampling_weights"):
                raise TypeError(
                    "sampling.weighted=true requires the train dataset to implement build_sampling_weights()."
                )
            weights, sampling_summary = dataset.build_sampling_weights(
                rules=sampling_cfg.get("rules", []),
                default_weight=float(sampling_cfg.get("default_weight", 1.0)),
                combine=str(sampling_cfg.get("combine", "max")),
            )
            num_samples_cfg = sampling_cfg.get("num_samples", None)
            num_samples = int(num_samples_cfg) if num_samples_cfg is not None else len(dataset)
            replacement = bool(sampling_cfg.get("replacement", True))
            self.train_sampler = ResumableWeightedEpochSampler(
                dataset=dataset,
                weights=weights,
                seed=self.seed,
                batch_size=self.batch_size,
                num_processes=self.accelerator.num_processes,
                num_samples=num_samples,
                replacement=replacement,
            )
            logger.info(
                "Weighted sampling enabled: replacement=%s num_samples=%d default_weight=%.3f "
                "combine=%s min=%.3f max=%.3f mean=%.3f rules=%s",
                replacement,
                num_samples,
                float(sampling_summary.get("default_weight", 1.0)),
                sampling_summary.get("combine", "max"),
                float(sampling_summary.get("min_weight", 0.0)),
                float(sampling_summary.get("max_weight", 0.0)),
                float(sampling_summary.get("mean_weight", 0.0)),
                sampling_summary.get("rules", []),
            )
        else:
            self.train_sampler = ResumableEpochSampler(
                dataset=dataset,
                seed=self.seed,
                batch_size=self.batch_size,
                num_processes=self.accelerator.num_processes,
            )
        # print(f"batch_size={self.batch_size} num_processes={self.accelerator.num_processes}")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_optimizer(self) -> torch.optim.AdamW:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(
            "Optimizer: lr=%.2e weight_decay=%.2e total_params=%.2fM",
            self.learning_rate,
            self.weight_decay,
            sum(p.numel() for p in trainable) / 1e6,
        )
        return torch.optim.AdamW(
            trainable,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _preload_weight_checkpoint_before_prepare(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if not resume_path.is_file():
            return
        logger.info("Pre-loading weight checkpoint before accelerator prepare: %s", resume)
        self.model.load_checkpoint(str(resume_path), optimizer=None)
        self._preloaded_weight_checkpoint = True
        logger.warning("Loaded .pt weights only before accelerator prepare; optimizer/scheduler/step were not restored.")

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        if getattr(self, "_preloaded_weight_checkpoint", False):
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self, log_trainable_params: bool = False):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model, finetune_mode=self.finetune_mode, log_trainable_params=log_trainable_params)

    @staticmethod
    def _apply_dit_only_train_mode(model, finetune_mode: str = "full_dit", log_trainable_params: bool = False):
        model.eval()
        model.requires_grad_(False)
        disable_video_dit_training = bool(getattr(model, "disable_video_dit_training", False))
        if disable_video_dit_training:
            model.dit.mixtures["action"].train()
            model.dit.mixtures["action"].requires_grad_(True)
            enabled_action_params = sum(
                int(param.numel()) for param in model.dit.mixtures["action"].parameters() if param.requires_grad
            )
            logger.info(
                "Train mode: disable video DiT training (first-frame video tokens only), "
                "action_params=%.2fM",
                enabled_action_params / 1e6,
            )
            return
        if finetune_mode == "action_only":
            model.dit.mixtures["action"].train()
            model.dit.mixtures["action"].requires_grad_(True)
            enabled_action_params = sum(
                int(param.numel()) for param in model.dit.mixtures["action"].parameters() if param.requires_grad
            )
            logger.info(
                "Train mode: action-only finetune (video expert frozen), "
                "action_params=%.2fM",
                enabled_action_params / 1e6,
            )
            return
        model.dit.train()
        model.dit.requires_grad_(True)

        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

        ### count the number of trainable parameters / total parameters
        trainable_params = 0
        total_params = 0
        trainable_param_keys = []
        frozen_param_keys = []
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                trainable_param_keys.append(name)
            else:
                frozen_param_keys.append(name)
        if log_trainable_params:
            logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M, Total parameters: {total_params / 1e6:.2f}M")

            # print all trainable parameter keys
            logger.info(f"Trainable parameter keys: {trainable_param_keys}")
            logger.info(f"Frozen parameter keys: {frozen_param_keys}")

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        # was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample, train_step=self.global_step)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        # if was_dit_training:
        self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)

        # Release GPU memory held by eval inference (infer + VAE encode/decode).
        # Without this, CUDA memory fragmentation after eval can cause OOM ~100 steps later.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self, save_weights: bool = True, save_state: bool = True):
        step_tag = f"step_{self.global_step:06d}"
        ckpt_path = None
        state_path = None

        if save_weights:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
            self.accelerator.wait_for_everyone()

        if save_state:
            state_path = os.path.join(self.state_dir, step_tag)
            ensure_dir(state_path)
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode(log_trainable_params=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        split_accum_loss_tensors: list[torch.Tensor] = []
        split_accum_loss_dicts: list[dict[str, float]] = []

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                if self.split_video_action_loss_backward:
                    if self.split_video_action_loss_mode == "same_batch":
                        with self.accelerator.autocast():
                            loss_action_pass, loss_dict_action = train_model.training_loss(
                                sample,
                                train_step=self.global_step,
                                loss_mode="action_only",
                            )
                        self.accelerator.backward(loss_action_pass)

                        with self.accelerator.autocast():
                            loss_video_pass, loss_dict_video = train_model.training_loss(
                                sample,
                                train_step=self.global_step,
                                loss_mode="video_only",
                            )
                        self.accelerator.backward(loss_video_pass)

                        loss = loss_action_pass.detach() + loss_video_pass.detach()
                        loss_dict = {
                            "loss_video": float(loss_dict_video.get("loss_video", 0.0)),
                            "loss_action": float(loss_dict_action.get("loss_action", 0.0)),
                            "loss_action_raw": float(loss_dict_action.get("loss_action_raw", 0.0)),
                            "action_loss_active": float(loss_dict_action.get("action_loss_active", 1.0)),
                            "action_loss_masked_ratio": float(loss_dict_action.get("action_loss_masked_ratio", 0.0)),
                            "video_target_due": float(loss_dict_video.get("video_target_due", 1.0)),
                            "split_video_action_loss_backward": 1.0,
                            "split_video_action_loss_same_batch": 1.0,
                            "split_loss_mode_video": 1.0,
                            "split_loss_mode_action": 1.0,
                            "split_loss_video_pass": float(loss_video_pass.detach().item()),
                            "split_loss_action_pass": float(loss_action_pass.detach().item()),
                        }
                    else:
                        split_micro_step = (int(self.batch_in_epoch) - 1) % self.gradient_accumulation_steps
                        split_loss_mode = "video_only" if split_micro_step % 2 == 0 else "action_only"
                        with self.accelerator.autocast():
                            loss, loss_dict = train_model.training_loss(
                                sample,
                                train_step=self.global_step,
                                loss_mode=split_loss_mode,
                            )
                        self.accelerator.backward(loss)
                        loss_dict = dict(loss_dict)
                        loss_dict["split_video_action_loss_backward"] = 1.0
                        loss_dict["split_video_action_loss_same_batch"] = 0.0
                        loss_dict["split_micro_step"] = float(split_micro_step)
                        loss_dict["split_loss_mode_video"] = float(split_loss_mode == "video_only")
                        loss_dict["split_loss_mode_action"] = float(split_loss_mode == "action_only")
                        split_accum_loss_tensors.append(loss.detach())
                        split_accum_loss_dicts.append({key: float(value) for key, value in loss_dict.items()})
                else:
                    with self.accelerator.autocast():
                        loss, loss_dict = train_model.training_loss(sample, train_step=self.global_step)
                    self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    if (
                        self.split_video_action_loss_backward
                        and self.split_video_action_loss_mode == "alternating_batches"
                        and len(split_accum_loss_tensors) > 0
                    ):
                        loss = torch.stack([x.to(device=loss.device) for x in split_accum_loss_tensors]).sum()
                        merged_loss_dict: dict[str, float] = {}
                        for item in split_accum_loss_dicts:
                            for key, value in item.items():
                                if key in {
                                    "split_video_action_loss_backward",
                                    "split_loss_mode_video",
                                    "split_loss_mode_action",
                                    "action_loss_active",
                                    "video_target_due",
                                }:
                                    merged_loss_dict[key] = merged_loss_dict.get(key, 0.0) + float(value)
                                elif key == "split_micro_step":
                                    merged_loss_dict[key] = float(value)
                                else:
                                    merged_loss_dict[key] = merged_loss_dict.get(key, 0.0) + float(value)
                        denom = float(max(len(split_accum_loss_dicts), 1))
                        for key in (
                            "split_video_action_loss_backward",
                            "split_loss_mode_video",
                            "split_loss_mode_action",
                            "action_loss_active",
                            "video_target_due",
                        ):
                            if key in merged_loss_dict:
                                merged_loss_dict[key] /= denom
                        loss_dict = merged_loss_dict
                        split_accum_loss_tensors.clear()
                        split_accum_loss_dicts.clear()
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "%s speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            "lr=%.2e" % current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    _save_weights = self.save_every > 0 and self.global_step % self.save_every == 0
                    _save_state = self.save_state_every > 0 and self.global_step % self.save_state_every == 0
                    if _save_weights or _save_state:
                        ckpt_info = self.save_checkpoint(save_weights=_save_weights, save_state=_save_state)
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
