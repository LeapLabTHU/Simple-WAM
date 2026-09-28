import logging
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from simplewam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from simplewam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from simplewam.utils.config_resolvers import register_default_resolvers
from simplewam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)


class _VideoLatentPrecomputeDataset(Dataset):
    def __init__(self, dataset: RobotVideoDataset, assigned_indices: list[int], overwrite: bool):
        self.dataset = dataset
        self.assigned_indices = assigned_indices
        self.overwrite = bool(overwrite)

    def __len__(self) -> int:
        return len(self.assigned_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        data_idx = int(self.assigned_indices[idx])
        # Resolve to source index first so the cache path is correct when a subset is active
        # (subset dataset positions are small integers, but files are keyed by source index)
        source_idx = int(self.dataset._source_index(data_idx))
        cache_path = self.dataset.get_video_latent_cache_path(source_idx)
        existed_before = os.path.exists(cache_path)
        if existed_before and (not self.overwrite):
            return {
                "skip": True,
            }

        sample_idx, video = self.dataset.get_video_for_latent_precompute(data_idx)
        sample_idx = int(sample_idx)
        if sample_idx != source_idx:
            cache_path = self.dataset.get_video_latent_cache_path(sample_idx)
            existed_before = os.path.exists(cache_path)
            if existed_before and (not self.overwrite):
                return {
                    "skip": True,
                }

        return {
            "skip": False,
            "sample_idx": sample_idx,
            "video": video,
            "cache_path": cache_path,
            "existed_before": existed_before,
        }


def _precompute_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    skip_count = 0
    to_encode = []
    for item in batch:
        if item["skip"]:
            skip_count += 1
            continue
        to_encode.append(item)

    payload = {
        "batch_size": len(batch),
        "skip_count": skip_count,
    }
    if not to_encode:
        payload["videos"] = None
        payload["sample_idx"] = []
        payload["cache_paths"] = []
        payload["existed_before"] = []
        return payload

    payload["videos"] = torch.stack([item["video"] for item in to_encode], dim=0)
    payload["sample_idx"] = [int(item["sample_idx"]) for item in to_encode]
    payload["cache_paths"] = [str(item["cache_path"]) for item in to_encode]
    payload["existed_before"] = [bool(item["existed_before"]) for item in to_encode]
    return payload


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _mixed_precision_to_dtype(mixed_precision: str) -> torch.dtype:
    key = str(mixed_precision).strip().lower()
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    if key == "no":
        return torch.float32
    raise ValueError(f"Unsupported mixed_precision: {mixed_precision}")


def _atomic_torch_save(payload: dict[str, Any], output_path: str):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = os.path.join(output_dir, f".{Path(output_path).name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, output_path)


def _atomic_npz_save(payload: dict[str, Any], output_path: str, *, compressed: bool):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = os.path.join(output_dir, f".{Path(output_path).name}.tmp.{os.getpid()}")
    if compressed:
        np.savez_compressed(tmp_path, **payload)
    else:
        np.savez(tmp_path, **payload)
    if not tmp_path.endswith(".npz"):
        source_path = f"{tmp_path}.npz"
    else:
        source_path = tmp_path
    os.replace(source_path, output_path)


def _save_latent_payload(
    *,
    output_path: str,
    latents: torch.Tensor,
    sample_idx: int,
    storage_format: str,
):
    if storage_format == "pt":
        payload = {
            "input_latents": latents,
            "sample_idx": int(sample_idx),
        }
        _atomic_torch_save(payload, output_path)
        return

    payload_np = {
        "input_latents": latents.cpu().numpy(),
        "sample_idx": np.array(int(sample_idx), dtype=np.int64),
    }
    if storage_format == "npz":
        _atomic_npz_save(payload_np, output_path, compressed=False)
        return
    if storage_format == "npz_compressed":
        _atomic_npz_save(payload_np, output_path, compressed=True)
        return
    raise ValueError(
        f"Unsupported `video_latent_storage_format`: {storage_format}. "
        "Expected one of: ['pt', 'npz', 'npz_compressed']."
    )


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    is_distributed, rank, world_size, local_rank = _init_distributed()

    if cfg.data is None or cfg.data.get("train") is None:
        raise ValueError("`cfg.data.train` is required.")
    if cfg.model is None:
        raise ValueError("`cfg.model` is required.")

    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    cache_dir = train_cfg.get("video_latent_cache_dir")
    if cache_dir is None or not str(cache_dir).strip():
        raise ValueError(
            "Please set `data.train.video_latent_cache_dir` to store precomputed latents."
        )
    train_cfg["use_precomputed_video_latents"] = False
    train_cfg["strict_video_latent_loading"] = True
    if bool(train_cfg.get("skip_padding_as_possible", False)):
        logger.warning(
            "`skip_padding_as_possible=true` may make latent precompute non-deterministic. "
            "Force setting it to false for stable cache indexing."
        )
        train_cfg["skip_padding_as_possible"] = False

    dataset = instantiate(train_cfg)
    if not isinstance(dataset, RobotVideoDataset):
        raise TypeError(
            f"Expected RobotVideoDataset from `cfg.data.train`, got {type(dataset)}"
        )

    overwrite = _to_bool(cfg.get("overwrite_video_latents", False))
    latent_storage_format = str(train_cfg.get("video_latent_storage_format", "pt")).strip().lower()
    if latent_storage_format not in {"pt", "npz", "npz_compressed"}:
        raise ValueError(
            f"Unsupported `data.train.video_latent_storage_format`: {latent_storage_format}. "
            "Expected one of: ['pt', 'npz', 'npz_compressed']."
        )
    save_dtype_key = str(cfg.get("video_latent_save_dtype", "fp16")).strip().lower()
    save_dtype = torch.float16 if save_dtype_key == "fp16" else torch.bfloat16
    if save_dtype_key not in {"fp16", "bf16"}:
        raise ValueError("`video_latent_save_dtype` must be one of: fp16, bf16")
    latent_batch_size = int(cfg.get("video_latent_batch_size", 1))
    if latent_batch_size <= 0:
        raise ValueError("`video_latent_batch_size` must be > 0.")
    latent_num_workers = int(cfg.get("video_latent_num_workers", cfg.get("num_workers", 4)))
    if latent_num_workers < 0:
        raise ValueError("`video_latent_num_workers` must be >= 0.")

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"
    model_dtype = _mixed_precision_to_dtype(cfg.get("mixed_precision", "bf16"))

    model_id = str(cfg.model.get("model_id"))
    tokenizer_model_id = str(cfg.model.get("tokenizer_model_id"))
    redirect_common_files = bool(cfg.model.get("redirect_common_files", True))
    _, _, vae_config, _ = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    vae_config.download_if_necessary()
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=model_dtype,
        device=device,
    ).eval()

    if rank == 0:
        dataset.write_video_latent_manifest(
            cache_dir=str(cache_dir),
            num_frames=int(dataset.num_frames),
            action_video_freq_ratio=int(dataset.action_video_freq_ratio),
            video_size=[int(dataset.video_size[0]), int(dataset.video_size[1])],
            concat_multi_camera=str(dataset.concat_multi_camera),
            extra={
                "vae_path": str(vae_config.path),
                "vae_temporal_downsample_factor": int(vae.temporal_downsample_factor),
                "vae_upsampling_factor": int(vae.upsampling_factor),
                "vae_z_dim": int(vae.z_dim),
                "save_dtype": save_dtype_key,
                "video_latent_storage_format": latent_storage_format,
            },
        )
    if is_distributed:
        dist.barrier()

    indices = list(range(rank, len(dataset), world_size)) if is_distributed else list(range(len(dataset)))
    precompute_dataset = _VideoLatentPrecomputeDataset(
        dataset=dataset,
        assigned_indices=indices,
        overwrite=overwrite,
    )
    loader_kwargs = {
        "dataset": precompute_dataset,
        "batch_size": latent_batch_size,
        "shuffle": False,
        "num_workers": latent_num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "collate_fn": _precompute_collate,
    }
    if latent_num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    precompute_loader = DataLoader(**loader_kwargs)

    stats = {"new": 0, "overwrite": 0, "skip": 0}
    logger.info(
        "Precompute settings: batch_size=%d num_workers=%d overwrite=%s save_dtype=%s storage_format=%s assigned_samples=%d",
        latent_batch_size,
        latent_num_workers,
        overwrite,
        save_dtype_key,
        latent_storage_format,
        len(indices),
    )
    with tqdm(
        total=len(precompute_dataset),
        desc=f"Precompute video latents (rank {rank}/{world_size})" if is_distributed else "Precompute video latents",
        unit="sample",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for batch in precompute_loader:
                stats["skip"] += int(batch["skip_count"])
                if batch["videos"] is not None:
                    video_batch = batch["videos"].to(
                        device=device,
                        dtype=model_dtype,
                        non_blocking=True,
                    )
                    latents_batch = vae.encode(video_batch, device=device, tiled=False)
                    if not isinstance(latents_batch, torch.Tensor):
                        raise TypeError(
                            f"Expected tensor latents from VAE encode, got {type(latents_batch)}"
                        )
                    if latents_batch.ndim != 5 or latents_batch.shape[0] != len(batch["sample_idx"]):
                        raise ValueError(
                            "Unexpected latent shape from VAE encode: "
                            f"got {tuple(latents_batch.shape)}, expected batch={len(batch['sample_idx'])}"
                        )
                    latents_batch = latents_batch.detach().to(device="cpu", dtype=save_dtype).contiguous()

                    for i, sample_idx in enumerate(batch["sample_idx"]):
                        _save_latent_payload(
                            output_path=batch["cache_paths"][i],
                            latents=latents_batch[i],
                            sample_idx=int(sample_idx),
                            storage_format=latent_storage_format,
                        )
                        if batch["existed_before"][i]:
                            stats["overwrite"] += 1
                        else:
                            stats["new"] += 1

                pbar.update(int(batch["batch_size"]))

    logger.info(
        "Rank %d/%d finished. cache_dir=%s new=%d overwrite=%d skip=%d",
        rank,
        world_size,
        str(cache_dir),
        stats["new"],
        stats["overwrite"],
        stats["skip"],
    )

    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
