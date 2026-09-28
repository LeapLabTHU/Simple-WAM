import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Dataset.select() deep-copies the entire DatasetInfo (including nested feature schema)
# on every call, which dominates per-sample cost (~420ms/sample, 13M deepcopy calls per
# 20 samples).  The copied info is only read, never mutated, so returning self is safe.
from datasets.info import DatasetInfo as _HFDatasetInfo
_HFDatasetInfo.copy = lambda self, **_: self

from simplewam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute action/proprio/pad/instruction metadata into sharded .npy mmap arrays."
    )
    parser.add_argument("--config", required=True, help="Path to a resolved train/run config.yaml.")
    parser.add_argument("--output-cache-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--subset-sample-indices-file",
        default=None,
        help="Path to a JSON file produced by robotwin_task_filter.py. "
             "When set, only the listed sample indices are processed.",
    )
    return parser.parse_args()


class _MetadataPrecomputeDataset(Dataset):
    def __init__(self, dataset: RobotVideoDataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.dataset.get_metadata_for_cache_precompute(int(self.indices[idx]))


def _collate_metadata(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in batch[0]:
        vals = [s[key] for s in batch]
        v0 = vals[0]
        if isinstance(v0, torch.Tensor):
            result[key] = torch.stack(vals).numpy()
        elif isinstance(v0, np.ndarray):
            result[key] = np.stack(vals)
        elif isinstance(v0, (int, float, np.integer)):
            result[key] = np.array(vals)
        else:
            result[key] = vals  # strings, etc.
    return result


def _to_numpy(value: torch.Tensor | np.ndarray | int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _open_memmap(path: str, *, dtype: np.dtype, shape: tuple[int, ...], overwrite: bool):
    if os.path.exists(path) and not overwrite:
        return None
    tmp_path = os.path.join(os.path.dirname(path), f".{Path(path).name}.tmp.{os.getpid()}")
    return np.lib.format.open_memmap(tmp_path, mode="w+", dtype=dtype, shape=shape), tmp_path


def _finalize_memmap(memmap_and_tmp, final_path: str):
    if memmap_and_tmp is None:
        return
    memmap, tmp_path = memmap_and_tmp
    memmap.flush()
    del memmap
    os.replace(tmp_path, final_path)


def _load_dataset(config_path: str, subset_sample_indices_file: str | None = None) -> RobotVideoDataset:
    cfg = OmegaConf.load(config_path)
    if cfg.get("data") is None or cfg.data.get("train") is None:
        raise ValueError("Config must contain `data.train`.")
    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    # Metadata precompute only needs action/proprio/instruction — no video access required.
    train_cfg["use_precomputed_video_latents"] = False
    train_cfg["use_precomputed_metadata_cache"] = False
    if subset_sample_indices_file is not None:
        train_cfg["subset_sample_indices_file"] = os.path.abspath(subset_sample_indices_file)
    dataset = instantiate(train_cfg)
    if not isinstance(dataset, RobotVideoDataset):
        raise TypeError(f"Expected RobotVideoDataset, got {type(dataset)}")
    return dataset


def _write_manifest(
    *,
    dataset: RobotVideoDataset,
    output_cache_dir: str,
    shard_size: int,
    num_samples: int,
    first_sample: dict[str, Any],
):
    extra = {
        "action_shape": [int(x) for x in first_sample["action"].shape],
        "proprio_shape": [int(x) for x in first_sample["proprio"].shape],
        "image_is_pad_shape": [int(x) for x in first_sample["image_is_pad"].shape],
        "action_is_pad_shape": [int(x) for x in first_sample["action_is_pad"].shape],
        "proprio_is_pad_shape": [int(x) for x in first_sample["proprio_is_pad"].shape],
        "action_dtype": str(_to_numpy(first_sample["action"]).dtype),
        "proprio_dtype": str(_to_numpy(first_sample["proprio"]).dtype),
    }
    dataset.write_metadata_cache_manifest(
        output_cache_dir,
        num_frames=int(dataset.num_frames),
        action_video_freq_ratio=int(dataset.action_video_freq_ratio),
        shard_size=int(shard_size),
        num_samples=int(num_samples),
        extra=extra,
    )


def main():
    args = _parse_args()
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")

    output_cache_dir = os.path.abspath(args.output_cache_dir)
    shards_root = os.path.join(output_cache_dir, "shards")
    os.makedirs(shards_root, exist_ok=True)

    dataset = _load_dataset(args.config, subset_sample_indices_file=args.subset_sample_indices_file)
    num_samples = int(args.num_samples) if args.num_samples is not None else len(dataset)
    if num_samples <= 0 or num_samples > len(dataset):
        raise ValueError(f"--num-samples must be in [1, {len(dataset)}], got {num_samples}")

    first_sample = dataset.get_metadata_for_cache_precompute(0)
    instruction_to_id: dict[str, int] = {}
    instructions: list[str] = []

    def instruction_id(text: str) -> int:
        found = instruction_to_id.get(text)
        if found is not None:
            return found
        new_id = len(instructions)
        instruction_to_id[text] = new_id
        instructions.append(text)
        return new_id

    # Shape info derived once from first_sample — same for all samples.
    action0 = _to_numpy(first_sample["action"])
    proprio0 = _to_numpy(first_sample["proprio"])
    image_is_pad0 = _to_numpy(first_sample["image_is_pad"]).astype(np.bool_)
    action_is_pad0 = _to_numpy(first_sample["action_is_pad"]).astype(np.bool_)
    proprio_is_pad0 = _to_numpy(first_sample["proprio_is_pad"]).astype(np.bool_)

    total_shards = (num_samples + args.shard_size - 1) // args.shard_size

    # Pre-scan: collect pending indices and build per-shard metadata.
    shard_info: dict[int, dict] = {}  # shard_id -> {paths, rows, start}
    pending_indices: list[int] = []
    skipped_count = 0
    for shard_id in range(total_shards):
        start = shard_id * args.shard_size
        end = min(start + args.shard_size, num_samples)
        rows = end - start
        shard_dir = os.path.join(shards_root, f"{shard_id:06d}")
        os.makedirs(shard_dir, exist_ok=True)
        paths = {
            "action":          os.path.join(shard_dir, "action.npy"),
            "proprio":         os.path.join(shard_dir, "proprio.npy"),
            "image_is_pad":    os.path.join(shard_dir, "image_is_pad.npy"),
            "action_is_pad":   os.path.join(shard_dir, "action_is_pad.npy"),
            "proprio_is_pad":  os.path.join(shard_dir, "proprio_is_pad.npy"),
            "num_video_frames": os.path.join(shard_dir, "num_video_frames.npy"),
            "dataset_index":   os.path.join(shard_dir, "dataset_index.npy"),
            "instruction_id":  os.path.join(shard_dir, "instruction_id.npy"),
        }
        if all(os.path.exists(p) for p in paths.values()) and not args.overwrite:
            skipped_count += rows
            continue
        shard_info[shard_id] = {"paths": paths, "rows": rows, "start": start}
        pending_indices.extend(range(start, end))

    # Lazily-opened shard memmaps: open on first write, finalize when shard complete.
    open_maps:   dict[int, dict] = {}
    open_arrays: dict[int, dict] = {}
    write_count: dict[int, int]  = {}

    def _get_arrays(shard_id: int) -> dict:
        if shard_id not in open_arrays:
            info = shard_info[shard_id]
            rows = info["rows"]
            maps = {
                "action":          _open_memmap(info["paths"]["action"],          dtype=action0.dtype,   shape=(rows, *action0.shape),          overwrite=True),
                "proprio":         _open_memmap(info["paths"]["proprio"],         dtype=proprio0.dtype,  shape=(rows, *proprio0.shape),         overwrite=True),
                "image_is_pad":    _open_memmap(info["paths"]["image_is_pad"],    dtype=np.bool_,        shape=(rows, *image_is_pad0.shape),    overwrite=True),
                "action_is_pad":   _open_memmap(info["paths"]["action_is_pad"],   dtype=np.bool_,        shape=(rows, *action_is_pad0.shape),   overwrite=True),
                "proprio_is_pad":  _open_memmap(info["paths"]["proprio_is_pad"],  dtype=np.bool_,        shape=(rows, *proprio_is_pad0.shape),  overwrite=True),
                "num_video_frames": _open_memmap(info["paths"]["num_video_frames"], dtype=np.int16,      shape=(rows,),                         overwrite=True),
                "dataset_index":   _open_memmap(info["paths"]["dataset_index"],   dtype=np.int16,        shape=(rows,),                         overwrite=True),
                "instruction_id":  _open_memmap(info["paths"]["instruction_id"],  dtype=np.int32,        shape=(rows,),                         overwrite=True),
            }
            open_maps[shard_id]   = maps
            open_arrays[shard_id] = {k: v[0] for k, v in maps.items()}
            write_count[shard_id] = 0
        return open_arrays[shard_id]

    def _finalize_shard(shard_id: int):
        for key, final_path in shard_info[shard_id]["paths"].items():
            _finalize_memmap(open_maps[shard_id][key], final_path)
        del open_maps[shard_id]
        del open_arrays[shard_id]

    def _worker_init_fn(worker_id):
        # Re-apply the DatasetInfo.copy patch in each worker process.
        # Workers may use 'spawn' and not inherit module-level monkey-patches.
        from datasets.info import DatasetInfo as _DI
        _DI.copy = lambda self, **_: self

    wrap_dataset = _MetadataPrecomputeDataset(dataset, pending_indices)
    loader = DataLoader(
        wrap_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_metadata,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
    )

    with tqdm(total=num_samples, desc="Precompute metadata cache", unit="sample",
              dynamic_ncols=True, initial=skipped_count) as pbar:
        for batch in loader:
            sample_idxs = np.asarray(batch["sample_idx"])
            batch_shard_ids = sample_idxs // args.shard_size

            # Vectorized write per shard (batches rarely cross shard boundary).
            for sid in np.unique(batch_shard_ids):
                mask = batch_shard_ids == sid
                rows = sample_idxs[mask] - sid * args.shard_size
                arrs = _get_arrays(int(sid))

                arrs["action"][rows]          = batch["action"][mask]
                arrs["proprio"][rows]         = batch["proprio"][mask]
                arrs["image_is_pad"][rows]    = batch["image_is_pad"][mask].astype(np.bool_)
                arrs["action_is_pad"][rows]   = batch["action_is_pad"][mask].astype(np.bool_)
                arrs["proprio_is_pad"][rows]  = batch["proprio_is_pad"][mask].astype(np.bool_)
                arrs["num_video_frames"][rows] = batch["num_video_frames"][mask]
                arrs["dataset_index"][rows]   = batch["dataset_index"][mask]
                arrs["instruction_id"][rows]  = np.array([
                    instruction_id(str(s)) for s in np.array(batch["instruction"])[mask]
                ], dtype=np.int32)

                write_count[int(sid)] += int(mask.sum())
                if write_count[int(sid)] == shard_info[int(sid)]["rows"]:
                    _finalize_shard(int(sid))

            pbar.update(len(sample_idxs))

    instructions_path = os.path.join(output_cache_dir, "instructions.json")
    if len(instructions) == 0 and os.path.exists(instructions_path):
        # All shards were skipped (already complete). Keep the existing valid instruction table
        # rather than overwriting it with an empty list.
        print(f"All shards already complete — keeping existing instructions.json ({instructions_path})")
    else:
        with open(instructions_path, "w", encoding="utf-8") as f:
            json.dump(instructions, f, ensure_ascii=True, indent=2)
    _write_manifest(
        dataset=dataset,
        output_cache_dir=output_cache_dir,
        shard_size=args.shard_size,
        num_samples=num_samples,
        first_sample=first_sample,
    )
    print(
        f"Wrote metadata cache: {output_cache_dir} "
        f"num_samples={num_samples} shard_size={args.shard_size} instructions={len(instructions)}"
    )


if __name__ == "__main__":
    main()
