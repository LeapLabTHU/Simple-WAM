import hashlib
import json
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from simplewam.utils.logging_config import get_logger
from simplewam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

ROBOTWIN_TASK_ORDER = [
    "pick_diverse_bottles",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_dual_bottles",
    "adjust_bottle",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]
ROBOTWIN_TASK_GROUP_SIZE = 550


class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        use_precomputed_video_latents: bool = False,
        video_latent_cache_dir: Optional[str] = None,
        strict_video_latent_loading: bool = True,
        video_latent_storage_format: str = "pt",
        cache_text_embeddings_in_memory: bool = True,
        video_latent_shard_size: int = 10000,
        force_return_images: bool = False,
        use_precomputed_metadata_cache: bool = False,
        metadata_cache_dir: Optional[str] = None,
        metadata_cache_shard_size: int = 10000,
        subset_sample_indices_file: Optional[str] = None,
        prompt_subset_file: Optional[str] = None,
        libero_fewshot_per_task: Optional[int] = None,
        libero_fewshot_seed: int = 0,
        libero_fewshot_allow_less: bool = False,
        libero_include_suites: Optional[list[str]] = None,
        libero_drop_task_ids=None,
        robotwin_clean_only: bool = False,
        robotwin_clean_group_size: int = 550,
        robotwin_clean_episodes_per_group: int = 50,
        robotwin_clean_random_select: bool = False,
        robotwin_clean_random_seed: int = 0,
        robotwin_clean_pool_episodes_per_group: int = 50,
        robotwin_drop_tasks: Optional[list[str]] = None,
        robotwin_action_loss_drop_tasks: Optional[list[str]] = None,
        action_chunk_size: Optional[int] = None,
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        self.dataset_dirs = [str(ds) for ds in dataset_dirs]

        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        if action_chunk_size is not None:
            assert 1 <= action_chunk_size <= num_frames - 1, \
                f"action_chunk_size must be in [1, num_frames-1], got {action_chunk_size} vs num_frames-1={num_frames-1}"
        self.action_chunk_size = action_chunk_size

        self.camera_key = camera_key
        self.use_precomputed_video_latents = bool(use_precomputed_video_latents)
        self.video_latent_cache_dir = (
            os.path.abspath(str(video_latent_cache_dir))
            if video_latent_cache_dir not in (None, "")
            else None
        )
        self.strict_video_latent_loading = bool(strict_video_latent_loading)
        storage_format = str(video_latent_storage_format).strip().lower()
        if storage_format not in {"pt", "npz", "npz_compressed", "sharded_npy"}:
            raise ValueError(
                f"Unsupported `video_latent_storage_format`: {video_latent_storage_format}. "
                "Expected one of: ['pt', 'npz', 'npz_compressed', 'sharded_npy']."
            )
        self.video_latent_storage_format = storage_format
        self.video_latent_shard_size = int(video_latent_shard_size)
        if self.video_latent_shard_size <= 0:
            raise ValueError("`video_latent_shard_size` must be > 0.")
        self._video_latent_shard_cache = {}
        needs_images = bool(force_return_images) or (
            (not self.use_precomputed_video_latents) or (not self.strict_video_latent_loading)
        )
        self.lerobot_dataset._set_return_images(needs_images)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.cache_text_embeddings_in_memory = bool(cache_text_embeddings_in_memory)
        # Bounded LRU cache: 921K unique prompts × 1MB each would exhaust RAM with many workers.
        # Cap at 512 entries per worker (~512MB max) — enough to benefit from locality without OOM.
        self._text_embedding_memory_cache: "collections.OrderedDict[str, tuple]" = __import__("collections").OrderedDict()
        self._text_embedding_memory_cache_maxsize = 512
        self.use_precomputed_metadata_cache = bool(use_precomputed_metadata_cache)
        self.metadata_cache_dir = (
            os.path.abspath(str(metadata_cache_dir))
            if metadata_cache_dir not in (None, "")
            else None
        )
        self.metadata_cache_shard_size = int(metadata_cache_shard_size)
        if self.metadata_cache_shard_size <= 0:
            raise ValueError("`metadata_cache_shard_size` must be > 0.")
        self._metadata_shard_cache = {}
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self._source_num_samples = int(len(self.lerobot_dataset))
        self.subset_sample_indices_file = (
            os.path.abspath(str(subset_sample_indices_file))
            if subset_sample_indices_file not in (None, "")
            else None
        )
        self.prompt_subset_file = (
            os.path.abspath(str(prompt_subset_file))
            if prompt_subset_file not in (None, "")
            else None
        )
        self.libero_fewshot_per_task = (
            None if libero_fewshot_per_task in (None, "") else int(libero_fewshot_per_task)
        )
        self.libero_fewshot_seed = int(libero_fewshot_seed)
        self.libero_fewshot_allow_less = bool(libero_fewshot_allow_less)
        self.libero_include_suites = self._normalize_libero_suite_names(
            libero_include_suites,
            field_name="libero_include_suites",
        )
        self.libero_drop_task_ids = self._normalize_libero_drop_task_ids(
            libero_drop_task_ids,
            field_name="libero_drop_task_ids",
        )
        if self.libero_fewshot_per_task is not None and self.libero_fewshot_per_task <= 0:
            raise ValueError(f"libero_fewshot_per_task must be > 0, got {self.libero_fewshot_per_task}.")
        self.robotwin_clean_only = bool(robotwin_clean_only)
        self.robotwin_clean_group_size = int(robotwin_clean_group_size)
        self.robotwin_clean_episodes_per_group = int(robotwin_clean_episodes_per_group)
        self.robotwin_clean_random_select = bool(robotwin_clean_random_select)
        self.robotwin_clean_random_seed = int(robotwin_clean_random_seed)
        self.robotwin_clean_pool_episodes_per_group = int(robotwin_clean_pool_episodes_per_group)
        self._robotwin_clean_random_offsets: dict[int, set[int]] = {}
        self.robotwin_drop_tasks = self._normalize_robotwin_task_names(
            robotwin_drop_tasks,
            field_name="robotwin_drop_tasks",
        )
        self.robotwin_action_loss_drop_tasks = self._normalize_robotwin_task_names(
            robotwin_action_loss_drop_tasks,
            field_name="robotwin_action_loss_drop_tasks",
        )
        overlap = self.robotwin_drop_tasks & self.robotwin_action_loss_drop_tasks
        if overlap:
            raise ValueError(
                "A RoboTwin task cannot be both dropped and action-loss-dropped: "
                f"{sorted(overlap)}"
            )
        if self.robotwin_clean_group_size <= 0:
            raise ValueError("robotwin_clean_group_size must be > 0.")
        if not (0 < self.robotwin_clean_episodes_per_group <= self.robotwin_clean_group_size):
            raise ValueError(
                "robotwin_clean_episodes_per_group must be in "
                f"[1, robotwin_clean_group_size], got {self.robotwin_clean_episodes_per_group} "
                f"vs group_size={self.robotwin_clean_group_size}."
            )
        if not (0 < self.robotwin_clean_pool_episodes_per_group <= self.robotwin_clean_group_size):
            raise ValueError(
                "robotwin_clean_pool_episodes_per_group must be in "
                f"[1, robotwin_clean_group_size], got {self.robotwin_clean_pool_episodes_per_group} "
                f"vs group_size={self.robotwin_clean_group_size}."
            )
        if self.robotwin_clean_random_select and (
            self.robotwin_clean_episodes_per_group > self.robotwin_clean_pool_episodes_per_group
        ):
            raise ValueError(
                "robotwin_clean_episodes_per_group must be <= robotwin_clean_pool_episodes_per_group "
                "when robotwin_clean_random_select=true, got "
                f"{self.robotwin_clean_episodes_per_group} vs {self.robotwin_clean_pool_episodes_per_group}."
            )
        # Load subset indices first so manifest validation can use the correct num_samples.
        self._sample_indices = self._load_subset_sample_indices()
        self._metadata_cache_indices: Optional[np.ndarray] = None
        if self.libero_include_suites is not None:
            self._sample_indices, self._metadata_cache_indices = self._apply_libero_suite_filter(
                self._sample_indices,
                self._metadata_cache_indices,
            )
        if self.libero_drop_task_ids:
            self._sample_indices, self._metadata_cache_indices = self._apply_libero_drop_task_id_filter(
                self._sample_indices,
                self._metadata_cache_indices,
            )
        if self.libero_fewshot_per_task is not None:
            self._sample_indices, self._metadata_cache_indices = self._apply_libero_fewshot_filter(
                self._sample_indices,
                self._metadata_cache_indices,
            )
        if self.robotwin_clean_only:
            self._sample_indices, self._metadata_cache_indices = self._apply_robotwin_clean_filter(
                self._sample_indices
            )
        if self.robotwin_drop_tasks:
            self._sample_indices, self._metadata_cache_indices = self._apply_robotwin_drop_task_filter(
                self._sample_indices,
                self._metadata_cache_indices,
            )
        self._robotwin_action_loss_source_mask = self._build_robotwin_action_loss_source_mask()
        if self.use_precomputed_metadata_cache and self.metadata_cache_dir is None:
            raise ValueError("`metadata_cache_dir` must be set when `use_precomputed_metadata_cache=true`.")
        if self.metadata_cache_dir is not None:
            self._validate_metadata_cache_manifest()
        if self.use_precomputed_video_latents and self.video_latent_cache_dir is None:
            raise ValueError(
                "`video_latent_cache_dir` must be set when `use_precomputed_video_latents=true`."
            )
        if self.video_latent_cache_dir is not None:
            os.makedirs(self.video_latent_cache_dir, exist_ok=True)
            self._validate_video_latent_manifest()
            self._validate_video_latent_cache_population()
        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        if self._sample_indices is not None:
            return int(len(self._sample_indices))
        return len(self.lerobot_dataset)

    def _source_index(self, idx: int) -> int:
        idx = int(idx)
        if self._sample_indices is None:
            return idx
        return int(self._sample_indices[idx])

    def _random_source_index(self) -> int:
        if self._sample_indices is None:
            return int(np.random.randint(len(self.lerobot_dataset)))
        return int(self._sample_indices[np.random.randint(len(self._sample_indices))])

    def _load_subset_sample_indices(self) -> Optional[np.ndarray]:
        if self.subset_sample_indices_file is not None and self.prompt_subset_file is not None:
            raise ValueError(
                "Set only one of subset_sample_indices_file or prompt_subset_file, not both."
            )
        if self.subset_sample_indices_file is not None:
            indices = self._read_sample_indices_file(self.subset_sample_indices_file)
        elif self.prompt_subset_file is not None:
            indices = self._build_sample_indices_from_prompt_subset(self.prompt_subset_file)
        else:
            return None

        if indices.ndim != 1:
            raise ValueError(f"Subset sample indices must be 1D, got shape {tuple(indices.shape)}.")
        if len(indices) == 0:
            raise ValueError("Subset sample indices are empty.")
        indices = np.asarray(indices, dtype=np.int64)
        if int(indices.min()) < 0 or int(indices.max()) >= self._source_num_samples:
            raise ValueError(
                "Subset sample indices out of bounds: "
                f"min={int(indices.min())}, max={int(indices.max())}, "
                f"source_num_samples={self._source_num_samples}."
            )
        if np.any(indices[1:] < indices[:-1]):
            indices = np.sort(indices)
        logger.info(
            "Using dataset subset with %d/%d samples (%.2f%%).",
            len(indices),
            self._source_num_samples,
            100.0 * len(indices) / max(self._source_num_samples, 1),
        )
        return indices

    @staticmethod
    def _stable_uint32_from_text(text: str) -> int:
        digest = hashlib.sha1(text.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="little", signed=False)

    @staticmethod
    def _canonical_libero_suite_name(path_or_name: str) -> str:
        name = os.path.basename(str(path_or_name).rstrip("/"))
        for suffix in ("_no_noops_lerobot", "_no_noops", "_lerobot"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name

    @staticmethod
    def _libero_suite_aliases(canonical_name: str) -> set[str]:
        aliases = {canonical_name}
        if canonical_name.startswith("libero_"):
            aliases.add(canonical_name[len("libero_"):])
        if canonical_name.startswith("libero_plus_"):
            aliases.add(canonical_name[len("libero_plus_"):])
        return aliases

    def _normalize_libero_suite_names(self, value, field_name: str) -> Optional[set[str]]:
        if value in (None, ""):
            return None
        if isinstance(value, str):
            raw_names = [item.strip() for item in value.split(",")]
        else:
            raw_names = [str(item).strip() for item in value]
        raw_names = [name for name in raw_names if name]
        if not raw_names:
            return None

        alias_to_canonical = {}
        for dataset_dir in self.dataset_dirs:
            canonical = self._canonical_libero_suite_name(dataset_dir)
            for alias in self._libero_suite_aliases(canonical):
                alias_to_canonical[alias] = canonical

        selected = set()
        unknown = []
        for name in raw_names:
            canonical = alias_to_canonical.get(name, alias_to_canonical.get(self._canonical_libero_suite_name(name)))
            if canonical is None:
                unknown.append(name)
            else:
                selected.add(canonical)
        if unknown:
            raise ValueError(
                f"Unknown LIBERO suite names in {field_name}: {unknown}. "
                f"Known suites/aliases: {sorted(alias_to_canonical)}"
            )
        return selected

    def _build_libero_suite_source_mask(self, suite_names: set[str]) -> np.ndarray:
        keep_mask = np.zeros(self._source_num_samples, dtype=bool)
        sample_offset = 0
        matched_samples = 0
        matched_suites = []
        for dataset_idx, dataset in enumerate(self.lerobot_dataset.multi_dataset._datasets):
            dataset_dir = self.dataset_dirs[dataset_idx] if dataset_idx < len(self.dataset_dirs) else str(dataset.root)
            suite_name = self._canonical_libero_suite_name(dataset_dir)
            num_samples = int(dataset.num_frames)
            if suite_name in suite_names:
                start = sample_offset
                end = sample_offset + num_samples
                keep_mask[start:end] = True
                matched_samples += num_samples
                matched_suites.append(suite_name)
            sample_offset += num_samples

        logger.info(
            "LIBERO suite filter %s matched suites %s with %d source samples.",
            sorted(suite_names),
            matched_suites,
            matched_samples,
        )
        return keep_mask

    def _normalize_libero_drop_task_ids(self, value, field_name: str) -> dict[str, set[int]]:
        if value in (None, ""):
            return {}
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if not isinstance(value, dict):
            raise TypeError(
                f"{field_name} must be a mapping from LIBERO suite name to task IDs, "
                f"got {type(value).__name__}."
            )

        alias_to_canonical: dict[str, str] = {}
        known_task_ids: dict[str, set[int]] = {}
        for dataset_idx, dataset_dir in enumerate(self.dataset_dirs):
            canonical = self._canonical_libero_suite_name(dataset_dir)
            for alias in self._libero_suite_aliases(canonical):
                alias_to_canonical[alias] = canonical
            dataset = self.lerobot_dataset.multi_dataset._datasets[dataset_idx]
            known_task_ids.setdefault(canonical, set()).update(
                int(task_id) for task_id in dataset.meta.tasks
            )

        normalized: dict[str, set[int]] = {}
        unknown_suites = []
        invalid_entries = []
        for raw_suite, raw_task_ids in value.items():
            suite_text = str(raw_suite).strip()
            canonical = alias_to_canonical.get(
                suite_text,
                alias_to_canonical.get(self._canonical_libero_suite_name(suite_text)),
            )
            if canonical is None:
                unknown_suites.append(suite_text)
                continue

            if isinstance(raw_task_ids, str):
                task_id_values = [item.strip() for item in raw_task_ids.split(",") if item.strip()]
            elif isinstance(raw_task_ids, (int, np.integer)):
                task_id_values = [raw_task_ids]
            else:
                try:
                    task_id_values = list(raw_task_ids)
                except TypeError:
                    invalid_entries.append((suite_text, raw_task_ids))
                    continue

            task_ids = set()
            try:
                task_ids = {int(task_id) for task_id in task_id_values}
            except (TypeError, ValueError):
                invalid_entries.append((suite_text, raw_task_ids))
                continue
            if any(task_id < 0 for task_id in task_ids):
                invalid_entries.append((suite_text, raw_task_ids))
                continue

            unknown_ids = sorted(task_ids - known_task_ids.get(canonical, set()))
            if unknown_ids:
                raise ValueError(
                    f"Unknown LIBERO task IDs for suite `{canonical}` in {field_name}: {unknown_ids}. "
                    f"Known task IDs: {sorted(known_task_ids.get(canonical, set()))}"
                )
            normalized.setdefault(canonical, set()).update(task_ids)

        if unknown_suites:
            raise ValueError(
                f"Unknown LIBERO suite names in {field_name}: {unknown_suites}. "
                f"Known suites/aliases: {sorted(alias_to_canonical)}"
            )
        if invalid_entries:
            raise ValueError(
                f"Invalid LIBERO task ID entries in {field_name}: {invalid_entries}. "
                "Each value must be an integer or a list/comma-separated string of integers."
            )
        return {suite: task_ids for suite, task_ids in normalized.items() if task_ids}

    def _build_libero_drop_task_source_mask(self) -> tuple[np.ndarray, dict[str, dict[int, int]]]:
        keep_mask = np.ones(self._source_num_samples, dtype=bool)
        dropped_episode_counts: dict[str, dict[int, int]] = {}
        sample_offset = 0

        for dataset_idx, dataset in enumerate(self.lerobot_dataset.multi_dataset._datasets):
            dataset_dir = self.dataset_dirs[dataset_idx] if dataset_idx < len(self.dataset_dirs) else str(dataset.root)
            suite_name = self._canonical_libero_suite_name(dataset_dir)
            drop_ids = self.libero_drop_task_ids.get(suite_name)
            episodes = dataset.episodes
            if episodes is None:
                episodes = list(range(dataset.meta.total_episodes))
            ep_from = dataset.episode_data_index["from"]
            ep_to = dataset.episode_data_index["to"]
            if len(episodes) != len(ep_from):
                raise ValueError(
                    "LIBERO task-ID filtering expected one episode_data_index entry per episode: "
                    f"suite={suite_name} episodes={len(episodes)} from={len(ep_from)}."
                )

            if drop_ids:
                counts = dropped_episode_counts.setdefault(suite_name, {})
                for local_ep_idx, episode_index in enumerate(episodes):
                    episode_index = int(episode_index)
                    episode_meta = dataset.meta.episodes.get(episode_index, {})
                    tasks = episode_meta.get("tasks", None)
                    if not tasks:
                        raise ValueError(
                            "LIBERO task-ID filtering requires episode task metadata. "
                            f"Missing tasks for suite={suite_name} dataset={dataset.root} "
                            f"episode={episode_index}."
                        )
                    episode_task_ids = set()
                    for task_name in tasks:
                        task_id = dataset.meta.task_to_task_index.get(str(task_name))
                        if task_id is None:
                            raise ValueError(
                                "LIBERO episode references an unknown task name: "
                                f"suite={suite_name} episode={episode_index} task={task_name!r}."
                            )
                        episode_task_ids.add(int(task_id))
                    matched_ids = episode_task_ids & drop_ids
                    if not matched_ids:
                        continue
                    start = sample_offset + int(ep_from[local_ep_idx])
                    end = sample_offset + int(ep_to[local_ep_idx])
                    keep_mask[start:end] = False
                    for task_id in matched_ids:
                        counts[task_id] = counts.get(task_id, 0) + 1

            sample_offset += int(dataset.num_frames)

        return keep_mask, dropped_episode_counts

    def _apply_libero_drop_task_id_filter(
        self,
        indices: Optional[np.ndarray],
        metadata_cache_indices: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        keep_source_mask, dropped_episode_counts = self._build_libero_drop_task_source_mask()
        if indices is None:
            filtered = np.nonzero(keep_source_mask)[0].astype(np.int64)
            filtered_metadata_indices = filtered.copy()
            before = self._source_num_samples
        else:
            indices = np.asarray(indices, dtype=np.int64)
            keep_rows = np.nonzero(keep_source_mask[indices])[0].astype(np.int64)
            filtered = indices[keep_rows]
            if metadata_cache_indices is None:
                filtered_metadata_indices = keep_rows
            else:
                filtered_metadata_indices = np.asarray(metadata_cache_indices, dtype=np.int64)[keep_rows]
            before = len(indices)

        if len(filtered) == 0:
            formatted = {
                suite: sorted(task_ids) for suite, task_ids in self.libero_drop_task_ids.items()
            }
            raise ValueError(f"libero_drop_task_ids removed all samples. Dropped tasks: {formatted}")
        logger.info(
            "Dropped LIBERO task IDs %s with episode counts %s: kept %d/%d samples (%.2f%%).",
            {suite: sorted(task_ids) for suite, task_ids in self.libero_drop_task_ids.items()},
            dropped_episode_counts,
            len(filtered),
            before,
            100.0 * len(filtered) / max(before, 1),
        )
        return filtered.astype(np.int64, copy=False), filtered_metadata_indices.astype(np.int64, copy=False)

    def _apply_libero_suite_filter(
        self,
        indices: Optional[np.ndarray],
        metadata_cache_indices: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.libero_include_suites is None:
            raise ValueError("_apply_libero_suite_filter called with libero_include_suites=None.")
        keep_source_mask = self._build_libero_suite_source_mask(self.libero_include_suites)
        if indices is None:
            filtered = np.nonzero(keep_source_mask)[0].astype(np.int64)
            filtered_metadata_indices = filtered.copy()
            before = self._source_num_samples
        else:
            indices = np.asarray(indices, dtype=np.int64)
            keep_rows = np.nonzero(keep_source_mask[indices])[0].astype(np.int64)
            filtered = indices[keep_rows]
            if metadata_cache_indices is None:
                filtered_metadata_indices = keep_rows
            else:
                filtered_metadata_indices = np.asarray(metadata_cache_indices, dtype=np.int64)[keep_rows]
            before = len(indices)

        if len(filtered) == 0:
            raise ValueError(
                "libero_include_suites removed all samples. "
                f"Selected suites: {sorted(self.libero_include_suites)}"
            )
        logger.info(
            "Using LIBERO suites %s with %d/%d samples (%.2f%%).",
            sorted(self.libero_include_suites),
            len(filtered),
            before,
            100.0 * len(filtered) / max(before, 1),
        )
        return filtered.astype(np.int64, copy=False), filtered_metadata_indices.astype(np.int64, copy=False)

    def _apply_libero_fewshot_filter(
        self,
        indices: Optional[np.ndarray],
        metadata_cache_indices: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.libero_fewshot_per_task is None:
            raise ValueError("_apply_libero_fewshot_filter called with libero_fewshot_per_task=None.")

        keep_source_mask = np.zeros(self._source_num_samples, dtype=bool)
        sample_offset = 0
        total_task_pools = 0
        total_selected_episodes = 0
        total_available_episodes = 0
        insufficient = []

        for dataset_idx, dataset in enumerate(self.lerobot_dataset.multi_dataset._datasets):
            episodes = dataset.episodes
            if episodes is None:
                episodes = list(range(dataset.meta.total_episodes))
            ep_from = dataset.episode_data_index["from"]
            ep_to = dataset.episode_data_index["to"]
            if len(episodes) != len(ep_from):
                raise ValueError(
                    "LIBERO few-shot filtering expected one episode_data_index entry per episode: "
                    f"episodes={len(episodes)} from={len(ep_from)}."
                )

            task_to_episode_rows: dict[str, list[tuple[int, int]]] = {}
            for local_ep_idx, episode_index in enumerate(episodes):
                episode_index = int(episode_index)
                episode_meta = dataset.meta.episodes.get(episode_index, {})
                tasks = episode_meta.get("tasks", None)
                if not tasks:
                    raise ValueError(
                        "LIBERO few-shot filtering requires episode task metadata. "
                        f"Missing tasks for dataset={dataset.root} episode={episode_index}."
                    )
                task_name = str(tasks[0])
                task_to_episode_rows.setdefault(task_name, []).append((local_ep_idx, episode_index))

            for task_name, episode_rows in sorted(task_to_episode_rows.items()):
                episode_rows = sorted(episode_rows, key=lambda item: item[1])
                available = len(episode_rows)
                total_task_pools += 1
                total_available_episodes += available
                if available < self.libero_fewshot_per_task and not self.libero_fewshot_allow_less:
                    insufficient.append((str(dataset.root), task_name, available))
                    continue
                select_count = min(self.libero_fewshot_per_task, available)
                seed_text = f"{self.libero_fewshot_seed}:{dataset_idx}:{dataset.root}:{task_name}"
                rng = np.random.default_rng(self._stable_uint32_from_text(seed_text))
                selected_positions = rng.choice(available, size=select_count, replace=False)
                for pos in selected_positions.tolist():
                    local_ep_idx, _ = episode_rows[int(pos)]
                    start = sample_offset + int(ep_from[local_ep_idx])
                    end = sample_offset + int(ep_to[local_ep_idx])
                    keep_source_mask[start:end] = True
                    total_selected_episodes += 1

            sample_offset += int(dataset.num_frames)

        if insufficient:
            details = "; ".join(
                f"{os.path.basename(ds)} | {task}: {count}"
                for ds, task, count in insufficient[:10]
            )
            suffix = "" if len(insufficient) <= 10 else f"; ... and {len(insufficient) - 10} more"
            raise ValueError(
                "Some LIBERO tasks have fewer episodes than libero_fewshot_per_task="
                f"{self.libero_fewshot_per_task}. Set libero_fewshot_allow_less=true to keep all "
                f"available episodes for those tasks. Insufficient tasks: {details}{suffix}"
            )

        if indices is None:
            filtered = np.nonzero(keep_source_mask)[0].astype(np.int64)
            filtered_metadata_indices = filtered.copy()
            before = self._source_num_samples
        else:
            indices = np.asarray(indices, dtype=np.int64)
            keep_rows = np.nonzero(keep_source_mask[indices])[0].astype(np.int64)
            filtered = indices[keep_rows]
            if metadata_cache_indices is None:
                filtered_metadata_indices = keep_rows
            else:
                filtered_metadata_indices = np.asarray(metadata_cache_indices, dtype=np.int64)[keep_rows]
            before = len(indices)

        if len(filtered) == 0:
            raise ValueError(
                "LIBERO few-shot filtering removed all samples. "
                f"libero_fewshot_per_task={self.libero_fewshot_per_task}, seed={self.libero_fewshot_seed}."
            )
        logger.info(
            "Using LIBERO few-shot subset with %d/%d samples (%.2f%%). "
            "Selected %d/%d episodes across %d task pools, per_task=%d, seed=%d, allow_less=%s.",
            len(filtered),
            before,
            100.0 * len(filtered) / max(before, 1),
            total_selected_episodes,
            total_available_episodes,
            total_task_pools,
            self.libero_fewshot_per_task,
            self.libero_fewshot_seed,
            self.libero_fewshot_allow_less,
        )
        return filtered.astype(np.int64, copy=False), filtered_metadata_indices.astype(np.int64, copy=False)

    @staticmethod
    def _normalize_robotwin_task_names(value, field_name: str) -> set[str]:
        if value in (None, ""):
            return set()
        if isinstance(value, str):
            raw_names = [item.strip() for item in value.split(",")]
        else:
            raw_names = [str(item).strip() for item in value]
        names = {name for name in raw_names if name}
        known = set(ROBOTWIN_TASK_ORDER)
        unknown = sorted(names - known)
        if unknown:
            raise ValueError(
                f"Unknown RoboTwin task names in {field_name}: {unknown}. "
                f"Known tasks: {ROBOTWIN_TASK_ORDER}"
            )
        return names

    def _robotwin_task_name_for_episode(self, episode_index: int) -> Optional[str]:
        task_idx = int(episode_index) // ROBOTWIN_TASK_GROUP_SIZE
        if 0 <= task_idx < len(ROBOTWIN_TASK_ORDER):
            return ROBOTWIN_TASK_ORDER[task_idx]
        return None

    def _build_robotwin_task_source_mask(self, task_names: set[str], *, active_value: bool) -> np.ndarray:
        mask = np.full(self._source_num_samples, not active_value, dtype=bool)
        if not task_names:
            return mask
        sample_offset = 0
        matched_episodes = 0
        matched_samples = 0
        for dataset in self.lerobot_dataset.multi_dataset._datasets:
            episodes = dataset.episodes
            if episodes is None:
                episodes = list(range(dataset.meta.total_episodes))
            ep_from = dataset.episode_data_index["from"]
            ep_to = dataset.episode_data_index["to"]
            if len(episodes) != len(ep_from):
                raise ValueError(
                    "RoboTwin task filtering expected one episode_data_index entry per episode: "
                    f"episodes={len(episodes)} from={len(ep_from)}."
                )
            for local_ep_idx, episode_index in enumerate(episodes):
                task_name = self._robotwin_task_name_for_episode(int(episode_index))
                if task_name not in task_names:
                    continue
                start = sample_offset + int(ep_from[local_ep_idx])
                end = sample_offset + int(ep_to[local_ep_idx])
                mask[start:end] = active_value
                matched_episodes += 1
                matched_samples += max(0, end - start)
            sample_offset += int(dataset.num_frames)
        logger.info(
            "RoboTwin task mask for %s matched %d episodes and %d source samples.",
            sorted(task_names),
            matched_episodes,
            matched_samples,
        )
        return mask

    def _apply_robotwin_drop_task_filter(
        self,
        indices: Optional[np.ndarray],
        metadata_cache_indices: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        keep_source_mask = self._build_robotwin_task_source_mask(
            self.robotwin_drop_tasks,
            active_value=False,
        )
        if indices is None:
            filtered = np.nonzero(keep_source_mask)[0].astype(np.int64)
            filtered_metadata_indices = filtered.copy()
            before = self._source_num_samples
        else:
            indices = np.asarray(indices, dtype=np.int64)
            keep_rows = np.nonzero(keep_source_mask[indices])[0].astype(np.int64)
            filtered = indices[keep_rows]
            if metadata_cache_indices is None:
                filtered_metadata_indices = keep_rows
            else:
                filtered_metadata_indices = np.asarray(metadata_cache_indices, dtype=np.int64)[keep_rows]
            before = len(indices)
        if len(filtered) == 0:
            raise ValueError(
                "robotwin_drop_tasks removed all samples. "
                f"Dropped tasks: {sorted(self.robotwin_drop_tasks)}"
            )
        logger.info(
            "Dropped RoboTwin tasks %s: kept %d/%d samples (%.2f%%).",
            sorted(self.robotwin_drop_tasks),
            len(filtered),
            before,
            100.0 * len(filtered) / max(before, 1),
        )
        return filtered.astype(np.int64, copy=False), filtered_metadata_indices.astype(np.int64, copy=False)

    def _build_robotwin_action_loss_source_mask(self) -> Optional[np.ndarray]:
        if not self.robotwin_action_loss_drop_tasks:
            return None
        action_active = self._build_robotwin_task_source_mask(
            self.robotwin_action_loss_drop_tasks,
            active_value=False,
        )
        logger.info(
            "RoboTwin action loss disabled for tasks: %s.",
            sorted(self.robotwin_action_loss_drop_tasks),
        )
        return action_active

    def _apply_robotwin_clean_filter(self, indices: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        clean_mask = self._build_robotwin_clean_source_mask()
        if indices is None:
            filtered = np.nonzero(clean_mask)[0].astype(np.int64)
            metadata_cache_indices = filtered.copy()
            before = self._source_num_samples
        else:
            indices = np.asarray(indices, dtype=np.int64)
            keep_rows = np.nonzero(clean_mask[indices])[0].astype(np.int64)
            filtered = indices[keep_rows]
            metadata_cache_indices = keep_rows
            before = len(indices)
        if len(filtered) == 0:
            raise ValueError(
                "robotwin_clean_only=true removed all samples. "
                "Check robotwin_clean_group_size and robotwin_clean_episodes_per_group."
            )
        if self.robotwin_clean_random_select:
            logger.info(
                "Using RobotWin random clean few-shot subset with %d/%d samples (%.2f%%). "
                "Rule: fixed random %d/%d clean episodes per %d-episode task block, seed=%d.",
                len(filtered),
                before,
                100.0 * len(filtered) / max(before, 1),
                self.robotwin_clean_episodes_per_group,
                self.robotwin_clean_pool_episodes_per_group,
                self.robotwin_clean_group_size,
                self.robotwin_clean_random_seed,
            )
        else:
            logger.info(
                "Using RobotWin clean-only subset with %d/%d samples (%.2f%%). "
                "Rule: episode_index %% %d < %d.",
                len(filtered),
                before,
                100.0 * len(filtered) / max(before, 1),
                self.robotwin_clean_group_size,
                self.robotwin_clean_episodes_per_group,
            )
        return filtered.astype(np.int64, copy=False), metadata_cache_indices.astype(np.int64, copy=False)

    def _robotwin_selected_clean_offsets(self, group_idx: int) -> set[int]:
        if not self.robotwin_clean_random_select:
            return set(range(self.robotwin_clean_episodes_per_group))
        if group_idx not in self._robotwin_clean_random_offsets:
            rng = np.random.default_rng(self.robotwin_clean_random_seed + int(group_idx))
            offsets = rng.choice(
                self.robotwin_clean_pool_episodes_per_group,
                size=self.robotwin_clean_episodes_per_group,
                replace=False,
            )
            self._robotwin_clean_random_offsets[group_idx] = {int(offset) for offset in offsets.tolist()}
        return self._robotwin_clean_random_offsets[group_idx]

    def _build_robotwin_clean_source_mask(self) -> np.ndarray:
        clean_mask = np.zeros(self._source_num_samples, dtype=bool)
        sample_offset = 0
        clean_episodes = 0
        total_episodes = 0
        for dataset in self.lerobot_dataset.multi_dataset._datasets:
            episodes = dataset.episodes
            if episodes is None:
                episodes = list(range(dataset.meta.total_episodes))
            ep_from = dataset.episode_data_index["from"]
            ep_to = dataset.episode_data_index["to"]
            if len(episodes) != len(ep_from):
                raise ValueError(
                    "RobotWin clean-only filtering expected one episode_data_index entry per episode: "
                    f"episodes={len(episodes)} from={len(ep_from)}."
                )
            for local_ep_idx, episode_index in enumerate(episodes):
                total_episodes += 1
                episode_index = int(episode_index)
                group_idx = episode_index // self.robotwin_clean_group_size
                group_offset = episode_index % self.robotwin_clean_group_size
                if group_offset not in self._robotwin_selected_clean_offsets(group_idx):
                    continue
                clean_episodes += 1
                start = sample_offset + int(ep_from[local_ep_idx])
                end = sample_offset + int(ep_to[local_ep_idx])
                clean_mask[start:end] = True
            sample_offset += int(dataset.num_frames)
        if self.robotwin_clean_random_select:
            logger.info(
                "RobotWin random clean few-shot episode filter kept %d/%d episodes. "
                "Rule: choose %d/%d clean episodes per %d-episode task block with seed=%d.",
                clean_episodes,
                total_episodes,
                self.robotwin_clean_episodes_per_group,
                self.robotwin_clean_pool_episodes_per_group,
                self.robotwin_clean_group_size,
                self.robotwin_clean_random_seed,
            )
        else:
            logger.info(
                "RobotWin clean-only episode filter kept %d/%d episodes.",
                clean_episodes,
                total_episodes,
            )
        return clean_mask

    @staticmethod
    def _read_sample_indices_file(path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing subset sample indices file: {path}")
        if path.endswith(".npy"):
            return np.load(path, allow_pickle=False)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("sample_indices", payload.get("indices"))
        if payload is None:
            raise ValueError(f"Cannot find sample indices in {path}")
        return np.asarray(payload, dtype=np.int64)

    @staticmethod
    def _read_prompt_subset_file(path: str) -> list[str]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing prompt subset file: {path}")
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                payload = payload.get("prompts", payload.get("instructions"))
            if not isinstance(payload, list):
                raise ValueError(f"Prompt subset JSON must contain a list: {path}")
            return [str(x) for x in payload]
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _build_sample_indices_from_prompt_subset(self, path: str) -> np.ndarray:
        if self.metadata_cache_dir is None:
            raise ValueError("metadata_cache_dir is required when using prompt_subset_file.")
        prompts = set(self._read_prompt_subset_file(path))
        if not prompts:
            raise ValueError(f"Prompt subset is empty: {path}")
        instructions = self._load_metadata_cache_instructions()
        selected_ids = np.asarray(
            [i for i, text in enumerate(instructions) if str(text) in prompts],
            dtype=np.int64,
        )
        if len(selected_ids) == 0:
            raise ValueError(f"No prompt subset entries matched metadata instructions: {path}")
        selected_set = set(int(x) for x in selected_ids.tolist())
        shards_root = os.path.join(self.metadata_cache_dir, "shards")
        if not os.path.isdir(shards_root):
            raise FileNotFoundError(f"Missing metadata cache shards directory: {shards_root}")
        chunks: list[np.ndarray] = []
        for shard_name in sorted(os.listdir(shards_root)):
            shard_dir = os.path.join(shards_root, shard_name)
            instruction_path = os.path.join(shard_dir, "instruction_id.npy")
            if not os.path.isfile(instruction_path):
                continue
            instruction_ids = np.load(instruction_path, mmap_mode="r", allow_pickle=False)
            mask = np.isin(instruction_ids, list(selected_set))
            if not bool(mask.any()):
                continue
            shard_start = int(shard_name) * self.metadata_cache_shard_size
            rows = np.nonzero(mask)[0].astype(np.int64)
            chunks.append(rows + shard_start)
        if not chunks:
            raise ValueError(f"Prompt subset matched instructions but no samples: {path}")
        return np.concatenate(chunks)

    @property
    def _metadata_cache_manifest_path(self) -> Optional[str]:
        if self.metadata_cache_dir is None:
            return None
        return os.path.join(self.metadata_cache_dir, "manifest.json")

    def _expected_metadata_cache_manifest(self) -> dict:
        num_samples = len(self._sample_indices) if self._sample_indices is not None else self._source_num_samples
        return {
            "version": 1,
            "num_frames": int(self.num_frames),
            "action_video_freq_ratio": int(self.action_video_freq_ratio),
            "metadata_cache_storage_format": "sharded_npy",
            "metadata_cache_shard_size": int(self.metadata_cache_shard_size),
            "num_samples": int(num_samples),
        }

    def _metadata_cache_key(self, idx: int, sample_idx: Optional[int] = None) -> int:
        idx = int(idx)
        if self._metadata_cache_indices is not None:
            return int(self._metadata_cache_indices[idx])
        if self._sample_indices is not None:
            return idx
        if sample_idx is None:
            return idx
        return int(sample_idx)

    def _validate_metadata_cache_manifest(self):
        manifest_path = self._metadata_cache_manifest_path
        if manifest_path is None or not os.path.exists(manifest_path):
            if self.use_precomputed_metadata_cache:
                raise FileNotFoundError(
                    f"Missing metadata cache manifest: {manifest_path}. "
                    "Please run scripts/precompute_metadata_cache.py first."
                )
            return
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        expected = self._expected_metadata_cache_manifest()
        for key, expected_value in expected.items():
            loaded_value = payload.get(key)
            if key == "num_samples" and self._metadata_cache_indices is not None:
                min_required = int(self._metadata_cache_indices.max()) + 1
                if loaded_value is None or int(loaded_value) < min_required:
                    raise ValueError(
                        f"Metadata cache manifest mismatch for `{key}`: "
                        f"expected at least {min_required} rows for mapped cache reuse, "
                        f"got={loaded_value}. Cache: {manifest_path}"
                    )
                continue
            if loaded_value != expected_value:
                raise ValueError(
                    f"Metadata cache manifest mismatch for `{key}`: "
                    f"expected={expected_value}, got={loaded_value}. Cache: {manifest_path}"
                )
        shards_dir = os.path.join(self.metadata_cache_dir, "shards")
        if self.use_precomputed_metadata_cache and not os.path.isdir(shards_dir):
            raise FileNotFoundError(f"Missing metadata cache shards directory: {shards_dir}")

    @staticmethod
    def write_metadata_cache_manifest(
        cache_dir: str,
        *,
        num_frames: int,
        action_video_freq_ratio: int,
        shard_size: int,
        num_samples: int,
        extra: Optional[dict] = None,
    ):
        os.makedirs(cache_dir, exist_ok=True)
        payload = {
            "version": 1,
            "num_frames": int(num_frames),
            "action_video_freq_ratio": int(action_video_freq_ratio),
            "metadata_cache_storage_format": "sharded_npy",
            "metadata_cache_shard_size": int(shard_size),
            "num_samples": int(num_samples),
        }
        if extra:
            payload.update(extra)
        with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    @property
    def _video_latent_manifest_path(self) -> Optional[str]:
        if self.video_latent_cache_dir is None:
            return None
        return os.path.join(self.video_latent_cache_dir, "manifest.json")

    def _expected_video_latent_manifest(self) -> dict:
        expected = {
            "version": 1,
            "num_frames": int(self.num_frames),
            "action_video_freq_ratio": int(self.action_video_freq_ratio),
            "video_size": [int(self.video_size[0]), int(self.video_size[1])],
            "concat_multi_camera": str(self.concat_multi_camera),
            "video_latent_storage_format": str(self.video_latent_storage_format),
        }
        if self.video_latent_storage_format == "sharded_npy":
            expected["video_latent_shard_size"] = int(self.video_latent_shard_size)
            expected["num_samples"] = int(self._source_num_samples)
        return expected

    def _validate_video_latent_manifest(self):
        manifest_path = self._video_latent_manifest_path
        if manifest_path is None:
            return
        if not os.path.exists(manifest_path):
            return
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        expected = self._expected_video_latent_manifest()
        for key, expected_value in expected.items():
            loaded_value = payload.get(key)
            if loaded_value != expected_value:
                raise ValueError(
                    f"Video latent cache manifest mismatch for `{key}`: "
                    f"expected={expected_value}, got={loaded_value}. "
                    f"Cache: {manifest_path}"
                )

    def _validate_video_latent_cache_population(self):
        if not (self.use_precomputed_video_latents and self.strict_video_latent_loading):
            return
        if self.video_latent_cache_dir is None:
            return
        ext = ".npy" if self.video_latent_storage_format == "sharded_npy" else self._video_latent_file_extension()
        has_any_latent = False
        for root, _, files in os.walk(self.video_latent_cache_dir):
            for filename in files:
                if filename.endswith(ext):
                    has_any_latent = True
                    break
            if has_any_latent:
                break
        if not has_any_latent:
            raise FileNotFoundError(
                "Precomputed video latent cache is empty. "
                f"Expected files with extension `{ext}` under `{self.video_latent_cache_dir}`. "
                "Please run `scripts/precompute_video_latents.py` first."
            )

    @staticmethod
    def write_video_latent_manifest(
        cache_dir: str,
        *,
        num_frames: int,
        action_video_freq_ratio: int,
        video_size: list[int],
        concat_multi_camera: str,
        extra: Optional[dict] = None,
    ):
        os.makedirs(cache_dir, exist_ok=True)
        payload = {
            "version": 1,
            "num_frames": int(num_frames),
            "action_video_freq_ratio": int(action_video_freq_ratio),
            "video_size": [int(video_size[0]), int(video_size[1])],
            "concat_multi_camera": str(concat_multi_camera),
        }
        if extra:
            payload.update(extra)
        with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _video_latent_file_extension(self) -> str:
        if self.video_latent_storage_format == "pt":
            return ".pt"
        if self.video_latent_storage_format == "sharded_npy":
            return ".npy"
        return ".npz"

    def _latent_cache_path(self, sample_idx: int) -> str:
        if self.video_latent_cache_dir is None:
            raise ValueError("`video_latent_cache_dir` is not set.")
        shard = f"{sample_idx // 10000:06d}"
        shard_dir = os.path.join(self.video_latent_cache_dir, shard)
        os.makedirs(shard_dir, exist_ok=True)
        return os.path.join(shard_dir, f"{sample_idx:09d}{self._video_latent_file_extension()}")

    def _sharded_latent_cache_path(self, sample_idx: int) -> str:
        if self.video_latent_cache_dir is None:
            raise ValueError("`video_latent_cache_dir` is not set.")
        shard = f"{sample_idx // self.video_latent_shard_size:06d}"
        return os.path.join(self.video_latent_cache_dir, "shards", f"{shard}.npy")

    def get_video_latent_cache_path(self, sample_idx: int) -> str:
        if self.video_latent_storage_format == "sharded_npy":
            return self._sharded_latent_cache_path(sample_idx)
        return self._latent_cache_path(sample_idx)

    def _load_sharded_video_latents(self, sample_idx: int) -> Optional[torch.Tensor]:
        shard_path = self._sharded_latent_cache_path(sample_idx)
        if not os.path.exists(shard_path):
            if self.strict_video_latent_loading:
                raise FileNotFoundError(
                    f"Missing precomputed video latent shard: {shard_path}. "
                    "Please run scripts/convert_video_latents_to_shards.py first."
                )
            return None
        shard = self._video_latent_shard_cache.get(shard_path)
        if shard is None:
            shard = np.load(shard_path, mmap_mode="r", allow_pickle=False)
            self._video_latent_shard_cache[shard_path] = shard
        row = int(sample_idx) % self.video_latent_shard_size
        if row >= int(shard.shape[0]):
            if self.strict_video_latent_loading:
                raise FileNotFoundError(
                    f"Missing sample_idx={sample_idx} in latent shard {shard_path}; "
                    f"row={row}, shard_rows={int(shard.shape[0])}."
                )
            return None
        latents = torch.from_numpy(np.asarray(shard[row]).copy())
        if latents.ndim != 4:
            raise ValueError(
                f"Cached `input_latents` must be 4D [C,T,H,W], got {tuple(latents.shape)} in {shard_path}"
            )
        return latents.contiguous()

    def _load_cached_video_latents(self, sample_idx: int) -> Optional[torch.Tensor]:
        if self.video_latent_storage_format == "sharded_npy":
            return self._load_sharded_video_latents(sample_idx)
        cache_path = self._latent_cache_path(sample_idx)
        if not os.path.exists(cache_path):
            if self.strict_video_latent_loading:
                raise FileNotFoundError(
                    f"Missing precomputed video latent: {cache_path}. "
                    "Please run scripts/precompute_video_latents.py first."
                )
            return None

        if self.video_latent_storage_format == "pt":
            payload = torch.load(cache_path, map_location="cpu")
            if isinstance(payload, torch.Tensor):
                latents = payload
            elif isinstance(payload, dict):
                latents = payload.get("input_latents")
            else:
                raise ValueError(f"Unsupported latent payload type in {cache_path}: {type(payload)}")
        else:
            with np.load(cache_path, allow_pickle=False) as payload:
                if "input_latents" not in payload:
                    raise ValueError(f"Missing `input_latents` array in latent cache: {cache_path}")
                latents = torch.from_numpy(payload["input_latents"])
        if not isinstance(latents, torch.Tensor):
            raise ValueError(f"Missing tensor `input_latents` in latent cache: {cache_path}")
        if latents.ndim != 4:
            raise ValueError(
                f"Cached `input_latents` must be 4D [C,T,H,W], got {tuple(latents.shape)} in {cache_path}"
            )
        return latents.contiguous()


    def _metadata_shard_dir(self, sample_idx: int) -> str:
        if self.metadata_cache_dir is None:
            raise ValueError("`metadata_cache_dir` is not set.")
        shard = f"{sample_idx // self.metadata_cache_shard_size:06d}"
        return os.path.join(self.metadata_cache_dir, "shards", shard)

    def _load_metadata_shard(self, sample_idx: int) -> dict:
        shard_dir = self._metadata_shard_dir(sample_idx)
        cached = self._metadata_shard_cache.get(shard_dir)
        if cached is not None:
            return cached
        required = [
            "action.npy",
            "proprio.npy",
            "image_is_pad.npy",
            "action_is_pad.npy",
            "proprio_is_pad.npy",
            "num_video_frames.npy",
            "dataset_index.npy",
            "instruction_id.npy",
        ]
        missing = [name for name in required if not os.path.exists(os.path.join(shard_dir, name))]
        if missing:
            raise FileNotFoundError(
                f"Missing metadata cache files in {shard_dir}: {missing}. "
                "Please run scripts/precompute_metadata_cache.py first."
            )
        shard = {
            "action": np.load(os.path.join(shard_dir, "action.npy"), mmap_mode="r", allow_pickle=False),
            "proprio": np.load(os.path.join(shard_dir, "proprio.npy"), mmap_mode="r", allow_pickle=False),
            "image_is_pad": np.load(os.path.join(shard_dir, "image_is_pad.npy"), mmap_mode="r", allow_pickle=False),
            "action_is_pad": np.load(os.path.join(shard_dir, "action_is_pad.npy"), mmap_mode="r", allow_pickle=False),
            "proprio_is_pad": np.load(os.path.join(shard_dir, "proprio_is_pad.npy"), mmap_mode="r", allow_pickle=False),
            "num_video_frames": np.load(os.path.join(shard_dir, "num_video_frames.npy"), mmap_mode="r", allow_pickle=False),
            "dataset_index": np.load(os.path.join(shard_dir, "dataset_index.npy"), mmap_mode="r", allow_pickle=False),
            "instruction_id": np.load(os.path.join(shard_dir, "instruction_id.npy"), mmap_mode="r", allow_pickle=False),
        }
        self._metadata_shard_cache[shard_dir] = shard
        return shard

    def _load_metadata_cache_instructions(self) -> list[str]:
        cache = getattr(self, "_metadata_instruction_cache", None)
        if cache is not None:
            return cache
        if self.metadata_cache_dir is None:
            raise ValueError("`metadata_cache_dir` is not set.")
        path = os.path.join(self.metadata_cache_dir, "instructions.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing metadata cache instruction table: {path}. "
                "Please run scripts/precompute_metadata_cache.py first."
            )
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if not isinstance(cache, list):
            raise ValueError(f"Metadata instruction table must be a list: {path}")
        self._metadata_instruction_cache = cache
        return cache

    def _load_cached_metadata(self, sample_idx: int) -> dict:
        shard = self._load_metadata_shard(sample_idx)
        row = int(sample_idx) % self.metadata_cache_shard_size
        if row >= int(shard["action"].shape[0]):
            raise FileNotFoundError(
                f"Missing sample_idx={sample_idx} in metadata shard {self._metadata_shard_dir(sample_idx)}; "
                f"row={row}, shard_rows={int(shard['action'].shape[0])}."
            )
        instructions = self._load_metadata_cache_instructions()
        instruction_id = int(shard["instruction_id"][row])
        if instruction_id < 0 or instruction_id >= len(instructions):
            raise ValueError(
                f"Invalid instruction_id={instruction_id} for sample_idx={sample_idx}; "
                f"instruction table size={len(instructions)}."
            )
        return {
            "action": torch.from_numpy(np.asarray(shard["action"][row]).copy()),
            "proprio": torch.from_numpy(np.asarray(shard["proprio"][row]).copy()),
            "image_is_pad": torch.from_numpy(np.asarray(shard["image_is_pad"][row]).copy()).bool(),
            "action_is_pad": torch.from_numpy(np.asarray(shard["action_is_pad"][row]).copy()).bool(),
            "proprio_is_pad": torch.from_numpy(np.asarray(shard["proprio_is_pad"][row]).copy()).bool(),
            "num_video_frames": int(shard["num_video_frames"][row]),
            "dataset_index": int(shard["dataset_index"][row]),
            "instruction": instructions[instruction_id],
        }


    def get_instruction_for_index(self, idx: int) -> str:
        sample_idx = self._source_index(idx)
        if self.override_instruction is not None:
            return str(self.override_instruction)
        if self.metadata_cache_dir is not None:
            cache_key = self._metadata_cache_key(idx, sample_idx)
            shard = self._load_metadata_shard(cache_key)
            row = int(cache_key) % self.metadata_cache_shard_size
            instructions = self._load_metadata_cache_instructions()
            instruction_id = int(shard["instruction_id"][row])
            if instruction_id < 0 or instruction_id >= len(instructions):
                raise ValueError(
                    f"Invalid instruction_id={instruction_id} for sample_idx={sample_idx}; "
                    f"instruction table size={len(instructions)}."
                )
            return str(instructions[instruction_id])
        sample = self.lerobot_dataset[sample_idx]
        return str(sample["instruction"])

    @staticmethod
    def _as_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(x) for x in value]

    def _normalize_sampling_rules(self, rules) -> list[dict]:
        if isinstance(rules, DictConfig):
            rules = OmegaConf.to_container(rules, resolve=True)
        if rules is None:
            return []
        if isinstance(rules, dict):
            rules = [rules]
        normalized = []
        for i, rule in enumerate(rules):
            if isinstance(rule, DictConfig):
                rule = OmegaConf.to_container(rule, resolve=True)
            if not isinstance(rule, dict):
                raise ValueError(f"Sampling rule #{i} must be a dict, got {type(rule)}.")
            weight = rule.get("weight", rule.get("factor", None))
            if weight is None:
                raise ValueError(f"Sampling rule #{i} is missing weight.")
            weight = float(weight)
            if weight < 0:
                raise ValueError(f"Sampling rule #{i} weight must be non-negative, got {weight}.")
            prompts = set(str(x) for x in self._as_list(rule.get("prompts")))
            prompt_file = rule.get("prompt_file", rule.get("prompt_subset_file", None))
            if prompt_file not in (None, ""):
                prompts.update(self._read_prompt_subset_file(os.path.expanduser(os.path.expandvars(str(prompt_file)))))
            substrings = self._as_list(
                rule.get(
                    "instruction_substrings",
                    rule.get("substrings", rule.get("contains", rule.get("match", None))),
                )
            )
            all_substrings = self._as_list(
                rule.get("all_instruction_substrings", rule.get("all_substrings", rule.get("contains_all", None)))
            )
            exclude_substrings = self._as_list(rule.get("exclude_instruction_substrings", rule.get("exclude_substrings", None)))
            if not prompts and not substrings and not all_substrings:
                raise ValueError(
                    f"Sampling rule #{i} must set one of prompts/prompt_file/instruction_substrings/all_instruction_substrings."
                )
            normalized.append(
                {
                    "name": str(rule.get("name", f"rule_{i}")),
                    "weight": weight,
                    "prompts": prompts,
                    "substrings": substrings,
                    "all_substrings": all_substrings,
                    "exclude_substrings": exclude_substrings,
                    "case_sensitive": bool(rule.get("case_sensitive", False)),
                }
            )
        return normalized

    @staticmethod
    def _instruction_matches_rule(instruction: str, rule: dict) -> bool:
        text = str(instruction)
        prompts = rule["prompts"]
        if text in prompts:
            matched = True
        else:
            if rule["case_sensitive"]:
                text_cmp = text
                substrings = rule["substrings"]
                all_substrings = rule["all_substrings"]
                excludes = rule["exclude_substrings"]
            else:
                text_cmp = text.lower()
                substrings = [s.lower() for s in rule["substrings"]]
                all_substrings = [s.lower() for s in rule["all_substrings"]]
                excludes = [s.lower() for s in rule["exclude_substrings"]]
            matched_any = any(sub in text_cmp for sub in substrings) if substrings else True
            matched_all = all(sub in text_cmp for sub in all_substrings) if all_substrings else True
            matched = matched_any and matched_all
        if not matched:
            return False
        if rule["exclude_substrings"]:
            if rule["case_sensitive"]:
                text_cmp = text
                excludes = rule["exclude_substrings"]
            else:
                text_cmp = text.lower()
                excludes = [s.lower() for s in rule["exclude_substrings"]]
            if any(sub in text_cmp for sub in excludes):
                return False
        return True

    def build_sampling_weights(self, rules, default_weight: float = 1.0, combine: str = "max") -> tuple[torch.Tensor, dict]:
        rules = self._normalize_sampling_rules(rules)
        default_weight = float(default_weight)
        if default_weight < 0:
            raise ValueError(f"default_weight must be non-negative, got {default_weight}.")
        combine = str(combine).strip().lower()
        if combine not in {"max", "multiply", "replace"}:
            raise ValueError(f"Unsupported sampling combine mode: {combine}. Expected max/multiply/replace.")
        if not rules:
            weights = torch.full((len(self),), default_weight, dtype=torch.double)
            return weights, {"enabled": False, "num_samples": len(self), "rules": []}

        if self.metadata_cache_dir is not None:
            instructions = [str(x) for x in self._load_metadata_cache_instructions()]
            matched_ids_by_rule = []
            for rule in rules:
                matched_ids = {
                    i for i, instruction in enumerate(instructions)
                    if self._instruction_matches_rule(instruction, rule)
                }
                matched_ids_by_rule.append(matched_ids)
            weights_source = np.full(self._source_num_samples, default_weight, dtype=np.float64)
            matched_counts_source = [0 for _ in rules]
            shards_root = os.path.join(self.metadata_cache_dir, "shards")
            if not os.path.isdir(shards_root):
                raise FileNotFoundError(f"Missing metadata cache shards directory: {shards_root}")
            for shard_name in sorted(os.listdir(shards_root)):
                shard_dir = os.path.join(shards_root, shard_name)
                instruction_path = os.path.join(shard_dir, "instruction_id.npy")
                if not os.path.isfile(instruction_path):
                    continue
                instruction_ids = np.load(instruction_path, mmap_mode="r", allow_pickle=False)
                shard_start = int(shard_name) * self.metadata_cache_shard_size
                shard_end = min(shard_start + len(instruction_ids), self._source_num_samples)
                shard_weights = weights_source[shard_start:shard_end]
                instruction_ids = np.asarray(instruction_ids[: len(shard_weights)])
                for rule_idx, (rule, matched_ids) in enumerate(zip(rules, matched_ids_by_rule)):
                    if not matched_ids:
                        continue
                    matched = np.isin(instruction_ids, list(matched_ids))
                    count = int(matched.sum())
                    if count <= 0:
                        continue
                    matched_counts_source[rule_idx] += count
                    if combine == "multiply":
                        shard_weights[matched] *= float(rule["weight"])
                    elif combine == "replace":
                        shard_weights[matched] = float(rule["weight"])
                    else:
                        shard_weights[matched] = np.maximum(shard_weights[matched], float(rule["weight"]))
            if self._sample_indices is not None:
                selected_cache_indices = (
                    self._metadata_cache_indices
                    if self._metadata_cache_indices is not None
                    else np.arange(len(self._sample_indices), dtype=np.int64)
                )
                weights = weights_source[selected_cache_indices]
                matched_counts = []
                for rule, matched_ids in zip(rules, matched_ids_by_rule):
                    count = 0
                    for cache_key in selected_cache_indices:
                        shard = self._load_metadata_shard(int(cache_key))
                        row = int(cache_key) % self.metadata_cache_shard_size
                        if int(shard["instruction_id"][row]) in matched_ids:
                            count += 1
                    matched_counts.append(count)
            else:
                weights = weights_source
                matched_counts = matched_counts_source
        else:
            weights = np.full(len(self), default_weight, dtype=np.float64)
            matched_counts = [0 for _ in rules]
            for idx in range(len(self)):
                instruction = self.get_instruction_for_index(idx)
                for rule_idx, rule in enumerate(rules):
                    if not self._instruction_matches_rule(instruction, rule):
                        continue
                    matched_counts[rule_idx] += 1
                    if combine == "multiply":
                        weights[idx] *= float(rule["weight"])
                    elif combine == "replace":
                        weights[idx] = float(rule["weight"])
                    else:
                        weights[idx] = max(float(weights[idx]), float(rule["weight"]))

        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        summary = {
            "enabled": True,
            "num_samples": int(len(self)),
            "default_weight": default_weight,
            "combine": combine,
            "min_weight": float(weights_tensor.min().item()),
            "max_weight": float(weights_tensor.max().item()),
            "mean_weight": float(weights_tensor.mean().item()),
            "rules": [
                {
                    "name": rule["name"],
                    "weight": float(rule["weight"]),
                    "matched_samples": int(count),
                    "matched_fraction": float(count / max(len(self), 1)),
                }
                for rule, count in zip(rules, matched_counts)
            ],
        }
        return weights_tensor, summary

    def _extract_video_and_pad(self, sample):
        image_is_pad = sample["image_is_pad"]
        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :]  # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :]  # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]
        video = video.permute(1, 0, 2, 3)  # [C, T_video, H, W], range [-1, 1]
        return video, image_is_pad

    def _get(self, idx):
        sample_idx = self._source_index(idx)
        sample = None
        image_is_pad = None
        metadata = None
        if self.use_precomputed_metadata_cache:
            # Metadata caches are normally indexed by dataset position when a subset
            # is active. Clean-only filtering can reuse a larger cache via an index map.
            cache_key = self._metadata_cache_key(idx, sample_idx)
            metadata = self._load_cached_metadata(cache_key)
        else:
            for attempt in range(self.max_padding_retry + 1):
                sample = self.lerobot_dataset[sample_idx]

                if not self.skip_padding_as_possible:
                    break

                action_is_pad = sample["action_is_pad"]
                image_is_pad = sample["image_is_pad"]
                proprio_is_pad = sample["proprio_is_pad"]
                has_pad = False
                if bool(action_is_pad.any().item()):
                    has_pad = True
                if bool(image_is_pad.any().item()):
                    has_pad = True
                if bool(proprio_is_pad.any().item()):
                    has_pad = True

                if not has_pad or attempt >= self.max_padding_retry:
                    break

                sample_idx = self._random_source_index()

            image_is_pad = sample["image_is_pad"][self.video_sample_indices]

        video = None
        input_latents = None
        if self.use_precomputed_video_latents:
            input_latents = self._load_cached_video_latents(sample_idx)
            if input_latents is None:
                if sample is None:
                    sample = self.lerobot_dataset[sample_idx]
                if "pixel_values" not in sample:
                    raise FileNotFoundError(
                        "Video latent cache miss and `pixel_values` is unavailable for fallback. "
                        "Set `strict_video_latent_loading=true` after precompute, or allow images during fallback."
                    )
                video, image_is_pad = self._extract_video_and_pad(sample)
        else:
            if sample is None:
                sample = self.lerobot_dataset[sample_idx]
            video, image_is_pad = self._extract_video_and_pad(sample)

        if metadata is None:
            metadata = self._build_metadata_from_lerobot_sample(sample, image_is_pad, video)
        action = metadata["action"]
        action_is_pad = metadata["action_is_pad"]
        if self.action_chunk_size is not None:
            action = action[:self.action_chunk_size]
            action_is_pad = action_is_pad[:self.action_chunk_size]
        proprio = metadata["proprio"]
        num_video_frames = metadata["num_video_frames"]
        instruction = DEFAULT_PROMPT.format(task=metadata["instruction"])

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        action_loss_mask = 1.0
        if self._robotwin_action_loss_source_mask is not None:
            action_loss_mask = float(bool(self._robotwin_action_loss_source_mask[int(sample_idx)]))

        data = {
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": metadata["image_is_pad"],
            "action_is_pad": action_is_pad,
            "proprio_is_pad": metadata["proprio_is_pad"],
            "num_video_frames": torch.tensor(num_video_frames, dtype=torch.long),
            "loss_action_mask": torch.tensor(action_loss_mask, dtype=torch.float32),
        }
        if video is not None:
            data["video"] = video
        if input_latents is not None:
            data["input_latents"] = input_latents
        dataset_index = int(metadata["dataset_index"])

        data["dataset_index"] = torch.tensor(dataset_index, dtype=torch.long)
        if 0 <= dataset_index < len(self.dataset_dirs):
            data["dataset_repo_id"] = self.dataset_dirs[dataset_index]
        else:
            data["dataset_repo_id"] = ""
        return data

    def _build_metadata_from_lerobot_sample(self, sample, image_is_pad=None, video=None):
        if self.use_precomputed_metadata_cache:
            return self._load_cached_metadata(int(sample["idx"]) if isinstance(sample, dict) and "idx" in sample else int(sample))

        if image_is_pad is None:
            image_is_pad = sample["image_is_pad"][self.video_sample_indices]

        # Proxy (from lerobot):
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"]  # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :]  # [T-1, state_dim], to align with action
        num_video_frames = len(self.video_sample_indices)
        if video is not None:
            num_video_frames = int(video.shape[1])
            if video.shape[1] <= 1:
                raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if self.action_chunk_size is None and action.shape[0] % (num_video_frames - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {num_video_frames - 1}"
            )

        task = sample["instruction"]
        if self.override_instruction is not None:
            task = self.override_instruction

        dataset_index_raw = sample.get("dataset_index")
        if dataset_index_raw is None:
            dataset_index = -1
        elif isinstance(dataset_index_raw, torch.Tensor):
            dataset_index = int(dataset_index_raw.item())
        else:
            dataset_index = int(dataset_index_raw)

        return {
            "action": action,
            "proprio": proprio,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
            "num_video_frames": int(num_video_frames),
            "instruction": task,
            "dataset_index": int(dataset_index),
        }

    def get_video_for_latent_precompute(self, idx):
        sample_idx = self._source_index(idx)
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))

        if "pixel_values" not in sample:
            raise KeyError(
                "Missing `pixel_values` in sample while precomputing latents. "
                "Please ensure dataset was initialized with image loading enabled."
            )
        video, _ = self._extract_video_and_pad(sample)
        return sample_idx, video

    def get_metadata_for_cache_precompute(self, idx):
        sample_idx = self._source_index(idx)
        # Disable image loading — metadata precompute needs action/proprio/instruction only.
        self.lerobot_dataset._set_return_images(False)
        try:
            sample = self.lerobot_dataset[sample_idx]
        finally:
            self.lerobot_dataset._set_return_images(
                (not self.use_precomputed_video_latents) or (not self.strict_video_latent_loading)
            )
        image_is_pad = sample["image_is_pad"][self.video_sample_indices]
        use_cache = self.use_precomputed_metadata_cache
        self.use_precomputed_metadata_cache = False
        try:
            metadata = self._build_metadata_from_lerobot_sample(sample, image_is_pad=image_is_pad, video=None)
        finally:
            self.use_precomputed_metadata_cache = use_cache
        # Return idx (dataset position) as the cache key so the precompute script
        # can write to the correct shard row. When no subset is active, idx == sample_idx.
        metadata["sample_idx"] = idx
        return metadata

    def _load_text_context_from_disk(self, cache_path: str):
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )
        return context.contiguous(), context_mask.contiguous()

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        if self.cache_text_embeddings_in_memory:
            cached = self._text_embedding_memory_cache.get(cache_path)
            if cached is None:
                cached = self._load_text_context_from_disk(cache_path)
                if len(self._text_embedding_memory_cache) >= self._text_embedding_memory_cache_maxsize:
                    self._text_embedding_memory_cache.popitem(last=False)
                self._text_embedding_memory_cache[cache_path] = cached
            else:
                self._text_embedding_memory_cache.move_to_end(cache_path)
            context, context_mask = cached
        else:
            context, context_mask = self._load_text_context_from_disk(cache_path)

        return context.clone(), context_mask.clone()

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
