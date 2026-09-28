import json
import inspect
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from simplewam.utils.video_io import save_mp4
from simplewam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from simplewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from simplewam.utils.pytorch_utils import set_global_seed
from simplewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark
from action_ensembler import ActionEnsembler

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def maybe_compile_inference_paths(model: torch.nn.Module, cfg: DictConfig) -> None:
    compile_action = bool(cfg.EVALUATION.get("torch_compile_action", False))
    compile_video = bool(cfg.EVALUATION.get("torch_compile_video", False))
    if not compile_action and not compile_video:
        return
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile inference options require torch.compile support.")

    compile_mode = cfg.EVALUATION.get("torch_compile_mode", "reduce-overhead")
    compile_backend = cfg.EVALUATION.get("torch_compile_backend", "inductor")
    compile_kwargs = {
        "backend": None if compile_backend in (None, "null", "") else str(compile_backend),
        "mode": None if compile_mode in (None, "null", "") else str(compile_mode),
        "fullgraph": bool(cfg.EVALUATION.get("torch_compile_fullgraph", False)),
        "dynamic": bool(cfg.EVALUATION.get("torch_compile_dynamic", False)),
    }
    if compile_video:
        try:
            import torch._inductor.config as inductor_config

            inductor_config.triton.cudagraph_trees = bool(
                cfg.EVALUATION.get("torch_compile_cudagraph_trees", False)
            )
        except Exception as exc:
            logging.warning("Failed to set torch._inductor cudagraph_trees option: %s", exc)

    compiled_paths: list[str] = []
    if compile_action:
        if not hasattr(model, "mot"):
            raise ValueError("EVALUATION.torch_compile_action=true requires model.mot.")
        infer_action = getattr(model, "infer_action", None)
        supports_official_compile = (
            infer_action is not None
            and "compile_action_infer" in inspect.signature(infer_action).parameters
        )
        if supports_official_compile:
            compiled_paths.append("infer_action.tensor_cache_cuda_graph")
        else:
            setattr(model.mot, "_fastwam_torch_compile_action", True)
            for name in (
                "forward_action_with_video_cache",
                "forward",
            ):
                fn = getattr(model.mot, name, None)
                if fn is None:
                    continue
                setattr(model.mot, name, torch.compile(fn, **compile_kwargs))
                compiled_paths.append(f"mot.{name}")

    if compile_video:
        if not hasattr(model, "video_expert"):
            raise ValueError("EVALUATION.torch_compile_video=true requires model.video_expert.")
        if hasattr(model, "mot"):
            setattr(model.mot, "_fastwam_torch_compile_video", True)
        for name in ("pre_dit", "post_dit"):
            fn = getattr(model.video_expert, name, None)
            if fn is None:
                continue
            setattr(model.video_expert, name, torch.compile(fn, **compile_kwargs))
            compiled_paths.append(f"video_expert.{name}")
        if hasattr(model, "mot"):
            for name in ("prefill_video_cache",):
                fn = getattr(model.mot, name, None)
                if fn is None:
                    continue
                setattr(model.mot, name, torch.compile(fn, **compile_kwargs))
                compiled_paths.append(f"mot.{name}")

    logging.info(
        "Enabled torch.compile for inference paths %s "
        "(backend=%s mode=%s fullgraph=%s dynamic=%s)",
        compiled_paths,
        compile_kwargs["backend"],
        compile_kwargs["mode"],
        compile_kwargs["fullgraph"],
        compile_kwargs["dynamic"],
    )


def maybe_compile_action_inference(model: torch.nn.Module, cfg: DictConfig) -> None:
    maybe_compile_inference_paths(model, cfg)


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs, rgb


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if bool(cfg.EVALUATION.get("freeze_future_video_noise", False)) and bool(
        cfg.EVALUATION.get("visualize_future_video", False)
    ):
        raise ValueError(
            "EVALUATION.freeze_future_video_noise=true is an action-only ablation and "
            "cannot be combined with EVALUATION.visualize_future_video=true."
        )
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _infer_video_token_layout(
    *,
    video_key_len: int,
    image_h: int,
    image_w: int,
    preferred_latent_frames: Optional[int] = None,
) -> tuple[int, int, int]:
    if video_key_len <= 0:
        raise ValueError(f"`video_key_len` must be positive, got {video_key_len}")

    target_aspect = float(image_w) / float(max(image_h, 1))
    best_score = float("inf")
    best_layout: Optional[tuple[int, int, int]] = None
    for latent_frames in range(1, video_key_len + 1):
        if video_key_len % latent_frames != 0:
            continue
        spatial_tokens = video_key_len // latent_frames
        for token_h in range(1, int(math.sqrt(spatial_tokens)) + 1):
            if spatial_tokens % token_h != 0:
                continue
            token_w = spatial_tokens // token_h
            aspect = float(token_w) / float(token_h)
            aspect_penalty = abs(aspect - target_aspect) * 1000.0
            frame_penalty = 0.0
            if preferred_latent_frames is not None:
                frame_penalty = abs(int(latent_frames) - int(preferred_latent_frames)) * 10.0
            score = aspect_penalty + frame_penalty
            if score < best_score:
                best_score = score
                best_layout = (latent_frames, token_h, token_w)
    if best_layout is None:
        raise ValueError(f"Failed to infer token layout for video_key_len={video_key_len}")
    return best_layout


def _heat_to_rgb(heat01: np.ndarray) -> np.ndarray:
    x = np.clip(heat01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _build_attention_overlay(
    *,
    attention: torch.Tensor,
    phase: str,
    observation_image: np.ndarray,
    alpha: float = 0.45,
) -> tuple[np.ndarray, dict[str, int]]:
    if attention.ndim != 3:
        raise ValueError(f"`attention` must be [B,Q,K], got shape {tuple(attention.shape)}")
    _, q_len, k_len = [int(v) for v in attention.shape]
    if k_len < q_len:
        raise ValueError(f"Expected key_len >= query_len, got Q={q_len}, K={k_len}")

    key_mean = attention.detach().to(device="cpu", dtype=torch.float32).mean(dim=0).mean(dim=0).numpy()
    if str(phase) == "action_denoise" and k_len > q_len:
        video_key = key_mean[: k_len - q_len]
    else:
        video_key = key_mean

    image_h, image_w = int(observation_image.shape[0]), int(observation_image.shape[1])
    preferred_frames = 1 if str(phase) in {"action_denoise", "video_prefill"} else None
    latent_frames, token_h, token_w = _infer_video_token_layout(
        video_key_len=int(video_key.shape[0]),
        image_h=image_h,
        image_w=image_w,
        preferred_latent_frames=preferred_frames,
    )

    video_tokens_fhw = video_key.reshape(latent_frames, token_h, token_w)
    token_map = video_tokens_fhw.mean(axis=0)
    token_min = float(token_map.min())
    token_max = float(token_map.max())
    denom = max(token_max - token_min, 1e-8)
    token_map_norm = (token_map - token_min) / denom

    heat_img = Image.fromarray((token_map_norm * 255.0).astype(np.uint8))
    heat_img = heat_img.resize((image_w, image_h), resample=Image.BILINEAR)
    heat = np.asarray(heat_img, dtype=np.float32) / 255.0
    heat_rgb = _heat_to_rgb(heat)

    obs_rgb = observation_image.astype(np.float32)
    overlay = (1.0 - alpha) * obs_rgb + alpha * heat_rgb.astype(np.float32)
    overlay = np.clip(overlay, 0.0, 255.0).astype(np.uint8)
    meta = {
        "q_len": int(q_len),
        "k_len": int(k_len),
        "video_key_len": int(video_key.shape[0]),
        "latent_frames": int(latent_frames),
        "token_h": int(token_h),
        "token_w": int(token_w),
    }
    return overlay, meta


def _save_attention_maps(
    *,
    attention_maps: list[dict[str, Any]],
    output_dir: Path,
    task_id: int,
    episode_idx: int,
    replan_idx: int,
    observation_image: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = output_dir / "observations"
    overlay_dir = output_dir / "overlays"
    obs_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    obs_path = obs_dir / f"task{task_id}_trial{episode_idx}_replan{replan_idx:03d}_obs.png"
    Image.fromarray(observation_image).save(obs_path)
    overlay_lookup: dict[tuple[int, int], np.ndarray] = {}

    for record_idx, record in enumerate(attention_maps):
        attention = record.get("attention")
        if not isinstance(attention, torch.Tensor):
            continue
        layer_idx = int(record.get("layer_idx", -1))
        step_idx = int(record.get("step_idx", -1))
        phase = str(record.get("phase", "unknown"))
        file_name = (
            f"task{task_id}_trial{episode_idx}_replan{replan_idx:03d}_"
            f"{phase}_layer{layer_idx:02d}_step{step_idx:03d}_{record_idx:03d}.pt"
        )
        save_path = output_dir / file_name
        overlay_meta: Optional[dict[str, int]] = None
        overlay_file_name = (
            f"task{task_id}_trial{episode_idx}_replan{replan_idx:03d}_"
            f"{phase}_layer{layer_idx:02d}_step{step_idx:03d}_{record_idx:03d}_overlay.png"
        )
        overlay_path = overlay_dir / overlay_file_name
        try:
            overlay_img, overlay_meta = _build_attention_overlay(
                attention=attention,
                phase=phase,
                observation_image=observation_image,
            )
            Image.fromarray(overlay_img).save(overlay_path)
            if phase == "action_denoise" and step_idx >= 0 and layer_idx >= 0:
                overlay_lookup[(int(step_idx), int(layer_idx))] = overlay_img
        except Exception as e:
            logging.warning(
                "Skip attention overlay due to mapping failure: task=%s trial=%s replan=%s phase=%s layer=%s step=%s err=%s",
                task_id,
                episode_idx,
                replan_idx,
                phase,
                layer_idx,
                step_idx,
                e,
            )
        torch.save(
            {
                "task_id": int(task_id),
                "episode_idx": int(episode_idx),
                "replan_idx": int(replan_idx),
                "phase": phase,
                "layer_idx": layer_idx,
                "step_idx": step_idx,
                "attention": attention,
                "observation_image_path": str(obs_path),
                "overlay_image_path": str(overlay_path) if overlay_meta is not None else None,
                "overlay_meta": overlay_meta,
            },
            save_path,
        )
    return overlay_lookup


def _compose_attention_grid_frame(
    *,
    overlay_lookup: dict[tuple[int, int], np.ndarray],
    step_indices: list[int],
    layer_indices: list[int],
    tile_h: int,
    tile_w: int,
) -> np.ndarray:
    if len(step_indices) == 0 or len(layer_indices) == 0:
        raise ValueError("`step_indices` and `layer_indices` must be non-empty.")
    # Swap layout for video grid visualization:
    # row -> layer, col -> denoise step
    layer_to_row = {int(layer): idx for idx, layer in enumerate(layer_indices)}
    step_to_col = {int(step): idx for idx, step in enumerate(step_indices)}
    canvas_h = len(layer_indices) * int(tile_h)
    canvas_w = len(step_indices) * int(tile_w)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    for (step_idx, layer_idx), overlay_img in overlay_lookup.items():
        if int(step_idx) not in step_to_col:
            continue
        if int(layer_idx) not in layer_to_row:
            continue
        tile = Image.fromarray(overlay_img).resize((tile_w, tile_h), resample=Image.BILINEAR)
        y0 = int(layer_to_row[int(layer_idx)]) * int(tile_h)
        x0 = int(step_to_col[int(step_idx)]) * int(tile_w)
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = np.asarray(tile, dtype=np.uint8)
    return canvas


def _resolve_index_list(
    value: Any,
    *,
    default: list[int],
    name: str,
) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, ListConfig):
        items = [int(x) for x in list(value)]
    elif isinstance(value, (list, tuple)):
        items = [int(x) for x in value]
    elif isinstance(value, str):
        text = value.strip()
        if text == "":
            return list(default)
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        if text == "":
            return list(default)
        items = [int(x.strip()) for x in text.split(",") if x.strip() != ""]
    else:
        items = [int(value)]
    if len(items) == 0:
        raise ValueError(f"`{name}` resolved to an empty list.")
    deduped: list[int] = []
    for x in items:
        if x < 0:
            raise ValueError(f"`{name}` contains negative index: {x}")
        if x not in deduped:
            deduped.append(x)
    return deduped


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    capture_attention: bool = False,
    profile_action_chunk_time: bool = False,
    prompt_cache: Optional[dict[str, torch.Tensor]] = None,
) -> tuple[np.ndarray, dict, np.ndarray, Optional[list[Image.Image]], list[dict[str, Any]], Optional[float]]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    image, proprio, imgs, input_rgb = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    cached_context = None if prompt_cache is None else prompt_cache.get("context")
    cached_context_mask = None if prompt_cache is None else prompt_cache.get("context_mask")
    use_cached_prompt_context = cached_context is not None and cached_context_mask is not None

    infer_kwargs = {
        "prompt": (None if use_cached_prompt_context else prompt),
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    if use_cached_prompt_context:
        infer_kwargs["context"] = cached_context
        infer_kwargs["context_mask"] = cached_context_mask
    if "text_encoder_offload" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["text_encoder_offload"] = bool(cfg.EVALUATION.get("offload_text_encoder", False))
    freeze_future_video_noise = bool(cfg.EVALUATION.get("freeze_future_video_noise", False))
    if freeze_future_video_noise:
        if "freeze_future_video_noise" not in inspect.signature(model.infer_action).parameters:
            raise ValueError(
                "EVALUATION.freeze_future_video_noise=true requires a joint model whose "
                "infer_action() supports the pure-noise future-video ablation."
            )
        infer_kwargs["freeze_future_video_noise"] = True
    if (
        not use_cached_prompt_context
        and "return_context" in inspect.signature(model.infer_action).parameters
    ):
        infer_kwargs["return_context"] = True
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    predicted_future_frames = None
    if visualize_future_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    compile_action_infer = bool(cfg.EVALUATION.get("compile_action_infer", False)) or bool(
        cfg.EVALUATION.get("torch_compile_action", False)
    )
    infer_method = model.infer_joint if visualize_future_video else model.infer_action
    if "compile_action_infer" in inspect.signature(infer_method).parameters:
        infer_kwargs["compile_action_infer"] = compile_action_infer
    elif bool(cfg.EVALUATION.get("compile_action_infer", False)):
        raise ValueError(
            f"{type(model).__name__}.{infer_method.__name__} does not support "
            "EVALUATION.compile_action_infer=true."
        )

    timing_scope = str(cfg.EVALUATION.get("action_chunk_timing_scope", "end_to_end")).strip().lower()
    if timing_scope not in {"end_to_end", "main_model"}:
        raise ValueError(
            "EVALUATION.action_chunk_timing_scope must be 'end_to_end' or 'main_model', "
            f"got {timing_scope!r}."
        )
    if profile_action_chunk_time and timing_scope == "main_model":
        if "profile_main_model_infer_time" not in inspect.signature(infer_method).parameters:
            raise ValueError(
                f"{type(model).__name__}.{infer_method.__name__} does not support "
                "EVALUATION.action_chunk_timing_scope=main_model."
            )
        infer_kwargs["profile_main_model_infer_time"] = True

    captured_attention_maps: list[dict[str, Any]] = []
    action_chunk_infer_time_sec: Optional[float] = None
    if capture_attention:
        if not hasattr(model, "mot"):
            raise ValueError("Attention capture requires model.mot, but current model does not expose it.")
        if not hasattr(model.mot, "enable_attention_capture"):
            raise ValueError("Attention capture requires MoT `enable_attention_capture` API.")
        model.mot.enable_attention_capture()

    with torch.no_grad():
        try:
            if profile_action_chunk_time and timing_scope == "end_to_end":
                infer_start = time.perf_counter()
            if visualize_future_video:
                pred = model.infer_joint(**infer_kwargs)
                predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
            else:
                pred = model.infer_action(**infer_kwargs)
            if profile_action_chunk_time and timing_scope == "end_to_end":
                action_chunk_infer_time_sec = float(time.perf_counter() - infer_start)
            elif profile_action_chunk_time:
                action_chunk_infer_time_sec = float(pred["main_model_infer_time_sec"])
        finally:
            if capture_attention:
                captured_attention_maps = model.mot.pop_captured_attention_maps()
                model.mot.disable_attention_capture()
    if (
        prompt_cache is not None
        and not use_cached_prompt_context
        and "context" in pred
        and "context_mask" in pred
    ):
        if isinstance(pred["context"], torch.Tensor) and isinstance(pred["context_mask"], torch.Tensor):
            prompt_cache["context"] = pred["context"]
            prompt_cache["context_mask"] = pred["context_mask"]

    max_denoise_steps = int(cfg.EVALUATION.get("attention_max_denoise_steps", 0))
    if max_denoise_steps > 0 and len(captured_attention_maps) > 0:
        captured_attention_maps = [
            item
            for item in captured_attention_maps
            if int(item.get("step_idx", -1)) < 0 or int(item.get("step_idx", -1)) < max_denoise_steps
        ]
    action = pred["action"]  # [T, D]

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return (
        action,
        imgs,
        input_rgb,
        predicted_future_frames,
        captured_attention_maps,
        action_chunk_infer_time_sec,
    )


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def ensure_num_initial_states(initial_states, num_trials: int):
    """Repeat LIBERO init states if fewer are available than requested trials."""
    if len(initial_states) >= num_trials:
        return initial_states

    missing = num_trials - len(initial_states)
    if isinstance(initial_states, np.ndarray):
        repeats = math.ceil(num_trials / len(initial_states))
        return np.concatenate([initial_states] * repeats, axis=0)[:num_trials]

    initial_states = list(initial_states)
    while missing > 0:
        chunk = initial_states[:missing]
        initial_states.extend(chunk)
        missing = num_trials - len(initial_states)
    return initial_states


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    attention_dir: Optional[Path] = None,
    profile_action_chunk_time: bool = False,
    prompt_cache: Optional[dict[str, torch.Tensor]] = None,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float], Optional[float]]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1
    attention_grid_frames: list[Image.Image] = []
    action_chunk_infer_times: list[float] = []

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            next_replan_idx = current_replan_idx + 1
            save_attention_maps = bool(cfg.EVALUATION.get("save_attention_maps", False))
            attention_max_replans = int(cfg.EVALUATION.get("attention_max_replans", 1))
            should_capture_attention = save_attention_maps and next_replan_idx < attention_max_replans

            (
                action_chunk,
                imgs,
                input_rgb,
                predicted_future_frames,
                captured_attention_maps,
                action_chunk_infer_time_sec,
            ) = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                capture_attention=should_capture_attention,
                profile_action_chunk_time=profile_action_chunk_time,
                prompt_cache=prompt_cache,
            )
            if action_chunk_infer_time_sec is not None:
                action_chunk_infer_times.append(float(action_chunk_infer_time_sec))
            current_replan_idx = next_replan_idx
            if should_capture_attention and attention_dir is not None and len(captured_attention_maps) > 0:
                overlay_lookup = _save_attention_maps(
                    attention_maps=captured_attention_maps,
                    output_dir=attention_dir,
                    task_id=int(cfg.EVALUATION.task_id),
                    episode_idx=int(episode_idx),
                    replan_idx=int(current_replan_idx),
                    observation_image=input_rgb,
                )
                default_step_indices = [0, 5, 9]
                default_layer_indices = [0, 15, 29]
                step_indices = _resolve_index_list(
                    cfg.EVALUATION.get("attention_grid_step_indices", None),
                    default=default_step_indices,
                    name="EVALUATION.attention_grid_step_indices",
                )
                layer_indices = _resolve_index_list(
                    cfg.EVALUATION.get("attention_grid_layer_indices", None),
                    default=default_layer_indices,
                    name="EVALUATION.attention_grid_layer_indices",
                )
                tile_h = int(cfg.EVALUATION.get("attention_grid_tile_h", 56))
                tile_w = int(cfg.EVALUATION.get("attention_grid_tile_w", 112))
                if len(step_indices) > 0 and len(layer_indices) > 0:
                    grid_np = _compose_attention_grid_frame(
                        overlay_lookup=overlay_lookup,
                        step_indices=step_indices,
                        layer_indices=layer_indices,
                        tile_h=tile_h,
                        tile_w=tile_w,
                    )
                    attention_grid_frames.append(Image.fromarray(grid_np))
            if predicted_future_frames is not None:
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        obs, _, done, _ = env.step(pending_actions.pop(0))
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    episode_avg_action_chunk_infer_time = (
        float(np.mean(action_chunk_infer_times)) if len(action_chunk_infer_times) > 0 else None
    )
    if attention_dir is not None and len(attention_grid_frames) > 0:
        episode_video_dir = attention_dir / "episode_grid_videos"
        episode_video_dir.mkdir(parents=True, exist_ok=True)
        video_path = episode_video_dir / (
            f"task{int(cfg.EVALUATION.task_id)}_trial{int(episode_idx)}_attention_grid.mp4"
        )
        save_mp4(attention_grid_frames, str(video_path), fps=2)
        logging.info("Saved episode attention-grid video: %s", video_path)
    return (
        bool(done),
        replay_images,
        predicted_future_video_clips,
        episode_mean_psnr,
        episode_avg_action_chunk_infer_time,
    )


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    attention_dir: Optional[Path] = None,
    profile_action_chunk_time: bool = False,
) -> dict:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None
    if profile_action_chunk_time:
        results["episode_action_chunk_infer_time_sec"] = []
        results["action_chunk_infer_time_sec_mean"] = None
        results["action_chunk_timing_scope"] = str(
            cfg.EVALUATION.get("action_chunk_timing_scope", "end_to_end")
        )

    prompt_cache: dict[str, torch.Tensor] = {}
    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        (
            success,
            replay_images,
            predicted_future_video_clips,
            episode_mean_psnr,
            episode_avg_action_chunk_infer_time,
        ) = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            attention_dir=attention_dir,
            profile_action_chunk_time=profile_action_chunk_time,
            prompt_cache=prompt_cache,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)
        if profile_action_chunk_time:
            results["episode_action_chunk_infer_time_sec"].append(episode_avg_action_chunk_infer_time)

        save_rollout_video(
            video_dir,
            replay_images,
            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success,
            task_description=task_description,
        )
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    if profile_action_chunk_time:
        valid_chunk_times = [x for x in results["episode_action_chunk_infer_time_sec"] if x is not None]
        if len(valid_chunk_times) > 0:
            results["action_chunk_infer_time_sec_mean"] = float(np.mean(valid_chunk_times))
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
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
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    # Build + load on CPU first to reduce peak GPU memory during checkpoint deserialization.
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
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)
    attention_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "attention_maps"
    if bool(cfg.EVALUATION.get("save_attention_maps", False)):
        attention_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)

    initial_states = ensure_num_initial_states(initial_states, int(cfg.EVALUATION.num_trials))

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
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

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    if results.get("action_chunk_infer_time_sec_mean") is not None:
        print(
            "Task "
            f"{cfg.EVALUATION.task_id} action-chunk inference mean time: "
            f"{results['action_chunk_infer_time_sec_mean']:.4f}s"
        )
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
