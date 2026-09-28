import time
from typing import Any, Optional

import torch

from simplewam.utils.logging_config import get_logger

from .fastwam import FastWAM

logger = get_logger(__name__)


class FastWAMJoint(FastWAM):
    """FastWAM variant where action normally attends to all video latent tokens."""

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        video_dit_config = kwargs.get("video_dit_config", None)
        if not isinstance(video_dit_config, dict):
            raise ValueError(
                "`video_dit_config` must be provided as dict for FastWAMJoint."
            )
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError(
                "FastWAMJoint requires `video_dit_config['action_conditioned']=false`."
            )
        return super().from_wan22_pretrained(**kwargs)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        video_end = video_seq_len

        # video -> video
        mask[:video_end, :video_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_end:, video_end:] = True
        mask[video_end:, :video_end] = True
        return mask

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        if test_action_with_infer_action:
            logger.warning(
                "`FastWAMJoint.infer_joint` ignores `test_action_with_infer_action=True` "
                "and always runs with `test_action_with_infer_action=False`."
            )
        return super().infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            test_action_with_infer_action=False,
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        text_encoder_offload: bool = False,
        return_context: bool = False,
        freeze_future_video_noise: bool = False,
        compile_action_infer: bool = False,
        profile_main_model_infer_time: bool = False,
    ) -> dict[str, Any]:
        """Predict actions, optionally keeping all future video latents at pure noise.

        When ``freeze_future_video_noise`` is enabled, the clean encoded current
        frame is still supplied in latent slot zero. All future latent slots stay
        equal to their initial Gaussian noise, and the video timestep stays at
        the first (maximum-noise) inference timestep throughout action denoising.
        The video branch is prefetched once into a per-layer K/V cache; subsequent
        denoising steps run only the action branch. The video output head and video
        scheduler step are both skipped.
        """
        self.eval()

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt, text_encoder_offload=text_encoder_offload)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        base_context = context
        base_context_mask = context_mask
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        capture_attention = bool(getattr(self.mot, "_attention_capture_enabled", False))
        tensor_core_compatible = not capture_attention
        if compile_action_infer and not tensor_core_compatible:
            raise ValueError(
                "`compile_action_infer=True` only supports the standard Joint inference paths; "
                "incompatible feature: attention_capture."
            )

        main_model_infer_start = None
        if profile_main_model_infer_time:
            if latents_action.is_cuda:
                torch.cuda.synchronize(latents_action.device)
            main_model_infer_start = time.perf_counter()

        if freeze_future_video_noise:
            # The video inputs and timestep are constant for every action denoising
            # step. Since video queries cannot attend to action tokens, its full
            # layer-wise trajectory is action-independent and can be prefetched once.
            timestep_video = infer_timesteps_video[0].unsqueeze(0).to(
                dtype=latents_video.dtype,
                device=self.device,
            )
            if compile_action_infer:
                (
                    video_tokens,
                    video_t,
                    video_t_mod,
                    video_context,
                    video_context_mask,
                    video_freqs,
                    video_f,
                    video_h,
                    video_w,
                    tokens_per_frame,
                ) = self.video_expert.prepare(
                    x=latents_video,
                    timestep=timestep_video,
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
                video_pre = {
                    "tokens": video_tokens,
                    "freqs": video_freqs,
                    "t": video_t,
                    "t_mod": video_t_mod,
                    "context": video_context,
                    "context_mask": video_context_mask,
                    "meta": {
                        "grid_size": (video_f, video_h, video_w),
                        "tokens_per_frame": tokens_per_frame,
                        "batch_size": latents_video.shape[0],
                    },
                }
            else:
                video_pre = self.video_expert.pre_dit(
                    x=latents_video,
                    timestep=timestep_video,
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
            video_seq_len = int(video_pre["tokens"].shape[1])
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=latents_action.shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            )
            action_attention_mask = attention_mask[video_seq_len:, :]
            if compile_action_infer:
                if not hasattr(self, "_prefill_video_cache_compiled"):
                    self._prefill_video_cache_compiled = torch.compile(
                        self.mot.prefill_video_cache_tensor,
                        mode="reduce-overhead",
                        fullgraph=True,
                    )
                if not hasattr(self, "_denoise_action_with_video_cache_compiled"):
                    self._denoise_action_with_video_cache_compiled = torch.compile(
                        self._denoise_action_with_video_cache,
                        mode="reduce-overhead",
                        fullgraph=True,
                    )
                torch.compiler.cudagraph_mark_step_begin()
                video_cache_k, video_cache_v = self._prefill_video_cache_compiled(
                    video_tokens=video_pre["tokens"],
                    video_freqs=video_pre["freqs"],
                    video_t_mod=video_pre["t_mod"],
                    video_context=video_pre["context"],
                    video_context_mask=video_pre["context_mask"],
                    video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
                )
                # Inductor reduce-overhead outputs may alias graph-owned replay buffers.
                video_cache_k = [cache.clone() for cache in video_cache_k]
                video_cache_v = [cache.clone() for cache in video_cache_v]
            else:
                video_context_payload = {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                }
                video_kv_cache = self.mot.prefill_video_cache(
                    video_tokens=video_pre["tokens"],
                    video_freqs=video_pre["freqs"],
                    video_t_mod=video_pre["t_mod"],
                    video_context_payload=video_context_payload,
                    video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
                )
                video_kv_cache = self._clone_video_kv_cache_if_needed(video_kv_cache)

            for denoise_step_idx, (step_t_action, step_delta_action) in enumerate(
                zip(infer_timesteps_action, infer_deltas_action)
            ):
                if compile_action_infer:
                    torch.compiler.cudagraph_mark_step_begin()
                timestep_action = step_t_action.unsqueeze(0).to(
                    dtype=latents_action.dtype,
                    device=self.device,
                )
                if compile_action_infer:
                    pred_action = self._denoise_action_with_video_cache_compiled(
                        latents_action=latents_action,
                        timestep_action=timestep_action,
                        context=context,
                        context_mask=context_mask,
                        video_cache_k=video_cache_k,
                        video_cache_v=video_cache_v,
                        action_attention_mask=action_attention_mask,
                    )
                else:
                    pred_action = self._predict_action_noise_with_cache(
                        latents_action=latents_action,
                        timestep_action=timestep_action,
                        context=context,
                        context_mask=context_mask,
                        video_kv_cache=video_kv_cache,
                        attention_mask=attention_mask,
                        video_seq_len=video_seq_len,
                        denoise_step_idx=denoise_step_idx,
                    )
                latents_action = self.infer_action_scheduler.step(
                    pred_action,
                    step_delta_action,
                    latents_action,
                )
        else:
            if compile_action_infer:
                patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
                tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
                joint_attention_mask = self._build_mot_attention_mask(
                    video_seq_len=(latent_t // patch_t) * tokens_per_frame,
                    action_seq_len=latents_action.shape[1],
                    video_tokens_per_frame=tokens_per_frame,
                    device=self.device,
                )
                if not hasattr(self, "_joint_denoise_core_compiled_inference"):
                    self._joint_denoise_core_compiled_inference = torch.compile(
                        self._joint_denoise_core,
                        mode="reduce-overhead",
                        fullgraph=True,
                    )
            for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
                infer_timesteps_video,
                infer_deltas_video,
                infer_timesteps_action,
                infer_deltas_action,
            ):
                if compile_action_infer:
                    torch.compiler.cudagraph_mark_step_begin()
                timestep_video = step_t_video.unsqueeze(0).to(
                    dtype=latents_video.dtype,
                    device=self.device,
                )
                timestep_action = step_t_action.unsqueeze(0).to(
                    dtype=latents_action.dtype,
                    device=self.device,
                )
                if compile_action_infer:
                    pred_video_posi, pred_action_posi = self._joint_denoise_core_compiled_inference(
                        latents_video=latents_video,
                        latents_action=latents_action,
                        timestep_video=timestep_video,
                        timestep_action=timestep_action,
                        context=context,
                        context_mask=context_mask,
                        attention_mask=joint_attention_mask,
                        fuse_vae_embedding_in_latents=fuse_flag,
                        action_condition=None,
                    )
                else:
                    pred_video_posi, pred_action_posi = self._predict_joint_noise(
                        latents_video=latents_video,
                        latents_action=latents_action,
                        timestep_video=timestep_video,
                        timestep_action=timestep_action,
                        context=context,
                        context_mask=context_mask,
                        fuse_vae_embedding_in_latents=fuse_flag,
                        gt_action=None,
                    )
                latents_video = self.infer_video_scheduler.step(pred_video_posi, step_delta_video, latents_video)
                latents_video[:, :, 0:1] = first_frame_latents
                latents_action = self.infer_action_scheduler.step(
                    pred_action_posi,
                    step_delta_action,
                    latents_action,
                )

        main_model_infer_time_sec = None
        if main_model_infer_start is not None:
            if latents_action.is_cuda:
                torch.cuda.synchronize(latents_action.device)
            main_model_infer_time_sec = float(time.perf_counter() - main_model_infer_start)

        output = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
        if main_model_infer_time_sec is not None:
            output["main_model_infer_time_sec"] = main_model_infer_time_sec
        if return_context:
            output["context"] = base_context.detach().to(device="cpu")
            output["context_mask"] = base_context_mask.detach().to(device="cpu", dtype=torch.bool)
        return output
