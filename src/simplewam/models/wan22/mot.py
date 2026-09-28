from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from einops import rearrange

from .wan_video_dit import flash_attention, modulate, rope_apply
from simplewam.utils.logging_config import get_logger

logger = get_logger(__name__)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention. This will save memory but use more computation.")

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim
        self.num_video_layers = len(self.mixtures["video"].blocks)
        self.num_action_layers = len(self.mixtures["action"].blocks)
        # Backward-compatible alias: action depth is the controlling depth for action denoising.
        self.num_layers = self.num_action_layers
        self._attention_capture_enabled = False
        self._attention_capture_max_records: Optional[int] = None
        self._captured_attention_maps: list[dict[str, Any]] = []

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )
        
        logger.info(
            "Initialized MoT with experts: %s, video_layers=%d action_layers=%d",
            self.expert_order,
            self.num_video_layers,
            self.num_action_layers,
        )
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")

    def enable_attention_capture(self, *, max_records: Optional[int] = None) -> None:
        self._attention_capture_enabled = True
        self._attention_capture_max_records = max_records
        self._captured_attention_maps = []

    def disable_attention_capture(self) -> None:
        self._attention_capture_enabled = False
        self._attention_capture_max_records = None

    def pop_captured_attention_maps(self) -> list[dict[str, Any]]:
        maps = list(self._captured_attention_maps)
        self._captured_attention_maps = []
        return maps

    def _can_capture_more_attention(self) -> bool:
        if not self._attention_capture_enabled:
            return False
        if self._attention_capture_max_records is None:
            return True
        return len(self._captured_attention_maps) < int(self._attention_capture_max_records)

    def _capture_attention_map(
        self,
        *,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        attention_mask: torch.Tensor,
        capture_meta: dict[str, Any],
    ) -> None:
        if not self._can_capture_more_attention():
            return

        with torch.no_grad():
            q = rearrange(q_cat, "b s (n d) -> b n s d", n=self.num_heads).to(dtype=torch.float32)
            k = rearrange(k_cat, "b s (n d) -> b n s d", n=self.num_heads).to(dtype=torch.float32)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.attn_head_dim))

            mask = attention_mask.to(device=scores.device)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            else:
                scores = scores + mask.unsqueeze(0).unsqueeze(0).to(dtype=scores.dtype)

            attn = torch.softmax(scores, dim=-1)
            attn = attn.mean(dim=1)  # [B, Lq, Lk], head-averaged for compactness.
            attn = attn.detach().to(device="cpu", dtype=torch.float32)

        payload = dict(capture_meta)
        payload["attention"] = attn
        self._captured_attention_maps.append(payload)

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
        capture_meta: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)
        if capture_meta is not None:
            self._capture_attention_map(
                q_cat=q_cat,
                k_cat=k_cat,
                attention_mask=attn_mask,
                capture_meta=capture_meta,
            )

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    @staticmethod
    def _apply_expert_post_block_tensor(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Tensor-only post block used by the compiled inference core."""
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))
        x = x + block.cross_attn(
            block.norm3(x),
            context,
            ctx_mask=context_mask.unsqueeze(1),
        )
        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        return block.gate(x, gate_mlp, block.ffn(mlp_input))

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Prefill video branch once and cache per-layer K/V for action denoising.

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE frequencies, shape [Sv, 1, rope_dim].
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Video self-attention mask, shape [Sv, Sv].

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: video key tensor [B, Sv, H*Dh]
                - `v`: video value tensor [B, Sv, H*Dh]
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            # Video prefill uses only video self-attention mask.
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
                capture_meta={
                    "phase": "video_prefill",
                    "layer_idx": layer_idx,
                    "step_idx": -1,
                },
            )
            # Update video tokens for the next layer and persist current layer K/V.
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    def prefill_video_cache_tensor(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        video_attention_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Prefill video K/V through a tensor-only CUDA Graph friendly path."""
        expert = self.mixtures["video"]
        x = video_tokens
        cache_k_list: list[torch.Tensor] = []
        cache_v_list: list[torch.Tensor] = []
        for layer_idx in range(self.num_video_layers):
            block = expert.blocks[layer_idx]
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                _use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            mixed = flash_attention(
                q=q,
                k=k,
                v=v,
                num_heads=self.num_heads,
                ctx_mask=video_attention_mask.to(device=q.device),
            )
            x = self._apply_expert_post_block_tensor(
                block=block,
                residual_x=residual_x,
                mixed_attn_out=mixed,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                context=video_context,
                context_mask=video_context_mask,
            )
            cache_k_list.append(k)
            cache_v_list.append(v)
        return cache_k_list, cache_v_list


    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        denoise_step_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        # Use the action query rows from the joint [video+action] mask.
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )

            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            # Mixed attention: action queries attend to cached video K/V plus current action K/V.
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
                capture_meta={
                    "phase": "action_denoise",
                    "layer_idx": layer_idx,
                    "step_idx": -1 if denoise_step_idx is None else int(denoise_step_idx),
                },
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x

    def forward_action_with_video_cache_tensor(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run cached action denoising through a tensor-only compiled path."""
        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_action_layers):
            block = expert.blocks[layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                _use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            mixed = flash_attention(
                q=q_action,
                k=torch.cat([video_cache_k[layer_idx], k_action], dim=1),
                v=torch.cat([video_cache_v[layer_idx], v_action], dim=1),
                num_heads=self.num_heads,
                ctx_mask=action_attention_mask.to(device=q_action.device),
            )
            x = self._apply_expert_post_block_tensor(
                block=block,
                residual_x=residual_x,
                mixed_attn_out=mixed,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                context=action_context,
                context_mask=action_context_mask,
            )
        return x

    def _forward_joint_layer(
        self,
        video_block: nn.Module,
        action_block: nn.Module,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one tensor-only joint video/action layer."""
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        (
            q_video,
            k_video,
            v_video,
            residual_video,
            gate_msa_video,
            shift_mlp_video,
            scale_mlp_video,
            gate_mlp_video,
            _video_checkpointing,
        ) = self._build_expert_attention_io(
            expert=video_expert,
            block=video_block,
            x=video_tokens,
            freqs=video_freqs,
            t_mod=video_t_mod,
        )
        (
            q_action,
            k_action,
            v_action,
            residual_action,
            gate_msa_action,
            shift_mlp_action,
            scale_mlp_action,
            gate_mlp_action,
            _action_checkpointing,
        ) = self._build_expert_attention_io(
            expert=action_expert,
            block=action_block,
            x=action_tokens,
            freqs=action_freqs,
            t_mod=action_t_mod,
        )
        mixed = flash_attention(
            q=torch.cat([q_video, q_action], dim=1),
            k=torch.cat([k_video, k_action], dim=1),
            v=torch.cat([v_video, v_action], dim=1),
            num_heads=self.num_heads,
            ctx_mask=attention_mask.to(device=q_video.device),
        )
        video_seq_len = video_tokens.shape[1]
        return (
            self._apply_expert_post_block_tensor(
                block=video_block,
                residual_x=residual_video,
                mixed_attn_out=mixed[:, :video_seq_len],
                gate_msa=gate_msa_video,
                shift_mlp=shift_mlp_video,
                scale_mlp=scale_mlp_video,
                gate_mlp=gate_mlp_video,
                context=video_context,
                context_mask=video_context_mask,
            ),
            self._apply_expert_post_block_tensor(
                block=action_block,
                residual_x=residual_action,
                mixed_attn_out=mixed[:, video_seq_len:],
                gate_msa=gate_msa_action,
                shift_mlp=shift_mlp_action,
                scale_mlp=scale_mlp_action,
                gate_mlp=gate_mlp_action,
                context=action_context,
                context_mask=action_context_mask,
            ),
        )

    def forward_joint_core(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the equal-depth MoT using only tensor inputs and outputs."""
        if self.num_video_layers != self.num_action_layers:
            raise ValueError(
                "Tensor-only joint core requires equal video/action depth. "
                f"Got video_layers={self.num_video_layers}, action_layers={self.num_action_layers}."
            )

        x_video = video_tokens
        x_action = action_tokens
        for layer_idx in range(self.num_video_layers):
            x_video, x_action = self._forward_joint_layer(
                video_block=self.mixtures["video"].blocks[layer_idx],
                action_block=self.mixtures["action"].blocks[layer_idx],
                video_tokens=x_video,
                action_tokens=x_action,
                video_freqs=video_freqs,
                action_freqs=action_freqs,
                video_t_mod=video_t_mod,
                action_t_mod=action_t_mod,
                video_context=video_context,
                video_context_mask=video_context_mask,
                action_context=action_context,
                action_context_mask=action_context_mask,
                attention_mask=attention_mask,
            )
        return x_video, x_action

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        capture_layers: Optional[set[int]] = None,
        capture_experts: Optional[set[str]] = None,
    ):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")
        if self.num_video_layers != self.num_action_layers:
            raise ValueError(
                "Standard MoT.forward requires equal video/action depth. "
                f"Got video_layers={self.num_video_layers}, action_layers={self.num_action_layers}."
            )

        tokens_all = {k: v for k, v in embeds_all.items()}
        captured_hidden_states: dict[str, dict[int, torch.Tensor]] = {}
        capture_layers = set() if capture_layers is None else set(int(i) for i in capture_layers)
        capture_experts = (
            set(self.expert_order)
            if capture_experts is None
            else set(str(name) for name in capture_experts)
        )

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            # 3. concat all tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if attention_mask.shape[0] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                )

            mixed = self._mixed_attention(
                q_cat=q_cat,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=attention_mask,
                capture_meta={
                    "phase": "joint_mixed",
                    "layer_idx": layer_idx,
                    "step_idx": -1,
                },
            )

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                if layer_idx in capture_layers and name in capture_experts:
                    captured_hidden_states.setdefault(name, {})[layer_idx] = updated_tokens
                start = end

        if captured_hidden_states:
            tokens_all["_hidden_states"] = captured_hidden_states
        return tokens_all
