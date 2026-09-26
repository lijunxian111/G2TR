# --------------------------------------------------------
# InternVL-U
# Modifications Copyright (c) 2026 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy
import math
from typing import List, Optional, Tuple, Union

import torch
import transformers

from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import (
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
)

from .configuration_internvlu_chat_prune import InternVLUChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .constants import SPECIAL_TOKEN_LIST, CLIP_MEAN, CLIP_STD
from .token_merge_new import token_merging, latent_guided_keep_indices, rebuild_packed_vit_after_prune

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op="eq"):
    """Compare two version strings using a provided operator."""
    import operator

    from packaging import version

    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLUChatModel(PreTrainedModel):
    """Multimodal chat model combining a vision encoder with a language model."""

    config_class = InternVLUChatConfig
    main_input_name = "pixel_values"
    base_model_prefix = ""
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "InternVisionModel",
        "Qwen3DecoderLayer",
    ]

    # support transformers 4.51.+
    _tp_plan = ""

    def __init__(
        self,
        config: InternVLUChatConfig,
        vision_model=None,
        language_model=None,
        use_flash_attn=None,
    ):
        super().__init__(config)

        assert version_cmp(transformers.__version__, "4.37.0", "ge")
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.patch_aspect_ratio = 1.0
        self.ps_version = config.ps_version
        use_flash_attn = getattr(config.vision_config, "use_flash_attn", True) if use_flash_attn is None else use_flash_attn
        use_flash_attn = bool(use_flash_attn) and has_flash_attn
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = (
            "flash_attention_2" if use_flash_attn else "eager"
        )

        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            architecture: str = config.llm_config.architectures[0]
            if architecture == "LlamaForCausalLM":
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif architecture == "Qwen2ForCausalLM":
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            elif architecture == "Qwen3MoeForCausalLM":
                self.language_model = Qwen3MoeForCausalLM(config.llm_config)
            elif architecture == "Qwen3ForCausalLM":
                self.language_model = Qwen3ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f"{architecture} is not implemented.")

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        self.im_start_token_id = None
        self.im_end_token_id = None
        self.img_context_token_id = None
        self.img_start_token_id = None
        self.img_end_token_id = None
        self.img_uncond_token_id = None
        self.img_line_break_token_id = None
        self.img_frame_break_token_id = None
        self.pad_token_id = None
        self.conv_template = get_conv_template(self.template)

        if hasattr(config, "system_message"):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message

        ##### ---- Special token embeddings ---- #####
        self.special_token_embedding = nn.Embedding(
            len(SPECIAL_TOKEN_LIST), config.llm_config.hidden_size
        )
        self.special_token_list = copy.deepcopy(SPECIAL_TOKEN_LIST)
        self.special_token_id_list = None  # Remember to initialize this in the training script after tokenizer is loaded

    def replace_img_special_tokens(self, input_embeds, input_ids):
        assert (
            self.special_token_id_list is not None
        ), "model's special_token_id_list is not initialized"
        for i, token_id in enumerate(self.special_token_id_list):
            token_pos = input_ids == token_id
            input_embeds[token_pos] = (
                input_embeds[token_pos] * 0.0 + self.special_token_embedding.weight[i]
            )

        return input_embeds


    def _retrieve_latents_for_prune(self, encoder_output, sample_mode: str = "argmax"):
        """Extract latents from diffusers-style VAE encoder outputs."""
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            return encoder_output.latent_dist.sample()
        if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
        if hasattr(encoder_output, "latents"):
            return encoder_output.latents
        if isinstance(encoder_output, (tuple, list)):
            return encoder_output[0]
        raise AttributeError("Could not access latents from the provided VAE encoder output.")

    def _infer_2d_grid_for_prune(self, num_tokens: int):
        side = int(round(math.sqrt(num_tokens)))
        if side * side == num_tokens:
            return side, side
        return 1, num_tokens

    def _find_img_context_segments(self, input_ids: torch.LongTensor):
        """Return contiguous <IMG_CONTEXT> runs as (batch_idx, start, end)."""
        segments = []
        ctx_mask = input_ids == self.img_context_token_id
        for b in range(ctx_mask.shape[0]):
            idx = torch.nonzero(ctx_mask[b], as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            start = int(idx[0].item())
            prev = start
            for cur_t in idx[1:]:
                cur = int(cur_t.item())
                if cur != prev + 1:
                    segments.append((b, start, prev + 1))
                    start = cur
                prev = cur
            segments.append((b, start, prev + 1))
        return segments

    def _fill_visual_tokens_without_prune(
        self,
        input_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        vit_embeds: torch.Tensor,
    ):
        """Original InternVL-U behavior: fill all <IMG_CONTEXT> positions."""
        B, N, C = input_embeds.shape
        flat_embeds = input_embeds.reshape(B * N, C)
        flat_ids = input_ids.reshape(B * N)
        selected = flat_ids == self.img_context_token_id
        assert selected.sum() != 0
        vit_flat = vit_embeds.reshape(-1, C).to(flat_embeds.device, flat_embeds.dtype)
        if selected.sum().item() != vit_flat.shape[0]:
            n_token = min(int(selected.sum().item()), vit_flat.shape[0])
            selected_idx = torch.nonzero(selected, as_tuple=False).flatten()[:n_token]
            flat_embeds[selected_idx] = flat_embeds[selected_idx] * 0.0 + vit_flat[:n_token]
        else:
            flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_flat
        return flat_embeds.reshape(B, N, C)

    def _make_vae_pixels_from_vlm_pixels(self, pixel_values: torch.Tensor):
        """
        Convert InternVL image-processor pixels, normally CLIP-normalized, back to
        the [-1, 1] range expected by the Qwen-Image/InternVL-U VAE.
        """
        x = pixel_values.float()
        if x.ndim != 4 or x.shape[1] != 3:
            return pixel_values
        mean = torch.tensor(CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x * std + mean).clamp(0.0, 1.0)
        x = x * 2.0 - 1.0
        return x

    def _pool_latent_grid_for_prune(self, latent_grid: torch.Tensor, h: int, w: int, pool: int):
        """
        latent_grid: [h*w, C]
        return pooled_latent_grid: [h2*w2, C], (h2, w2)
        """
        if pool <= 1:
            return latent_grid, (h, w)

        grid = latent_grid.view(h, w, -1)
        pooled_rows = []
        for r in range(0, h, pool):
            row_chunks = []
            for c in range(0, w, pool):
                block = grid[r : min(r + pool, h), c : min(c + pool, w)]
                row_chunks.append(block.reshape(-1, grid.shape[-1]).mean(dim=0))
            pooled_rows.append(torch.stack(row_chunks, dim=0))
        pooled = torch.stack(pooled_rows, dim=0)
        h2, w2 = pooled.shape[:2]
        return pooled.reshape(-1, pooled.shape[-1]), (h2, w2)

    def _resize_latent_feature_dim(self, latent_features: torch.Tensor, target_dim: int, dtype: torch.dtype):
        """Parameter-free projection so VAE anchors can be compared with VLM tokens."""
        if latent_features.shape[-1] == target_dim:
            return latent_features.to(dtype=dtype)
        x = latent_features.float().unsqueeze(1)  # [N, 1, C_lat]
        x = F.interpolate(x, size=target_dim, mode="linear", align_corners=False)
        return x.squeeze(1).to(dtype=dtype)

    @torch.no_grad()
    def _build_vae_anchor_features_for_prune(
        self,
        vae_model,
        pixel_values: torch.Tensor,
        pixel_values_gen: Optional[torch.Tensor],
        expected_images: int,
        target_dim: int,
        target_dtype: torch.dtype,
    ):
        """
        Encode input images with the generation VAE and build pooled latent anchors.
        If pixel_values_gen does not align with ViT patches, fall back to a VAE-style
        denormalization of the VLM pixel_values so each ViT patch gets one VAE grid.
        """
        if vae_model is None:
            return None, None

        if (
            pixel_values_gen is not None
            and pixel_values_gen.ndim == 4
            and pixel_values_gen.shape[0] == expected_images
        ):
            vae_pixels = pixel_values_gen
        else:
            vae_pixels = self._make_vae_pixels_from_vlm_pixels(pixel_values)

        if vae_pixels.shape[0] < expected_images:
            return None, None
        vae_pixels = vae_pixels[:expected_images]

        try:
            first_param = next(vae_model.parameters())
            vae_device = first_param.device
            vae_dtype = first_param.dtype
        except StopIteration:
            vae_device = vae_pixels.device
            vae_dtype = vae_pixels.dtype

        vae_pixels = vae_pixels.to(device=vae_device, dtype=vae_dtype)
        max_bs = max(1, int(getattr(self.config, "umm_vae_encode_max_bs", 32)))
        latent_chunks = []
        for s in range(0, vae_pixels.shape[0], max_bs):
            x_bs = vae_pixels[s : s + max_bs]
            try:
                enc = vae_model.encode(x_bs.unsqueeze(2))
            except Exception:
                enc = vae_model.encode(x_bs)
            z = self._retrieve_latents_for_prune(enc, sample_mode="argmax")
            latent_chunks.append(z)
        latents = torch.cat(latent_chunks, dim=0)

        if latents.ndim == 5:
            # Qwen-Image VAE uses [B, C, T, H, W] for image/video; T=1 for images.
            latents = latents[:, :, 0]
        if hasattr(vae_model, "config") and hasattr(vae_model.config, "latents_mean") and hasattr(vae_model.config, "latents_std"):
            latents_mean = torch.tensor(vae_model.config.latents_mean, device=latents.device, dtype=latents.dtype)
            latents_std = torch.tensor(vae_model.config.latents_std, device=latents.device, dtype=latents.dtype)
            while latents_mean.ndim < latents.ndim:
                latents_mean = latents_mean.view(1, -1, *([1] * (latents.ndim - 2)))
                latents_std = latents_std.view(1, -1, *([1] * (latents.ndim - 2)))
            latents = (latents - latents_mean) / latents_std

        pool = max(1, int(getattr(self.config, "umm_latent_pool", 1)))
        packed_latents = []
        latent_shapes = []
        for z in latents:
            c, h, w = z.shape
            latent_grid = z.permute(1, 2, 0).reshape(h * w, c)
            latent_grid, pooled_shape = self._pool_latent_grid_for_prune(latent_grid, h, w, pool)
            packed_latents.append(latent_grid)
            latent_shapes.append(pooled_shape)

        packed_latents = torch.cat(packed_latents, dim=0).to(pixel_values.device)
        anchor_features = self._resize_latent_feature_dim(packed_latents, target_dim, target_dtype)
        return anchor_features, latent_shapes

    def _rebuild_inputs_with_pruned_visual_tokens(
        self,
        input_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        labels: Optional[torch.LongTensor],
        segments,
        merged_segments,
        kept_local_indices,
    ):
        B, N, C = input_embeds.shape
        device = input_embeds.device
        seg_by_batch = {b: [] for b in range(B)}
        for i, (b, s, e) in enumerate(segments):
            seg_by_batch[b].append((s, e, i))

        rows_embeds, rows_ids, rows_attn, rows_pos, rows_labels = [], [], [], [], []
        for b in range(B):
            cur = 0
            emb_parts, id_parts = [], []
            attn_parts, pos_parts, label_parts = [], [], []
            for s, e, seg_i in seg_by_batch.get(b, []):
                if s > cur:
                    emb_parts.append(input_embeds[b, cur:s])
                    id_parts.append(input_ids[b, cur:s])
                    if attention_mask is not None:
                        attn_parts.append(attention_mask[b, cur:s])
                    if position_ids is not None:
                        pos_parts.append(position_ids[b, cur:s])
                    if labels is not None:
                        label_parts.append(labels[b, cur:s])

                merged = merged_segments[seg_i].to(device=device, dtype=input_embeds.dtype)
                local_keep = kept_local_indices[seg_i].to(device=device)
                emb_parts.append(merged)
                id_parts.append(input_ids[b, s].expand(merged.shape[0]))
                if attention_mask is not None:
                    attn_parts.append(attention_mask[b, s + local_keep])
                if position_ids is not None:
                    pos_parts.append(position_ids[b, s + local_keep])
                if labels is not None:
                    label_parts.append(labels[b, s + local_keep])
                cur = e

            if cur < N:
                emb_parts.append(input_embeds[b, cur:N])
                id_parts.append(input_ids[b, cur:N])
                if attention_mask is not None:
                    attn_parts.append(attention_mask[b, cur:N])
                if position_ids is not None:
                    pos_parts.append(position_ids[b, cur:N])
                if labels is not None:
                    label_parts.append(labels[b, cur:N])

            rows_embeds.append(torch.cat(emb_parts, dim=0))
            rows_ids.append(torch.cat(id_parts, dim=0))
            if attention_mask is not None:
                rows_attn.append(torch.cat(attn_parts, dim=0))
            if position_ids is not None:
                rows_pos.append(torch.cat(pos_parts, dim=0))
            if labels is not None:
                rows_labels.append(torch.cat(label_parts, dim=0))

        max_len = max(x.shape[0] for x in rows_embeds)
        pad_token_id = self.pad_token_id
        if pad_token_id is None:
            pad_token_id = getattr(self.language_model.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        new_embeds = input_embeds.new_zeros((B, max_len, C))
        new_ids = input_ids.new_full((B, max_len), int(pad_token_id))
        new_attention = None if attention_mask is None else attention_mask.new_zeros((B, max_len))
        new_position = None if position_ids is None else position_ids.new_zeros((B, max_len))
        new_labels = None if labels is None else labels.new_full((B, max_len), -100)

        for b in range(B):
            l = rows_embeds[b].shape[0]
            new_embeds[b, :l] = rows_embeds[b]
            new_ids[b, :l] = rows_ids[b]
            if new_attention is not None:
                new_attention[b, :l] = rows_attn[b]
            if new_position is not None:
                new_position[b, :l] = rows_pos[b]
            if new_labels is not None:
                new_labels[b, :l] = rows_labels[b]

        return new_embeds, new_ids, new_attention, new_position, new_labels

    def _insert_visual_tokens_maybe_vae_prune(
        self,
        input_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        labels: Optional[torch.LongTensor],
        pixel_values: torch.Tensor,
        grid_thw: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        vae_model=None,
        pixel_values_gen: Optional[torch.Tensor] = None,
        image_grid_thw_gen: Optional[torch.LongTensor] = None,
        prune_ratio: Optional[float] = None,
    ):
        """Fill visual tokens, optionally doing VAE-guided true prune before LLM prefill."""
        vit_embeds = self.extract_feature(pixel_values, grid_thw)
        if image_flags is not None:
            image_flags = image_flags.squeeze(-1).to(vit_embeds.device)
            vit_embeds = vit_embeds[image_flags == 1]

        keep_ratio = float(prune_ratio if prune_ratio is not None else getattr(self.config, "umm_prune_ratio", 0.5))
        enable_prune = (
            (bool(getattr(self.config, "umm_vae_guided_prune", False)) or keep_ratio < 1.0)
            and keep_ratio < 1.0
            and vae_model is not None
            and pixel_values is not None
            and not (self.training and not bool(getattr(self.config, "umm_prune_training", False)))
        )
        if not enable_prune:
            return (
                self._fill_visual_tokens_without_prune(input_embeds, input_ids, vit_embeds),
                input_ids,
                attention_mask,
                position_ids,
                labels,
                None,
            )

        segments = self._find_img_context_segments(input_ids)
        if len(segments) == 0 or vit_embeds.ndim != 3 or vit_embeds.shape[0] < len(segments):
            return (
                self._fill_visual_tokens_without_prune(input_embeds, input_ids, vit_embeds),
                input_ids,
                attention_mask,
                position_ids,
                labels,
                None,
            )

        C = input_embeds.shape[-1]
        vit_list = []
        image_shapes = []
        seg_lens = []
        for i, (_, s, e) in enumerate(segments):
            seg_len = e - s
            feat = vit_embeds[i]
            if feat.shape[0] < seg_len:
                return (
                    self._fill_visual_tokens_without_prune(input_embeds, input_ids, vit_embeds),
                    input_ids,
                    attention_mask,
                    position_ids,
                    labels,
                    None,
                )
            feat = feat[:seg_len]
            vit_list.append(feat)
            seg_lens.append(seg_len)
            h, w = self._infer_2d_grid_for_prune(seg_len)
            image_shapes.append([feat.shape[-1], h, w])

        vit_packed = torch.cat(vit_list, dim=0)
        vit_lens = torch.tensor(seg_lens, device=vit_packed.device, dtype=torch.int32)
        vit_cu_seqlens = F.pad(torch.cumsum(vit_lens, dim=0), (1, 0)).to(torch.int32)

        latent_anchor_features, latent_anchor_shapes = self._build_vae_anchor_features_for_prune(
            vae_model=vae_model,
            pixel_values=pixel_values,
            pixel_values_gen=pixel_values_gen,
            expected_images=len(segments),
            target_dim=C,
            target_dtype=vit_packed.dtype,
        )
        if latent_anchor_features is None or latent_anchor_shapes is None:
            return (
                self._fill_visual_tokens_without_prune(input_embeds, input_ids, vit_embeds),
                input_ids,
                attention_mask,
                position_ids,
                labels,
                None,
            )

        keep_indices, token_scores = latent_guided_keep_indices(
            vit_features=vit_packed,
            vit_cu_seqlens=vit_cu_seqlens,
            image_shapes=image_shapes,
            latent_anchor_features=latent_anchor_features.to(vit_packed.device, vit_packed.dtype),
            latent_anchor_shapes=latent_anchor_shapes,
            keep_ratio=keep_ratio,
            min_keep_tokens=int(getattr(self.config, "umm_min_keep_tokens", 1)),
        )
        keep_indices = torch.sort(torch.unique(keep_indices)).values
        print(vit_packed.shape)
        merged_vit = token_merging(
            vit_packed,
            keep_indices,
            scaling=float(getattr(self.config, "umm_merge_scaling", 1.0)),
        )
        print(merged_vit.shape)
        _, new_cu_seqlens = rebuild_packed_vit_after_prune(
            packed_flattened_position_ids=torch.arange(vit_packed.shape[0], device=vit_packed.device),
            cu_seqlens=vit_cu_seqlens,
            keep_indices=keep_indices,
        )

        merged_segments = []
        kept_local_indices = []
        old_start = 0
        new_start = 0
        for i, seg_len in enumerate(seg_lens):
            old_end = old_start + seg_len
            keep_local = keep_indices[(keep_indices >= old_start) & (keep_indices < old_end)] - old_start
            new_len = int((new_cu_seqlens[i + 1] - new_cu_seqlens[i]).item())
            merged_segments.append(merged_vit[new_start : new_start + new_len])
            kept_local_indices.append(torch.sort(keep_local).values)
            old_start = old_end
            new_start += new_len

        new_inputs = self._rebuild_inputs_with_pruned_visual_tokens(
            input_embeds=input_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            segments=segments,
            merged_segments=merged_segments,
            kept_local_indices=kept_local_indices,
        )

        prune_info = {
            "old_visual_tokens": int(vit_packed.shape[0]),
            "new_visual_tokens": int(merged_vit.shape[0]),
            "keep_ratio": float(merged_vit.shape[0] / max(1, vit_packed.shape[0])),
        }
        if bool(getattr(self.config, "umm_prune_debug", False)):
            logger.info(
                f"VAE-guided prune: {prune_info['old_visual_tokens']} -> "
                f"{prune_info['new_visual_tokens']} visual tokens "
                f"({prune_info['keep_ratio']:.3f})"
            )
        return (*new_inputs, prune_info)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        padding_type: Optional[str] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        # Optional inputs for VAE-guided visual token pruning.
        vae_model=None,
        pixel_values_gen: Optional[torch.Tensor] = None,
        image_grid_thw_gen: Optional[torch.LongTensor] = None,
        prune_ratio: Optional[float] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        input_embeds = self.replace_img_special_tokens(input_embeds, input_ids)

        if video_grid_thw is not None:
            grid_thw = video_grid_thw
        else:
            grid_thw = image_grid_thw

        prune_info = None
        if pixel_values is not None:
            (
                input_embeds,
                input_ids,
                attention_mask,
                position_ids,
                labels,
                prune_info,
            ) = self._insert_visual_tokens_maybe_vae_prune(
                input_embeds=input_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                image_flags=image_flags,
                vae_model=vae_model,
                pixel_values_gen=pixel_values_gen,
                image_grid_thw_gen=image_grid_thw_gen,
                prune_ratio=prune_ratio,
            )

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            padding_type=padding_type,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle_v2(self, x, scale_factor=0.5, patch_aspect_ratio=1.0):
        # input shape: N, L, C or N, H, W, C
        # output shape: N, L * (scale_factor ** 2), C / (scale_factor ** 2)

        if x.ndim == 3:
            n, l, c = x.size()
            h = w = int(l**0.5)
            # N, L, C --> N, H, W, C
            x = x.reshape(n, h, w, c)

        n, h, w, c = x.size()

        h_scale_factor = scale_factor * (patch_aspect_ratio**0.5)
        w_scale_factor = scale_factor / (patch_aspect_ratio**0.5)

        # N, H, W, C --> N, H, W * w_scale_factor, C // w_scale_factor
        x = x.reshape(n, h, int(w * w_scale_factor), int(c / w_scale_factor))
        # N, H, W * w_scale_factor, C // w_scale_factor --> N, W * w_scale_factor, H, C // w_scale_factor
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, W * w_scale_factor, H, C // w_scale_factor -->
        # N, W * w_scale_factor, H * h_scale_factor, C // (w_scale_factor * h_scale_factor)
        x = x.reshape(
            n,
            int(w * w_scale_factor),
            int(h * h_scale_factor),
            int(c / (w_scale_factor * h_scale_factor)),
        )
        # N, W * w_scale_factor, H * h_scale_factor, C // (w_scale_factor * h_scale_factor) -->
        # N, H * h_scale_factor, W * w_scale_factor, C // (w_scale_factor * h_scale_factor)
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * h_scale_factor, W * w_scale_factor, C // (w_scale_factor * h_scale_factor) -->
        # N, L * (scale_factor ** 2), C // (scale_factor ** 2)
        x = x.reshape(
            n,
            int(h * h_scale_factor * w * w_scale_factor),
            int(c / (h_scale_factor * w_scale_factor)),
        )

        return x

    def extract_feature(self, pixel_values, grid_thw=None):
        if not self.config.anyres_image_size:
            if self.select_layer == -1:
                vit_embeds = self.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=False,
                    return_dict=True,
                ).last_hidden_state
            else:
                vit_embeds = self.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states[self.select_layer]
            vit_embeds = vit_embeds[:, 1:, :]
        else:
            if grid_thw is not None:
                grid_thw = grid_thw.to(pixel_values.device)

            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
                grid_thw=grid_thw,
            ).last_hidden_state

        vit_embeds = self.pixel_shuffle_v2(
            vit_embeds,
            scale_factor=self.downsample_ratio,
            patch_aspect_ratio=self.patch_aspect_ratio,
        )
        vit_embeds_after_mlp = self.mlp1(vit_embeds)

        return vit_embeds_after_mlp

    def batch_chat(
        self,
        tokenizer,
        pixel_values,
        questions,
        generation_config,
        num_patches_list=None,
        history=None,
        return_history=False,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
        image_counts=None,
    ):
        if history is not None or return_history:
            print("Now multi-turn chat is not supported in batch_chat.")
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print(
                "Warning: `image_counts` is deprecated. Please use `num_patches_list` instead."
            )

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and "<image>" not in question:
                question = "<image>\n" + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = "left"
        model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [
            response.split(template.sep.strip())[0].strip() for response in responses
        ]
        return responses

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
    ):

        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        if num_patches_list is None:
            num_patches_list = (
                [pixel_values.shape[0]] if pixel_values is not None else []
            )
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[
            0
        ]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
            query_to_print = query_to_print.replace(
                f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>"
            )
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        vae_model=None,
        pixel_values_gen: Optional[torch.Tensor] = None,
        image_grid_thw_gen: Optional[torch.LongTensor] = None,
        prune_ratio: Optional[float] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """Generate text tokens from multimodal inputs.

        Args:
            pixel_values (`torch.FloatTensor`, *optional*):
                Image tensor of shape `(B, C, H, W)` used to extract visual features.
            input_ids (`torch.LongTensor`, *optional*):
                Token IDs for the language model inputs.
            attention_mask (`torch.LongTensor`, *optional*):
                Attention mask for the input tokens.
            visual_features (`torch.FloatTensor`, *optional*):
                Precomputed vision features to insert at image token positions.
            generation_config (`GenerationConfig`, *optional*):
                Generation configuration for the language model.
            output_hidden_states (`bool`, *optional*):
                Whether to return hidden states from the language model.
            generate_kwargs:
                Additional kwargs forwarded to `language_model.generate`.

        Returns:
            `torch.LongTensor`: Generated token IDs.
        """

        assert self.img_context_token_id is not None
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        input_embeds = self.replace_img_special_tokens(input_embeds, input_ids)

        if pixel_values is not None:
            if visual_features is not None:
                input_embeds = self._fill_visual_tokens_without_prune(input_embeds, input_ids, visual_features)
            else:
                (
                    input_embeds,
                    input_ids,
                    attention_mask,
                    _,
                    _,
                    _,
                ) = self._insert_visual_tokens_maybe_vae_prune(
                    input_embeds=input_embeds,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=None,
                    labels=None,
                    pixel_values=pixel_values,
                    grid_thw=None,
                    image_flags=None,
                    vae_model=vae_model,
                    pixel_values_gen=pixel_values_gen,
                    image_grid_thw_gen=image_grid_thw_gen,
                    prune_ratio=prune_ratio,
                )

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )
        return outputs

    @torch.no_grad()
    def generate_hidden_states(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        vae_model=None,
        pixel_values_gen: Optional[torch.Tensor] = None,
        image_grid_thw_gen: Optional[torch.LongTensor] = None,
        prune_ratio: Optional[float] = None,
        return_pruned_inputs: bool = False,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """Return hidden states from a forward pass with multimodal inputs.

        Args:
            pixel_values (`torch.FloatTensor`, *optional*):
                Image tensor of shape `(B, C, H, W)` used to extract visual features.
            input_ids (`torch.LongTensor`, *optional*):
                Token IDs for the language model inputs.
            attention_mask (`torch.LongTensor`, *optional*):
                Attention mask for the input tokens.
            visual_features (`torch.FloatTensor`, *optional*):
                Precomputed vision features to insert at image token positions.
            generate_kwargs:
                Additional kwargs forwarded to the language model forward pass.

        Returns:
            `CausalLMOutputWithPast`: Output containing hidden states.
        """

        assert self.img_context_token_id is not None
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        input_embeds = self.replace_img_special_tokens(input_embeds, input_ids)
        pruned_input_ids = input_ids
        pruned_attention_mask = attention_mask

        if pixel_values is not None:
            if visual_features is not None:
                input_embeds = self._fill_visual_tokens_without_prune(input_embeds, input_ids, visual_features)
            else:
                (
                    input_embeds,
                    pruned_input_ids,
                    pruned_attention_mask,
                    _,
                    _,
                    _,
                ) = self._insert_visual_tokens_maybe_vae_prune(
                    input_embeds=input_embeds,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=None,
                    labels=None,
                    pixel_values=pixel_values,
                    grid_thw=None,
                    image_flags=None,
                    vae_model=vae_model,
                    pixel_values_gen=pixel_values_gen,
                    image_grid_thw_gen=image_grid_thw_gen,
                    prune_ratio=prune_ratio,
                )

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=pruned_attention_mask,
            position_ids=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=True,
            padding_type="pad",
        )
        if return_pruned_inputs:
            outputs.pruned_input_ids = pruned_input_ids
            outputs.pruned_attention_mask = pruned_attention_mask
        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)
