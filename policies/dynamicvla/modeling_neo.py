# -*- coding: utf-8 -*-
#
# @File:   modeling_neo.py
# @Author: Haozhe Xie
# @Date:   2025-10-22 09:51:08
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-10-22 19:48:42
# @Email:  root@haozhexie.com

import logging
import typing

import torch
from transformers import (
    AutoModel,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
    Qwen3Config,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3Model,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


class NeoVLMBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: typing.Optional[torch.FloatTensor] = None
    past_key_values: typing.Optional[Cache] = None
    hidden_states: typing.Optional[tuple[torch.FloatTensor]] = None
    attentions: typing.Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: typing.Optional[tuple[torch.FloatTensor]] = None


class NeoVLMCausalLMOutputWithPast(ModelOutput):
    loss: typing.Optional[torch.FloatTensor] = None
    logits: typing.Optional[torch.FloatTensor] = None
    past_key_values: typing.Optional[Cache] = None
    hidden_states: typing.Optional[tuple[torch.FloatTensor]] = None
    attentions: typing.Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: typing.Optional[tuple[torch.FloatTensor]] = None


class NeoVisionConfig(PretrainedConfig):
    model_type = "neo_vision"

    def __init__(
        self,
        in_channels=3,
        patch_size=16,
        hidden_size=1024,
        llm_hidden_size=1024,
        downsample_ratio=0.5,
        rope_theta_vision=10_000,
        max_position_embeddings_vision=10_000,
        min_pixels=65_536,
        max_pixels=4_194_304,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.downsample_ratio = downsample_ratio
        self.rope_theta_vision = rope_theta_vision
        self.max_position_embeddings_vision = max_position_embeddings_vision
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels


class NeoLMConfig(Qwen3Config):
    model_type = "neo_lm"

    def __init__(
        self, rope_theta_hw=10000.0, max_position_embeddings_hw=10000, **kwargs
    ):
        super().__init__(**kwargs)
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw


class NeoVLMConfig(PretrainedConfig):
    model_type = "neo_vlm"
    sub_configs = {"text_config": Qwen3Config, "vision_config": NeoVisionConfig}

    def __init__(
        self,
        use_cache=True,
        downsample_ratio=0.5,
        tie_word_embeddings=False,
        max_position_embeddings=None,
        text_config=None,
        vision_config=None,
        **kwargs,
    ) -> None:
        # NeoVisionConfig
        if vision_config is None:
            self.vision_config = NeoVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = NeoVisionConfig(**vision_config)
        elif isinstance(vision_config, NeoVisionConfig):
            self.vision_config = vision_config
        else:
            raise ValueError("No valid vision_config is provided.")

        # NeoLMConfig
        if text_config is None:
            self.text_config = Qwen3Config()
        elif isinstance(text_config, dict):
            self.text_config = Qwen3Config(**text_config)
        elif isinstance(text_config, Qwen3Config):
            self.text_config = text_config
        else:
            raise ValueError("No valid text_config is provided.")

        self.use_cache = use_cache
        self.downsample_ratio = downsample_ratio
        self.tie_word_embeddings = tie_word_embeddings
        if max_position_embeddings is not None:
            self.text_config.max_position_embeddings = max_position_embeddings

        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


class NeoVLMPreTrainedModel(PreTrainedModel):
    config: NeoVLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["QWen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }


class NeoVLMForConditionalGeneration(NeoVLMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: NeoVLMConfig):
        super().__init__(config)
        self.model = NeoVLMModel(config)
        self.lm_head = torch.nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.vocab_size = config.text_config.vocab_size
        # Initialize weights and apply final processing
        self.post_init()

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = (
            self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        )
        self._vision_require_grads_hook = (
            self.model.vision_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grads
            )
        )

    def disable_input_require_grads(self):
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    def get_input_embeddings(self):
        return self.model.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.text_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        # indexes: typing.Optional[torch.LongTensor] = None,
        input_ids: typing.Optional[torch.LongTensor] = None,
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_ids: typing.Optional[torch.LongTensor] = None,
        past_key_values: typing.Optional[Cache] = None,
        inputs_embeds: typing.Optional[torch.FloatTensor] = None,
        pixel_values: typing.Optional[torch.FloatTensor] = None,
        pixel_attention_mask: typing.Optional[torch.BoolTensor] = None,
        image_hidden_states: typing.Optional[torch.FloatTensor] = None,
        labels: typing.Optional[torch.LongTensor] = None,
        output_attentions: typing.Optional[bool] = None,
        output_hidden_states: typing.Optional[bool] = None,
        use_cache: typing.Optional[bool] = None,
        cache_position: typing.Optional[torch.LongTensor] = None,
        return_dict: typing.Optional[bool] = None,
        logits_to_keep: typing.Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> NeoVLMCausalLMOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            # indexes=indexes,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not
        # computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return NeoVLMCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


class NeoVLMModel(NeoVLMPreTrainedModel):
    def __init__(self, config: NeoVLMConfig):
        super().__init__(config)
        self.vision_model = NeoVisionModel(config.vision_config)
        self.text_model: Qwen3Model = AutoModel.from_config(config.text_config)
        self.post_init()

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings.

        This is useful for lora when using gradient checkpointing.
        c.f. https://github.com/huggingface/peft/issues/1402#issuecomment-1913675032

        Override to set output.requires_grad = True for both the decoder's and vision model's
        embeddings.
        """

        def get_lowest_module(module):
            if len(list(module.children())) == 0:
                # If the module has no children, it is a leaf module (e.g., Linear, Conv2d, etc.)
                return module
            else:
                # Recursively call the function on each child module
                return get_lowest_module(list(module.children())[0])

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = (
            self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        )
        self._vision_require_grads_hook = get_lowest_module(
            self.vision_model
        ).register_forward_hook(make_inputs_require_grads)

    def disable_input_require_grads(self) -> None:
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.text_model.set_input_embeddings(value)

    def get_image_features(self, pixel_values: torch.FloatTensor):
        return self.vision_model(pixel_values).last_hidden_state


class NeoVisionModel(PreTrainedModel):
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    config_class = NeoVisionConfig

    def __init__(self, config: NeoVisionConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = NeoVisionEmbeddings(config)

    def forward(
        self,
        pixel_values: typing.Optional[torch.FloatTensor] = None,
        pixel_embeddings: typing.Optional[torch.FloatTensor] = None,
        output_hidden_states: typing.Optional[bool] = None,
        return_dict: typing.Optional[bool] = None,
        grid_hw: typing.Optional[torch.Tensor] = None,
    ) -> typing.Union[typing.Tuple, BaseModelOutput]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if pixel_values is None and pixel_embeddings is None:
            raise ValueError("You have to specify pixel_values or pixel_embeddings")

        if grid_hw is None:
            grid_hw = torch.tensor(
                [
                    [
                        pixel_values.size(-2) // self.config.patch_size,
                        pixel_values.size(-1) // self.config.patch_size,
                    ]
                ],
                device=pixel_values.device,
            ).repeat(pixel_values.size(0), 1)

        if pixel_embeddings is not None:
            hidden_states = pixel_embeddings
        else:
            if pixel_values.dim() != 2:
                pixel_values = pixel_values.view(pixel_values.size(0), -1)

            hidden_states = self.embeddings(pixel_values, grid_hw=grid_hw)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
        )


class NeoVisionEmbeddings(torch.nn.Module):
    def __init__(self, config: NeoVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.llm_embed_dim = config.llm_hidden_size
        self.downsample_factor = int(1 / config.downsample_ratio)
        self.patch_size = config.patch_size
        self.patch_embedding = torch.nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.dense_embedding = torch.nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.llm_embed_dim,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor,
        )
        self.gelu = torch.nn.GELU()
        self.rope_dim_part = self.embed_dim // 2
        cos_x, sin_x = self._get_rope_freqs_sincos(
            self.rope_dim_part,
            config.max_position_embeddings_vision,
            base=config.rope_theta_vision,
            device=None,
        )
        cos_y, sin_y = self._get_rope_freqs_sincos(
            self.rope_dim_part,
            config.max_position_embeddings_vision,
            base=config.rope_theta_vision,
            device=None,
        )
        self.register_buffer("cos_cached_x", cos_x, persistent=False)
        self.register_buffer("sin_cached_x", sin_x, persistent=False)
        self.register_buffer("cos_cached_y", cos_y, persistent=False)
        self.register_buffer("sin_cached_y", sin_y, persistent=False)

    def _get_rope_freqs_sincos(
        self, dim: int, max_position: int, base: float = 10000.0, device=None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device).float() / dim)
        )
        t = torch.arange(max_position, device=device).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        return torch.cos(freqs), torch.sin(freqs)

    def _apply_rotary_emb_2d(
        self, patch_embeds: torch.Tensor, grid_hw: torch.Tensor
    ) -> torch.Tensor:
        abs_pos_x, abs_pos_y = self._get_patch_abs_pos(grid_hw)
        half_dim = patch_embeds.shape[-1] // 2
        patch_embeds = patch_embeds.to(torch.float32)

        rotated_part_1 = self._apply_rotary_emb_1d(
            patch_embeds[..., :half_dim],
            self.cos_cached_x,
            self.sin_cached_x,
            abs_pos_x,
        )
        rotated_part_2 = self._apply_rotary_emb_1d(
            patch_embeds[..., half_dim:],
            self.cos_cached_y,
            self.sin_cached_y,
            abs_pos_y,
        )
        return torch.cat((rotated_part_1, rotated_part_2), dim=-1).to(
            self.patch_embedding.weight.dtype
        )

    def _apply_rotary_emb_1d(
        self,
        x: torch.Tensor,
        cos_cached: torch.Tensor,
        sin_cached: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated_x1 = x1 * cos_cached[positions] - x2 * sin_cached[positions]
        rotated_x2 = x1 * sin_cached[positions] + x2 * cos_cached[positions]

        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = rotated_x1
        x_rotated[..., 1::2] = rotated_x2
        return x_rotated

    def _get_patch_abs_pos(
        self, grid_hw: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        B = grid_hw.shape[0]
        # Get the number of patches per image
        H, W = grid_hw[:, 0], grid_hw[:, 1]
        N = H * W
        # Create the batch index for each patch (B x patch count)
        patch_to_sample = torch.repeat_interleave(
            torch.arange(B, device=grid_hw.device), N
        )
        # Generate intra-image patch index (row-major order)
        patch_id_within_image = torch.arange(N.sum(), device=grid_hw.device)
        patch_id_within_image = (
            patch_id_within_image
            - torch.cumsum(
                torch.cat([torch.tensor([0], device=grid_hw.device), N[:-1]]), dim=0
            )[patch_to_sample]
        )
        # Get H/W for each patch according to its image
        W_per_patch = W[patch_to_sample]
        abs_x = patch_id_within_image % W_per_patch
        abs_y = patch_id_within_image // W_per_patch
        return abs_x, abs_y

    def forward(
        self, pixel_values: torch.FloatTensor, grid_hw: torch.Tensor
    ) -> torch.Tensor:
        batch_size = pixel_values.size(0)
        pixel_values = pixel_values.view(
            -1,
            self.config.in_channels,
            self.patch_size,
            self.patch_size,
        )
        patch_embeds = self.gelu(self.patch_embedding(pixel_values)).view(
            -1, self.embed_dim
        )
        self.cos_cached_x = self.cos_cached_x.to(patch_embeds.device)
        self.sin_cached_x = self.sin_cached_x.to(patch_embeds.device)
        self.cos_cached_y = self.cos_cached_y.to(patch_embeds.device)
        self.sin_cached_y = self.sin_cached_y.to(patch_embeds.device)
        patch_embeds = self._apply_rotary_emb_2d(patch_embeds, grid_hw)
        assert (grid_hw[:, 0] * grid_hw[:, 1]).sum() == patch_embeds.shape[0]

        patches = []
        cur_position = 0
        for i in range(grid_hw.shape[0]):
            h, w = grid_hw[i]
            patches_per_img = (
                patch_embeds[cur_position : cur_position + h * w]
                .view(h, w, -1)
                .unsqueeze(0)
            )
            patches_per_img = self.dense_embedding(patches_per_img.permute(0, 3, 1, 2))
            patches_per_img = patches_per_img.permute(0, 2, 3, 1)
            patches.append(patches_per_img.view(-1, patches_per_img.shape[-1]))
            cur_position += h * w

        embeddings = torch.cat(patches, dim=0)
        assert cur_position == patch_embeds.shape[0]
        assert embeddings.shape[0] == int(
            patch_embeds.shape[0] / self.downsample_factor**2
        )
        return embeddings.view(batch_size, -1, self.llm_embed_dim)
