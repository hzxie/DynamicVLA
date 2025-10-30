# -*- coding: utf-8 -*-
#
# @File:   modeling_neo.py
# @Author: Haozhe Xie
# @Date:   2025-10-22 09:51:08
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-10-29 16:58:53
# @Email:  root@haozhexie.com

import copy
import typing

import torch
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel, Qwen3Config
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    ModelOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


def get_patch_abs_pos(
    grid_hw: torch.Tensor,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    B = grid_hw.shape[0]
    # Get the number of patches per image
    H, W = grid_hw[:, 0], grid_hw[:, 1]
    N = H * W
    # Create the batch index for each patch (B x patch count)
    patch_to_sample = torch.repeat_interleave(torch.arange(B, device=grid_hw.device), N)
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


class NeoTextConfig(Qwen3Config):
    model_type = "neo_text"

    def __init__(
        self,
        rope_theta_hw: float = 10000.0,
        max_position_embeddings_hw: int = 10000,
        **kwargs,
    ):
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw
        # HACK: reinitialize layer_types in Qwen3Config
        kwargs["layer_types"] = None
        super().__init__(**kwargs)


class NeoVLMConfig(PretrainedConfig):
    model_type = "neo_vlm"
    sub_configs = {"text_config": NeoTextConfig, "vision_config": NeoVisionConfig}

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

        # NeoTextConfig
        if text_config is None:
            self.text_config = NeoTextConfig()
        elif isinstance(text_config, dict):
            self.text_config = NeoTextConfig(**text_config)
        elif isinstance(text_config, NeoTextConfig):
            self.text_config = text_config
        else:
            raise ValueError("No valid text_config is provided.")

        self.use_cache = use_cache
        self.downsample_ratio = downsample_ratio
        self.tie_word_embeddings = tie_word_embeddings
        if max_position_embeddings is not None:
            self.text_config.max_position_embeddings = max_position_embeddings

        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


class NeoAttention(torch.nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = torch.nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.q_proj_hw = torch.nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = torch.nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj_hw = torch.nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = torch.nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = torch.nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_norm_h = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.q_norm_w = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_h = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.k_norm_w = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)

        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        hw_config = copy.deepcopy(config)
        hw_config.head_dim = config.head_dim // 2
        hw_config.rope_theta = config.rope_theta_hw
        hw_config.max_position_embeddings = config.max_position_embeddings_hw
        self.rotary_emb_hw = Qwen3RotaryEmbedding(config=hw_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        indexes: typing.Optional[torch.LongTensor],
        attention_mask: typing.Optional[torch.Tensor],
        past_key_values: typing.Optional[Cache] = None,
        cache_position: typing.Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, typing.Optional[torch.Tensor]]:
        assert self.config._attn_implementation == "eager"
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states_t = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        query_states_h, query_states_w = (
            self.q_proj_hw(hidden_states)
            .view(hidden_shape)
            .transpose(1, 2)
            .chunk(2, dim=-1)
        )
        query_states_h, query_states_w = self.q_norm_h(query_states_h), self.q_norm_w(
            query_states_w
        )

        key_states_t = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states_h, key_states_w = (
            self.k_proj_hw(hidden_states)
            .view(hidden_shape)
            .transpose(1, 2)
            .chunk(2, dim=-1)
        )
        key_states_h, key_states_w = self.k_norm_h(key_states_h), self.k_norm_w(
            key_states_w
        )

        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
        query_states_t, key_states_t = apply_rotary_pos_emb(
            query_states_t, key_states_t, cos_t, sin_t
        )

        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
        query_states_h, key_states_h = apply_rotary_pos_emb(
            query_states_h, key_states_h, cos_h, sin_h
        )

        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
        query_states_w, key_states_w = apply_rotary_pos_emb(
            query_states_w, key_states_w, cos_w, sin_w
        )

        query_states = torch.cat(
            [query_states_t, query_states_h, query_states_w], dim=-1
        )
        key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs=None
            )

        attention_interface: typing.Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class NeoDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeoAttention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        indexes: typing.Optional[torch.LongTensor] = None,
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_ids: typing.Optional[torch.LongTensor] = None,
        past_key_values: typing.Optional[Cache] = None,
        use_cache: typing.Optional[bool] = False,
        cache_position: typing.Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class NeoVLMPreTrainedModel(PreTrainedModel):
    config: NeoVLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NeoDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": NeoDecoderLayer,
        "attentions": NeoAttention,
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
        indexes: typing.Optional[torch.LongTensor] = None,
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
        if indexes is None:
            indexes = self.model.get_thw_indexes(input_ids, pixel_values)

        outputs = self.model(
            indexes=indexes,
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
        self.text_model = NeoTextModel(config.text_config)
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

    def get_thw_indexes(
        self, input_ids: torch.LongTensor, pixel_values: torch.FloatTensor
    ) -> torch.LongTensor:
        grid_hw = self.vision_model.get_grid_hw(pixel_values=pixel_values)
        img_start_shift = torch.cat(
            [
                torch.zeros(1, dtype=torch.long).to(input_ids.device),
                (input_ids == self.img_start_token_id).long(),
            ],
            dim=0,
        )[:-1]

        not_img_token = (input_ids != self.img_context_token_id).long()
        t_indexes = (img_start_shift + not_img_token).cumsum(0) - 1
        h_indexes = torch.zeros_like(t_indexes).to(t_indexes.device)
        w_indexes = torch.zeros_like(t_indexes).to(t_indexes.device)

        selected = input_ids == self.img_context_token_id
        if selected.long().sum() > 0:
            abs_pos_w, abs_pos_h = get_patch_abs_pos(
                grid_hw // int(1 / self.downsample_ratio)
            )
            h_indexes[selected] = abs_pos_h.to(t_indexes.device, t_indexes.dtype)
            w_indexes[selected] = abs_pos_w.to(t_indexes.device, t_indexes.dtype)

        return torch.stack([t_indexes, h_indexes, w_indexes], dim=0)


class NeoVisionModel(PreTrainedModel):
    config_class = NeoVisionConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(self, config: NeoVisionConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = NeoVisionEmbeddings(config)

    def get_grid_hw(
        self, pixel_values: torch.FloatTensor = None, pixel_size: torch.Size = None
    ) -> torch.Tensor:
        if pixel_size is not None:
            assert (
                pixel_values.ndim >= 4
            ), "pixel_values should have shape (batch, channels, height, width) or higher."
            return torch.tensor(
                [
                    [
                        pixel_values.size(-2) // self.config.patch_size,
                        pixel_values.size(-1) // self.config.patch_size,
                    ]
                ],
                device=pixel_values.device,
            ).repeat(pixel_values.size(0), 1)
        elif pixel_values is not None:
            batch_size = pixel_values.size(0)
            img_height = pixel_values.size(-2)
            img_width = pixel_values.size(-1)
            patch_size = self.config.patch_size
            h_patches = img_height // patch_size
            w_patches = img_width // patch_size
            return torch.tensor(
                [[h_patches, w_patches]], device=pixel_values.device
            ).repeat(batch_size, 1)
        else:
            raise ValueError("Either pixel_values or pixel_size must be provided.")

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

        if pixel_embeddings is not None:
            hidden_states = pixel_embeddings
        else:
            if grid_hw is None:
                grid_hw = self.get_grid_hw(pixel_values=pixel_values)
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
        abs_pos_x, abs_pos_y = get_patch_abs_pos(grid_hw)
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


class NeoTextModel(PreTrainedModel):
    config_class = NeoTextConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(self, config: NeoTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = torch.nn.ModuleList(
            [
                NeoDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.current_index = -1
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        indexes: typing.Optional[torch.LongTensor] = None,
        input_ids: typing.Optional[torch.LongTensor] = None,
        attention_mask: typing.Optional[torch.Tensor] = None,
        position_ids: typing.Optional[torch.LongTensor] = None,
        past_key_values: typing.Optional[Cache] = None,
        inputs_embeds: typing.Optional[torch.FloatTensor] = None,
        use_cache: typing.Optional[bool] = None,
        cache_position: typing.Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        assert position_ids is not None
        assert cache_position is not None
        assert past_key_values is not None
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            if input_ids is not None:
                mask_kwargs = {
                    "config": self.config,
                    "input_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                    "position_ids": position_ids,
                }
                # Create the masks
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                self.current_index += 1
                indexes = torch.LongTensor([[self.current_index], [0], [0]]).to(
                    input_ids.device
                )
            else:
                causal_mask_mapping = {
                    "full_attention": self._get_block_causal_mask(indexes[0]),
                }
                self.current_index = indexes[0].max()
        else:
            raise NotImplementedError(
                "not isinstance(causal_mask_mapping := attention_mask, dict)"
            )

        hidden_states = inputs_embeds
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                indexes=indexes,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def _get_block_causal_mask(self, index: torch.Tensor) -> torch.Tensor:
        """
        index: (L)
        return: (1, 1, L, L) block-wise causal attention mask
        """
        L = index.size(0)
        idx_i = index.unsqueeze(1).expand(L, L)
        idx_j = index.unsqueeze(0).expand(L, L)

        arange = torch.arange(L, device=index.device)
        mask = (idx_j == idx_i) | (arange.unsqueeze(0) <= arange.unsqueeze(1))

        return torch.where(
            mask[None, None, :, :] > 0, torch.tensor(0.0), torch.tensor(float("-inf"))
        )
