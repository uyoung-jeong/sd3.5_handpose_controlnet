#!/usr/bin/env python
# coding=utf-8
"""Model construction/loading helpers.

SD3 text-encoder loading and prompt encoding are copied from the diffusers
dreambooth sd3 example; the ControlNet builder carries the
force_zeros_for_pooled_projection train<->inference contract fix.

posectrl-env only (imports transformers + diffusers at module level) — do not
import from the WiLoR stage of tools/infer_sd3_pose_controlnet.py.
"""

import torch
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

from diffusers import SD3ControlNetModel, StableDiffusion3ControlNetPipeline


# Copied from dreambooth sd3 example
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_text_encoders_and_tokenizers(pretrained_model_name_or_path, revision=None, variant=None):
    """Load SD3's [CLIP, CLIP, T5] tokenizers and text encoders (on CPU, fp32)."""
    tokenizers = [
        CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", revision=revision),
        CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2", revision=revision),
        T5TokenizerFast.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_3", revision=revision),
    ]
    text_encoders = []
    for subfolder in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision, subfolder=subfolder)
        text_encoders.append(
            cls.from_pretrained(
                pretrained_model_name_or_path, subfolder=subfolder, revision=revision, variant=variant
            )
        )
    return tokenizers, text_encoders


# Copied from dreambooth sd3 example
def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


# Copied from dreambooth sd3 example
def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


# Copied from dreambooth sd3 example
def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def build_pose_controlnet(transformer, num_layers, pretrained_path, logger):
    """Create the trainable SD3ControlNetModel and pin the train<->inference contract.

    Warm-starts from `pretrained_path` if given, otherwise initializes
    `num_layers` blocks from the frozen transformer.
    """
    if num_layers >= transformer.config.num_layers:
        # The last transformer block is context_pre_only and would copy incorrectly through
        # from_transformer's strict=False state-dict load.
        raise ValueError(
            f"--num_controlnet_layers must be < the transformer's num_layers ({transformer.config.num_layers})."
        )

    if pretrained_path:
        logger.info("Loading existing controlnet weights")
        controlnet = SD3ControlNetModel.from_pretrained(pretrained_path)
        if controlnet.config.force_zeros_for_pooled_projection:
            # Warm-starting from an InstantX-convention checkpoint (trained with zero pooled
            # projections and no shift subtraction): this script trains with the opposite
            # convention, so the first steps run off-distribution for the loaded weights.
            logger.warning(
                "Loaded controlnet was trained with force_zeros_for_pooled_projection=True;"
                " this script re-trains it under the False convention (real pooled embeds,"
                " shift-subtracted control latents)."
            )
    else:
        logger.info(f"Initializing a {num_layers}-layer controlnet from the transformer")
        # num_extra_conditioning_channels=0 must be passed explicitly: the classmethod default
        # is 1, which changes pos_embed_input to 17 in-channels and breaks the 16-channel
        # VAE-latent conditioning used here.
        # Side effect: from_transformer mutates the passed transformer's config *dict* in
        # place (num_layers etc.); harmless here because the transformer config is never
        # saved or re-read after this point — do not add code that does.
        controlnet = SD3ControlNetModel.from_transformer(
            transformer,
            num_layers=num_layers,
            num_extra_conditioning_channels=0,
        )

    # ------------------------------------------------------------------
    # CRITICAL train<->inference contract fix.
    # from_transformer leaves force_zeros_for_pooled_projection=True (the InstantX convention)
    # in the controlnet config. With that flag, StableDiffusion3ControlNetPipeline
    # (1) does NOT subtract the VAE shift_factor from the control-image latent and
    # (2) passes ZEROED pooled projections to the controlnet — the exact opposite of what
    # the training loop does (shift subtraction + real pooled embeds, as upstream).
    # Setting the flag to False makes the pipeline branch match training exactly.
    # register_to_config mutates the config that save_pretrained writes, so the flag persists
    # into every saved config.json (final save and the accelerate checkpoint hooks).
    # ------------------------------------------------------------------
    controlnet.register_to_config(force_zeros_for_pooled_projection=False)

    controlnet_num_params = sum(p.numel() for p in controlnet.parameters())
    logger.info(f"ControlNet has {controlnet_num_params:,} parameters ({controlnet_num_params / 1e9:.2f}B)")
    return controlnet


def load_controlnet_pipeline(controlnet_path, base_model, device, cpu_offload=False):
    """SD3.5 + trained pose ControlNet inference pipeline (bf16, progress bar off)."""
    controlnet = SD3ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.bfloat16)
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe
