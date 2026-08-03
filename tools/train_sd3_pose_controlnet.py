#!/usr/bin/env python
# coding=utf-8
"""SD3.5-Large pose ControlNet training on FreiHAND.

Adapted from diffusers v0.37.0 `examples/controlnet/train_controlnet_sd3.py`
(Apache-2.0, Copyright 2025 The HuggingFace Inc. team).

Machine-local default paths defined in lib/config.py.

### Single-GPU debug run:
python /share_home/uyoung/gen/sd3_5_pose_controlnet/tools/train_sd3_pose_controlnet.py \
    --max_train_samples 64 \
    --max_train_steps 8 \
    --checkpointing_steps 4 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --use_8bit_adam

### Full 4-GPU training (4x RTX A6000 48GB):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --num_processes 4 --mixed_precision bf16 \
    /share_home/uyoung/gen/sd3_5_pose_controlnet/tools/train_sd3_pose_controlnet.py \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --checkpoints_total_limit 6

VRAM guidance: --gradient_checkpointing and --use_8bit_adam are required to fit 48 GB
(fp32 Adam adds ~15 GiB, no checkpointing adds ~7-10 GiB activations — either alone OOMs).
The default 6-layer ControlNet (1.34B params — larger relative capacity than the InstantX
~0.6B / alimama ~0.85B SD3-Medium precedents) fits at resolution 512 / batch 1.
--num_controlnet_layers 12 was MEASURED to OOM on 48 GB A6000s.

Disk: each checkpoint is ~16 GB (fp32 controlnet + 8-bit Adam state).
Keep the limit >= 6 so the 12k-17k collapse window stays covered by checkpoints.
"""

import argparse
import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path

import accelerate
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3ControlNetModel,
    SD3Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import free_memory
from diffusers.utils import check_min_version
from diffusers.utils.torch_utils import is_compiled_module

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for lib.*

from lib import config
from lib.datasets import FreiHANDPoseDataset, collate_fn
from lib.loss import SD3FlowMatchLoss
from lib.models import build_pose_controlnet, encode_prompt, load_text_encoders_and_tokenizers


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.37.0")

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="SD3.5-Large pose ControlNet training on FreiHAND.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=config.BASE_MODEL,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from the transformer.",
    )
    parser.add_argument(
        "--num_controlnet_layers",
        type=int,
        default=6,
        help="Number of transformer blocks in the controlnet (must be < the base transformer's 38 layers)."
        " Default 6 (1.34B params) fits 48GB GPUs; 12 measured to OOM on 48GB — needs 80GB-class.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.TRAIN_OUTPUT_DIR,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=15000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training"
            " via `--resume_from_checkpoint`, and `checkpoint-N/controlnet` can be evaluated directly with"
            " tools/eval_sd3_pose_controlnet.py."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass."
        " Enabled on BOTH the controlnet and the frozen transformer.",
    )
    parser.add_argument(
        "--upcast_vae",
        action="store_true",
        help="Whether or not to upcast vae to fp32",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--freihand_root",
        type=str,
        default=config.FREIHAND_ROOT,
        help="Root of the FreiHAND dataset (contains training/rgb and training/pose).",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Debugging only: deterministically truncate to the first N samples. Note N <= 32560"
        " draws exclusively from the first (green-screen) background set.",
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0.5,
        help="Proportion of image prompts to be replaced with the empty string (classifier-free guidance dropout).",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=256,
        # 256 (not upstream-training's 77) so the [N, 77+256=333, 4096] prompt embeddings
        # match what StableDiffusion3ControlNetPipeline.encode_prompt produces by default at inference.
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sd3_pose_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.num_controlnet_layers < 1:
        raise ValueError("`--num_controlnet_layers` must be >= 1.")

    if args.resolution % 16 != 0:
        # VAE downsamples by 8 and the transformer patchifies by 2: a resolution that is
        # divisible by 8 but not 16 produces an odd latent size, and unpatchify emits a
        # different shape than the target -> opaque broadcast crash in the loss.
        raise ValueError("`--resolution` must be divisible by 16 (VAE factor 8 x patch size 2).")

    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # gradient_as_bucket_view avoids DDP's duplicate flat gradient buffers (~10 GiB for the
    # 12-layer fp32 controlnet — the difference between fitting and OOM on 48 GB cards);
    # broadcast_buffers=False stops re-broadcasting the controlnet's large fp32 sincos
    # pos_embed buffer every forward (it is constant).
    kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True, gradient_as_bucket_view=True, broadcast_buffers=False
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now. device_specific=True decorrelates the
    # flow-matching timestep/noise draws across DDP ranks (identical seeds would make the
    # 4 ranks sample identical (timestep, noise) pairs every micro-step).
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ------------------------------------------------------------------
    # Precompute the fixed caption pool embeddings.
    # This replaces the upstream tokenize/`.map` machinery: EVERY rank loads the tokenizers
    # + 3 text encoders on its own device, encodes the small config.CAPTIONS pool once, keeps
    # the resulting tensors on device, and frees the encoders before the diffusion models load.
    # ------------------------------------------------------------------
    tokenizers, text_encoders = load_text_encoders_and_tokenizers(
        args.pretrained_model_name_or_path, args.revision, args.variant
    )
    for text_encoder in text_encoders:
        text_encoder.requires_grad_(False)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    with torch.no_grad():
        pool_prompt_embeds, pool_pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, list(config.CAPTIONS), args.max_sequence_length, device=accelerator.device
        )
    pool_prompt_embeds = pool_prompt_embeds.to(dtype=weight_dtype)
    pool_pooled_prompt_embeds = pool_pooled_prompt_embeds.to(dtype=weight_dtype)
    logger.info(
        f"Precomputed caption pool embeddings: prompt_embeds {tuple(pool_prompt_embeds.shape)},"
        f" pooled {tuple(pool_pooled_prompt_embeds.shape)}"
    )

    del text_encoders, tokenizers, text_encoder
    free_memory()

    # Load scheduler and models. The scheduler is only consumed by the flow-matching
    # objective (which keeps its own deepcopy).
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    fm_loss = SD3FlowMatchLoss(
        noise_scheduler,
        weighting_scheme=args.weighting_scheme,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        mode_scale=args.mode_scale,
        precondition_outputs=bool(args.precondition_outputs),
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    # torch_dtype=weight_dtype halves the transient host RAM per rank during loading (the
    # on-disk weights are bf16; the default would upcast to fp32 on CPU: ~32 GB x num ranks).
    # from_transformer still builds the controlnet in fp32 (from_config default) and
    # load_state_dict upcasts the copied weights, so trainable precision is unaffected.
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    controlnet = build_pose_controlnet(
        transformer, args.num_controlnet_layers, args.controlnet_model_name_or_path, logger
    )

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    controlnet.train()

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = SD3ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        # Deviation from upstream (which only checkpoints the controlnet): gradients flow
        # through the frozen 38-layer transformer, so its activations dominate memory.
        transformer.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Move vae and transformer to device and cast to weight_dtype
    if args.upcast_vae:
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    train_dataset = FreiHANDPoseDataset(
        root=args.freihand_root,
        resolution=args.resolution,
        proportion_empty_prompts=args.proportion_empty_prompts,
        max_train_samples=args.max_train_samples,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                # Sample noise and density-weighted timesteps, then add noise
                # according to flow matching (zt = (1 - texp) * x + texp * z1).
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                timesteps = fm_loss.sample_timesteps(bsz, model_input.device)
                sigmas = fm_loss.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = fm_loss.add_noise(model_input, noise, sigmas)

                # Get the text embedding for conditioning from the precomputed pool
                caption_idx = batch["caption_idx"].to(pool_prompt_embeds.device)
                prompt_embeds = pool_prompt_embeds[caption_idx].to(dtype=weight_dtype)
                pooled_prompt_embeds = pool_pooled_prompt_embeds[caption_idx].to(dtype=weight_dtype)

                # controlnet(s) inference
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=vae.dtype)
                controlnet_image = vae.encode(controlnet_image).latent_dist.sample()
                # shift_factor IS subtracted here; matches the pipeline branch selected by
                # force_zeros_for_pooled_projection=False (see lib/models.build_pose_controlnet).
                controlnet_image = (controlnet_image - vae.config.shift_factor) * vae.config.scaling_factor
                controlnet_image = controlnet_image.to(dtype=weight_dtype)

                control_block_res_samples = controlnet(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )[0]
                control_block_res_samples = [sample.to(dtype=weight_dtype) for sample in control_block_res_samples]

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=control_block_res_samples,
                    return_dict=False,
                )[0]

                # Flow-matching loss (preconditioning + weighting live in lib/loss.py).
                loss = fm_loss(model_pred, model_input, noise, noisy_model_input, sigmas)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the trained controlnet.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)

        # Sanity check: the inference-contract flag must persist into the saved config
        # (the pipeline reads it to decide shift-factor subtraction and pooled zeroing).
        with open(os.path.join(args.output_dir, "config.json")) as f:
            saved_config = json.load(f)
        if saved_config.get("force_zeros_for_pooled_projection") is not False:
            raise RuntimeError(
                "force_zeros_for_pooled_projection was not persisted as False in the saved config.json;"
                " inference with StableDiffusion3ControlNetPipeline would mismatch training."
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
