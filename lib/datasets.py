#!/usr/bin/env python
# coding=utf-8
"""FreiHAND dataset pieces: training dataset (train) and pose listing (eval)."""

import os
import random

import torch
from PIL import Image
from torchvision import transforms

from lib.config import CAPTIONS


class FreiHANDPoseDataset(torch.utils.data.Dataset):
    """FreiHAND training split: RGB targets + OpenPose-style hand skeleton conditioning.

    Layout under `root`:
        training/rgb/%08d.jpg   i in [0, 130240)  (224x224 RGB photos)
        training/pose/%08d.png  i in [0, 32560)   (224x224 skeleton on black background)

    The 130240 rgb frames are 4 renderings (different backgrounds) of 32560 unique poses,
    so the conditioning image for rgb index i is pose index i % 32560 (pixel-exact validated).
    """

    NUM_RGB = 130240
    NUM_POSES = 32560

    def __init__(self, root, resolution=512, proportion_empty_prompts=0.5, max_train_samples=None):
        self.root = root
        self.rgb_dir = os.path.join(root, "training", "rgb")
        self.pose_dir = os.path.join(root, "training", "pose")
        self.proportion_empty_prompts = proportion_empty_prompts
        self._rng = None  # created lazily, once per dataloader worker

        first_rgb = os.path.join(self.rgb_dir, "00000000.jpg")
        first_pose = os.path.join(self.pose_dir, "00000000.png")
        if not (os.path.isfile(first_rgb) and os.path.isfile(first_pose)):
            raise FileNotFoundError(
                f"FreiHAND training data not found under {root} "
                f"(expected {first_rgb} and {first_pose}). Pass --freihand_root."
            )

        # Deterministic first-N subsample (debugging).
        self.num_samples = self.NUM_RGB if max_train_samples is None else min(max_train_samples, self.NUM_RGB)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        # NOTE: Normalize([0.5], [0.5]) on the conditioning image is REQUIRED (the upstream
        # v0.37.0 example feeds [0, 1] here). At inference StableDiffusion3ControlNetPipeline
        # preprocesses the control image with VaeImageProcessor (-> [-1, 1]) before VAE-encoding
        # it, so training must feed the VAE the same range.
        self.conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def _get_rng(self):
        # Per-worker RNG: dataloader workers are forked, so the parent's global `random`
        # state would be copied and every worker would emit identical caption streams.
        if self._rng is None:
            worker_info = torch.utils.data.get_worker_info()
            seed = torch.initial_seed() if worker_info is None else worker_info.seed
            self._rng = random.Random(seed % 2**32)
        return self._rng

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.rgb_dir, f"{idx:08d}.jpg")).convert("RGB")
        # i % 32560: each unique pose was rendered over 4 backgrounds (see class docstring).
        pose = Image.open(os.path.join(self.pose_dir, f"{idx % self.NUM_POSES:08d}.png")).convert("RGB")

        rng = self._get_rng()
        if rng.random() < self.proportion_empty_prompts:
            caption_idx = 0  # CAPTIONS[0] == "" (unconditional)
        else:
            caption_idx = rng.randrange(1, len(CAPTIONS))

        return {
            "pixel_values": self.image_transforms(image),
            "conditioning_pixel_values": self.conditioning_image_transforms(pose),
            "caption_idx": caption_idx,
        }


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    caption_idx = torch.tensor([example["caption_idx"] for example in examples], dtype=torch.long)

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "caption_idx": caption_idx,
    }


def list_eval_pose_paths(freihand_root):
    """Sorted absolute paths of evaluation/pose/%08d.png (index i == filename stem)."""
    pose_dir = os.path.join(freihand_root, "evaluation", "pose")
    if not os.path.isdir(pose_dir):
        raise FileNotFoundError(f"Evaluation pose directory not found: {pose_dir}")
    names = sorted(n for n in os.listdir(pose_dir) if n.endswith(".png"))
    return [os.path.join(pose_dir, n) for n in names]
