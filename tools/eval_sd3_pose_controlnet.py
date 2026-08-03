#!/usr/bin/env python
# coding=utf-8
"""Evaluate an SD3.5-Large pose ControlNet on the FreiHAND evaluation split.

Generates images from the `evaluation/pose/%08d.png` conditioning maps.
Evaluates both the generated images and the matching ground-truth images.

We use WiLoR hand detector: mean max-confidence, detection rate @0.3, and the
right-class fraction among best detections. Results go to `out_dir/metrics.json`.

Example (single spare GPU, 200 samples):

python tools/eval_sd3_pose_controlnet.py \
    --controlnet_path /share_home/uyoung/gen/sd3_5_pose_controlnet/outputs/sd35_large_pose_cn \
    --out_dir /share_home/uyoung/gen/sd3_5_pose_controlnet/outputs/sd35_large_pose_cn/eval \
    --device cuda:0

    # a mid-training checkpoint works too:
    #   --controlnet_path .../outputs/sd35_large_pose_cn/checkpoint-5000/controlnet

Notes:
    * If the generations misalign, sweep `--controlnet_conditioning_scale` in [0.6, 1.0].
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for lib.*

from lib import config
from lib.datasets import list_eval_pose_paths
from lib.utils import compose_grid, load_control_image, run_detector, summarize_detections


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Evaluate an SD3.5 pose ControlNet on FreiHAND evaluation poses.")
    parser.add_argument(
        "--controlnet_path",
        type=str,
        required=True,
        help="Path to the trained SD3ControlNetModel (final output dir or checkpoint-N/controlnet).",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=config.BASE_MODEL,
        help="Base SD3.5 model id or path.",
    )
    parser.add_argument("--freihand_root", type=str, default=config.FREIHAND_ROOT, help="FreiHAND dataset root.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory (images/, grids/, metrics.json).")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of evaluation poses to use (first N).")
    parser.add_argument("--all", action="store_true", help="Use all evaluation poses (3960); overrides --num_samples.")
    parser.add_argument("--resolution", type=int, default=512, help="Generation resolution; must match training.")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default="a photo of a right human hand")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0, help="Per-image generator seed is `seed + image_index`.")
    parser.add_argument(
        "--cpu_offload", action="store_true", help="Use enable_model_cpu_offload instead of pipe.to(device)."
    )
    parser.add_argument("--save_grids", type=int, default=16, help="Number of [rgb | pose | generated] grids to save.")
    parser.add_argument(
        "--skip_generation", action="store_true", help="Skip generation; compute metrics over existing out_dir/images."
    )
    parser.add_argument("--skip_metrics", action="store_true", help="Skip the WiLoR detector metric.")
    parser.add_argument(
        "--detector_path", type=str, default=config.WILOR_DETECTOR_PATH, help="WiLoR YOLO hand detector weights."
    )
    args = parser.parse_args(input_args)
    if args.resolution % 16 != 0:
        # VAE factor 8 x transformer patch size 2 — odd latent sizes crash the pipeline.
        parser.error("--resolution must be divisible by 16.")
    return args


def generate_images(args, pose_paths, images_dir):
    from lib.models import load_controlnet_pipeline

    pipe = load_controlnet_pipeline(args.controlnet_path, args.base_model, args.device, cpu_offload=args.cpu_offload)

    for start in tqdm(range(0, len(pose_paths), args.batch_size), desc="Generating"):
        batch_paths = pose_paths[start : start + args.batch_size]
        indices = [int(os.path.splitext(os.path.basename(p))[0]) for p in batch_paths]
        control_images = [load_control_image(p, args.resolution) for p in batch_paths]
        # One CPU generator per image seeded seed+idx: results are reproducible
        # and independent of batch size / batching order.
        generators = [torch.Generator(device="cpu").manual_seed(args.seed + idx) for idx in indices]
        images = pipe(
            prompt=[args.prompt] * len(batch_paths),
            negative_prompt=[args.negative_prompt] * len(batch_paths),
            control_image=control_images,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            generator=generators,
        ).images
        for idx, image in zip(indices, images):
            image.save(os.path.join(images_dir, f"{idx:08d}.png"))

    # Free pipeline VRAM before the YOLO detector is loaded.
    del pipe
    from diffusers.training_utils import free_memory

    free_memory()


def main(args):
    images_dir = os.path.join(args.out_dir, "images")
    grids_dir = os.path.join(args.out_dir, "grids")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)

    pose_paths = list_eval_pose_paths(args.freihand_root)
    num_samples = len(pose_paths) if args.all else min(args.num_samples, len(pose_paths))
    pose_paths = pose_paths[:num_samples]

    if not args.skip_generation:
        generate_images(args, pose_paths, images_dir)

    # Generated images actually on disk, restricted to the poses selected THIS run —
    # stale pngs from an earlier larger run (or a different checkpoint) must not leak
    # into grids/metrics. --skip_generation still works over a pre-existing out_dir/images.
    expected_stems = {os.path.splitext(os.path.basename(p))[0] for p in pose_paths}
    all_names = sorted(n for n in os.listdir(images_dir) if n.endswith(".png"))
    gen_names = [n for n in all_names if os.path.splitext(n)[0] in expected_stems]
    if len(gen_names) < len(all_names):
        print(
            f"WARNING: ignoring {len(all_names) - len(gen_names)} images in {images_dir} "
            f"outside the {len(expected_stems)} selected poses (stale from an earlier run?)."
        )
    if len(gen_names) < len(expected_stems):
        print(f"WARNING: only {len(gen_names)}/{len(expected_stems)} selected poses have generated images.")
    gen_paths = [os.path.join(images_dir, n) for n in gen_names]
    if not gen_paths:
        raise FileNotFoundError(f"No generated images found in {images_dir} (did you mean to drop --skip_generation?).")
    rgb_dir = os.path.join(args.freihand_root, "evaluation", "rgb")
    pose_dir = os.path.join(args.freihand_root, "evaluation", "pose")

    num_grids = min(args.save_grids, len(gen_paths))
    for gen_path in gen_paths[:num_grids]:
        stem = os.path.splitext(os.path.basename(gen_path))[0]
        grid = compose_grid(
            [os.path.join(rgb_dir, f"{stem}.jpg"), os.path.join(pose_dir, f"{stem}.png"), gen_path],
            args.resolution,
        )
        grid.save(os.path.join(grids_dir, f"{stem}.png"))
    if num_grids:
        print(f"Saved {num_grids} grids to {grids_dir}")

    if args.skip_metrics:
        return

    gt_paths = [os.path.join(rgb_dir, os.path.splitext(os.path.basename(p))[0] + ".jpg") for p in gen_paths]
    print(f"Running WiLoR detector on {len(gen_paths)} generated images...")
    gen_summary = summarize_detections(run_detector(gen_paths, args.detector_path, args.device))
    print(f"Running WiLoR detector on {len(gt_paths)} GT rgb images (baseline)...")
    baseline_summary = summarize_detections(run_detector(gt_paths, args.detector_path, args.device))

    metrics = dict(gen_summary)
    metrics.update({f"baseline_{k}": v for k, v in baseline_summary.items()})
    metrics["args"] = vars(args)
    metrics["controlnet_path"] = args.controlnet_path
    metrics["timestamp"] = datetime.now().isoformat(timespec="seconds")
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'metric':<16}{'generated':>12}{'gt-baseline':>14}")
    for key in ("n", "mean_max_conf", "det_rate@0.3", "right_frac"):
        gen_v, base_v = gen_summary[key], baseline_summary[key]
        if key == "n":
            print(f"{key:<16}{gen_v:>12d}{base_v:>14d}")
        else:
            print(f"{key:<16}{gen_v:>12.4f}{base_v:>14.4f}")
    print(f"\nWrote {metrics_path}")


if __name__ == "__main__":
    main(parse_args())
