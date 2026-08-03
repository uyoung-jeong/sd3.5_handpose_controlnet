#!/usr/bin/env python
# coding=utf-8
"""Run the SD3.5 pose ControlNet on in-the-wild images.

Pipeline per image (README "Inference"):
  1. WiLoR YOLO detector finds hands (left/right class + box).
  2. WiLoR estimates 3D hand keypoints per detection, projected to 2D
     full-image coordinates (same math as WiLoR demo.py).
  3. Each hand is cropped to a square matching FreiHAND framing and rendered
     as the OpenPose-style skeleton map the ControlNet was trained on.
  4. StableDiffusion3ControlNetPipeline generates one image per hand.

Steps 1-2 need the WiLoR conda env, so this script re-invokes itself with that env's
python (`--stage wilor`, cwd = --wilor_root) and receives keypoints back via
`out_dir/poses.json`. Steps 3-4 run in the posectrl env. Both env paths are
flags (--wilor_root / --wilor_python, defaults in lib/config.py).

Left hands: the ControlNet is right-hand-only (FreiHAND), so left-hand
skeletons are mirrored to right-hand orientation before rendering and the
generated image is mirrored back, yielding a left hand in the input's
orientation. `pose/` holds the maps exactly as fed to the model; grids
display everything in input orientation.

The rendering contract (validated pixel-exact against training pose maps)
lives in lib/utils.render_pose_map + lib/config; the FreiHAND framing
constant is lib/config.HAND_SPAN_FRAC. The pipeline upscales the 224 maps to
--resolution bilinearly, exactly like the training dataloader did.

Example (defaults: samples/*.jpg, checkpoint-13000, one spare GPU):

python tools/infer_sd3_pose_controlnet.py --device cuda:0

    # pose maps only, no SD3.5 load:      --skip_generation
    # reuse existing out_dir/poses.json:  --skip_pose_extraction
    # pose over-/under-followed: sweep --controlnet_conditioning_scale in [0.6, 1.0]

Outputs: out_dir/{poses.json, pose/, images/, grids/} with per-hand stems
`<image>_h<i>`; grids are [input crop | pose | generated].
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for lib.*

from lib import config
from lib.utils import compose_grid, hand_crop_box, load_control_image, project_full_img, render_pose_map


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="SD3.5 pose ControlNet inference on in-the-wild images.")
    parser.add_argument("--img_dir", type=str, default=config.SAMPLES_DIR, help="Input image folder.")
    parser.add_argument(
        "--controlnet_path",
        type=str,
        default=config.CONTROLNET_PATH,
        help="Path to the trained SD3ControlNetModel (final output dir or checkpoint-N/controlnet).",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=config.BASE_MODEL,
        help="Base SD3.5 model id or path.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=config.INFER_OUT_DIR,
        help="Output directory (poses.json, pose/, images/, grids/).",
    )
    parser.add_argument(
        "--wilor_root",
        type=str,
        default=config.WILOR_ROOT,
        help="WiLoR repo root (contains pretrained_models/ and mano_data/).",
    )
    parser.add_argument(
        "--wilor_python",
        type=str,
        default=config.WILOR_PYTHON,
        help="Python binary of the WiLoR conda env (runs the --stage wilor subprocess).",
    )
    parser.add_argument("--resolution", type=int, default=512, help="Generation resolution; must match training.")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default="a photo of a right human hand")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0, help="Per-hand generator seed is `seed + hand_index`.")
    parser.add_argument(
        "--cpu_offload", action="store_true", help="Use enable_model_cpu_offload instead of pipe.to(device)."
    )
    parser.add_argument("--det_conf", type=float, default=0.3, help="WiLoR YOLO detection confidence threshold.")
    parser.add_argument(
        "--skip_pose_extraction", action="store_true", help="Reuse an existing out_dir/poses.json (skip WiLoR)."
    )
    parser.add_argument("--skip_generation", action="store_true", help="Only extract poses and render pose maps.")
    parser.add_argument("--stage", type=str, choices=["wilor"], default=None, help="Internal (subprocess use only).")
    args = parser.parse_args(input_args)
    if args.resolution % 16 != 0:
        # VAE factor 8 x transformer patch size 2 — odd latent sizes crash the pipeline.
        parser.error("--resolution must be divisible by 16.")
    return args


def run_wilor_stage(args):
    """Detect hands and estimate 2D keypoints; write out_dir/poses.json.

    Runs inside the WiLoR conda env (see main). load_wilor and the MANO config
    use cwd-relative paths, so this chdirs into --wilor_root first.
    """
    img_dir = os.path.abspath(args.img_dir)
    out_dir = os.path.abspath(args.out_dir)
    sys.path.insert(0, args.wilor_root)
    os.chdir(args.wilor_root)
    import cv2
    from ultralytics import YOLO
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.models import load_wilor
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full

    model, model_cfg = load_wilor(
        checkpoint_path="./pretrained_models/wilor_final.ckpt", cfg_path="./pretrained_models/model_config.yaml"
    )
    detector = YOLO("./pretrained_models/detector.pt")
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    detector = detector.to(device)

    img_paths = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in Path(img_dir).glob(ext))
    if not img_paths:
        raise FileNotFoundError(f"No .jpg/.jpeg/.png images found in {img_dir}")

    hands = []
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        det = detector(img_cv2, conf=args.det_conf, verbose=False)[0]
        boxes = det.boxes.xyxy.cpu().numpy()
        confs = det.boxes.conf.cpu().numpy()
        is_right = det.boxes.cls.cpu().numpy()  # {0: left, 1: right}
        if len(boxes) == 0:
            print(f"{img_path.name}: no hands detected")
            continue

        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, is_right, rescale_factor=2.0)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        hand_id = 0
        for batch in loader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = 2 * batch["right"] - 1
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = (
                cam_crop_to_full(pred_cam, batch["box_center"].float(), batch["box_size"].float(), img_size, scaled_focal_length)
                .detach()
                .cpu()
                .numpy()
            )

            for n in range(len(pred_cam_t_full)):
                joints = out["pred_keypoints_3d"][n].detach().cpu().numpy()
                right = batch["right"][n].cpu().numpy()
                joints[:, 0] = (2 * right - 1) * joints[:, 0]
                kpts_2d = project_full_img(
                    joints, pred_cam_t_full[n], float(scaled_focal_length), img_size[n].cpu().numpy()
                )
                hands.append(
                    {
                        "image": str(img_path.resolve()),
                        "hand_id": hand_id,
                        "is_right": float(right),
                        "conf": float(confs[hand_id]),
                        "kpts": kpts_2d.tolist(),
                    }
                )
                hand_id += 1
        print(f"{img_path.name}: {hand_id} hand(s)")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "poses.json"), "w") as f:
        json.dump(hands, f)
    print(f"Wrote {len(hands)} hand(s) to {os.path.join(out_dir, 'poses.json')}")


def build_conditions(hands, pose_dir):
    """Render one pose map per detected hand; return per-hand records."""
    records = []
    for h in hands:
        kpts = np.asarray(h["kpts"], dtype=float)
        x0, y0, side = hand_crop_box(kpts)
        canvas_kpts = (kpts - (x0, y0)) * (config.POSE_CANVAS / side)
        is_right = h["is_right"] >= 0.5
        if not is_right:
            canvas_kpts[:, 0] = config.POSE_CANVAS - canvas_kpts[:, 0]  # mirror to the trained right-hand domain
        stem = f"{os.path.splitext(os.path.basename(h['image']))[0]}_h{h['hand_id']}"
        pose_path = os.path.join(pose_dir, f"{stem}.png")
        Image.fromarray(render_pose_map(canvas_kpts)).save(pose_path)
        records.append(
            {"stem": stem, "image": h["image"], "crop_box": (x0, y0, side), "is_right": is_right, "pose_path": pose_path}
        )
    return records


def generate_images(args, records, images_dir):
    from lib.models import load_controlnet_pipeline

    pipe = load_controlnet_pipeline(args.controlnet_path, args.base_model, args.device, cpu_offload=args.cpu_offload)

    for start in tqdm(range(0, len(records), args.batch_size), desc="Generating"):
        batch = records[start : start + args.batch_size]
        control_images = [load_control_image(r["pose_path"], args.resolution) for r in batch]
        generators = [torch.Generator(device="cpu").manual_seed(args.seed + start + j) for j in range(len(batch))]
        images = pipe(
            prompt=[args.prompt] * len(batch),
            negative_prompt=[args.negative_prompt] * len(batch),
            control_image=control_images,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            generator=generators,
        ).images
        for r, image in zip(batch, images):
            if not r["is_right"]:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)  # back to the input's left-hand orientation
            image.save(os.path.join(images_dir, f"{r['stem']}.png"))


def compose_grids(args, records, images_dir, grids_dir):
    """[input crop | pose | generated] per hand, all in input orientation."""
    for r in records:
        x0, y0, side = r["crop_box"]
        crop = Image.open(r["image"]).convert("RGB").crop(
            (round(x0), round(y0), round(x0 + side), round(y0 + side))
        )
        pose = Image.open(r["pose_path"]).convert("RGB")
        gen = Image.open(os.path.join(images_dir, f"{r['stem']}.png")).convert("RGB")
        if not r["is_right"]:
            pose = pose.transpose(Image.FLIP_LEFT_RIGHT)  # generated image is already mirrored back
        compose_grid([crop, pose, gen], args.resolution).save(os.path.join(grids_dir, f"{r['stem']}.png"))


def main(args):
    if args.stage == "wilor":
        return run_wilor_stage(args)

    os.makedirs(args.out_dir, exist_ok=True)
    poses_path = os.path.join(args.out_dir, "poses.json")

    if not args.skip_pose_extraction:
        if not os.path.isfile(args.wilor_python):
            raise FileNotFoundError(f"WiLoR env python not found: {args.wilor_python} (pass --wilor_python).")
        print("Extracting hand poses with WiLoR (its own conda env)...")
        subprocess.run(
            [
                args.wilor_python, os.path.abspath(__file__), "--stage", "wilor",
                "--img_dir", os.path.abspath(args.img_dir),
                "--out_dir", os.path.abspath(args.out_dir),
                "--wilor_root", args.wilor_root,
                "--det_conf", str(args.det_conf),
                "--device", args.device,
            ],
            cwd=args.wilor_root,
            check=True,
        )

    if not os.path.isfile(poses_path):
        raise FileNotFoundError(f"{poses_path} not found (rerun without --skip_pose_extraction).")
    with open(poses_path) as f:
        hands = json.load(f)
    if not hands:
        print(f"No hands detected in {args.img_dir}; nothing to generate.")
        return

    pose_dir = os.path.join(args.out_dir, "pose")
    images_dir = os.path.join(args.out_dir, "images")
    grids_dir = os.path.join(args.out_dir, "grids")
    for d in (pose_dir, images_dir, grids_dir):
        os.makedirs(d, exist_ok=True)

    records = build_conditions(hands, pose_dir)
    n_left = sum(not r["is_right"] for r in records)
    print(f"Rendered {len(records)} pose map(s) to {pose_dir} ({n_left} left hand(s) mirrored)")
    if args.skip_generation:
        return

    generate_images(args, records, images_dir)
    compose_grids(args, records, images_dir, grids_dir)
    print(f"Done: {len(records)} image(s) in {images_dir}, grids in {grids_dir}")


if __name__ == "__main__":
    main(parse_args())
