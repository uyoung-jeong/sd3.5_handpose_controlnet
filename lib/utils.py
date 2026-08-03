#!/usr/bin/env python
# coding=utf-8
"""Image, pose-rendering and hand-detection utilities shared by the tools.

Import note: this module must stay importable from BOTH the posectrl env and
the WiLoR env (tools/infer_sd3_pose_controlnet.py --stage wilor) — keep it
free of top-level diffusers/transformers/torch imports (ultralytics is
imported lazily inside run_detector).
"""

import os

import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb
from PIL import Image

from lib.config import HAND_EDGES, HAND_SPAN_FRAC, POSE_CANVAS, RIGHT_CLASS_ID


def load_control_image(path, resolution):
    """Conditioning image exactly as the training dataloader saw it (bilinear resize)."""
    return Image.open(path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)


def compose_grid(panels, resolution):
    """Horizontal grid; each panel (a PIL.Image or a path) resized to resolution^2."""
    images = [
        Image.open(p).convert("RGB") if isinstance(p, (str, os.PathLike)) else p.convert("RGB") for p in panels
    ]
    images = [im.resize((resolution, resolution), Image.BILINEAR) for im in images]
    grid = Image.new("RGB", (resolution * len(images), resolution))
    for i, im in enumerate(images):
        grid.paste(im, (i * resolution, 0))
    return grid


def run_detector(image_paths, detector_path, device):
    """Run the WiLoR hand detector; per image return (max box confidence, class of best box or None)."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics is required for the WiLoR detection metric but could not be imported. "
            "Activate the posectrl env (`conda activate posectrl`, ships ultralytics 8.3.116) "
            "or rerun with --skip_metrics."
        ) from e
    if not os.path.isfile(detector_path):
        raise FileNotFoundError(f"WiLoR detector weights not found: {detector_path} (or pass --detector_path).")
    model = YOLO(detector_path)
    results = model.predict(
        source=list(image_paths), imgsz=640, conf=0.001, device=device, save=False, stream=True, verbose=False
    )
    per_image = []
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            best = int(r.boxes.conf.argmax())
            per_image.append((float(r.boxes.conf[best]), int(r.boxes.cls[best])))
        else:
            per_image.append((0.0, None))
    return per_image


def summarize_detections(per_image):
    n = len(per_image)
    confs = [c for c, _ in per_image]
    best_classes = [k for _, k in per_image if k is not None]
    return {
        "n": n,
        "mean_max_conf": sum(confs) / n if n else 0.0,
        "det_rate@0.3": sum(c >= 0.3 for c in confs) / n if n else 0.0,
        "right_frac": sum(k == RIGHT_CLASS_ID for k in best_classes) / len(best_classes) if best_classes else 0.0,
    }


def hand_crop_box(kpts):
    """Square crop (x0, y0, side) centered on the keypoint bbox, sized so the
    hand span occupies HAND_SPAN_FRAC of the crop like FreiHAND."""
    lo, hi = kpts.min(0), kpts.max(0)
    center = (lo + hi) / 2.0
    side = max(float((hi - lo).max()) / HAND_SPAN_FRAC, 1.0)
    return float(center[0] - side / 2.0), float(center[1] - side / 2.0), side


def render_pose_map(canvas_kpts):
    """OpenPose-style skeleton map, pixel-exact wrt the FreiHAND training maps
    (224x224 black canvas, thickness-2 HSV edges, radius-4 blue keypoints,
    int-truncated coordinates)."""
    canvas = np.zeros((POSE_CANVAS, POSE_CANVAS, 3), np.uint8)
    for ie, (a, b) in enumerate(HAND_EDGES):
        color = hsv_to_rgb([ie / len(HAND_EDGES), 1.0, 1.0]) * 255
        cv2.line(
            canvas,
            (int(canvas_kpts[a, 0]), int(canvas_kpts[a, 1])),
            (int(canvas_kpts[b, 0]), int(canvas_kpts[b, 1])),
            color,
            thickness=2,
        )
    for x, y in canvas_kpts:
        cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), thickness=-1)
    return canvas


def project_full_img(points, cam_trans, focal_length, img_res):
    """Perspective-project 3D joints to full-image pixel coords (WiLoR demo.py math)."""
    K = np.array(
        [[focal_length, 0.0, img_res[0] / 2.0], [0.0, focal_length, img_res[1] / 2.0], [0.0, 0.0, 1.0]]
    )
    points = points + cam_trans
    points = points / points[:, -1:]
    return (K @ points.T).T[:, :2]
