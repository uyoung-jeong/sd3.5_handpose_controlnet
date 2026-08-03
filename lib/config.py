#!/usr/bin/env python
# coding=utf-8
"""Central configuration: machine-local default paths and shared constants.

The paths below describe THIS machine's layout. They are only argparse
DEFAULTS — every tool exposes a flag for each one, so override per run with
the flag, or edit here once when moving machines.
"""

import os

# Repo root (this file lives in <repo>/lib/).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Machine-local paths (argparse defaults)
# ---------------------------------------------------------------------------
FREIHAND_ROOT = os.path.join(REPO_ROOT, "data", "FreiHAND")
TRAIN_OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "sd35_large_pose_cn")
CONTROLNET_PATH = os.path.join(REPO_ROOT, "models", "sd3.5-handpose-controlnet")
SAMPLES_DIR = os.path.join(REPO_ROOT, "samples")
INFER_OUT_DIR = os.path.join(REPO_ROOT, "outputs", "infer")
BASE_MODEL = "stabilityai/stable-diffusion-3.5-large"

# WiLoR install used for hand detection + 3D pose. It has its own conda env;
# tools/infer_sd3_pose_controlnet.py bridges to it via WILOR_PYTHON.
WILOR_ROOT = "/share_home/uyoung/hand/WiLoR"
WILOR_PYTHON = "/home/uyoung/anaconda3/envs/wilor/bin/python"
WILOR_DETECTOR_PATH = os.path.join(WILOR_ROOT, "pretrained_models", "detector.pt")

# ---------------------------------------------------------------------------
# Shared constants (not machine-local)
# ---------------------------------------------------------------------------
# Fixed training caption pool. Index 0 is reserved for the empty prompt and is
# sampled with probability --proportion_empty_prompts (classifier-free guidance
# dropout). FreiHANDPoseDataset draws indices into this list; the train script
# precomputes one text embedding per entry.
CAPTIONS = [
    "",
    "a photo of a right human hand",
    "a close-up photo of a person's right hand",
    "a photograph of a right hand, fingers clearly visible",
    "a high quality photo of a single right hand",
    "a picture of someone's right hand",
]

# WiLoR YOLO detector class ids: {0: "left", 1: "right"}. FreiHAND is right hands only.
RIGHT_CLASS_ID = 1

# OpenPose hand skeleton over the FreiHAND/WiLoR joint order
# (0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky).
HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
    [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20],
]
POSE_CANVAS = 224  # FreiHAND native resolution the training pose maps were drawn at
# Median keypoint span / canvas over FreiHAND training poses (measured 0.382,
# p10 0.329, p90 0.433): crop side = span / this puts in-the-wild hands at
# FreiHAND's framing.
HAND_SPAN_FRAC = 0.38
