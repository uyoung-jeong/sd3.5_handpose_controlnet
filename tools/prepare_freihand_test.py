"""Pose and mask maps for the FreiHAND evaluation (test) split.

  evaluation/pose/<id>.png   colored skeleton on black, training/pose convention
  evaluation/mask/<id>.jpg   3-channel 0/255 silhouette, training/mask convention
"""
from PIL import Image
import os
import os.path as osp
import json
import glob
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.colors
from argparse import ArgumentParser

import sys
lib_path = osp.join(osp.dirname(__file__), '..', '..')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
from hamer.freihand_utils.fh_utils import projectPoints

def parse_args():
    parser = ArgumentParser(description="FreiHAND evaluation pose/mask extraction")
    parser.add_argument('--freihand_dir', type=str, default='data/FreiHAND')

    args = parser.parse_args()
    return args

# same drawing as generate_freihand_depth.py:get_pose (canvas already is HWC3)
def get_pose(xyz, camera_intrinsics, pose_dir, fname, width, height):
    xyz_2d = projectPoints(xyz, camera_intrinsics) # [21, 2]

    eps = 0.01

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # draw pose
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for ie, (e1, e2) in enumerate(edges):
        k1 = xyz_2d[e1]
        k2 = xyz_2d[e2]
        x1 = int(k1[0])
        y1 = int(k1[1])
        x2 = int(k2[0])
        y2 = int(k2[1])
        if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
            cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

    for keypoint in xyz_2d:
        x = int(keypoint[0])
        y = int(keypoint[1])
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)

    pose_path = os.path.join(pose_dir, f"{fname}.png")
    Image.fromarray(canvas).save(pose_path)

# binarize the GT part labels into a training/mask-style hand silhouette
def get_mask(segmap_path, mask_dir, fname):
    segmap = cv2.imread(segmap_path, cv2.IMREAD_GRAYSCALE)
    mask = np.where(segmap > 0, 255, 0).astype(np.uint8)
    cv2.imwrite(os.path.join(mask_dir, f"{fname}.jpg"), np.dstack([mask] * 3))

def main(args):
    freihand_dir = args.freihand_dir
    freihand_rgb_dir = os.path.join(freihand_dir, 'evaluation/rgb')
    freihand_segmap_dir = os.path.join(freihand_dir, 'evaluation/segmap')

    rgb_img_paths = glob.glob(os.path.join(freihand_rgb_dir, '*.jpg'))
    rgb_img_paths = sorted(rgb_img_paths)

    pose_dir = os.path.join(freihand_dir, 'evaluation', 'pose')
    os.makedirs(pose_dir, exist_ok=True)

    mask_dir = os.path.join(freihand_dir, 'evaluation', 'mask')
    os.makedirs(mask_dir, exist_ok=True)

    # get gt
    gt_K_json_path = os.path.join(freihand_dir, 'evaluation_K.json')
    gt_xyz_json_path = os.path.join(freihand_dir, 'evaluation_xyz.json')

    with open(gt_K_json_path, 'r') as fi:
        gt_K = json.load(fi)
    gt_K = np.array(gt_K) # [3960, 3, 3]

    with open(gt_xyz_json_path, 'r') as fi:
        gt_xyz = json.load(fi)
    gt_xyz = np.array(gt_xyz) # [3960, 21, 3]

    n_data = len(gt_xyz)
    assert len(rgb_img_paths) == n_data, f'{len(rgb_img_paths)} images vs {n_data} annotations'

    pbar = tqdm(rgb_img_paths)
    for idx, rgb_img_path in enumerate(pbar):
        pbar.set_description(rgb_img_path)
        fname = os.path.basename(rgb_img_path)
        fname = os.path.splitext(fname)[0]

        rgb_img = Image.open(rgb_img_path)
        width, height = rgb_img.size

        # get pose
        get_pose(gt_xyz[idx], gt_K[idx], pose_dir, fname, width, height)

        # get mask
        get_mask(os.path.join(freihand_segmap_dir, f"{fname}.png"), mask_dir, fname)

    print("Finished")

if __name__ == '__main__':
    args = parse_args()
    main(args)
