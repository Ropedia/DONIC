# Packaging changes: configurable dataset/model/output paths; inference unchanged.
import os
from os.path import join
import cv2
import numpy as np
import json
import argparse
import trimesh
import torch

from FMHR.train_gcn import run_infer_json as gcn_mano
from FMHR.models.gcn import GCNDecoder



def mano_from_mvimages(name, hand_side="left", data_dir=None, model_dir="./mano", out_dir=None):
    gcn_net_left = GCNDecoder(join(model_dir, 'TEMPLATE_LEFT.obj'), 21 * 3).cuda()
    gcn_net_right = GCNDecoder(join(model_dir, 'TEMPLATE_RIGHT.obj'), 21 * 3).cuda()

    gcn_net_left.load_state_dict(torch.load(join(model_dir, "gcn_left.pth")))
    gcn_net_right.load_state_dict(torch.load(join(model_dir, "gcn_right.pth")))

    gcn_net_left.eval()
    gcn_net_right.eval()

    gcn_net = {
        'left': gcn_net_left,
        'right': gcn_net_right
    }

    os.makedirs("output", exist_ok=True)

    mano_output = "manol" if hand_side == "left" else "manor"
    json_path = join(data_dir or f"../{name}", "output", mano_output, "keypoints3d", "000000.json")

    print(f"[info] hand_side={hand_side}, keypoints={json_path}")
    gcn_mano(gcn_net, json_path=json_path, data_path=out_dir, hand_side=hand_side)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
       '--name',
        type=str,
        required=True,
        help='Path to the candidate for hand calib .'
    )
    parser.add_argument(
        '--hand_side', '--hand-side', '--hand',
        dest='hand_side',
        type=str,
        default='left',
        choices=['left', 'right'],
        help='Hand side to calibrate (default: left).',
    )
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--model-dir', default='./mano')
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--scan_id', type=int, default=2)
    parser.add_argument('--range', type=int, default=None)
    args = parser.parse_args()
    if args.range is not None:
        start = 1
        end = args.range + 1
    else:
        start = args.scan_id
        end = args.scan_id + 1
    for i in range(start, end):
        mano_from_mvimages(args.name, args.hand_side, args.data_dir, args.model_dir, args.out_dir)
