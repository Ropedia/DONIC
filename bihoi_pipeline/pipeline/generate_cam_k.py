#!/usr/bin/env python3
"""
Generate cam_K.txt from an EasyMocap/OpenCV intri.yml.

Default behavior matches the common resize case:
original image is resized to target width while preserving aspect ratio.
For your intri.yml, K_1 has cx=820, so original width is inferred as 1640.
With --w 640, scale = 640 / 1640, giving:
528.696585 0.000000 320.000000
0.000000 529.555122 240.390244
0.000000 0.000000 1.000000
"""

import argparse
import re
from pathlib import Path
import numpy as np


def read_K_from_intri(intri_path: str, cam_id: str = "1") -> np.ndarray:
    """Read K_<cam_id> 3x3 matrix from OpenCV-style intri.yml without requiring cv2."""
    text = Path(intri_path).read_text()

    # Match blocks like:
    # K_1: !!opencv-matrix
    #   rows: 3
    #   cols: 3
    #   dt: d
    #   data: [...]
    pattern = rf"K_{re.escape(str(cam_id))}\s*:\s*!!opencv-matrix.*?data\s*:\s*\[([^\]]+)\]"
    m = re.search(pattern, text, flags=re.S)
    if m is None:
        available = sorted(set(re.findall(r"\bK_([^:\s]+)\s*:", text)), key=lambda x: int(x) if x.isdigit() else x)
        raise KeyError(f"Cannot find K_{cam_id} in {intri_path}. Available camera IDs: {available}")

    vals = [float(x.strip()) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
    if len(vals) != 9:
        raise ValueError(f"K_{cam_id} should have 9 values, but got {len(vals)}")

    return np.array(vals, dtype=np.float64).reshape(3, 3)


def resize_K(K: np.ndarray, w: int, h: int, mode: str = "width") -> np.ndarray:
    """
    Resize intrinsic matrix.

    mode='width':
        Uniform scale using target width. This is the default and preserves aspect ratio.
        original width is inferred as 2 * cx.

    mode='xy':
        Separate x/y scaling using original width=2*cx and original height=2*cy.
    """
    K_new = K.copy().astype(np.float64)

    orig_w = 2.0 * K[0, 2]
    orig_h = 2.0 * K[1, 2]

    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid principal point in K, cannot infer original size: cx={K[0,2]}, cy={K[1,2]}")

    if mode == "width":
        s = float(w) / orig_w
        sx = sy = s
    elif mode == "xy":
        sx = float(w) / orig_w
        sy = float(h) / orig_h
    else:
        raise ValueError(f"Unknown mode: {mode}")

    K_new[0, 0] *= sx  # fx
    K_new[0, 2] *= sx  # cx
    K_new[1, 1] *= sy  # fy
    K_new[1, 2] *= sy  # cy
    K_new[0, 1] *= sx  # skew, normally 0
    K_new[1, 0] *= sy  # normally 0
    K_new[2, :] = [0.0, 0.0, 1.0]
    return K_new


def save_cam_K(K: np.ndarray, out_path: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, K, fmt="%.6f")


def main():
    parser = argparse.ArgumentParser(description="Generate cam_K.txt from intri.yml")
    parser.add_argument("--intri", type=str, default="intri.yml", help="Path to intri.yml")
    parser.add_argument("--out", type=str, default="cam_K.txt", help="Output cam_K.txt path")
    parser.add_argument("--cam_id", type=str, default="1", help="Camera ID to read, e.g. 1 for K_1")
    parser.add_argument("--w", type=int, default=640, help="Target image width")
    parser.add_argument("--h", type=int, default=481, help="Target image height")
    parser.add_argument(
        "--mode",
        type=str,
        default="width",
        choices=["width", "xy"],
        help="width: uniform scale by target width; xy: separate width/height scaling",
    )
    args = parser.parse_args()

    K = read_K_from_intri(args.intri, args.cam_id)
    K_new = resize_K(K, args.w, args.h, args.mode)
    save_cam_K(K_new, args.out)

    print(f"Saved: {args.out}")
    print(K_new)


if __name__ == "__main__":
    main()
