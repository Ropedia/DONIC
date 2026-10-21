#!/usr/bin/env python3
"""Inspect first-frame object masks (<dataset>/mask/<cam>.png) and choose usable camera sets.

Modes:
  check   : print per-camera mask coverage; exit 1 when fewer than --min_cams cameras
            have a non-empty mask (the SAM3 text prompt probably failed).
  select  : print shell assignments MESH_CAMS=... DEPTH_CAMS=... where BOTH lists contain
            every camera with a non-empty mask. If all 31 cameras have valid masks, all 31
            cameras are used for both mesh reconstruction and depth fusion.
            Use with:  eval "$(python mask_views.py select --data_dir <ds>)"
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def mask_pixels(mask_dir: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in sorted(mask_dir.iterdir(), key=lambda p: natural_key(p.name)):
        if p.suffix.lower() not in MASK_EXTS:
            continue
        cam = re.sub(r"_mask$", "", p.stem)
        if not cam.isdigit():
            continue
        try:
            a = np.asarray(Image.open(p).convert("L"))
            out[cam] = int((a > 127).sum())
        except Exception as exc:  # noqa: BLE001
            print(f"[masks] WARNING: cannot read {p.name}: {exc}", file=sys.stderr)
            out[cam] = 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "select"])
    ap.add_argument("--data_dir", required=True, help="Dataset root containing mask/")
    ap.add_argument("--mask_dir_name", default="mask")
    ap.add_argument(
        "--prefer",
        default="",
        help=argparse.SUPPRESS,  # backward compatibility; select now uses all usable cameras
    )
    ap.add_argument("--min_cams", type=int, default=8, help="Minimum number of cameras with a non-empty mask")
    ap.add_argument("--min_pixels", type=int, default=50, help="A mask with fewer foreground pixels counts as empty")
    args = ap.parse_args()

    mask_dir = Path(args.data_dir) / args.mask_dir_name
    if not mask_dir.is_dir():
        print(f"[masks] ERROR: {mask_dir} does not exist (run the SAM3 reconstruction-mask step first)", file=sys.stderr)
        return 1
    px = mask_pixels(mask_dir)
    if not px:
        print(f"[masks] ERROR: no per-camera masks in {mask_dir}", file=sys.stderr)
        return 1
    good = [c for c, n in px.items() if n >= args.min_pixels]
    empty = [c for c, n in px.items() if n < args.min_pixels]
    print(f"[masks] {len(px)} first-frame masks: {len(good)} usable, {len(empty)} empty" +
          (f" (cams {', '.join(empty)})" if empty else ""), file=sys.stderr)
    print("[masks] pixels per cam: " + " ".join(f"{c}={n}" for c, n in px.items()), file=sys.stderr)

    if len(good) < args.min_cams:
        print(f"[masks] ERROR: only {len(good)} camera(s) have an object mask (need >= {args.min_cams}). "
              "The SAM3 text prompt probably did not find the object — set a specific prompt for this dataset "
              "(e.g. 'nail clipper') and rerun '1b · SAM3 masks only'.", file=sys.stderr)
        return 1

    if args.mode == "check":
        if empty:
            print("[masks] WARNING: empty masks will be excluded from MV-SAM3D views / depth fusion.", file=sys.stderr)
        return 0

    good_ids = sorted(int(c) for c in good)

    # Use every camera with a valid first-frame object mask for BOTH mesh
    # reconstruction and depth fusion. MV-SAM3D refuses empty masks, so cameras
    # whose masks are empty/too small remain excluded automatically.
    chosen: List[int] = list(good_ids)

    print(
        f"[masks] using ALL usable cameras: "
        f"MESH_CAMS={len(chosen)} cams  DEPTH_CAMS={len(good_ids)} cams",
        file=sys.stderr,
    )
    print(
        f"MESH_CAMS={','.join(map(str, chosen))}; "
        f"DEPTH_CAMS={','.join(map(str, good_ids))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
