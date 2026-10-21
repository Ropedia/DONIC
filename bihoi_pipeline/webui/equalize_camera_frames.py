#!/usr/bin/env python3
"""Trim every camera folder to the same number of frames.

Frames are sampled on a uniform grid from --start-sec (frame n is at n/fps), so
cameras whose source video is slightly shorter end up with fewer trailing
frames. Multi-view stages (e.g. Depth-Anything-3 --all_da3) require identical,
synchronized counts, so we drop the surplus trailing frames of the longer
cameras. Deleting only the tail keeps the remaining frames synchronized.

Usage:
    python equalize_camera_frames.py --data_dir <dataset_root> [--dirs images masks] [--dry-run]
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

FRAME_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def camera_dirs(root: Path) -> List[Path]:
    return sorted((p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")), key=natural_key)


def frame_files(cam: Path) -> List[Path]:
    return sorted((p for p in cam.iterdir() if p.is_file() and p.suffix.lower() in FRAME_EXTS), key=natural_key)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="Dataset root containing images/ (and optionally masks/)")
    ap.add_argument("--dirs", nargs="+", default=["images", "masks"],
                    help="Per-camera frame folders under data_dir to equalize (missing ones are skipped)")
    ap.add_argument("--dry-run", action="store_true", help="Only report, do not delete")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        print(f"[equalize] ERROR: not a directory: {data_dir}", file=sys.stderr)
        return 1

    counts: Dict[str, Dict[str, List[Path]]] = {}
    for name in args.dirs:
        root = data_dir / name
        if not root.is_dir():
            print(f"[equalize] {name}/: absent, skipped")
            continue
        cams = camera_dirs(root)
        if not cams:
            print(f"[equalize] {name}/: no camera folders, skipped")
            continue
        counts[name] = {cam.name: frame_files(cam) for cam in cams}

    if not counts:
        print("[equalize] nothing to do (no frame folders found)")
        return 0

    all_counts = [len(files) for per_cam in counts.values() for files in per_cam.values()]
    target = min(all_counts)
    if target <= 0:
        empty = [f"{d}/{c}" for d, per_cam in counts.items() for c, f in per_cam.items() if not f]
        print(f"[equalize] ERROR: empty camera folder(s): {', '.join(empty)}", file=sys.stderr)
        return 1

    for name, per_cam in counts.items():
        summary = " ".join(f"{c}={len(f)}" for c, f in per_cam.items())
        print(f"[equalize] {name}/ before: min={min(len(f) for f in per_cam.values())} "
              f"max={max(len(f) for f in per_cam.values())} | {summary}")

    removed = 0
    for name, per_cam in counts.items():
        for cam, files in per_cam.items():
            surplus = files[target:]
            for p in surplus:
                if not args.dry_run:
                    p.unlink()
                removed += 1
            if surplus:
                print(f"[equalize] {name}/{cam}: {len(files)} -> {target} "
                      f"({'would remove' if args.dry_run else 'removed'} {surplus[0].name}..{surplus[-1].name})")

    # Sanity: after trimming, every camera must expose the same frame names.
    mismatched = []
    for name, per_cam in counts.items():
        ref = None
        for cam, files in per_cam.items():
            names = [p.name for p in files[:target]]
            if ref is None:
                ref = names
            elif names != ref:
                mismatched.append(f"{name}/{cam}")
    if mismatched:
        print(f"[equalize] WARNING: frame names differ from the first camera in: {', '.join(mismatched)}")

    verb = "would remove" if args.dry_run else "removed"
    print(f"[equalize] target frames per camera: {target}; {verb} {removed} surplus file(s) in {data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
