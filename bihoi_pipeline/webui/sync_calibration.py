#!/usr/bin/env python3
"""Keep only the cameras that actually exist in a dataset inside its intri.yml / extri.yml.

The rig calibration lists 31 cameras, but a capture may miss one (e.g. 30 videos).
`rename_videos_31cams.py --allow-missing` keeps the calibration index of every present
camera, so the only thing left to do is to drop the absent cameras from the dataset's
own calibration copy; every downstream script that enumerates cameras from the
calibration (EasyMocap, FoundationPose render, ...) then sees a consistent set.

The untouched originals are kept once as intri.yml.full / extri.yml.full, and the
script is idempotent (re-running with the same cameras changes nothing).

Usage:
    python sync_calibration.py --data_dir <dataset>          # cameras = images/<cam>/ (or videos/<cam>.mp4)
    python sync_calibration.py --data_dir <dataset> --restore # put the full calibration back
Run inside the `easymocap` environment (needs cv2 + easymocap).
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))


def present_cameras(ds: Path):
    images = ds / "images"
    if images.is_dir():
        cams = [p.name for p in images.iterdir() if p.is_dir() and p.name.isdigit() and any(p.iterdir())]
        if cams:
            return sorted(cams, key=int), "images/"
    videos = ds / "videos"
    if videos.is_dir():
        cams = [p.stem for p in videos.iterdir() if p.is_file() and p.suffix.lower() == ".mp4" and p.stem.isdigit()]
        if cams:
            return sorted(cams, key=int), "videos/"
    return [], "none"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--restore", action="store_true", help="Restore intri.yml/extri.yml from the .full backups")
    ap.add_argument("--min_cams", type=int, default=12, help="Refuse to write a calibration with fewer cameras")
    args = ap.parse_args()
    ds = Path(args.data_dir).resolve()
    intri, extri = ds / "intri.yml", ds / "extri.yml"
    if not (intri.is_file() and extri.is_file()):
        print(f"[calib] ERROR: {intri} / {extri} missing (run the prepare step first)", file=sys.stderr)
        return 1

    if args.restore:
        for p in (intri, extri):
            full = p.with_suffix(".yml.full")
            if full.is_file():
                shutil.copy2(full, p)
                print(f"[calib] restored {p.name} from {full.name}")
        return 0

    from easymocap.mytools.camera_utils import read_camera, write_camera  # noqa: E402

    cams_present, source = present_cameras(ds)
    if not cams_present:
        print(f"[calib] ERROR: no images/<cam>/ or videos/<cam>.mp4 found under {ds}", file=sys.stderr)
        return 1

    # Always filter from the FULL rig calibration so a camera that appears later is picked up again.
    for p in (intri, extri):
        full = p.with_suffix(".yml.full")
        if not full.is_file():
            shutil.copy2(p, full)
    cams_full = read_camera(str(intri.with_suffix(".yml.full")), str(extri.with_suffix(".yml.full")))
    cams_full.pop("basenames", None)
    names_full = sorted(cams_full.keys(), key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)])

    keep = [c for c in names_full if c in cams_present]
    dropped = [c for c in names_full if c not in cams_present]
    extra = [c for c in cams_present if c not in cams_full]
    if extra:
        print(f"[calib] WARNING: {source} has cameras without calibration entry (ignored): {extra}", file=sys.stderr)
    if len(keep) < args.min_cams:
        print(f"[calib] ERROR: only {len(keep)} calibrated camera(s) present (need >= {args.min_cams})", file=sys.stderr)
        return 1

    current = read_camera(str(intri), str(extri))
    current.pop("basenames", None)
    if sorted(current.keys(), key=int) == sorted(keep, key=int):
        print(f"[calib] {len(keep)} camera(s) from {source}; calibration already in sync"
              + (f" (absent: {dropped})" if dropped else ""))
        return 0

    write_camera({c: cams_full[c] for c in keep}, str(ds))
    print(f"[calib] wrote intri.yml/extri.yml with {len(keep)}/{len(names_full)} camera(s) present in {source}"
          + (f"; dropped absent camera(s): {dropped}" if dropped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
