#!/usr/bin/env python3
"""Extract one full-resolution reference frame per camera at a given time into images1/.

Replaces `extract_first_frames.py` (which always takes the very first video frame). The
reference frames feed the SAM3 first-frame object masks (mask/) and MV-SAM3D, while the
depth maps and per-frame masks start at --time-sec (the pipeline's start_sec). Taking
images1/ at the same instant keeps mask/, depth/<cam>/000000.png and images/<cam>/000000.jpg
consistent — otherwise, when the object moves during the first second, MV-SAM3D fuses the
t=0 masks with t=start_sec depth and the metric scale comes out wrong.

Usage:
    python extract_reference_frames.py --video_dir <ds>/videos --output_dir <ds>/images1 --time-sec 1.0
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV"}


def extract(video: Path, out: Path, t: float) -> bool:
    # -ss before -i seeks accurately (decodes from the previous keyframe) and is fast.
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(video),
           "-frames:v", "1", "-q:v", "2", str(out)]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=120).returncode == 0 and out.is_file()
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video_dir", required=True, help="Folder with <cam>.mp4 (numeric names)")
    ap.add_argument("--output_dir", required=True, help="images1/ : one <cam>.jpg per camera")
    ap.add_argument("--time-sec", type=float, default=0.0, help="Timestamp of the reference frame (= start_sec)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    vdir, odir = Path(args.video_dir), Path(args.output_dir)
    videos = sorted((p for p in vdir.iterdir() if p.is_file() and p.suffix in VIDEO_EXTS and p.stem.isdigit()),
                    key=lambda p: int(p.stem)) if vdir.is_dir() else []
    if not videos:
        print(f"[ref-frames] ERROR: no <cam>.mp4 in {vdir}", file=sys.stderr)
        return 1
    odir.mkdir(parents=True, exist_ok=True)
    for old in odir.glob("*.jpg"):
        old.unlink()
    jobs = [(v, odir / f"{v.stem}.jpg", args.time_sec) for v in videos]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        ok = list(pool.map(lambda a: extract(*a), jobs))
    failed = [v.name for v, good in zip(videos, ok) if not good]
    print(f"[ref-frames] t={args.time_sec:.3f}s: {sum(ok)}/{len(videos)} frames -> {odir}")
    if failed:
        print(f"[ref-frames] WARNING: failed for {failed}", file=sys.stderr)
    # Time stamp of the reference frames, so later stages can verify consistency.
    (odir / ".time_sec").write_text(f"{args.time_sec:.3f}\n")
    return 0 if sum(ok) >= max(1, len(videos) - 2) else 1


if __name__ == "__main__":
    sys.exit(main())
