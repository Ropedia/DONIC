#!/usr/bin/env python3
"""Print the --end-sec to use so every camera yields the same number of frames.

Looks at all videos in --video_dir, takes the SHORTEST one, subtracts a small
safety margin (the last few source frames are often not decodable / not on the
sampling grid) and snaps the result down to the --fps grid so that the number of
output frames, (end - start) * fps, is an integer. If --max-end-sec > start is
given, the result is additionally capped by it.

Only the chosen end time is printed to stdout; diagnostics go to stderr, so the
value can be captured with $(...) in a shell.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Optional

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV"}


def probe_duration(path: Path) -> Optional[float]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration:format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
        info = json.loads(out)
        cands = []
        for s in info.get("streams", []):
            if s.get("duration") not in (None, "N/A"):
                cands.append(float(s["duration"]))
        if info.get("format", {}).get("duration") not in (None, "N/A"):
            cands.append(float(info["format"]["duration"]))
        return min(cands) if cands else None
    except Exception as exc:  # noqa: BLE001
        print(f"[auto-end] WARNING: ffprobe failed for {path.name}: {exc}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video_dir", required=True, help="Folder with one video per camera (non-recursive)")
    ap.add_argument("--fps", type=float, required=True, help="Output sampling fps (end is snapped to this grid)")
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--max-end-sec", type=float, default=0.0, help="Upper cap; ignored when <= start")
    ap.add_argument("--margin-sec", type=float, default=0.1, help="Safety margin subtracted from the shortest video")
    args = ap.parse_args()

    vdir = Path(args.video_dir)
    videos = sorted(p for p in vdir.iterdir() if p.is_file() and p.suffix in VIDEO_EXTS) if vdir.is_dir() else []
    if not videos:
        print(f"[auto-end] ERROR: no videos in {vdir}", file=sys.stderr)
        return 1

    durations = {}
    for v in videos:
        d = probe_duration(v)
        if d is not None:
            durations[v.name] = d
    if not durations:
        print("[auto-end] ERROR: could not read any video duration", file=sys.stderr)
        return 1

    shortest_name = min(durations, key=durations.get)
    shortest = durations[shortest_name]
    longest = max(durations.values())
    usable = shortest - float(args.margin_sec)
    n_frames = math.floor((usable - args.start_sec) * args.fps + 1e-6)
    if args.max_end_sec > args.start_sec:
        n_frames = min(n_frames, math.floor((args.max_end_sec - args.start_sec) * args.fps + 1e-6))
    if n_frames < 1:
        print(f"[auto-end] ERROR: shortest video {shortest_name} ({shortest:.3f}s) is not longer than "
              f"start {args.start_sec}s + margin {args.margin_sec}s", file=sys.stderr)
        return 1
    end_sec = args.start_sec + n_frames / args.fps

    print(f"[auto-end] {len(durations)} video(s): shortest {shortest_name}={shortest:.3f}s, longest={longest:.3f}s "
          f"(spread {longest - shortest:.3f}s)", file=sys.stderr)
    cap = f", capped by max {args.max_end_sec}s" if args.max_end_sec > args.start_sec else ""
    print(f"[auto-end] end-sec = {end_sec:.3f} -> {n_frames} frames per camera at {args.fps:g} fps{cap}", file=sys.stderr)
    print(f"{end_sec:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
