#!/usr/bin/env python3
"""
Rename camera videos and extract a timestamp-aware fixed-FPS image sequence.

Main change from the previous OpenCV implementation
----------------------------------------------------
The old extractor estimated a source frame timestamp as

    frame_index / CAP_PROP_FPS

which assumes a perfectly constant-rate source and cannot correctly represent
VFR videos, dropped-frame gaps, or irregular source PTS.

This version delegates temporal resampling to FFmpeg:

    trim=<requested time range>,setpts=PTS-STARTPTS,fps=<target fps>

`trim` uses the decoded video timestamps, and the `fps` filter drops/duplicates
source frames as needed to place output frames on a uniform time grid. Thus, at
--fps 10, output frame n represents approximately n/10 seconds after the chosen
segment start, even when source frame spacing is irregular.

Expected rename rule:
    rpizero000_test0000.mp4 -> 1.mp4
    rpizero001_test0000.mp4 -> 2.mp4
    ...

Output:
    <data_name>/images/<camera_id>/000000.jpg
    <data_name>/images/<camera_id>/000001.jpg
    ...

Examples
--------
Dry run:
    python rename_videos_extract_frames_ffmpeg_pts.py \
        --data-name <data-root>/handcap \
        --start-sec 2.0 \
        --end-sec 8.0 \
        --fps 10

Execute:
    python rename_videos_extract_frames_ffmpeg_pts.py \
        --data-name <data-root>/handcap \
        --start-sec 2.0 \
        --end-sec 8.0 \
        --fps 10 \
        --clear-output \
        --execute
"""

import argparse
import os
import json
import math
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple


VIDEO_PATTERN = re.compile(r"^rpizero(\d+)_test\d+(\.[^.]+)$", re.IGNORECASE)
NUMERIC_VIDEO_PATTERN = re.compile(r"^\d+(\.[^.]+)$", re.IGNORECASE)
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def camera_sort_key(path: Path):
    if path.stem.isdigit():
        return (0, int(path.stem), path.name)
    return (1, path.name)


def collect_video_renames(videos_dir: Path) -> List[Tuple[Path, Path]]:
    if not videos_dir.exists():
        raise FileNotFoundError(f"videos folder does not exist: {videos_dir}")

    renames: List[Tuple[Path, Path]] = []
    for child in sorted(videos_dir.iterdir()):
        if not is_video_file(child):
            continue

        match = VIDEO_PATTERN.match(child.name)
        if match is None:
            if NUMERIC_VIDEO_PATTERN.match(child.name):
                print(f"[keep] Already numeric video name: {child.name}")
            else:
                print(f"[skip] Video name not matched: {child.name}")
            continue

        cam_id = int(match.group(1))
        ext = match.group(2).lower()
        new_path = videos_dir / f"{cam_id + 1}{ext}"
        if child.resolve() != new_path.resolve():
            renames.append((child, new_path))

    return renames


def rename_videos(
    videos_dir: Path,
    execute: bool = False,
    overwrite: bool = False,
) -> None:
    """Rename in two stages so camera-file names cannot collide."""
    renames = collect_video_renames(videos_dir)

    if not renames:
        print(f"[info] No video files need renaming under: {videos_dir}")
        return

    print(f"\n[videos] Rename plan under: {videos_dir}")
    for old_path, new_path in renames:
        print(f"{old_path.name}  ->  {new_path.name}")
        if new_path.exists() and old_path.resolve() != new_path.resolve():
            if overwrite:
                if execute:
                    new_path.unlink()
                print(f"[overwrite] Existing target will be removed: {new_path.name}")
            else:
                raise FileExistsError(
                    f"Target already exists: {new_path}\n"
                    f"Use --overwrite to replace it, or manually check the folder."
                )

    if not execute:
        print("[dry-run] No actual video renaming performed. Use --execute to apply.")
        return

    temp_pairs: List[Tuple[Path, Path]] = []
    for old_path, new_path in renames:
        temp_path = old_path.with_name(old_path.name + "__tmp_rename__")
        if temp_path.exists():
            temp_path.unlink()
        old_path.rename(temp_path)
        temp_pairs.append((temp_path, new_path))

    for temp_path, new_path in temp_pairs:
        temp_path.rename(new_path)

    print("[done] Video renaming completed.")


def validate_time_range(start_sec: float, end_sec: Optional[float]) -> None:
    if start_sec < 0:
        raise ValueError("--start-sec must be >= 0")
    if end_sec is not None and end_sec <= start_sec:
        raise ValueError("--end-sec must be larger than --start-sec")


def parse_fraction(text: Optional[str]) -> Optional[float]:
    if not text or text in {"0/0", "N/A"}:
        return None
    try:
        return float(Fraction(text))
    except Exception:
        try:
            return float(text)
        except Exception:
            return None


def ffprobe_video(video_path: Path, ffprobe: str) -> Dict[str, Optional[float]]:
    """Read container/stream timing metadata without relying on OpenCV FPS."""
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,start_time,duration:format=duration,start_time",
        "-of", "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(proc.stdout or "{}")

    streams = data.get("streams") or []
    stream = streams[0] if streams else {}
    fmt = data.get("format") or {}

    def fnum(x):
        try:
            return float(x)
        except Exception:
            return None

    duration = fnum(stream.get("duration"))
    if duration is None:
        duration = fnum(fmt.get("duration"))

    start_time = fnum(stream.get("start_time"))
    if start_time is None:
        start_time = fnum(fmt.get("start_time"))

    nb_frames = fnum(stream.get("nb_frames"))

    return {
        "duration": duration,
        "start_time": start_time,
        "avg_fps": parse_fraction(stream.get("avg_frame_rate")),
        "r_fps": parse_fraction(stream.get("r_frame_rate")),
        "nb_frames": nb_frames,
    }


def ffmpeg_jpeg_q_from_quality(jpg_quality: int) -> int:
    """
    Approximate OpenCV JPEG quality 1..100 with FFmpeg MJPEG q:v 31..2.
    Higher jpg_quality -> lower q:v -> better quality.
    """
    q = int(round(31.0 - (jpg_quality / 100.0) * 29.0))
    return max(2, min(31, q))


def count_output_images(folder: Path, image_ext: str) -> int:
    ext = image_ext.lower()
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ext
    ) if folder.exists() else 0


def build_pts_filter(
    sample_fps: float,
    start_sec: float,
    end_sec: Optional[float],
) -> str:
    """
    Build a PTS-aware filter chain.

    1) trim selects the requested interval using decoded timestamps.
    2) setpts preserves all *relative* source timestamp spacing while rebasing
       the selected segment to zero.
    3) fps resamples that timestamped sequence to a uniform output time grid.

    If the source has a timestamp gap / dropped source frame, fps may duplicate
    a neighboring decoded frame instead of compressing time. This is the desired
    behavior for multi-camera temporal alignment.
    """
    if end_sec is None:
        trim = f"trim=start={start_sec:.9f}"
    else:
        trim = f"trim=start={start_sec:.9f}:end={end_sec:.9f}"

    return (
        f"{trim},"
        "setpts=PTS-STARTPTS,"
        f"fps=fps={sample_fps:.12g}:start_time=0:round=near"
    )


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    ffmpeg: str,
    sample_fps: float = 10.0,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    image_ext: str = ".jpg",
    jpg_quality: int = 95,
    png_compression: int = 1,
    ffmpeg_threads: int = 1,
    execute: bool = False,
    clear_output: bool = False,
    verify_count: bool = False,
) -> int:
    """Ultra-fast extraction for one camera.

    Speed-oriented differences from the compatibility version:
      * no per-video ffprobe in the hot path;
      * input-side -ss seeks before decoding when start_sec > 0;
      * no resize/scale filter at all;
      * writes directly to images/<cam>/ instead of a temporary directory;
      * no per-file move after extraction;
      * output counting is optional (disabled by default).

    FFmpeg still resamples decoded timestamps onto a fixed FPS grid. With an
    explicit [start_sec, end_sec) interval, the exact output frame count is
    capped by -frames:v.
    """
    if sample_fps <= 0:
        raise ValueError("--fps must be > 0")
    validate_time_range(start_sec, end_sec)

    expected_count = None
    if end_sec is not None:
        expected_count = max(
            0,
            int(math.floor((end_sec - start_sec) * sample_fps + 1e-9)),
        )

    if not execute:
        print(
            f"[dry-run] {video_path.name} -> {output_dir} "
            f"range=[{start_sec:.6f}, {'EOF' if end_sec is None else f'{end_sec:.6f}'}) "
            f"fps={sample_fps:g} expected={expected_count if expected_count is not None else 'N/A'}"
        )
        return 0

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Direct-write mode is much faster than temp-dir -> count -> move.
    # --clear-output is the recommended mode and removes the whole old camera dir.
    if clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If the caller did not request a full clear, remove only stale generated
    # numbered images so a shorter re-run cannot leave an old tail behind.
    if not clear_output:
        for old in output_dir.iterdir():
            if old.is_file() and old.suffix.lower() == image_ext.lower() and old.stem.isdigit():
                old.unlink()

    pattern = output_dir / f"%06d{image_ext}"

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-threads", str(ffmpeg_threads),
    ]

    # Fast input seek. For transcoding, FFmpeg seeks to the previous keyframe and
    # decodes/discards until the requested point, avoiding decoding the whole prefix.
    if start_sec > 0:
        cmd += ["-ss", f"{start_sec:.9f}"]

    cmd += [
        "-i", str(video_path),
        "-map", "0:v:0",
        "-an",
    ]

    # Keep original spatial resolution. There is intentionally NO scale/resize.
    # After input-side seeking, rebase timestamps and sample on a uniform FPS grid.
    cmd += [
        "-vf", f"setpts=PTS-STARTPTS,fps=fps={sample_fps:.12g}:start_time=0:round=near",
        "-filter_threads", "1",
        "-start_number", "0",
    ]

    if end_sec is not None:
        duration = max(0.0, end_sec - start_sec)
        cmd += ["-t", f"{duration:.9f}"]
        if expected_count is not None:
            cmd += ["-frames:v", str(expected_count)]

    if image_ext.lower() in {".jpg", ".jpeg"}:
        cmd += ["-q:v", str(ffmpeg_jpeg_q_from_quality(jpg_quality))]
    elif image_ext.lower() == ".png":
        cmd += ["-compression_level", str(png_compression)]

    # fps already decides temporal resampling; do not let the muxer do another pass.
    cmd += ["-vsync", "0", str(pattern)]

    subprocess.run(cmd, check=True)

    # Avoid O(N_frames) directory scans unless explicitly requested.
    if verify_count:
        saved_count = count_output_images(output_dir, image_ext)
        if saved_count <= 0:
            raise RuntimeError(f"FFmpeg produced no frames for {video_path.name}")
        if expected_count is not None and saved_count != expected_count:
            print(
                f"[WARNING] {video_path.name}: expected {expected_count} frames, got {saved_count}",
                flush=True,
            )
    else:
        # Fast sanity check: expected final file should exist when frame count is known.
        # This is O(1) instead of scanning the directory.
        if expected_count is not None and expected_count > 0:
            last = output_dir / f"{expected_count - 1:06d}{image_ext}"
            if not last.exists():
                # Fall back to one directory count only on an abnormal result.
                saved_count = count_output_images(output_dir, image_ext)
                if saved_count <= 0:
                    raise RuntimeError(f"FFmpeg produced no frames for {video_path.name}")
            else:
                saved_count = expected_count
        else:
            # EOF mode has no known expected count; count once for the return value.
            saved_count = count_output_images(output_dir, image_ext)
            if saved_count <= 0:
                raise RuntimeError(f"FFmpeg produced no frames for {video_path.name}")

    print(f"[done] {video_path.name}: {saved_count} frames -> {output_dir}", flush=True)
    return saved_count


def extract_all_videos(
    data_dir: Path,
    ffmpeg: str,
    sample_fps: float,
    start_sec: float,
    end_sec: Optional[float],
    image_ext: str,
    jpg_quality: int,
    png_compression: int,
    ffmpeg_threads: int,
    workers: int,
    execute: bool,
    clear_output: bool,
    verify_count: bool,
) -> None:
    videos_dir = data_dir / "videos"
    images_dir = data_dir / "images"

    if not videos_dir.exists():
        raise FileNotFoundError(f"videos folder does not exist: {videos_dir}")

    videos = sorted(
        [p for p in videos_dir.iterdir() if is_video_file(p) and p.stem.isdigit()],
        key=camera_sort_key,
    )
    if not videos:
        raise FileNotFoundError(
            f"No numeric video files found under {videos_dir}.\n"
            f"Expected files like 1.mp4, 2.mp4, ..."
        )

    if workers <= 0:
        cpu = os.cpu_count() or len(videos)
        # One FFmpeg process per camera is ideal for this workload; cap only by
        # available CPU execution capacity and the number of cameras.
        workers = max(1, min(len(videos), cpu // max(1, ffmpeg_threads)))
    workers = max(1, min(int(workers), len(videos)))

    print(
        f"\n[FAST] {len(videos)} cameras, workers={workers}, "
        f"ffmpeg_threads={ffmpeg_threads}, resize=OFF, ffprobe=OFF, direct_write=ON",
        flush=True,
    )

    def run_one(video_path: Path):
        cam_name = video_path.stem
        saved = extract_frames_from_video(
            video_path=video_path,
            output_dir=images_dir / cam_name,
            ffmpeg=ffmpeg,
            sample_fps=sample_fps,
            start_sec=start_sec,
            end_sec=end_sec,
            image_ext=image_ext,
            jpg_quality=jpg_quality,
            png_compression=png_compression,
            ffmpeg_threads=ffmpeg_threads,
            execute=execute,
            clear_output=clear_output,
            verify_count=verify_count,
        )
        return cam_name, saved

    counts: Dict[str, int] = {}
    failures = []

    # Thread workers only orchestrate independent FFmpeg subprocesses. The actual
    # decode/encode work happens in FFmpeg processes, so the Python GIL is irrelevant.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(run_one, v): v for v in videos}
        for fut in as_completed(future_map):
            video = future_map[fut]
            try:
                cam_name, saved = fut.result()
                if execute:
                    counts[cam_name] = saved
            except Exception as exc:
                failures.append((video.name, exc))
                print(f"[ERROR] {video.name}: {exc}", flush=True)

    if failures:
        details = "\n".join(f"  {name}: {exc}" for name, exc in failures)
        raise RuntimeError(f"{len(failures)} camera extraction(s) failed:\n{details}")

    if execute:
        total_saved = sum(counts.values())
        print(f"\n[summary] Total saved frames: {total_saved}", flush=True)
        if counts:
            unique_counts = sorted(set(counts.values()))
            if len(unique_counts) == 1:
                print(
                    f"[summary] All {len(counts)} cameras have exactly {unique_counts[0]} frames.",
                    flush=True,
                )
            else:
                print("[WARNING] Camera output frame counts differ:", flush=True)
                for cam in sorted(counts, key=lambda x: int(x) if x.isdigit() else x):
                    print(f"          cam {cam}: {counts[cam]}", flush=True)
    else:
        print("\n[dry-run] Extraction preview finished. Use --execute to write images.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rename videos and extract frames with FFmpeg PTS-aware fixed-FPS "
            "temporal resampling."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-name", "--data",
        type=str,
        required=True,
        help="Dataset root folder containing videos/.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Output sampling FPS. Output frame n is on a uniform n/fps time grid.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Start time in seconds on each video's timeline.",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="Exclusive end time in seconds. If omitted, extract until video EOF.",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default=".jpg",
        choices=[".jpg", ".jpeg", ".png"],
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality in familiar 1..100 scale (mapped to FFmpeg q:v).",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=1,
        choices=range(10),
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=1,
        help="FFmpeg threads per camera process. For maximum multi-camera throughput keep this at 1.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel camera FFmpeg processes. 0 = auto (normally all cameras if CPU allows).",
    )
    parser.add_argument(
        "--verify-count",
        action="store_true",
        help="Scan every output directory and verify frame counts. Disabled by default for speed.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually rename videos and extract frames. Otherwise preview only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing renamed video targets if needed.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete each existing images/<camera_id>/ before installing new output.",
    )
    parser.add_argument(
        "--skip-rename",
        action="store_true",
        help="Skip video renaming and only extract already numeric video files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if not (1 <= args.jpg_quality <= 100):
        raise ValueError("--jpg-quality must be in [1, 100]")
    if args.ffmpeg_threads < 1:
        raise ValueError("--ffmpeg-threads must be >= 1")
    validate_time_range(args.start_sec, args.end_sec)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH. Check: ffmpeg -version")

    data_dir = Path(args.data_name).resolve()
    videos_dir = data_dir / "videos"

    print(f"[data]    {data_dir}")
    print(f"[videos]  {videos_dir}")
    print(f"[ffmpeg]  {ffmpeg}")
    print(f"[mode]    FAST parallel extraction -> {args.fps:g} FPS (NO resize)")

    if not args.skip_rename:
        rename_videos(
            videos_dir=videos_dir,
            execute=args.execute,
            overwrite=args.overwrite,
        )
    else:
        print("[skip] Video renaming skipped by --skip-rename")

    extract_all_videos(
        data_dir=data_dir,
        ffmpeg=ffmpeg,
        sample_fps=args.fps,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        image_ext=args.image_ext,
        jpg_quality=args.jpg_quality,
        png_compression=args.png_compression,
        ffmpeg_threads=args.ffmpeg_threads,
        workers=args.workers,
        execute=args.execute,
        clear_output=args.clear_output,
        verify_count=args.verify_count,
    )


if __name__ == "__main__":
    main()
