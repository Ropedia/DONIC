#!/usr/bin/env python3
"""
File role / module responsibility:
    Rename camera video files under <data_name>/videos using the EasyMocap-style
    camera naming convention, then extract frames from each renamed video at a
    fixed sampling FPS into <data_name>/images/<camera_id>/.

Expected video rename rule:
    rpizero000_test0000.mp4 -> 1.mp4
    rpizero001_test0000.mp4 -> 2.mp4
    ...
    rpizero035_test0000.mp4 -> 36.mp4

Output structure:
    <data_name>/
        videos/
            1.mp4
            2.mp4
            ...
        images/
            1/
                000000.jpg
                000001.jpg
                ...
            2/
                000000.jpg
                000001.jpg
                ...

Example:

    # Dry run: preview rename and extraction plan only
    python rename_videos_extract_frames.py \
        --data-name <data-root>/handcap \
        --start-sec 2.0 \
        --end-sec 8.0

    # Actually rename videos and extract frames at 10 FPS
    python rename_videos_extract_frames.py \
        --data-name <data-root>/handcap \
        --start-sec 2.0 \
        --end-sec 8.0 \
        --fps 10 \
        --execute

    # Re-extract and overwrite existing image folders
    python rename_videos_extract_frames.py \
        --data-name <data-root>/handcap \
        --start-sec 0 \
        --end-sec 5 \
        --fps 10 \
        --clear-output \
        --execute
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "OpenCV is required for frame extraction. Install it with:\n"
        "    pip install opencv-python\n"
        "or, on a headless server:\n"
        "    pip install opencv-python-headless"
    ) from exc


VIDEO_PATTERN = re.compile(r"^rpizero(\d+)_test\d+(\.[^.]+)$", re.IGNORECASE)
NUMERIC_VIDEO_PATTERN = re.compile(r"^\d+(\.[^.]+)$", re.IGNORECASE)

VIDEO_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"
}


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def camera_sort_key(path: Path):
    """
    Sort numeric videos by camera id first:
        1.mp4, 2.mp4, ..., 10.mp4
    Non-numeric videos are sorted by name.
    """
    if path.stem.isdigit():
        return (0, int(path.stem), path.name)
    return (1, path.name)


def collect_video_renames(videos_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Collect video rename pairs.

    Only files matching rpizeroXXX_testYYYY.<ext> are renamed.
    Already-renamed numeric video files such as 1.mp4 are left unchanged.
    """
    if not videos_dir.exists():
        raise FileNotFoundError(f"videos folder does not exist: {videos_dir}")

    renames = []

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
        new_name = f"{cam_id + 1}{ext}"
        new_path = videos_dir / new_name

        if child.resolve() == new_path.resolve():
            continue

        renames.append((child, new_path))

    return renames


def rename_videos(videos_dir: Path, execute: bool = False, overwrite: bool = False) -> None:
    """
    Rename video files in two stages to avoid name collisions.

    Example:
        rpizero000_test0000.mp4 -> 1.mp4
    """
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

    temp_pairs = []

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


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    sample_fps: float = 10.0,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    image_ext: str = ".jpg",
    jpg_quality: int = 95,
    execute: bool = False,
    clear_output: bool = False,
) -> int:
    """
    Extract frames from one video at fixed sample_fps.

    The output frame index always starts from 000000.jpg for the selected segment.
    For example, with start_sec=2 and fps=10:
        000000.jpg corresponds approximately to t=2.0s
        000001.jpg corresponds approximately to t=2.1s
    """
    if sample_fps <= 0:
        raise ValueError("--fps must be > 0")

    validate_time_range(start_sec, end_sec)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / src_fps if src_fps and src_fps > 0 else None

    if src_fps is None or src_fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid source FPS for video: {video_path}")

    actual_end_sec = end_sec
    if actual_end_sec is None:
        actual_end_sec = duration_sec

    if actual_end_sec is None:
        cap.release()
        raise RuntimeError(f"Cannot determine video duration: {video_path}")

    if start_sec >= actual_end_sec:
        cap.release()
        raise ValueError(
            f"Invalid time range for {video_path.name}: "
            f"start_sec={start_sec}, end_sec={actual_end_sec}"
        )

    expected_count = int((actual_end_sec - start_sec) * sample_fps + 1e-9)

    print(
        f"[extract] {video_path.name}: "
        f"src_fps={src_fps:.3f}, duration={duration_sec:.3f}s, "
        f"range=[{start_sec:.3f}, {actual_end_sec:.3f}), "
        f"sample_fps={sample_fps}, expected≈{expected_count} frames"
    )

    if not execute:
        cap.release()
        print(f"[dry-run] Would write frames to: {output_dir}")
        return 0

    if clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Jump near the start time for speed.
    start_frame = max(0, int(start_sec * src_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    next_sample_time = start_sec
    sample_interval = 1.0 / sample_fps
    saved_count = 0

    encode_params = []
    if image_ext.lower() in [".jpg", ".jpeg"]:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]

    eps = 1e-6

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        current_time = frame_idx / src_fps

        if current_time + eps < start_sec:
            continue

        if current_time >= actual_end_sec - eps:
            break

        # Save the first frame whose timestamp reaches each target sample time.
        if current_time + eps >= next_sample_time:
            out_name = f"{saved_count:06d}{image_ext}"
            out_path = output_dir / out_name

            success = cv2.imwrite(str(out_path), frame, encode_params)
            if not success:
                cap.release()
                raise RuntimeError(f"Failed to write image: {out_path}")

            saved_count += 1
            next_sample_time = start_sec + saved_count * sample_interval

    cap.release()

    print(f"[done] {video_path.name}: saved {saved_count} frames -> {output_dir}")
    return saved_count


def extract_all_videos(
    data_dir: Path,
    sample_fps: float,
    start_sec: float,
    end_sec: Optional[float],
    image_ext: str,
    jpg_quality: int,
    execute: bool,
    clear_output: bool,
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

    print(f"\n[info] Found {len(videos)} numeric videos for extraction.")

    total_saved = 0

    for video_path in videos:
        cam_name = video_path.stem
        output_dir = images_dir / cam_name

        saved = extract_frames_from_video(
            video_path=video_path,
            output_dir=output_dir,
            sample_fps=sample_fps,
            start_sec=start_sec,
            end_sec=end_sec,
            image_ext=image_ext,
            jpg_quality=jpg_quality,
            execute=execute,
            clear_output=clear_output,
        )
        total_saved += saved

    if execute:
        print(f"\n[summary] Total saved frames: {total_saved}")
    else:
        print("\n[dry-run] Extraction preview finished. Use --execute to actually write images.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rename videos under <data_name>/videos and extract frames at fixed FPS "
            "to <data_name>/images/<camera_id>/."
        )
    )

    parser.add_argument(
        "--data-name",
        "--data",
        type=str,
        required=True,
        help="Dataset root folder containing videos/. Example: <data-root>/handcap",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frame extraction FPS. Default: 10.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Start time in seconds. Default: 0.",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="End time in seconds. If omitted, extract until video end.",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default=".jpg",
        choices=[".jpg", ".jpeg", ".png"],
        help="Output image extension. Default: .jpg",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality, only used for .jpg/.jpeg. Default: 95.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually rename videos and extract frames. Without this flag, only preview.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing renamed video targets if they already exist.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete existing images/<camera_id>/ folder before extracting frames.",
    )
    parser.add_argument(
        "--skip-rename",
        action="store_true",
        help="Skip video renaming and only extract from already numeric video files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_name).resolve()
    videos_dir = data_dir / "videos"

    print(f"[data] {data_dir}")
    print(f"[videos] {videos_dir}")

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
        sample_fps=args.fps,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        image_ext=args.image_ext,
        jpg_quality=args.jpg_quality,
        execute=args.execute,
        clear_output=args.clear_output,
    )


if __name__ == "__main__":
    main()
