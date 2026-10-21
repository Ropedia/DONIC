#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def copy_calibration(src_dir: Path, data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("intri.yml", "extri.yml", "corrected_gsformat.pkl"):
        src = src_dir / name
        dst = data_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Calibration file not found: {src}")
        shutil.copy2(src, dst)
        print(f"[COPY] {src} -> {dst}", flush=True)


def image_size(path: Path):
    # Reads image header/metadata only; does not fully decode pixels.
    with Image.open(path) as im:
        return im.size


def numeric_info(path: Path):
    if not path.stem.isdigit():
        return None
    return int(path.stem), len(path.stem), path.suffix.lower()


def leaf_image_dirs(root: Path):
    """Find image folders quickly.

    Fast path targets the normal layout images/<camera>/*.jpg and avoids an
    expensive recursive walk through every frame. Falls back to rglob only for
    unusual nested layouts.
    """
    if not root.exists():
        return []

    def sort_key(p):
        name = p.name
        if name.isdigit():
            return (0, int(name))
        return (1, str(p))

    direct = []
    try:
        for d in root.iterdir():
            if not d.is_dir():
                continue
            try:
                if any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in d.iterdir()):
                    direct.append(d)
            except OSError:
                pass
    except OSError:
        direct = []

    if direct:
        return sorted(direct, key=sort_key)

    dirs = {
        p.parent
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }
    return sorted(dirs, key=sort_key)


def split_runs(items):
    """Split sorted [(frame_id, path), ...] into consecutive numeric runs."""
    if not items:
        return []

    runs = [[items[0]]]
    for item in items[1:]:
        prev = runs[-1][-1]
        same_naming = (
            numeric_info(item[1])[1:] == numeric_info(prev[1])[1:]
        )
        if item[0] == prev[0] + 1 and same_naming:
            runs[-1].append(item)
        else:
            runs.append([item])
    return runs


def ffmpeg_resize_run(
    ffmpeg,
    run,
    width,
    height,
    ffmpeg_threads,
    jpeg_q,
    png_compression,
):
    """
    Resize one consecutive numeric image sequence with ONE ffmpeg process.
    Output goes to a temporary folder first, then atomically replaces originals.
    """
    first_num, first_path = run[0]
    _, digits, ext = numeric_info(first_path)
    parent = first_path.parent
    pattern = f"%0{digits}d{ext}"

    with tempfile.TemporaryDirectory(
        prefix=".resize_tmp_", dir=str(parent)
    ) as tmp_s:
        tmp = Path(tmp_s)

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-y",
            "-threads", str(ffmpeg_threads),
            "-filter_threads", "1",
            "-start_number", str(first_num),
            "-i", str(parent / pattern),
            "-frames:v", str(len(run)),
            "-vf", f"scale={width}:{height}:flags=bilinear",
        ]

        if ext in {".jpg", ".jpeg"}:
            cmd += ["-q:v", str(jpeg_q)]
        elif ext == ".png":
            cmd += ["-compression_level", str(png_compression)]

        cmd += [
            "-start_number", str(first_num),
            str(tmp / pattern),
        ]

        subprocess.run(cmd, check=True)

        # Verify all expected outputs before replacing originals.
        for frame_id, src in run:
            out = tmp / f"{frame_id:0{digits}d}{ext}"
            if not out.exists() or out.stat().st_size == 0:
                raise RuntimeError(f"Missing ffmpeg output: {out}")

        for frame_id, src in run:
            out = tmp / f"{frame_id:0{digits}d}{ext}"
            os.replace(out, src)


def ffmpeg_resize_single(
    ffmpeg,
    path,
    width,
    height,
    ffmpeg_threads,
    jpeg_q,
    png_compression,
):
    ext = path.suffix.lower()

    fd, tmp_s = tempfile.mkstemp(
        prefix=".resize_", suffix=ext, dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(tmp_s)

    try:
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-y",
            "-threads", str(ffmpeg_threads),
            "-i", str(path),
            "-frames:v", "1",
            "-vf", f"scale={width}:{height}:flags=bilinear",
        ]

        if ext in {".jpg", ".jpeg"}:
            cmd += ["-q:v", str(jpeg_q)]
        elif ext == ".png":
            cmd += ["-compression_level", str(png_compression)]

        cmd += [str(tmp)]
        subprocess.run(cmd, check=True)

        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError(f"Missing ffmpeg output: {tmp}")

        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def process_dir(
    folder,
    width,
    height,
    ffmpeg,
    ffmpeg_threads,
    jpeg_q,
    png_compression,
    verify_all_sizes=False,
):
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    files.sort(
        key=lambda p: (
            int(p.stem) if p.stem.isdigit() else 10**30,
            p.name,
        )
    )

    if not files:
        return 0, 0, 0, 0

    # FAST MODE: extracted frames from one camera/video are expected to have one
    # resolution. Probe only the first frame rather than opening every JPEG with PIL.
    # This removes thousands of metadata opens/stat calls per camera.
    if not verify_all_sizes:
        try:
            if image_size(files[0]) == (width, height):
                print(
                    f"[SKIP DIR {folder}] total={len(files)} already {width}x{height}",
                    flush=True,
                )
                return len(files), 0, len(files), 0
            need = files
            keep = []
            bad_header = []
        except Exception as e:
            print(f"[WARN] first-frame header read failed: {files[0]}: {e}", flush=True)
            # Let ffmpeg attempt all files; no slow per-file Pillow scan.
            need = files
            keep = []
            bad_header = []
    else:
        # Strict compatibility mode: check every frame individually.
        keep = []
        need = []
        bad_header = []
        for p in files:
            try:
                if image_size(p) == (width, height):
                    keep.append(p)
                else:
                    need.append(p)
            except Exception as e:
                print(f"[WARN] header read failed: {p}: {e}", flush=True)
                bad_header.append(p)

    print(
        f"[DIR {folder}] total={len(files)} "
        f"keep={len(keep)} resize={len(need)} "
        f"header_fail={len(bad_header)}",
        flush=True,
    )

    numeric = []
    single = []

    for p in need:
        info = numeric_info(p)
        if info is None:
            single.append(p)
        else:
            numeric.append((info[0], p))

    single.extend(bad_header)
    numeric.sort(key=lambda x: x[0])
    runs = split_runs(numeric)

    resized = 0
    failed = 0

    for run in runs:
        first = run[0][1].name
        last = run[-1][1].name
        print(f"  [FFMPEG {folder.name}] {first} -> {last} ({len(run)} frames)", flush=True)
        try:
            ffmpeg_resize_run(
                ffmpeg, run, width, height, ffmpeg_threads, jpeg_q, png_compression
            )
            resized += len(run)
        except Exception as e:
            failed += len(run)
            print(
                f"  [ERROR] {folder}: sequence {first}->{last}: {e}",
                file=sys.stderr,
                flush=True,
            )

    for p in single:
        try:
            ffmpeg_resize_single(
                ffmpeg, p, width, height, ffmpeg_threads, jpeg_q, png_compression
            )
            resized += 1
        except Exception as e:
            failed += 1
            print(f"  [ERROR] {p}: {e}", file=sys.stderr, flush=True)

    print(
        f"[DONE DIR] {folder}: resized={resized}, kept={len(keep)}, failed={failed}",
        flush=True,
    )
    return len(files), resized, len(keep), failed

def resize_tree(
    root,
    width,
    height,
    ffmpeg,
    ffmpeg_threads,
    jpeg_q,
    png_compression,
    workers,
    verify_all_sizes,
):
    if not root.exists():
        print(f"[WARNING] Folder does not exist: {root}", flush=True)
        return

    dirs = leaf_image_dirs(root)
    if not dirs:
        print(f"[WARNING] No images found under: {root}", flush=True)
        return

    if workers <= 0:
        # Keep the machine saturated without blindly oversubscribing CPU threads.
        cpu = os.cpu_count() or 8
        workers = max(1, min(len(dirs), cpu // max(1, ffmpeg_threads)))

    print(
        f"[INFO] {root}: {len(dirs)} image folder(s), "
        f"parallel camera workers={workers}, ffmpeg_threads/process={ffmpeg_threads}, "
        f"verify_all_sizes={verify_all_sizes}",
        flush=True,
    )

    total = resized = kept = failed = 0

    # Each camera is independent. Run multiple FFmpeg processes concurrently.
    # ThreadPoolExecutor is intentional: Python only orchestrates subprocesses,
    # while the real work happens in FFmpeg processes outside the GIL.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_dir,
                folder,
                width,
                height,
                ffmpeg,
                ffmpeg_threads,
                jpeg_q,
                png_compression,
                verify_all_sizes,
            ): folder
            for folder in dirs
        }

        for fut in as_completed(futures):
            folder = futures[fut]
            try:
                a, b, c, d = fut.result()
            except Exception as e:
                print(f"[ERROR] folder {folder}: {e}", file=sys.stderr, flush=True)
                failed += 1
                continue
            total += a
            resized += b
            kept += c
            failed += d

    print(
        f"[DONE] {root}: total={total}, resized={resized}, kept={kept}, failed={failed}",
        flush=True,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-name", default="my_sequence")
    parser.add_argument("--calib-dir", default=".")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)

    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=2,
        help="Threads per FFmpeg process. Default: 2.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Parallel camera-folder workers. 0 = auto: min(camera_count, "
            "CPU_count // ffmpeg_threads). For 31 cameras on a large server, "
            "auto commonly uses 16-31 concurrent FFmpeg processes."
        ),
    )
    parser.add_argument(
        "--verify-all-sizes",
        action="store_true",
        help=(
            "Strict slow mode: inspect every frame header before resizing. "
            "Default fast mode checks only the first frame of each camera folder, "
            "which is appropriate for frames extracted from one video."
        ),
    )

    parser.add_argument(
        "--jpeg-q",
        type=int,
        default=2,
        help="FFmpeg JPEG quality: lower is better quality. Default: 2.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=1,
        choices=range(10),
    )
    parser.add_argument(
        "--only",
        choices=["both", "images", "images1"],
        default="both",
    )
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found. Check `ffmpeg -version` first."
        )

    if args.ffmpeg_threads < 1:
        parser.error("--ffmpeg-threads must be >= 1")
    if args.workers < 0:
        parser.error("--workers must be >= 0")

    data_dir = Path(args.data_name).resolve()
    calib_dir = Path(args.calib_dir).resolve()

    print(f"[DATA]    {data_dir}")
    print(f"[CALIB]   {calib_dir}")
    print(f"[SIZE]    {args.width}x{args.height}")
    print(f"[FFMPEG]  {ffmpeg}")
    print(f"[FFMPEG THREADS] {args.ffmpeg_threads}")
    print(f"[CAM WORKERS]    {'auto' if args.workers == 0 else args.workers}")
    print(f"[FAST PROBE]     {not args.verify_all_sizes}")

    copy_calibration(calib_dir, data_dir)

    common = dict(
        width=args.width,
        height=args.height,
        ffmpeg=ffmpeg,
        ffmpeg_threads=args.ffmpeg_threads,
        jpeg_q=args.jpeg_q,
        png_compression=args.png_compression,
        workers=args.workers,
        verify_all_sizes=args.verify_all_sizes,
    )

    if args.only in ("both", "images"):
        resize_tree(data_dir / "images", **common)

    if args.only in ("both", "images1"):
        resize_tree(data_dir / "images1", **common)

    print("All done.", flush=True)


if __name__ == "__main__":
    main()
