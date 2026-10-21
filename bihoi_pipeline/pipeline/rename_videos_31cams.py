#!/usr/bin/env python3
from pathlib import Path
import argparse
import re
import shutil
import sys

# Exact camera order used by the modified intri.yml / extri.yml.
CALIB_CAMERA_ORDER = [
    "node13_cam0",
    "node13_cam1",
    "node25_cam1",
    "node26_cam0",
    "node26_cam1",
    "node27_cam0",
    "node27_cam1",
    "node28_cam0",
    "node28_cam1",
    "node33_cam0",
    "node33_cam1",
    "node34_cam0",
    "node34_cam1",
    "node35_cam0",
    "node35_cam1",
    "node36_cam0",
    "node36_cam1",
    "node37_cam0",
    "node37_cam1",
    "node38_cam0",
    "node38_cam1",
    "node39_cam0",
    "node39_cam1",
    "node40_cam0",
    "node40_cam1",
    "node41_cam0",
    "node41_cam1",
    "node42_cam0",
    "node42_cam1",
    "node47_cam0",
    "node47_cam1",
]


def natural_key(path: Path):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", path.stem)
    ]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy all .mp4 files from an input directory into an output directory "
            "and rename them to 1.mp4, 2.mp4, ... . If filenames match the "
            "calibration camera names, the exact intri/extri camera order is preserved."
        )
    )
    parser.add_argument(
        "--input_dir",
        "--input-video-dir",
        dest="input_dir",
        default="videos",
        help="Directory containing the original .mp4 videos (default: videos)",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        default="videos_renamed",
        help=(
            "Directory where renamed videos will be saved "
            "(default: videos_renamed)"
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Accept captures where some calibrated cameras have no video. "
            "Matched cameras keep their calibration index (gaps stay empty); "
            "extra unmatched videos are dropped."
        ),
    )
    parser.add_argument(
        "--strict-calibration",
        action="store_true",
        help=("For hand calibration, require exactly 31 calibrated videos. Accept "
              "complete nodeNN_camN, 1..31 or rpizero000..030 sets only; "
              "never guess camera positions from filenames."),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing numbered .mp4 files in the output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the input -> output mapping without copying any files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        sys.exit(f"ERROR: input directory does not exist: {input_dir}")

    # This version is intentionally input -> output, not in-place renaming.
    if input_dir == output_dir:
        sys.exit(
            "ERROR: input_dir and output_dir must be different.\n"
            "This script preserves the original videos and writes renamed copies "
            "to a separate output directory."
        )

    videos = sorted(input_dir.glob("*.mp4"), key=natural_key)
    n = len(videos)

    if n == 0:
        sys.exit(f"ERROR: no .mp4 files found in {input_dir}")

    n_calib = len(CALIB_CAMERA_ORDER)
    if args.strict_calibration and n != n_calib:
        sys.exit(f'ERROR: hand calibration requires all {n_calib} calibrated videos; found {n}')
    if n > n_calib + 1:
        sys.exit(
            f"ERROR: found {n} videos; expected at most {n_calib + 1} "
            f"({n_calib} calibrated cameras + at most 1 optional extra)"
        )

    by_stem = {p.stem: p for p in videos}
    matched_names = [name for name in CALIB_CAMERA_ORDER if name in by_stem]

    assignments = []

    if args.strict_calibration and not matched_names:
        # Never guess ordering in hand calibration: only known legacy schemes
        # have the same camera indices as the fixed intri.yml/extri.yml.
        numeric = [int(p.stem) for p in videos if p.stem.isdigit()]
        legacy_matches = [re.fullmatch(r"rpizero(\d+)_test\d+", p.stem, re.IGNORECASE) for p in videos]
        if len(numeric) == n_calib and sorted(numeric) == list(range(1, n_calib + 1)):
            assignments = list(zip(videos, numeric))
        elif all(legacy_matches) and sorted(int(m.group(1)) for m in legacy_matches) == list(range(n_calib)):
            assignments = [(p, int(m.group(1)) + 1) for p, m in zip(videos, legacy_matches)]
        else:
            sys.exit(
                "ERROR: ambiguous hand calibration camera names. "
                "Expected all CALIB_CAMERA_ORDER names, 1.mp4..31.mp4, or "
                "rpizero000_test*.mp4..rpizero030_test*.mp4."
            )

    elif len(matched_names) == len(CALIB_CAMERA_ORDER):
        # All calibrated camera videos are present.
        # Preserve the exact calibration order -> 1..N.
        used = set()
        for idx, cam_name in enumerate(CALIB_CAMERA_ORDER, start=1):
            p = by_stem[cam_name]
            assignments.append((p, idx))
            used.add(p)

        # Optional extra videos, if any, come after the calibrated cameras.
        extras = sorted([p for p in videos if p not in used], key=natural_key)
        assignments.extend(
            (p, len(CALIB_CAMERA_ORDER) + i)
            for i, p in enumerate(extras, start=1)
        )

    elif len(matched_names) == 0:
        # Source files use some other naming scheme.
        # Use natural filename order.
        print("WARNING: filenames do not match the calibration camera names.")
        print("         Falling back to natural filename order.")
        assignments = [(p, i) for i, p in enumerate(videos, start=1)]

    elif args.allow_missing:
        # Partial set: keep the calibration index of every matched camera so the
        # numbering still lines up with intri/extri; missing indices stay empty.
        missing = [name for name in CALIB_CAMERA_ORDER if name not in by_stem]
        print(f"WARNING: {len(missing)} calibrated camera(s) have no video, "
              f"indices left empty: {missing}")
        assignments = [
            (by_stem[cam_name], idx)
            for idx, cam_name in enumerate(CALIB_CAMERA_ORDER, start=1)
            if cam_name in by_stem
        ]

    else:
        # Partial matching is dangerous because it can silently misalign cameras.
        missing = [name for name in CALIB_CAMERA_ORDER if name not in by_stem]
        sys.exit(
            "ERROR: only part of the calibration camera set matched the video filenames.\n"
            f"Matched {len(matched_names)}/{n_calib} calibrated camera names.\n"
            f"Missing: {missing}\n"
            "Refusing to guess the remaining camera order."
        )

    assignments.sort(key=lambda x: x[1])

    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {n} video(s).")
    print("Copy/rename mapping:")

    destinations = []
    for src, idx in assignments:
        dst = output_dir / f"{idx}.mp4"
        destinations.append((src, dst, idx))
        note = "  [NO CALIBRATION ENTRY]" if idx > n_calib else ""
        print(f"  {src.name:30s} -> {dst.name}{note}")

    if args.dry_run:
        print("\nDry run only; no files were copied.")
        return

    # Check destination conflicts before copying anything.
    existing = [dst for _, dst, _ in destinations if dst.exists()]
    if existing and not args.overwrite:
        shown = "\n".join(f"  {p}" for p in existing[:10])
        more = "" if len(existing) <= 10 else f"\n  ... and {len(existing) - 10} more"
        sys.exit(
            "ERROR: destination file(s) already exist:\n"
            f"{shown}{more}\n"
            "Use --overwrite if you want to replace them."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for src, dst, _ in destinations:
        shutil.copy2(src, dst)

    print(
        f"\nDone: copied {len(assignments)} videos to "
        f"{output_dir}/1.mp4 ... {output_dir}/{len(assignments)}.mp4"
    )
    print("Original input videos were left unchanged.")

    if len(assignments) == n_calib:
        print(
            f"These {n_calib} video indices correspond to cameras "
            f"1..{n_calib} in the modified intri.yml/extri.yml."
        )
    elif len(assignments) > n_calib:
        print(
            f"NOTE: camera indices above {n_calib} do not have calibration "
            "entries in the supplied YAML files."
        )


if __name__ == "__main__":
    main()
