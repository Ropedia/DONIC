#!/usr/bin/env python3
"""Resize EasyMocap images and their corresponding masks in place.

Images:
  - All supported image files below --image_root are recursively resized.
  - Images are saved as PNG, matching the behavior of the original script.

Masks:
  - All supported mask files below --mask_root are recursively resized.
  - Masks keep their original filename and format.
  - Nearest-neighbor interpolation is used so binary/label values are not blended.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resize every image under image_root and every mask under mask_root "
            "recursively, in place."
        )
    )
    parser.add_argument(
        "--image_root",
        type=Path,
        default=Path("my_sequence/images"),
        help="Image root, e.g. <data-root>/my_sequence/images (default: my_sequence/images).",
    )
    parser.add_argument(
        "--mask_root",
        type=Path,
        default=None,
        help=(
            "Mask root, e.g. <data-root>/my_sequence/masks. If omitted, this is "
            "inferred as <image_root_parent>/masks."
        ),
    )
    parser.add_argument(
        "--w",
        type=int,
        default=640,
        help="Target width in pixels (default: 640).",
    )
    parser.add_argument(
        "--h",
        type=int,
        default=481,
        help="Target height in pixels (default: 481).",
    )
    parser.add_argument(
        "--skip_images",
        action="store_true",
        help="Only resize masks; do not resize files under image_root.",
    )
    parser.add_argument(
        "--skip_masks",
        action="store_true",
        help="Only resize images; do not resize files under mask_root.",
    )
    return parser.parse_args()


def iter_supported_files(root: Path) -> Iterable[Path]:
    """Yield supported image files recursively under root in a stable order."""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            yield path


def resize_array(array, width: int, height: int, interpolation: int):
    return cv2.resize(array, (width, height), interpolation=interpolation)


def resize_image_file(path: Path, width: int, height: int) -> bool:
    """Resize an ordinary image and save it as PNG, deleting old non-PNG input."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"[image] skip unreadable file: {path}")
        return False

    resized = resize_array(image, width, height, cv2.INTER_AREA)
    output_path = path.with_suffix(".png")

    if not cv2.imwrite(str(output_path), resized):
        print(f"[image] failed to save: {output_path}")
        return False

    if output_path != path:
        path.unlink()
        print(f"[image] converted + resized: {path} -> {output_path}")
    else:
        print(f"[image] resized: {output_path}")
    return True


def resize_mask_file(path: Path, width: int, height: int) -> bool:
    """Resize a mask in place without changing labels, bit depth, channels, or filename."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"[mask] skip unreadable file: {path}")
        return False

    # Never use linear/area interpolation for a segmentation or instance mask:
    # it would create intermediate, invalid label values along boundaries.
    resized = resize_array(mask, width, height, cv2.INTER_NEAREST)

    if not cv2.imwrite(str(path), resized):
        print(f"[mask] failed to save: {path}")
        return False

    print(f"[mask] resized: {path}")
    return True


def process_root(
    root: Path,
    width: int,
    height: int,
    kind: str,
) -> tuple[int, int]:
    """Process one root and return (successful_file_count, discovered_file_count)."""
    if not root.exists():
        print(f"[{kind}] root does not exist: {root}")
        return 0, 0
    if not root.is_dir():
        print(f"[{kind}] root is not a directory: {root}")
        return 0, 0

    files = list(iter_supported_files(root))
    print(f"\n[{kind}] root: {root}")
    print(f"[{kind}] supported files found: {len(files)}")

    success = 0
    for path in files:
        if kind == "image":
            success += resize_image_file(path, width, height)
        else:
            success += resize_mask_file(path, width, height)

    return success, len(files)


def main() -> None:
    args = parse_args()

    if args.w <= 0 or args.h <= 0:
        raise ValueError(f"--w and --h must be positive, got w={args.w}, h={args.h}")
    if args.skip_images and args.skip_masks:
        raise ValueError("Both --skip_images and --skip_masks were set; there is nothing to do.")

    image_root = args.image_root.expanduser().resolve()
    mask_root = (
        args.mask_root.expanduser().resolve()
        if args.mask_root is not None
        else image_root.parent / "masks"
    )

    print(f"Target size: {args.w} x {args.h}")
    print(f"Image root:  {image_root}")
    print(f"Mask root:   {mask_root}")

    image_success = image_total = mask_success = mask_total = 0

    if not args.skip_images:
        image_success, image_total = process_root(
            image_root, args.w, args.h, kind="image"
        )

    if not args.skip_masks:
        mask_success, mask_total = process_root(
            mask_root, args.w, args.h, kind="mask"
        )

    print("\nDone.")
    if not args.skip_images:
        print(f"Images resized: {image_success}/{image_total}")
    if not args.skip_masks:
        print(f"Masks resized:  {mask_success}/{mask_total}")


if __name__ == "__main__":
    main()
