#!/usr/bin/env python3
"""Link resources a secondary take shares with take01 of the same dataset.

    python link_shared_take_resources.py --dataset_dir <ds> --take_dir <ds>/takes/take02 \
        --require meshes/compressed.ply

Creates (or refreshes) relative symlinks <take_dir>/<name> -> <dataset_dir>/<name> for every
shared folder (calib/, meshes/) that exists in the dataset folder. A real folder of the same
name inside the take is left untouched. Every --require path must exist inside the take
afterwards, otherwise the step fails with a clear message (e.g. take01's Stage 2 never ran).
"""
import argparse
import os
import sys
from pathlib import Path

SHARED_DIRS = ("calib", "meshes")


def ensure_link(link: Path, target: Path) -> str:
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return f"[take-link] {link.name}/ already -> {os.readlink(link)}"
        link.unlink()
    elif link.exists():
        return f"[take-link] {link.name}/ is a real folder inside the take; left untouched"
    rel = os.path.relpath(target.resolve(), link.parent.resolve())
    link.symlink_to(rel, target_is_directory=True)
    return f"[take-link] {link.name}/ -> {rel}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True, help="dataset folder (take01 lives here)")
    ap.add_argument("--take_dir", required=True, help="working folder of the secondary take")
    ap.add_argument("--require", nargs="*", default=[], help="paths relative to the take that must exist afterwards")
    args = ap.parse_args()

    ds = Path(args.dataset_dir).resolve()
    take = Path(args.take_dir).resolve()
    if not ds.is_dir():
        print(f"[take-link] ERROR: dataset folder not found: {ds}", file=sys.stderr)
        return 1
    take.mkdir(parents=True, exist_ok=True)

    for name in SHARED_DIRS:
        src = ds / name
        if src.is_dir():
            print(ensure_link(take / name, src))
        else:
            print(f"[take-link] {name}/ not present in {ds.name}; nothing to link")

    missing = [r for r in args.require if not (take / r).exists()]
    if missing:
        print(f"[take-link] ERROR: shared resource(s) missing in {take}: {', '.join(missing)} — "
              f"run Stage 2 (object reconstruction) on take01 of {ds.name} first", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
