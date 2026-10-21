#!/usr/bin/env python3
"""Inspect first-frame object masks and choose camera sets for reconstruction.

Policy
------
DEPTH_CAMS:
    Use ALL cameras with a valid/non-empty first-frame object mask.

MESH_CAMS:
    Read <dataset>/extri.yml, recover each camera center C = -R^T T,
    split the camera rig into four rings from smallest -> largest using
    the three largest gaps in camera-center radius, then use ONLY
    Ring 1, Ring 2 and Ring 3.

    From each of those three rings, select exactly 4 usable cameras whose
    azimuths are as evenly spaced as possible around the ring.  The objective
    minimizes the squared deviation of the four circular angular gaps from 90°.

Thus, when all expected masks are valid:
    MESH_CAMS  = 4 + 4 + 4 = 12 cameras
    DEPTH_CAMS = all cameras (31 for the current rig)

Cameras with empty masks are never sent to MV-SAM3D because the downstream
reconstruction script rejects empty masks.

Usage:
    python mask_views.py check  --data_dir <dataset>
    eval "$(python mask_views.py select --data_dir <dataset>)"
"""

import argparse
import itertools
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def mask_pixels(mask_dir: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in sorted(mask_dir.iterdir(), key=lambda p: natural_key(p.name)):
        if p.suffix.lower() not in MASK_EXTS:
            continue
        cam = re.sub(r"_mask$", "", p.stem)
        if not cam.isdigit():
            continue
        try:
            a = np.asarray(Image.open(p).convert("L"))
            out[cam] = int((a > 127).sum())
        except Exception as exc:  # noqa: BLE001
            print(f"[masks] WARNING: cannot read {p.name}: {exc}", file=sys.stderr)
            out[cam] = 0
    return out


def _parse_opencv_matrix_data(text: str, key: str):
    """Parse `key: !!opencv-matrix ... data: [...]` without requiring OpenCV/PyYAML."""
    pat = re.compile(
        rf"(?ms)^\s*{re.escape(key)}\s*:\s*!!opencv-matrix\s*"
        rf".*?^\s*data\s*:\s*\[([^\]]+)\]"
    )
    m = pat.search(text)
    if m is None:
        return None
    vals = [
        float(x.strip())
        for x in m.group(1).replace("\n", " ").split(",")
        if x.strip()
    ]
    return np.asarray(vals, dtype=np.float64)


def load_camera_centers(extri_path: Path, cam_ids: List[int]) -> Dict[int, np.ndarray]:
    """Return world-space camera centers from EasyMocap/OpenCV world_to_cam extrinsics.

    This parser deliberately has no cv2 dependency.  It reads the already-stored
    3x3 Rot_<id> matrix and T_<id> vector directly from extri.yml.
    """
    if not extri_path.is_file():
        raise FileNotFoundError(f"Missing extrinsics: {extri_path}")

    raw = extri_path.read_text(encoding="utf-8", errors="replace")
    centers: Dict[int, np.ndarray] = {}

    for cam in cam_ids:
        R = None
        T = None

        for key in (f"Rot_{cam}", f"Rot{cam}"):
            arr = _parse_opencv_matrix_data(raw, key)
            if arr is not None and arr.size == 9:
                R = arr.reshape(3, 3)
                break

        for key in (f"T_{cam}", f"T{cam}", f"Trans_{cam}", f"trans_{cam}"):
            arr = _parse_opencv_matrix_data(raw, key)
            if arr is not None and arr.size >= 3:
                T = arr[:3].reshape(3)
                break

        if R is None or T is None:
            print(
                f"[rings] WARNING: missing Rot/T for cam {cam}; "
                f"excluding it from ring analysis",
                file=sys.stderr,
            )
            continue

        # world_to_cam:
        # X_cam = R @ X_world + T
        # camera center in world coordinates = -R^T @ T
        centers[cam] = -(R.T @ T)

    return centers


def split_four_rings(centers: Dict[int, np.ndarray]) -> List[List[int]]:
    """Split cameras into four radius clusters using the 3 largest radial gaps."""
    if len(centers) < 4:
        raise RuntimeError("Need at least 4 calibrated cameras to infer four rings.")

    ordered = sorted(
        ((float(np.linalg.norm(C)), cam) for cam, C in centers.items()),
        key=lambda x: (x[0], x[1]),
    )

    if len(ordered) < 8:
        raise RuntimeError(
            f"Only {len(ordered)} calibrated cameras available; cannot robustly infer four rings."
        )

    gaps = [
        (ordered[i + 1][0] - ordered[i][0], i)
        for i in range(len(ordered) - 1)
    ]
    cut_after = sorted(i for _gap, i in sorted(gaps, reverse=True)[:3])

    groups: List[List[int]] = []
    start = 0
    for cut in cut_after:
        groups.append([cam for _r, cam in ordered[start : cut + 1]])
        start = cut + 1
    groups.append([cam for _r, cam in ordered[start:]])

    if len(groups) != 4 or any(not g for g in groups):
        raise RuntimeError(f"Failed to split cameras into four non-empty rings: {groups}")

    # Already smallest -> largest because ordered is sorted by radius.
    return groups


def camera_azimuth_deg(
    cam: int,
    ring_all_ids: List[int],
    centers: Dict[int, np.ndarray],
) -> float:
    """Azimuth around the ring centroid in the XY plane."""
    ring_xyz = np.stack([centers[c] for c in ring_all_ids], axis=0)
    center = ring_xyz.mean(axis=0)
    d = centers[cam] - center
    return float(math.degrees(math.atan2(float(d[1]), float(d[0]))) % 360.0)


def choose_four_uniform(
    ring_all_ids: List[int],
    usable_ids: List[int],
    centers: Dict[int, np.ndarray],
) -> Tuple[List[int], List[float], List[float], float]:
    """Pick 4 usable cameras with circular angular gaps closest to 90 degrees."""
    candidates = [c for c in ring_all_ids if c in usable_ids and c in centers]
    if len(candidates) < 4:
        raise RuntimeError(
            f"Ring has only {len(candidates)} usable camera(s): {candidates}; need >= 4."
        )

    az = {
        c: camera_azimuth_deg(c, ring_all_ids, centers)
        for c in candidates
    }

    best = None
    for combo in itertools.combinations(candidates, 4):
        ordered = sorted(combo, key=lambda c: az[c])
        angles = [az[c] for c in ordered]
        gaps = [
            angles[1] - angles[0],
            angles[2] - angles[1],
            angles[3] - angles[2],
            angles[0] + 360.0 - angles[3],
        ]

        # Main objective: four ~90-degree sectors.
        score = float(sum((g - 90.0) ** 2 for g in gaps))

        # Deterministic tie-breaker by camera IDs.
        key = (score, tuple(ordered))
        if best is None or key < best[0]:
            best = (key, ordered, angles, gaps, score)

    assert best is not None
    return best[1], best[2], best[3], best[4]


def infer_and_select(
    data_dir: Path,
    usable_ids: List[int],
) -> Tuple[List[List[int]], List[int], List[dict], Dict[int, np.ndarray]]:
    """Infer rings and choose 4 uniform views from Rings 1-3."""
    all_mask_ids = sorted(usable_ids)
    # Ring geometry must be inferred from ALL calibrated cameras, not only good masks,
    # otherwise an empty mask could distort ring boundaries.
    extri_path = data_dir / "extri.yml"

    # Discover calibrated IDs from mask IDs + common 1..31 range.  For this rig the
    # mask directory contains all cameras; use those IDs for the geometry partition.
    mask_dir = data_dir / "mask"
    px = mask_pixels(mask_dir)
    all_ids = sorted(int(c) for c in px.keys())

    centers = load_camera_centers(extri_path, all_ids)
    rings = split_four_rings(centers)

    selected_all: List[int] = []
    details: List[dict] = []

    for ring_idx, ring_ids in enumerate(rings[:3], start=1):
        chosen, angles, gaps, score = choose_four_uniform(
            ring_all_ids=ring_ids,
            usable_ids=all_mask_ids,
            centers=centers,
        )
        selected_all.extend(chosen)
        details.append(
            {
                "ring": ring_idx,
                "all": ring_ids,
                "chosen": chosen,
                "angles": angles,
                "gaps": gaps,
                "score": score,
                "mean_radius": float(
                    np.mean([np.linalg.norm(centers[c]) for c in ring_ids])
                ),
            }
        )

    return rings, selected_all, details, centers


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("mode", choices=["check", "select"])
    ap.add_argument("--data_dir", required=True, help="Dataset root containing mask/ and extri.yml")
    ap.add_argument("--mask_dir_name", default="mask")
    ap.add_argument(
        "--min_cams",
        type=int,
        default=12,
        help="Minimum total cameras with a non-empty mask. Default: 12.",
    )
    ap.add_argument(
        "--min_pixels",
        type=int,
        default=50,
        help="A mask with fewer foreground pixels counts as empty.",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    mask_dir = data_dir / args.mask_dir_name
    if not mask_dir.is_dir():
        print(
            f"[masks] ERROR: {mask_dir} does not exist "
            f"(run the SAM3 reconstruction-mask step first)",
            file=sys.stderr,
        )
        return 1

    px = mask_pixels(mask_dir)
    if not px:
        print(f"[masks] ERROR: no per-camera masks in {mask_dir}", file=sys.stderr)
        return 1

    good = [c for c, n in px.items() if n >= args.min_pixels]
    empty = [c for c, n in px.items() if n < args.min_pixels]

    print(
        f"[masks] {len(px)} first-frame masks: "
        f"{len(good)} usable, {len(empty)} empty"
        + (f" (cams {', '.join(empty)})" if empty else ""),
        file=sys.stderr,
    )
    print(
        "[masks] pixels per cam: "
        + " ".join(f"{c}={n}" for c, n in px.items()),
        file=sys.stderr,
    )

    if len(good) < args.min_cams:
        print(
            f"[masks] ERROR: only {len(good)} camera(s) have an object mask "
            f"(need >= {args.min_cams}).",
            file=sys.stderr,
        )
        return 1

    good_ids = sorted(int(c) for c in good)

    try:
        rings, mesh_cams, details, centers = infer_and_select(
            data_dir=data_dir,
            usable_ids=good_ids,
        )
    except Exception as exc:
        print(f"[rings] ERROR: {exc}", file=sys.stderr)
        return 1

    print("[rings] inferred camera rings (small -> large):", file=sys.stderr)
    for i, ids in enumerate(rings, start=1):
        mean_r = np.mean([np.linalg.norm(centers[c]) for c in ids])
        good_here = [c for c in ids if c in good_ids]
        print(
            f"[rings] Ring {i}: mean_radius={mean_r:.6f}, "
            f"all={ids}, usable={good_here}",
            file=sys.stderr,
        )

    print(
        "[mesh] Ring1-3 uniform selection:",
        file=sys.stderr,
    )
    for d in details:
        print(
            f"[mesh] Ring {d['ring']}: chosen={d['chosen']} "
            f"azimuth={[round(a, 1) for a in d['angles']]} deg "
            f"gaps={[round(g, 1) for g in d['gaps']]} deg",
            file=sys.stderr,
        )

    if len(mesh_cams) != 12:
        print(
            f"[mesh] ERROR: expected 12 mesh cameras, got {len(mesh_cams)}: {mesh_cams}",
            file=sys.stderr,
        )
        return 1

    # Depth fusion uses every camera whose first-frame mask is valid.
    depth_cams = good_ids

    if args.mode == "check":
        if empty:
            print(
                "[masks] WARNING: empty-mask cameras are excluded from depth fusion "
                "because downstream MV-SAM3D rejects empty masks.",
                file=sys.stderr,
            )
        print(
            f"[check] OK: MESH_CAMS=12 (4 each from Ring1-3), "
            f"DEPTH_CAMS={len(depth_cams)} usable camera(s).",
            file=sys.stderr,
        )
        return 0

    print(
        f"[masks] MESH_CAMS={len(mesh_cams)} cams "
        f"(4 each from Ring1-3); DEPTH_CAMS={len(depth_cams)} cams",
        file=sys.stderr,
    )

    # stdout intentionally contains only shell assignments so callers can:
    #   eval "$(python mask_views.py select ...)"
    print(
        f"MESH_CAMS={','.join(map(str, mesh_cams))}; "
        f"DEPTH_CAMS={','.join(map(str, depth_cams))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
