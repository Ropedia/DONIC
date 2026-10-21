#!/usr/bin/env python3
"""Check / fix the metric scale of meshes/compressed.ply using calibrated multi-view silhouettes.

MV-SAM3D recovers the object's metric scale from Depth-Anything-3 depth inside the first-frame
masks (mask/, taken at t=0 from images1/). The depth maps, however, start at the pipeline's
start_sec (depth/<cam>/000000.png = images/<cam>/000000.jpg). When the object moves in between
— typical when the hand grabs it during the first second — the masked depth is just the table
where the object used to be, the fused cloud becomes a large flat blob and the mesh comes out
several times too big. DA3's absolute depth is also only accurate to tens of percent.

This script needs neither depth nor timing agreement:
  1. triangulate the object centre from the mask/ centroids of all calibrated cameras
     (greedy reprojection-based rejection),
  2. in every view, measure the mask's major/minor extent (robust PCA, 2-98 percentile) and
     convert it to metres at the triangulated distance (ext_px * z / f),
  3. the object's 3D major extent is approximated by a high percentile of the per-view major
     extents (some camera in the rig sees the object side-on),
  4. compare with the mesh's own robust major extent; if they disagree by more than
     --tolerance, rescale the mesh about its bounding-box centre and record everything in
     meshes/metric_scale_info.json (backup: meshes/compressed_prescale.ply, written once).

Runs in the `easymocap` environment (cv2, numpy, trimesh).
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def load_first_masks(ds: Path, cams, ref_hw):
    """{cam: bool mask at the calibration resolution}. Uses mask/<cam>.png — the masks MV-SAM3D
    itself consumed (same instant as images1/) — and falls back to masks/<cam>/<first frame>."""
    out = {}
    for c in cams:
        cand = []
        for ext in (".png", ".jpg", ".jpeg"):
            p = ds / "mask" / f"{c}{ext}"
            if p.is_file():
                cand = [p]
                break
        if not cand and (ds / "masks" / c).is_dir():
            cand = sorted((ds / "masks" / c).glob("*.png"), key=lambda p: natural_key(p.name))[:1]
        if not cand:
            continue
        m = cv2.imread(str(cand[0]), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != tuple(ref_hw):
            m = cv2.resize(m, (ref_hw[1], ref_hw[0]), interpolation=cv2.INTER_NEAREST)
        b = m > 127
        if b.sum() >= 50:
            out[c] = b
    return out


def reference_resolution(ds: Path, cams: dict) -> tuple:
    """Resolution the intrinsics refer to: the resized images/ frames, else 2*principal point."""
    for c in cams:
        d = ds / "images" / c
        if d.is_dir():
            for p in sorted(d.iterdir())[:1]:
                im = cv2.imread(str(p))
                if im is not None:
                    return im.shape[:2]
    K = next(iter(cams.values()))["K"]
    return (int(round(2 * K[1, 2])), int(round(2 * K[0, 2])))


def triangulate(cams, obs):
    A = []
    for c, (u, v) in obs.items():
        P = cams[c]["P"]
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    return X[:3] / X[3]


def reproj_err(cams, c, X, uv):
    p = cams[c]["P"] @ np.append(X, 1.0)
    return float(np.hypot(*(p[:2] / p[2] - np.asarray(uv))))


def robust_pca_extent(pts: np.ndarray, lo=2.0, hi=98.0) -> np.ndarray:
    c = pts - pts.mean(0)
    sub = c[:: max(1, len(c) // 50000)]
    _, _, Vt = np.linalg.svd(sub, full_matrices=False)
    proj = c @ Vt.T
    return np.percentile(proj, hi, axis=0) - np.percentile(proj, lo, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--mesh", default="meshes/compressed.ply")
    ap.add_argument("--tolerance", type=float, default=0.20, help="Relative disagreement above which the mesh is rescaled")
    ap.add_argument("--percentile", type=float, default=90.0, help="Percentile of per-view major extents used as the 3D major extent")
    ap.add_argument("--max_reproj_px", type=float, default=15.0)
    ap.add_argument("--min_views", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from easymocap.mytools.camera_utils import read_camera  # noqa: E402
    import trimesh  # noqa: E402

    ds = Path(args.data_dir).resolve()
    mesh_path = ds / args.mesh
    if not mesh_path.is_file():
        print(f"[scale] ERROR: {mesh_path} not found", file=sys.stderr)
        return 1
    cams = read_camera(str(ds / "intri.yml"), str(ds / "extri.yml"))
    cams.pop("basenames", None)
    ref_hw = reference_resolution(ds, cams)
    masks = load_first_masks(ds, sorted(cams, key=natural_key), ref_hw)
    if len(masks) < args.min_views:
        print(f"[scale] ERROR: only {len(masks)} usable first-frame masks (need >= {args.min_views})", file=sys.stderr)
        return 1

    # 1) object centre from mask centroids, with reprojection-based rejection
    obs = {}
    for c, m in masks.items():
        ys, xs = np.nonzero(m)
        obs[c] = (float(xs.mean()), float(ys.mean()))
    # Greedy rejection: drop the worst view until every remaining view reprojects within tolerance.
    good = dict(obs)
    while True:
        X = triangulate(cams, good)
        errs = {c: reproj_err(cams, c, X, uv) for c, uv in good.items()}
        worst = max(errs, key=errs.get)
        if errs[worst] <= args.max_reproj_px or len(good) <= args.min_views:
            break
        good.pop(worst)
    if len(good) < args.min_views or max(errs.values()) > args.max_reproj_px:
        print(f"[scale] ERROR: only {len(good)} views agree on the object centre "
              f"(worst reprojection {max(errs.values()):.1f} px)", file=sys.stderr)
        return 1

    # 2) per-view metric extents of the silhouette at the triangulated distance
    per_view = {}
    for c in good:
        ys, xs = np.nonzero(masks[c])
        pts = np.stack([xs, ys], 1).astype(np.float64)
        ext_px = robust_pca_extent(pts)  # major, minor in pixels
        z = float((cams[c]["R"] @ X + cams[c]["T"].reshape(3))[2])
        f = 0.5 * (cams[c]["K"][0, 0] + cams[c]["K"][1, 1])
        per_view[c] = (float(ext_px[0] * z / f), float(ext_px[1] * z / f), z)
    majors = np.array([v[0] for v in per_view.values()])
    minors = np.array([v[1] for v in per_view.values()])
    sil_major = float(np.percentile(majors, args.percentile))
    sil_minor_med = float(np.median(minors))

    # 3) the mesh's own major extent, same robust measure
    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    mesh_ext = robust_pca_extent(np.asarray(mesh.vertices, dtype=np.float64))
    mesh_major = float(mesh_ext[0])
    ratio = sil_major / mesh_major

    print(f"[scale] object centre (world): {np.round(X, 3).tolist()} from {len(good)}/{len(masks)} views, "
          f"reproj err median {np.median(list(errs.values())):.1f} px")
    print(f"[scale] silhouette major extent: median {np.median(majors):.3f} m, p{args.percentile:.0f} {sil_major:.3f} m, max {majors.max():.3f} m; "
          f"minor median {sil_minor_med:.3f} m")
    print(f"[scale] mesh robust extent (m): {np.round(mesh_ext, 3).tolist()} -> major {mesh_major:.3f} m")
    print(f"[scale] silhouette / mesh = {ratio:.3f}")

    info_path = ds / "meshes" / "metric_scale_info.json"
    info = {}
    if info_path.is_file():
        try:
            info = json.load(open(info_path))
        except Exception:
            info = {}
    report = {
        "checked_at": time.time(),
        "views_used": sorted(good, key=natural_key),
        "object_centre_world": [float(x) for x in X],
        "silhouette_major_m": sil_major,
        "silhouette_major_median_m": float(np.median(majors)),
        "silhouette_minor_median_m": sil_minor_med,
        "mesh_major_before_m": mesh_major,
        "ratio": ratio,
        "tolerance": args.tolerance,
        "applied": False,
    }
    if abs(np.log(ratio)) <= np.log(1.0 + args.tolerance):
        print(f"[scale] OK: mesh scale agrees with the silhouettes within {args.tolerance * 100:.0f}%")
    elif args.dry_run:
        print(f"[scale] DRY RUN: would rescale the mesh by {ratio:.3f} "
              f"(major extent {mesh_major:.3f} m -> {sil_major:.3f} m)")
    else:
        backup = mesh_path.with_name(mesh_path.stem + "_prescale.ply")
        if not backup.exists():
            shutil.copy2(mesh_path, backup)
        centre = mesh.bounding_box.centroid
        mesh.vertices = (np.asarray(mesh.vertices) - centre) * ratio + centre
        mesh.export(str(mesh_path))
        report["applied"] = True
        report["backup"] = str(backup)
        print(f"[scale] RESCALED mesh by {ratio:.3f}: major extent {mesh_major:.3f} m -> {sil_major:.3f} m "
              f"(original kept as {backup.name})")
    info["silhouette_scale_check"] = report
    if not args.dry_run:
        tmp = info_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(info, f, indent=2)
        tmp.replace(info_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
