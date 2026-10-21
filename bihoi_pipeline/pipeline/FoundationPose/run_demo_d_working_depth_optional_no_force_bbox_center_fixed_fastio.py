# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
File responsibility / module role:
    Per-frame multi-camera FoundationPose registration without temporal tracking.

    This version preserves the original working RGB-only pipeline and adds OPTIONAL
    depth use via --use_depth 0/1. For every frame, it loads RGB and object masks from
    all cameras, estimates the object center by the ORIGINAL RANSAC triangulation code,
    selects the camera whose object mask has the largest foreground area, and
    runs FoundationPose registration from the same triangulated object-center XYZ.

    Optional --tracking_cam <id> strictly locks FoundationPose registration/tracking/refinement
    to one camera, while multi-view RANSAC triangulation still uses all valid camera masks. In this
    mode the centered-mesh xyz is always locked to the triangulated center.

    When --use_depth 1, uint16 relative depth is divided by 1000.0. By default
    (--do-not-rescale-depth true), that depth is passed to FoundationPose without scaling
    it to the triangulated object-center Z. Set --do-not-rescale-depth false to restore the
    historical masked-median -> triangulated-camera-Z scale alignment.

    IMPORTANT pose convention: FoundationPose refinement/scoring/registration first run normally.
    By default, this script preserves the historical behavior: after the best pose is selected, the
    CENTERED-MESH translation is reset to the triangulated object center. Pass --no-force to disable
    that post-selection recentering and keep FoundationPose's refined translation unchanged. This
    applies to the normal path AND the fast / fast-fallback paths, except when --tracking_cam is used,
    which still forces the centered-mesh translation to the triangulated xyz. The triangulated xyz is
    always used as the initialization for registration / temporal hypotheses. When recentering is
    enabled, the ORIGINAL-MESH pose is converted through the centered-mesh frame before applying the
    xyz lock so mesh-origin offsets remain correct.

    BBOX CENTER FIX: visualization now uses est.get_tf_to_centered_mesh() directly.
    It no longer uses trimesh.bounds.oriented_bounds(mesh) as the visualization center, so the
    rendered bbox/XYZ origin is the same FoundationPose centered-mesh origin that is locked to the
    triangulated xyz. This removes the fixed AABB-center vs OBB-center offset.

    OFFLINE XYZ PIPELINE (10 FPS default): there is NO XYZ Kalman during FoundationPose inference.
    For a normal fast frame, the top-10 largest-mask views are independently refined from the
    previous accepted WORLD rotation. A deterministic RANSAC-style consensus search finds the
    largest subset (minimum 3 views) whose members are mutually <30 deg apart in WORLD rotation
    and within 5 cm in refined WORLD translation. The accepted cluster's XYZ translations are
    averaged with mask area as weight, while rotation is NOT fused: the final WORLD rotation is taken
    directly from the largest-mask (best) camera inside the accepted consensus cluster. If no satisfactory
    cluster exists, the frame falls
    back to full registration on the largest-mask view. For the first full-registration frame, the
    registration/refinement XYZ is used. AFTER the complete pass, XYZ spikes are detected with a process_dataset.py-style
    local median+MAD velocity threshold capped at 0.80 m/s (0.08 m/frame at 10 FPS), short bad
    bursts are linearly interpolated, and only then is the full repaired XYZ sequence run forward
    through the constant-velocity XYZ Kalman filter. Rotation is untouched by this XYZ stage.

    AFTER the XYZ interpolation+Kalman stage, the existing optional offline rotation-fix stage
    detects short SO(3) angular-velocity spike bursts and replaces only those burst rotations by
    geodesic SLERP between trustworthy boundary rotations. Raw pre-XYZ-postprocess poses are
    backed up separately.

    OPTIONAL POSTKAL ROTATION FILTER: when --postkal is enabled, AFTER the SLERP rotfix stage,
    the complete repaired CENTERED-object WORLD rotation sequence is run once from first to last
    through a causal multiplicative SO(3) constant-angular-velocity Kalman filter. Default FPS is
    10. Translation is copied exactly from the rotfix sequence; only rotation is filtered. The
    postkal outputs are saved separately so raw and SLERP-only poses remain available.

Expected scene layout:
    test_scene_dir/
        images/<cam_id>/<frame_id>.png
        masks/<cam_id>/<frame_id>.png or .jpg
        depth/<cam_id>/<frame_id>.png     # optional uint16 relative depth, stored x1000
        cam_K.txt                         # fallback shared intrinsic
        intri.yml                         # optional per-camera K_<cam_id>
        extri.yml                         # required for converting selected cam pose to main cam

Pose convention:
    FoundationPose.register returns ob_in_cam_selected:
        X_selected_cam = ob_in_cam_selected @ X_object

    The saved pose is ob_in_cam_main:
        X_main_cam = ob_in_cam_main @ X_object

    If selected_cam != main_cam:
        X_selected_cam = T_selected_from_main @ X_main_cam
        ob_in_cam_main = inv(T_selected_from_main) @ ob_in_cam_selected
"""

from estimater import *
from datareader import *

import argparse
import csv
import glob
import itertools
import logging
import os
import shutil
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import trimesh

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm as mpl_cm
    from matplotlib import colors as mpl_colors
    _HAS_MATPLOTLIB = True
    _MATPLOTLIB_IMPORT_ERROR = None
except Exception as _mpl_e:
    plt = None
    mpl_cm = None
    mpl_colors = None
    _HAS_MATPLOTLIB = False
    _MATPLOTLIB_IMPORT_ERROR = _mpl_e


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tif")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tif")
DEPTH_EXTS = (".png", ".tif", ".tif", ".npy")


def readextfromyml(video_dir):
    """
    Read all camera extrinsics from video_dir/extri.yml.

    Returns:
        cam_ext: dict
            cam_ext[cam_name]["ext"] is world-to-camera extrinsic [R|T].
    """
    extri_path = os.path.join(video_dir, "extri.yml")
    if not os.path.exists(extri_path):
        raise FileNotFoundError(f"Cannot find extri.yml: {extri_path}")

    fs = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open extri.yml: {extri_path}")

    names_node = fs.getNode("names")
    if names_node.empty():
        fs.release()
        raise ValueError("No 'names' field found in extri.yml")

    cam_names = []
    for i in range(names_node.size()):
        cam_names.append(names_node.at(i).string())

    cam_ext = {}
    for cam_name in cam_names:
        rvec_key = f"R_{cam_name}"
        rot_key = f"Rot_{cam_name}"
        t_key = f"T_{cam_name}"

        rvec_node = fs.getNode(rvec_key)
        rot_node = fs.getNode(rot_key)
        t_node = fs.getNode(t_key)

        if rvec_node.empty() or rot_node.empty() or t_node.empty():
            logging.warning(f"Missing extrinsic entry for camera {cam_name}, skip.")
            continue

        rvec = rvec_node.mat()
        rot = rot_node.mat()
        t = t_node.mat()
        if rvec is None or rot is None or t is None:
            logging.warning(f"Failed to read extrinsic matrices for camera {cam_name}, skip.")
            continue

        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t, dtype=np.float64).reshape(3, 1)

        ext = np.eye(4, dtype=np.float64)
        ext[:3, :3] = rot
        ext[:3, 3:] = t

        cam_ext[str(cam_name)] = {
            "Rvec": rvec,
            "Rot": rot,
            "T": t,
            "ext": ext,
        }

    fs.release()
    return cam_ext


def read_intrinsics_from_txt(video_dir, downscale=1.0, txt_name="cam_K.txt"):
    """
    Read the shared camera intrinsic matrix only from video_dir/cam_K.txt.
    This function does NOT read intri.yml.

    Expected cam_K.txt format can be either:

    1) 3x3 matrix:
        fx 0  cx
        0  fy cy
        0  0  1

    2) flattened 9 numbers:
        fx 0 cx 0 fy cy 0 0 1

    Return:
        K: np.ndarray with shape (3, 3)
    """
    txt_path = os.path.join(video_dir, txt_name)

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"[intrinsics] {txt_name} not found: {txt_path}")

    values = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Support both comma-separated and space-separated formats.
            line = line.replace(",", " ")
            for item in line.split():
                values.append(float(item))

    if len(values) != 9:
        raise ValueError(
            f"[intrinsics] expected 9 numbers in {txt_path}, "
            f"but got {len(values)} numbers: {values}"
        )

    K = np.asarray(values, dtype=np.float64).reshape(3, 3)

    # If image is downscaled, fx, fy, cx, cy must be downscaled too.
    K[:2] *= float(downscale)

    logging.info(f"[intrinsics] loaded shared K from txt: {txt_path}")
    logging.info(f"[intrinsics] K =\n{K}")

    return K


def read_intrinsics_from_yml(video_dir, downscale=1.0, txt_name="cam_K.txt"):
    """
    Backward-compatible wrapper.

    The old code calls read_intrinsics_from_yml(), but this modified version
    only reads video_dir/cam_K.txt and returns a single shared K matrix.
    """
    return read_intrinsics_from_txt(video_dir, downscale=downscale, txt_name=txt_name)


def pose_to_numpy_4x4(pose):
    """Convert FoundationPose pose tensor/array to a single 4x4 numpy matrix."""
    if torch.is_tensor(pose):
        pose_np = pose.detach().cpu().numpy()
    else:
        pose_np = np.asarray(pose)

    if pose_np.ndim == 3:
        pose_np = pose_np[0]

    return pose_np.reshape(4, 4).astype(np.float64)


def find_existing_file(folder, stem, exts):
    """Find frame file by trying several common extensions."""
    folder = str(folder)
    for ext in exts:
        p = os.path.join(folder, f"{stem}{ext}")
        if os.path.exists(p):
            return p

    # Fallback: case-insensitive glob, useful when extensions are unusual.
    candidates = []
    for p in glob.glob(os.path.join(folder, f"{stem}.*")):
        if os.path.splitext(p)[1].lower() in exts:
            candidates.append(p)
    if candidates:
        return sorted(candidates)[0]
    return None


def read_color_resized(path, target_hw):
    """Read RGB image and resize to the reader resolution."""
    color = imageio.imread(path)[..., :3]
    H, W = target_hw
    if color.shape[:2] != (H, W):
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
    return color


def read_depth_resized(path, target_hw):
    """
    Read uint16 NON-METRIC relative depth.

    The stored uint16 values are assumed to be relative depth quantized with x1000,
    so we restore the floating representation with /1000.0. This does NOT make the
    depth metric; metric scale alignment happens later using triangulated camera-Z.
    """
    suffix = os.path.splitext(path)[1].lower()
    if suffix not in (".png", ".tif", ".tif"):
        raise ValueError(f"Relative depth must be uint16 PNG/TIFF, got: {path}")

    depth_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Failed to read depth: {path}")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[..., 0]
    if depth_raw.dtype != np.uint16:
        raise TypeError(f"Depth must be uint16, but {path} has dtype={depth_raw.dtype}")

    H, W = target_hw
    if depth_raw.shape[:2] != (H, W):
        depth_raw = cv2.resize(depth_raw, (W, H), interpolation=cv2.INTER_NEAREST)

    depth_relative = depth_raw.astype(np.float32) / 1000.0
    depth_relative[~np.isfinite(depth_relative)] = 0.0
    depth_relative[depth_relative <= 0.0] = 0.0
    return depth_relative.astype(np.float32)


def rescale_relative_depth_to_triangulated_z(depth_relative, mask, z_reference, mask_only=False):
    """
    Align a non-metric relative depth map to triangulated object-center camera-Z.

        median_rel = median(depth_relative[mask & depth>0])
        scale      = z_reference / median_rel
        depth_used = depth_relative * scale

    Thus the valid masked median of depth_used equals the triangulated camera-Z.
    The same single scale is applied to the entire depth map so relative geometry is kept.
    """
    depth_relative = np.asarray(depth_relative, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if depth_relative.shape != mask.shape:
        raise ValueError(f"depth/mask shape mismatch: {depth_relative.shape} vs {mask.shape}")

    z_reference = float(z_reference)
    if (not np.isfinite(z_reference)) or z_reference <= 0.0:
        raise ValueError(f"Invalid triangulated camera-Z reference: {z_reference}")

    valid = mask & np.isfinite(depth_relative) & (depth_relative > 0.0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("No valid relative-depth pixels inside object mask")

    median_rel = float(np.median(depth_relative[valid].astype(np.float64)))
    if (not np.isfinite(median_rel)) or median_rel <= 1e-12:
        raise ValueError(f"Invalid masked relative-depth median: {median_rel}")

    scale = z_reference / median_rel
    depth_used = depth_relative.astype(np.float32) * np.float32(scale)
    depth_used[~np.isfinite(depth_used)] = 0.0
    depth_used[depth_relative <= 0.0] = 0.0
    depth_used[depth_used < 0.001] = 0.0

    if bool(mask_only):
        depth_used = depth_used.copy()
        depth_used[~mask] = 0.0

    valid_after = valid & (depth_used > 0.0)
    median_after = float(np.median(depth_used[valid_after].astype(np.float64))) if np.any(valid_after) else 0.0

    return depth_used.astype(np.float32), {
        "mode": "relative_depth_median_aligned",
        "relative_mask_median": median_rel,
        "z_reference": z_reference,
        "scale": float(scale),
        "aligned_mask_median": median_after,
        "valid_masked_pixels": n_valid,
    }


def prepare_depth_for_foundationpose(obs, init_xyz_selected, args):
    """
    Return the depth map passed to FoundationPose.

    Behavior when --use_depth 1:
      - default (--do-not-rescale-depth true): keep the loaded depth values unchanged
        after the existing uint16 / 1000 conversion; do NOT anchor them to triangulated Z.
      - --do-not-rescale-depth false: restore the historical masked-median alignment to
        the triangulated object-center camera-Z.

    --depth_mask_only is honored in both modes.
    """
    mask = np.asarray(obs["mask"], dtype=bool)
    z_reference = float(np.asarray(init_xyz_selected).reshape(3)[2])

    if not bool(int(getattr(args, "use_depth", 0))):
        return np.zeros(mask.shape, dtype=np.float32), {
            "mode": "depth_disabled",
            "relative_mask_median": 0.0,
            "z_reference": z_reference,
            "scale": 0.0,
            "aligned_mask_median": 0.0,
            "valid_masked_pixels": 0,
        }

    depth_relative = np.asarray(obs.get("depth", np.zeros(mask.shape)), dtype=np.float32)
    valid = mask & np.isfinite(depth_relative) & (depth_relative > 0.0)
    if not np.any(valid):
        if bool(getattr(args, "require_depth", False)):
            raise ValueError(
                f"Depth required but no valid masked depth: frame={obs.get('frame_id')} cam={obs.get('cam')}"
            )
        logging.warning(
            f"[depth] frame={obs.get('frame_id')} cam={obs.get('cam')}: no valid depth; fallback zero depth"
        )
        return np.zeros(mask.shape, dtype=np.float32), {
            "mode": "zero_depth_fallback",
            "relative_mask_median": 0.0,
            "z_reference": z_reference,
            "scale": 0.0,
            "aligned_mask_median": 0.0,
            "valid_masked_pixels": 0,
        }

    # New default: do NOT rescale depth using the triangulated object center.
    if bool(getattr(args, "do_not_rescale_depth", True)):
        depth_used = depth_relative.astype(np.float32).copy()
        depth_used[~np.isfinite(depth_used)] = 0.0
        depth_used[depth_used <= 0.0] = 0.0
        if bool(getattr(args, "depth_mask_only", False)):
            depth_used[~mask] = 0.0

        raw_median = float(np.median(depth_relative[valid].astype(np.float64)))
        valid_after = mask & np.isfinite(depth_used) & (depth_used > 0.0)
        output_median = (
            float(np.median(depth_used[valid_after].astype(np.float64)))
            if np.any(valid_after) else 0.0
        )
        return depth_used.astype(np.float32), {
            "mode": "relative_depth_unscaled",
            "relative_mask_median": raw_median,
            "z_reference": z_reference,  # logged only; NOT used to scale depth in this mode
            "scale": 1.0,
            "aligned_mask_median": output_median,
            "valid_masked_pixels": int(valid.sum()),
        }

    # Optional legacy behavior: align masked median depth to triangulated camera-Z.
    return rescale_relative_depth_to_triangulated_z(
        depth_relative=depth_relative,
        mask=mask,
        z_reference=z_reference,
        mask_only=bool(getattr(args, "depth_mask_only", False)),
    )


def read_mask_resized(path, target_hw):
    """Read object mask as bool array and resize to the reader resolution."""
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")

    if mask.ndim == 3:
        if mask.shape[2] == 4 and mask[..., 3].sum() > 0:
            mask = mask[..., 3]
        else:
            selected = None
            for c in range(mask.shape[2]):
                if mask[..., c].sum() > 0:
                    selected = mask[..., c]
                    break
            mask = selected if selected is not None else mask[..., 0]

    H, W = target_hw
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    max_val = float(np.nanmax(mask)) if mask.size else 0.0
    if max_val <= 1.0:
        return (mask > 0).astype(bool)
    return (mask > 127 if (mask > 127).sum() > 0 else mask > 0).astype(bool)


def compute_mask_total_brightness(color, mask):
    """
    Compute total brightness inside an object mask.

    The input image is RGB. Brightness is standard grayscale luminance:
        Y = 0.299 * R + 0.587 * G + 0.114 * B

    The camera-selection score is the SUM of Y over every foreground mask pixel,
    not the mean. Therefore a larger and/or brighter visible object region receives
    a larger score, exactly matching ``pixels * brightness`` summed over the mask.

    Returns:
        total_brightness: float
            Sum of grayscale luminance inside the mask.
        mean_brightness: float
            Mean grayscale luminance inside the mask, used only for logging/debug.
    """
    color = np.asarray(color)
    mask = np.asarray(mask, dtype=bool)

    if color.ndim != 3 or color.shape[2] < 3:
        raise ValueError(f"color must have shape (H,W,3+), got {color.shape}")
    if color.shape[:2] != mask.shape:
        raise ValueError(
            f"color/mask shape mismatch: color={color.shape[:2]}, mask={mask.shape}"
        )

    if not np.any(mask):
        return 0.0, 0.0

    rgb = color[..., :3].astype(np.float64)
    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )
    masked_gray = gray[mask]
    return float(masked_gray.sum(dtype=np.float64)), float(masked_gray.mean())


def compute_mask_center(mask, method="min_area_rect"):
    """
    Compute a 2D mask center for multi-view triangulation.

    Args:
        mask: bool or uint8 mask, shape (H, W).
        method:
            - "min_area_rect": OpenCV rotated bounding-box center. This is the
              default and is usually better than axis-aligned bbox center for
              tilted/slender masks.
            - "pca_rotated_box": PCA-oriented bounding-box center. This uses
              the principal axes of the mask pixels, projects the mask onto
              those axes, takes the center of the min/max ranges, and maps it
              back to image coordinates.
            - "centroid": foreground pixel centroid fallback/reference.
            - "bbox": original axis-aligned bbox center fallback/reference.

    Returns:
        center_uv: np.ndarray, shape (2,), [u, v], or None if the mask is empty.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    vs, us = np.where(mask_u8 > 0)
    if len(us) == 0:
        return None

    method = str(method).lower()

    if method == "bbox":
        u_center = float((us.min() + us.max()) / 2.0)
        v_center = float((vs.min() + vs.max()) / 2.0)
        return np.asarray([u_center, v_center], dtype=np.float64)

    if method == "centroid":
        M = cv2.moments(mask_u8, binaryImage=True)
        if abs(M["m00"]) < 1e-8:
            u_center = float(us.mean())
            v_center = float(vs.mean())
        else:
            u_center = float(M["m10"] / M["m00"])
            v_center = float(M["m01"] / M["m00"])
        return np.asarray([u_center, v_center], dtype=np.float64)

    if method == "min_area_rect":
        # pts must be (x, y) = (u, v), not (row, col).
        pts = np.stack([us, vs], axis=1).astype(np.float32)
        if len(pts) < 3:
            return np.asarray([float(us.mean()), float(vs.mean())], dtype=np.float64)
        rect = cv2.minAreaRect(pts)
        (u_center, v_center), _, _ = rect
        return np.asarray([float(u_center), float(v_center)], dtype=np.float64)

    if method == "pca_rotated_box":
        pts = np.stack([us, vs], axis=1).astype(np.float64)
        if len(pts) < 3:
            return np.asarray([float(us.mean()), float(vs.mean())], dtype=np.float64)

        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = centered.T @ centered / max(len(pts) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        axes = eigvecs[:, order]

        # Project mask pixels to PCA axes, take the oriented bbox center there,
        # then map the center back to image uv coordinates.
        local = centered @ axes
        local_center = 0.5 * (local.min(axis=0) + local.max(axis=0))
        center_uv = mean + local_center @ axes.T
        return center_uv.astype(np.float64)

    raise ValueError(
        f"Unknown mask center method: {method}. "
        f"Use one of: min_area_rect, pca_rotated_box, centroid, bbox."
    )


def load_all_camera_observations(
    reader, frame_idx, K_by_cam, min_mask_area=1, mask_center_method="min_area_rect",
    use_depth=False, depth_root=None, require_depth=False,
):
    """
    STAGE 1 (geometry only): load masks from ALL cameras, but DO NOT decode RGB/depth.

    The complete valid-mask set is kept for:
      - mask area ranking
      - mask-center extraction
      - all-camera RANSAC triangulation

    RGB/depth are intentionally deferred until AFTER triangulation.  The second-stage
    helper `hydrate_topk_foundationpose_observations()` loads them only for the top-K
    largest-mask views (default K=10 in FAST mode).

    `use_depth`, `depth_root`, and `require_depth` remain in this signature for backward
    call compatibility; no depth is decoded in this stage.
    """
    frame_id = reader.id_strs[frame_idx]
    target_hw = (reader.H, reader.W)
    observations = []

    for cam in reader.all_cam_folders:
        cam = str(cam)
        mask_file = find_existing_file(
            os.path.join(reader.mask_root, cam),
            frame_id,
            MASK_EXTS,
        )
        if mask_file is None:
            logging.warning(f"[frame {frame_id}] cam={cam}: missing mask, skip geometry observation")
            continue

        try:
            mask = read_mask_resized(mask_file, target_hw)
        except Exception as e:
            logging.warning(f"[frame {frame_id}] cam={cam}: failed to load mask: {e}")
            continue

        mask_area = int(mask.sum())
        if mask_area < int(min_mask_area):
            logging.info(
                f"[frame {frame_id}] cam={cam}: mask too small, "
                f"area={mask_area}, min_mask_area={min_mask_area}, skip"
            )
            continue

        mask_center = compute_mask_center(mask, method=mask_center_method)
        if mask_center is None:
            logging.info(f"[frame {frame_id}] cam={cam}: empty mask center, skip")
            continue

        K = np.asarray(K_by_cam.get(cam, reader.K), dtype=np.float64).reshape(3, 3)

        observations.append({
            "cam": cam,
            "frame_id": frame_id,
            "mask": mask,
            "K": K,
            "mask_area": mask_area,
            "mask_center": np.asarray(mask_center, dtype=np.float64),
            "mask_center_method": mask_center_method,
            "mask_file": mask_file,

            # Heavy data is deliberately NOT loaded in stage 1.
            "color": None,
            "depth": None,
            "color_file": None,
            "depth_file": None,
            "mask_total_brightness": 0.0,
            "mask_mean_brightness": 0.0,
            "depth_relative_mask_median": 0.0,
            "heavy_io_loaded": False,
        })

    return observations


def hydrate_topk_foundationpose_observations(
    observations,
    reader,
    frame_idx,
    topk=10,
    use_depth=False,
    depth_root=None,
    require_depth=False,
    tracking_cam=None,
):
    """
    STAGE 2 (FoundationPose inputs): decode RGB/depth ONLY for top-K mask-area views.

    The all-camera mask-only `observations` list is untouched and remains the source
    of truth for RANSAC triangulation.

    Normal automatic FAST mode:
        topk=10 -> at most 10 RGBs + 10 depths are decoded per frame.

    If --tracking_cam is explicitly used and that camera is outside top-K, it is also
    loaded so the existing strict camera-lock feature keeps working.
    """
    if len(observations) == 0:
        return []

    frame_id = reader.id_strs[frame_idx]
    target_hw = (reader.H, reader.W)
    H, W = target_hw

    ranked = sorted(
        observations,
        key=lambda x: (-int(x["mask_area"]), _camera_sort_key(x["cam"])),
    )
    k = max(1, int(topk))
    selected = list(ranked[:k])

    tracking_cam = str(tracking_cam or "").strip()
    if tracking_cam:
        locked = next((o for o in ranked if str(o["cam"]) == tracking_cam), None)
        if locked is not None and all(str(o["cam"]) != tracking_cam for o in selected):
            selected.append(locked)

    hydrated = []
    for base_obs in selected:
        obs = dict(base_obs)
        cam = str(obs["cam"])

        color_file = find_existing_file(
            os.path.join(reader.rgb_root, cam),
            frame_id,
            IMAGE_EXTS,
        )
        if color_file is None:
            logging.warning(
                f"[frame {frame_id}] cam={cam}: top-K RGB missing; skip FoundationPose view"
            )
            continue

        depth_file = None
        if bool(use_depth) and depth_root is not None:
            depth_file = find_existing_file(
                os.path.join(str(depth_root), cam),
                frame_id,
                DEPTH_EXTS,
            )
            if bool(require_depth) and depth_file is None:
                logging.warning(
                    f"[frame {frame_id}] cam={cam}: top-K required depth missing; "
                    "skip FoundationPose view"
                )
                continue

        try:
            color = read_color_resized(color_file, target_hw)
            if bool(use_depth) and depth_file is not None:
                depth = read_depth_resized(depth_file, target_hw)
            else:
                depth = np.zeros((H, W), dtype=np.float32)
        except Exception as e:
            logging.warning(
                f"[frame {frame_id}] cam={cam}: failed top-K RGB/depth load: {e}"
            )
            continue

        mask = np.asarray(obs["mask"], dtype=bool)
        mask_total_brightness, mask_mean_brightness = compute_mask_total_brightness(
            color, mask
        )
        valid_masked_depth = mask & np.isfinite(depth) & (depth > 0.0)
        raw_median = (
            float(np.median(depth[valid_masked_depth].astype(np.float64)))
            if np.any(valid_masked_depth)
            else 0.0
        )

        obs.update({
            "color": color,
            "depth": depth,
            "color_file": color_file,
            "depth_file": depth_file,
            "mask_total_brightness": mask_total_brightness,
            "mask_mean_brightness": mask_mean_brightness,
            "depth_relative_mask_median": raw_median,
            "heavy_io_loaded": True,
        })
        hydrated.append(obs)

    logging.info(
        f"[io2] frame={frame_id}: all-mask geometry views={len(observations)}, "
        f"RGB/depth loaded views={len(hydrated)}/{min(k, len(observations))}, "
        f"cams={[str(o['cam']) for o in hydrated]}"
    )
    return hydrated


def make_projection_matrix(K, ext):
    """Build P = K [R|t], where ext maps world coordinates to camera coordinates."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    ext = np.asarray(ext, dtype=np.float64).reshape(4, 4)
    return K @ ext[:3, :]


def triangulate_point_dlt(observations, cam_ext, obs_indices):
    """
    Multi-view DLT triangulation from mask centers.

    Args:
        observations: list returned by load_all_camera_observations().
        cam_ext: dict from readextfromyml(); cam_ext[cam]["ext"] is world-to-camera.
        obs_indices: indices of observations used for triangulation.

    Returns:
        xyz_world: np.ndarray, shape (3,)
    """
    A = []
    for idx in obs_indices:
        obs = observations[idx]
        cam = str(obs["cam"])
        if cam not in cam_ext:
            raise KeyError(f"Camera {cam} is missing in extri.yml")
        u, v = obs["mask_center"]
        P = make_projection_matrix(obs["K"], cam_ext[cam]["ext"])
        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])

    A = np.asarray(A, dtype=np.float64)
    if A.shape[0] < 4:
        raise ValueError("Need at least two views for triangulation")

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    if abs(X_h[3]) < 1e-12:
        raise ValueError("Triangulation failed: homogeneous coordinate is near zero")
    xyz_world = X_h[:3] / X_h[3]
    if not np.all(np.isfinite(xyz_world)):
        raise ValueError(f"Triangulation produced non-finite xyz_world: {xyz_world}")
    return xyz_world.astype(np.float64)


def reproject_world_point(xyz_world, K, ext):
    """Project a world-space 3D point into one camera."""
    xyz1 = np.concatenate([np.asarray(xyz_world, dtype=np.float64).reshape(3), [1.0]])
    P = make_projection_matrix(K, ext)
    proj = P @ xyz1
    if abs(proj[2]) < 1e-12 or proj[2] <= 0:
        return None
    return proj[:2] / proj[2]


def camera_center_and_ray_world(obs, ext):
    """Return camera center C_world and unit ray direction d_world for obs['mask_center']."""
    K = np.asarray(obs["K"], dtype=np.float64).reshape(3, 3)
    ext = np.asarray(ext, dtype=np.float64).reshape(4, 4)
    R_wc = ext[:3, :3]  # world -> camera
    t_wc = ext[:3, 3]

    u, v = np.asarray(obs["mask_center"], dtype=np.float64).reshape(2)
    ray_cam = np.linalg.inv(K) @ np.asarray([u, v, 1.0], dtype=np.float64)
    ray_cam = ray_cam / (np.linalg.norm(ray_cam) + 1e-12)

    # ext maps X_world to X_cam = R_wc X_world + t_wc.
    # Camera center in world: C = -R^T t. Ray direction in world: R^T ray_cam.
    C_world = -R_wc.T @ t_wc
    d_world = R_wc.T @ ray_cam
    d_world = d_world / (np.linalg.norm(d_world) + 1e-12)
    return C_world.astype(np.float64), d_world.astype(np.float64)


def triangulation_angle_deg(obs_i, obs_j, cam_ext):
    """Angle between two viewing rays in degrees; small angle means unstable depth."""
    cam_i = str(obs_i["cam"])
    cam_j = str(obs_j["cam"])
    _, d_i = camera_center_and_ray_world(obs_i, cam_ext[cam_i]["ext"])
    _, d_j = camera_center_and_ray_world(obs_j, cam_ext[cam_j]["ext"])
    cosang = float(np.clip(abs(np.dot(d_i, d_j)), -1.0, 1.0))
    # Use abs(dot) because two rays can be represented with opposite orientation numerically.
    return float(np.degrees(np.arccos(cosang)))


def triangulate_point_rays_lstsq(observations, cam_ext, obs_indices, weights=None):
    """
    Multi-view triangulation by closest point to all camera rays.

    This is usually more stable than homogeneous DLT for noisy 2D mask centers.
    It solves: min_X sum_i w_i ||(I - d_i d_i^T)(X - C_i)||^2.
    """
    if len(obs_indices) < 2:
        raise ValueError("Need at least two views for ray least-squares triangulation")

    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    if weights is None:
        weights = np.ones(len(obs_indices), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)

    for local_idx, idx in enumerate(obs_indices):
        obs = observations[idx]
        cam = str(obs["cam"])
        if cam not in cam_ext:
            raise KeyError(f"Camera {cam} is missing in extri.yml")
        C, d = camera_center_and_ray_world(obs, cam_ext[cam]["ext"])
        I_minus_dd = np.eye(3, dtype=np.float64) - np.outer(d, d)
        w = max(float(weights[local_idx]), 1e-8)
        A += w * I_minus_dd
        b += w * (I_minus_dd @ C)

    try:
        X = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(A, b, rcond=None)[0]

    if not np.all(np.isfinite(X)):
        raise ValueError(f"Ray least-squares triangulation produced non-finite xyz_world: {X}")
    return X.astype(np.float64)


def world_to_camera_xyz(xyz_world, ext):
    """Convert a world-space xyz point to one camera coordinate system."""
    ext = np.asarray(ext, dtype=np.float64).reshape(4, 4)
    xyz1 = np.concatenate([np.asarray(xyz_world, dtype=np.float64).reshape(3), [1.0]])
    xyz_cam = ext @ xyz1
    return xyz_cam[:3].astype(np.float64)


def ransac_triangulate_object_center(observations, cam_ext, reproj_thresh=30.0, min_inliers=3, min_pair_angle_deg=2.0, final_method="ray_lstsq"):
    """
    RANSAC over all camera pairs using mask centers.

    For each camera pair:
      1. triangulate one world-space point from that pair's mask centers;
      2. reproject it to every valid camera;
      3. count reprojection inliers under reproj_thresh pixels.

    The best pair is the pair with the most inliers. Ties are broken by lower
    mean inlier reprojection error and then by larger triangulation angle. The
    final xyz is re-triangulated using the best pair together with all its
    inlier cameras. Pairs with too small triangulation angle are skipped because
    they give very unstable depth.
    """
    usable_indices = [
        i for i, obs in enumerate(observations)
        if str(obs["cam"]) in cam_ext and "mask_center" in obs
    ]
    if len(usable_indices) < 2:
        raise RuntimeError(
            f"Need at least two cameras with mask centers and extrinsics; got {len(usable_indices)}"
        )

    best = None
    for i, j in itertools.combinations(usable_indices, 2):
        pair_angle = triangulation_angle_deg(observations[i], observations[j], cam_ext)
        if pair_angle < float(min_pair_angle_deg):
            logging.debug(
                f"[triangulation] pair=({observations[i]['cam']},{observations[j]['cam']}) "
                f"skipped due to small ray angle: {pair_angle:.3f} deg"
            )
            continue

        try:
            # Ray least-squares is less sensitive to numerical scaling than DLT.
            xyz_world_pair = triangulate_point_rays_lstsq(observations, cam_ext, [i, j])
        except Exception as e:
            logging.debug(
                f"[triangulation] pair=({observations[i]['cam']},{observations[j]['cam']}) failed: {e}"
            )
            continue

        errors = {}
        inlier_indices = []
        for k in usable_indices:
            obs_k = observations[k]
            cam_k = str(obs_k["cam"])
            uv_hat = reproject_world_point(xyz_world_pair, obs_k["K"], cam_ext[cam_k]["ext"])
            if uv_hat is None:
                err = np.inf
            else:
                err = float(np.linalg.norm(uv_hat - obs_k["mask_center"]))
            errors[cam_k] = err
            if np.isfinite(err) and err <= float(reproj_thresh):
                inlier_indices.append(k)

        if len(inlier_indices) == 0:
            mean_inlier_error = np.inf
        else:
            mean_inlier_error = float(np.mean([errors[str(observations[k]["cam"])] for k in inlier_indices]))

        candidate = {
            "pair_indices": (i, j),
            "pair_cams": (str(observations[i]["cam"]), str(observations[j]["cam"])),
            "xyz_world_pair": xyz_world_pair,
            "inlier_indices": inlier_indices,
            "num_inliers": len(inlier_indices),
            "mean_inlier_error": mean_inlier_error,
            "pair_angle_deg": pair_angle,
            "errors": errors,
        }

        if best is None:
            best = candidate
        else:
            better = candidate["num_inliers"] > best["num_inliers"]
            tie_better = (
                candidate["num_inliers"] == best["num_inliers"] and
                (
                    candidate["mean_inlier_error"] < best["mean_inlier_error"] or
                    (
                        abs(candidate["mean_inlier_error"] - best["mean_inlier_error"]) < 1e-6 and
                        candidate["pair_angle_deg"] > best.get("pair_angle_deg", -np.inf)
                    )
                )
            )
            if better or tie_better:
                best = candidate

    if best is None:
        raise RuntimeError("RANSAC triangulation failed for all camera pairs")

    final_indices = sorted(set(best["pair_indices"]).union(best["inlier_indices"]))
    if len(final_indices) < max(2, int(min_inliers)):
        raise RuntimeError(
            f"RANSAC triangulation has too few inliers: {len(final_indices)} < {min_inliers}; "
            f"best_pair={best['pair_cams']}"
        )

    if str(final_method).lower() == "dlt":
        xyz_world_final = triangulate_point_dlt(observations, cam_ext, final_indices)
    elif str(final_method).lower() == "ray_lstsq":
        # Weight larger visible masks a bit higher, but do not let one huge mask dominate.
        areas = np.asarray([observations[k]["mask_area"] for k in final_indices], dtype=np.float64)
        weights = np.sqrt(np.maximum(areas, 1.0))
        weights = weights / (np.mean(weights) + 1e-12)
        xyz_world_final = triangulate_point_rays_lstsq(observations, cam_ext, final_indices, weights=weights)
    else:
        raise ValueError(f"Unknown final triangulation method: {final_method}")

    # Recompute final reprojection errors using the final multi-view point.
    final_errors = {}
    final_inlier_indices = []
    for k in usable_indices:
        obs_k = observations[k]
        cam_k = str(obs_k["cam"])
        uv_hat = reproject_world_point(xyz_world_final, obs_k["K"], cam_ext[cam_k]["ext"])
        if uv_hat is None:
            err = np.inf
        else:
            err = float(np.linalg.norm(uv_hat - obs_k["mask_center"]))
        final_errors[cam_k] = err
        if np.isfinite(err) and err <= float(reproj_thresh):
            final_inlier_indices.append(k)

    result = {
        "xyz_world": xyz_world_final,
        "best_pair_cams": best["pair_cams"],
        "initial_num_inliers": best["num_inliers"],
        "initial_mean_inlier_error": best["mean_inlier_error"],
        "best_pair_angle_deg": best.get("pair_angle_deg", np.nan),
        "final_method": str(final_method),
        "final_inlier_indices": final_inlier_indices,
        "final_inlier_cams": [str(observations[k]["cam"]) for k in final_inlier_indices],
        "final_num_inliers": len(final_inlier_indices),
        "final_mean_inlier_error": float(np.mean([final_errors[str(observations[k]["cam"])] for k in final_inlier_indices])) if len(final_inlier_indices) > 0 else np.inf,
        "final_errors": final_errors,
        "used_indices_for_final_triangulation": final_indices,
        "used_cams_for_final_triangulation": [str(observations[k]["cam"]) for k in final_indices],
    }
    return result


def convert_pose_to_main_camera(pose_selected, selected_cam, main_cam, cam_pose_rest):
    """
    Convert ob_in_selected_cam to ob_in_main_cam.

    cam_pose_rest[selected_cam]["pose_4x4"] means:
        X_selected = T_selected_from_main @ X_main
    Therefore:
        ob_in_main = inv(T_selected_from_main) @ ob_in_selected
    """
    pose_selected_np = pose_to_numpy_4x4(pose_selected)
    selected_cam = str(selected_cam)
    main_cam = str(main_cam)

    if selected_cam == main_cam:
        return pose_selected_np.astype(np.float32)

    if selected_cam not in cam_pose_rest:
        raise KeyError(
            f"selected_cam={selected_cam} not found in cam_pose_rest. "
            f"Available cams: {list(cam_pose_rest.keys())}"
        )

    T_selected_from_main = np.asarray(
        cam_pose_rest[selected_cam]["pose_4x4"],
        dtype=np.float64,
    ).reshape(4, 4)
    T_main_from_selected = np.linalg.inv(T_selected_from_main)
    pose_main = T_main_from_selected @ pose_selected_np
    return pose_main.astype(np.float32)




def convert_pose_to_world(pose_cam, cam, cam_ext):
    """
    Convert ob_in_cam to ob_in_world.

    cam_ext[cam]["ext"] is assumed to be world-to-camera:
        X_cam = T_cam_from_world @ X_world

    pose_cam is object-to-camera:
        X_cam = ob_in_cam @ X_object

    Therefore:
        X_world = inv(T_cam_from_world) @ ob_in_cam @ X_object
        ob_in_world = inv(T_cam_from_world) @ ob_in_cam
    """
    cam = str(cam)
    if cam not in cam_ext:
        raise KeyError(f"Camera {cam} is missing in extri.yml")

    pose_cam_np = pose_to_numpy_4x4(pose_cam)
    T_cam_from_world = np.asarray(cam_ext[cam]["ext"], dtype=np.float64).reshape(4, 4)
    T_world_from_cam = np.linalg.inv(T_cam_from_world)
    pose_world = T_world_from_cam @ pose_cam_np
    return pose_world.astype(np.float32)


def rotation_matrix_to_quaternion_xyzw(R):
    """Convert a 3x3 rotation matrix to quaternion [qx, qy, qz, qw]."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    q = np.empty(4, dtype=np.float64)
    trace = np.trace(R)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0)) * 2.0
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0)) * 2.0
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0)) * 2.0
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s

    norm = np.linalg.norm(q)
    if norm < 1e-12 or not np.all(np.isfinite(q)):
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def rotation_matrix_to_euler_xyz_deg(R):
    """
    Convert a 3x3 rotation matrix to XYZ Euler angles in degrees.
    This is mainly for CSV/debug readability; the 4x4 txt is the source of truth.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0

    return np.degrees(np.asarray([x, y, z], dtype=np.float64))

def register_from_best_mask_camera(est, observations, triangulation, cam_ext, args):
    """
    Select the camera with the largest object-mask AREA and run
    FoundationPose.register.

    Camera ranking uses mask_area ONLY (foreground pixel count); RGB brightness is
    not part of the ranking or tie-break. The translation init is the RANSAC-
    triangulated object center transformed into the selected camera coordinate system.
    If --fallback_next_best is enabled, cameras with the next-largest masks are tried
    after a failure.
    """
    if len(observations) == 0:
        raise RuntimeError("No valid camera observation for this frame.")

    observations_sorted = sorted(
        observations,
        key=lambda x: x["mask_area"],
        reverse=True,
    )
    tracking_cam = str(getattr(args, "tracking_cam", "") or "").strip()
    if tracking_cam:
        candidates = [obs for obs in observations_sorted if str(obs["cam"]) == tracking_cam]
        if len(candidates) == 0:
            available = [str(obs["cam"]) for obs in observations_sorted]
            raise RuntimeError(
                f"Locked tracking camera {tracking_cam} has no valid observation in this frame. "
                f"Valid cameras: {available}"
            )
        logging.info(
            f"[tracking_cam] full register locked to cam={tracking_cam}; "
            "xyz still comes from multi-view triangulation"
        )
    else:
        candidates = observations_sorted if args.fallback_next_best else observations_sorted[:1]
    tried = []
    xyz_world = np.asarray(triangulation["xyz_world"], dtype=np.float64).reshape(3)

    for obs in candidates:
        cam = str(obs["cam"])
        tried.append(cam)
        try:
            if cam not in cam_ext:
                raise KeyError(f"selected cam {cam} is missing in extri.yml")

            init_xyz_selected = world_to_camera_xyz(xyz_world, cam_ext[cam]["ext"])
            if init_xyz_selected[2] <= 0:
                raise ValueError(
                    f"triangulated object is behind selected camera {cam}: {init_xyz_selected.tolist()}"
                )

            depth_used, depth_info = prepare_depth_for_foundationpose(
                obs=obs, init_xyz_selected=init_xyz_selected, args=args
            )
            logging.info(
                f"[registration] frame={obs['frame_id']} selected_cam={cam}, "
                f"mask_total_brightness={obs['mask_total_brightness']:.3f}, "
                f"mask_mean_brightness={obs['mask_mean_brightness']:.3f}, "
                f"mask_area={obs['mask_area']}, init_xyz_selected={init_xyz_selected.tolist()}, "
                f"depth_mode={depth_info['mode']}, depth_file={obs.get('depth_file')}, "
                f"depth_rel_median={depth_info['relative_mask_median']:.6f}, "
                f"z_ref={depth_info['z_reference']:.6f}, scale={depth_info['scale']:.8f}, "
                f"aligned_median={depth_info['aligned_mask_median']:.6f}, "
                f"best_pair={triangulation['best_pair_cams']}, "
                f"tri_inliers={triangulation['final_num_inliers']}, candidates_tried={tried}"
            )
            pose_selected = est.register(
                K=obs["K"],
                rgb=obs["color"],
                depth=depth_used,
                ob_mask=obs["mask"].astype(bool),
                iteration=(3 if bool(getattr(args, "fast", False)) else args.est_refine_iter),
                init_xyz=init_xyz_selected,
            )

            # Full/all-direction registration (used by the first frame, and whenever
            # temporal-cone mode is disabled) returns an ORIGINAL-MESH pose.
            #
            # Default behavior: keep FoundationPose's selected rotation, but force the
            # CENTERED-MESH object-center translation to the RANSAC-triangulated xyz only
            # after FoundationPose has finished registration.
            #
            # With --no-force: do NOT recenter after registration. Keep the complete
            # FoundationPose-refined ORIGINAL-MESH pose, including its refined translation.
            # The triangulated xyz is still used above as init_xyz for registration.
            #
            # Do NOT directly write init_xyz_selected into the ORIGINAL-MESH pose, because
            # the original mesh origin can be offset from the geometric/centered-mesh origin.
            pose_selected_np = pose_to_numpy_4x4(pose_selected).astype(np.float64)
            tf_to_centered_mesh = pose_to_numpy_4x4(est.get_tf_to_centered_mesh()).astype(np.float64)

            # pose_original = pose_centered @ tf_to_centered_mesh
            # => pose_centered = pose_original @ inv(tf_to_centered_mesh)
            pose_centered = pose_selected_np @ np.linalg.inv(tf_to_centered_mesh)
            refined_xyz_before_lock = pose_centered[:3, 3].copy()

            # A locked tracking camera changes ONLY which RGB/mask/depth view FoundationPose uses.
            # Its xyz must still come from the multi-view triangulated object center, so tracking_cam
            # intentionally overrides --no-force for translation (same xyz-lock semantics as --fast).
            tracking_cam_locked = bool(str(getattr(args, "tracking_cam", "") or "").strip())
            if bool(getattr(args, "no_force", False)) and not tracking_cam_locked:
                # Keep FoundationPose's full refined 6D pose unchanged.
                pose_selected = pose_selected_np.astype(np.float32)
                logging.info(
                    f"[registration_xyz_free] frame={obs['frame_id']} cam={cam}, "
                    f"keeping_refined_centered_xyz={refined_xyz_before_lock.tolist()}, "
                    f"triangulated_xyz_used_only_for_init={init_xyz_selected.tolist()}"
                )
            else:
                pose_centered[:3, 3] = init_xyz_selected
                pose_selected = (pose_centered @ tf_to_centered_mesh).astype(np.float32)
                lock_reason = "tracking_cam" if tracking_cam_locked else "default_force"
                logging.info(
                    f"[registration_xyz_lock] frame={obs['frame_id']} cam={cam}, reason={lock_reason}, "
                    f"centered_xyz_before={refined_xyz_before_lock.tolist()}, "
                    f"triangulated_xyz_selected={init_xyz_selected.tolist()}"
                )

            obs_used = dict(obs)
            obs_used["depth_used"] = depth_used
            obs_used["depth_scale_info"] = depth_info
            return obs_used, pose_selected, init_xyz_selected
        except Exception as e:
            logging.warning(
                f"[registration] frame={obs['frame_id']} cam={cam} failed: {e}"
            )
            if not args.fallback_next_best:
                raise

    raise RuntimeError(f"Registration failed for all tried cameras: {tried}")



def axis_angle_to_rotmat(axis, angle_rad):
    """Rodrigues formula: unit axis + angle -> 3x3 rotation matrix."""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / axis_norm
    x, y, z = axis
    K = np.asarray([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float64)
    I = np.eye(3, dtype=np.float64)
    return I + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)


def sample_rotation_cone_around_last(R_last, max_angle_deg=50.0, num_hypotheses=240, rng=None):
    """
    Sample exactly num_hypotheses rotations inside a geodesic cone around R_last.

    The first hypothesis is exactly R_last. The rest are deterministic principal-axis
    rotations plus random axis-angle perturbations whose SO(3) angle is <= max_angle_deg.
    Left perturbation is used:
        R_hyp = R_delta @ R_last
    which means the perturbation cone is expressed in world coordinates.
    """
    R_last = np.asarray(R_last, dtype=np.float64).reshape(3, 3)
    num_hypotheses = max(1, int(num_hypotheses))
    max_angle_rad = np.radians(float(max_angle_deg))
    if rng is None:
        rng = np.random.default_rng(0)

    rotations = [R_last.copy()]
    if num_hypotheses == 1 or max_angle_rad <= 1e-12:
        return np.asarray(rotations, dtype=np.float64)

    deterministic_axes = [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([-1.0, 0.0, 0.0]),
        np.asarray([0.0, -1.0, 0.0]),
        np.asarray([0.0, 0.0, -1.0]),
    ]
    deterministic_fracs = [0.15, 0.30, 0.50, 0.70, 0.90, 1.00]
    for frac in deterministic_fracs:
        for axis in deterministic_axes:
            if len(rotations) >= num_hypotheses:
                break
            rotations.append(axis_angle_to_rotmat(axis, frac * max_angle_rad) @ R_last)
        if len(rotations) >= num_hypotheses:
            break

    while len(rotations) < num_hypotheses:
        axis = rng.normal(size=3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            continue
        axis = axis / axis_norm
        # Uniform-in-volume sampling inside the angle ball: more samples near center.
        angle = max_angle_rad * (rng.random() ** (1.0 / 3.0))
        rotations.append(axis_angle_to_rotmat(axis, angle) @ R_last)

    return np.asarray(rotations[:num_hypotheses], dtype=np.float64)


def make_pose_from_R_t(R, t):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return pose


def to_numpy_pose_batch(poses):
    if torch.is_tensor(poses):
        poses = poses.detach().cpu().numpy()
    poses = np.asarray(poses)
    if poses.ndim == 2:
        poses = poses[None]
    return poses.astype(np.float32)


def scores_to_numpy(scores):
    if torch.is_tensor(scores):
        return scores.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def rotation_angle_deg(R_a, R_b):
    """
    Geodesic rotation difference between two rotation matrices, in degrees.

    angle = arccos((trace(R_a R_b^T) - 1) / 2)
    """
    R_a = np.asarray(R_a, dtype=np.float64).reshape(3, 3)
    R_b = np.asarray(R_b, dtype=np.float64).reshape(3, 3)

    R_delta = R_a @ R_b.T
    cos_theta = 0.5 * (np.trace(R_delta) - 1.0)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))

    return float(np.degrees(np.arccos(cos_theta)))



def _project_rotation_so3(R):
    """Project a numerically noisy 3x3 matrix to the closest proper rotation."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0.0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def rotation_matrix_to_rotvec_rad(R):
    """Return Rodrigues/SO(3) rotation vector [rx, ry, rz] in radians."""
    R = _project_rotation_so3(R)
    rvec, _ = cv2.Rodrigues(R)
    return np.asarray(rvec, dtype=np.float64).reshape(3)


def unwrap_rotvec_near(reference_rotvec, measured_rotvec, turns=2):
    """Choose the 2*pi-equivalent Rodrigues-vector branch nearest a reference."""
    ref = np.asarray(reference_rotvec, dtype=np.float64).reshape(3)
    meas = np.asarray(measured_rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(meas))
    candidates = [meas.copy()]
    if theta > 1e-10:
        axis = meas / theta
        for k in range(-int(turns), int(turns) + 1):
            candidates.append((theta + 2.0 * np.pi * k) * axis)
    else:
        ref_norm = float(np.linalg.norm(ref))
        if ref_norm > 1e-10:
            axis = ref / ref_norm
            for k in range(-int(turns), int(turns) + 1):
                candidates.append((2.0 * np.pi * k) * axis)
    return min(candidates, key=lambda c: float(np.linalg.norm(c - ref))).copy()


class ConstantVelocityKalman3D:
    """6-state CV Kalman filter: [p_x,p_y,p_z,v_x,v_y,v_z]."""

    def __init__(self, measurement_std, acceleration_std, initial_velocity_std):
        self.measurement_std = max(float(measurement_std), 1e-12)
        self.acceleration_std = max(float(acceleration_std), 1e-12)
        self.initial_velocity_std = max(float(initial_velocity_std), 1e-12)
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64)
        self.initialized = False
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[:, :3] = np.eye(3, dtype=np.float64)
        self.R = (self.measurement_std ** 2) * np.eye(3, dtype=np.float64)

    def initialize(self, measurement):
        z = np.asarray(measurement, dtype=np.float64).reshape(3)
        self.x[:] = 0.0
        self.x[:3] = z
        self.P[:] = 0.0
        self.P[:3, :3] = (self.measurement_std ** 2) * np.eye(3, dtype=np.float64)
        self.P[3:, 3:] = (self.initial_velocity_std ** 2) * np.eye(3, dtype=np.float64)
        self.initialized = True
        return self.x.copy()

    def predict(self, dt):
        if not self.initialized:
            raise RuntimeError("Kalman filter must be initialized before predict().")
        dt = max(float(dt), 1e-6)
        F = np.eye(6, dtype=np.float64)
        F[:3, 3:] = dt * np.eye(3, dtype=np.float64)
        q = self.acceleration_std ** 2
        I3 = np.eye(3, dtype=np.float64)
        Q = q * np.block([
            [0.25 * (dt ** 4) * I3, 0.5 * (dt ** 3) * I3],
            [0.5 * (dt ** 3) * I3,        (dt ** 2) * I3],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x.copy()

    def correct(self, measurement):
        if not self.initialized:
            return self.initialize(measurement)
        z = np.asarray(measurement, dtype=np.float64).reshape(3)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        I6 = np.eye(6, dtype=np.float64)
        A = I6 - K @ self.H
        self.P = A @ self.P @ A.T + K @ self.R @ K.T
        return self.x.copy()


# 10 FPS low-lag tuning: trust measurements more and allow faster acceleration changes.
class ObjectPoseKalmanFilter:
    """
    Final WORLD-frame object-pose filter with XYZ-only Kalman smoothing.

    Translation is Kalman-filtered with state [x,y,z,vx,vy,vz].
    Rotation Kalman filtering is intentionally DISABLED: the current frame's
    FoundationPose raw/refined rotation is copied through unchanged.

    rot_measurement_std_deg / rot_acceleration_std_deg are kept in the constructor
    only for backward CLI compatibility and are intentionally ignored.
    """

    def __init__(self, fps, pos_measurement_std=0.02, pos_acceleration_std=0.25,
                 rot_measurement_std_deg=5.0, rot_acceleration_std_deg=90.0):
        self.fps = max(float(fps), 1e-6)
        self.pos = ConstantVelocityKalman3D(
            measurement_std=pos_measurement_std,
            acceleration_std=pos_acceleration_std,
            # Stronger XYZ anti-jitter: do not allow the initial velocity state to
            # react too aggressively to frame-to-frame translation noise.
            initial_velocity_std=0.25,
        )
        # IMPORTANT: no rotation Kalman state is created.
        self.last_frame_idx = None

    def update(self, pose_centered_world_measurement, frame_idx):
        pose_meas = np.asarray(pose_centered_world_measurement, dtype=np.float64).reshape(4, 4)
        xyz_meas = pose_meas[:3, 3].copy()
        R_meas = _project_rotation_so3(pose_meas[:3, :3])
        rvec_meas = rotation_matrix_to_rotvec_rad(R_meas)
        frame_idx = int(frame_idx)

        if not self.pos.initialized:
            self.pos.initialize(xyz_meas)
            dt = 1.0 / self.fps
        else:
            frame_gap = max(frame_idx - int(self.last_frame_idx), 1)
            dt = float(frame_gap) / self.fps
            self.pos.predict(dt)
            self.pos.correct(xyz_meas)

        self.last_frame_idx = frame_idx
        xyz_f = self.pos.x[:3].copy()
        vel_f = self.pos.x[3:].copy()

        # Rotation is NOT filtered. Preserve the current FoundationPose measurement.
        pose_f = np.eye(4, dtype=np.float64)
        pose_f[:3, :3] = R_meas
        pose_f[:3, 3] = xyz_f
        return pose_f, {
            "dt": float(dt),
            "measurement_xyz": xyz_meas,
            "filtered_xyz": xyz_f,
            "filtered_velocity": vel_f,
            "measurement_rotvec_rad": rvec_meas.copy(),
            "filtered_rotvec_rad": rvec_meas.copy(),  # raw rotation passthrough
            "filtered_omega_rad_s": np.zeros(3, dtype=np.float64),
            "rotation_kalman_enabled": False,
        }

def _relative_rotvec_deg(R_world, R_reference_world):
    """SO(3) logarithm represented as axis*angle (degrees) relative to reference."""
    R_world = _project_rotation_so3(R_world)
    R_reference_world = _project_rotation_so3(R_reference_world)
    R_delta = _project_rotation_so3(R_world @ R_reference_world.T)
    rvec, _ = cv2.Rodrigues(R_delta)
    return rvec.reshape(3).astype(np.float64) * (180.0 / np.pi)

def compute_refined_rotation_avg_var_world(refined_selected_poses, R_cam_from_world, R_reference_world):
    """
    Compute the scalar rotation-distribution variance used for mask quality gating.

    Each refined selected-camera rotation is converted back to WORLD coordinates,
    represented by the SO(3) log-map relative to the previous accepted world rotation,
    and then the three component variances are averaged:

        avg_var = (var(dRx) + var(dRy) + var(dRz)) / 3

    Units are deg^2 because dRx/dRy/dRz are in degrees.
    """
    poses = to_numpy_pose_batch(refined_selected_poses)
    if len(poses) == 0:
        return float("inf"), np.full(3, np.nan), np.full(3, np.nan), np.empty((0, 3), dtype=np.float64)

    R_cam_from_world = _project_rotation_so3(R_cam_from_world)
    R_world_from_cam = R_cam_from_world.T
    R_reference_world = _project_rotation_so3(R_reference_world)

    rotvecs = []
    for pose in poses:
        R_selected = _project_rotation_so3(pose[:3, :3])
        R_world = _project_rotation_so3(R_world_from_cam @ R_selected)
        rotvecs.append(_relative_rotvec_deg(R_world, R_reference_world))
    rotvecs = np.asarray(rotvecs, dtype=np.float64)

    mean_xyz = np.mean(rotvecs, axis=0)
    var_xyz = np.var(rotvecs, axis=0)
    avg_var = float(np.mean(var_xyz))
    return avg_var, mean_xyz, var_xyz, rotvecs


def _score_colors_bgr(scores):
    """Map finite FoundationPose scores to TURBO colors; non-finite scores are gray."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    colors = np.tile(np.asarray([[150, 150, 150]], dtype=np.uint8), (len(scores), 1))
    finite = np.isfinite(scores)
    if not np.any(finite):
        return colors, np.nan, np.nan
    smin = float(np.min(scores[finite]))
    smax = float(np.max(scores[finite]))
    if abs(smax - smin) < 1e-12:
        normalized = np.full(np.count_nonzero(finite), 0.5, dtype=np.float64)
    else:
        normalized = (scores[finite] - smin) / (smax - smin)
    lut_idx = np.clip(np.round(normalized * 255.0), 0, 255).astype(np.uint8).reshape(-1, 1)
    mapped = cv2.applyColorMap(lut_idx, cv2.COLORMAP_TURBO).reshape(-1, 3)
    colors[finite] = mapped
    return colors, smin, smax


def _draw_star_bgr(img, center, radius, color, thickness=-1):
    """Draw a five-point star marker using OpenCV."""
    cx, cy = int(center[0]), int(center[1])
    pts = []
    for k in range(10):
        angle = -np.pi / 2.0 + k * np.pi / 5.0
        rr = float(radius) if (k % 2 == 0) else float(radius) * 0.43
        pts.append([int(round(cx + rr * np.cos(angle))), int(round(cy + rr * np.sin(angle)))])
    pts = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
    if thickness < 0:
        cv2.fillPoly(img, [pts], color)
    else:
        cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)


def _draw_distribution_placeholder(save_path, frame_id, reason):
    """Write a per-frame placeholder when that frame does not have 240 temporal hypotheses."""
    canvas = np.full((700, 1200, 3), 250, dtype=np.uint8)
    cv2.putText(canvas, f"Rotation distribution - frame {frame_id}", (55, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, "No temporal-cone hypothesis distribution for this frame.", (55, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (45, 45, 45), 2, cv2.LINE_AA)
    y = 285
    words = str(reason).split()
    line = ""
    for word in words:
        trial = (line + " " + word).strip()
        if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 1)[0][0] > 1050:
            cv2.putText(canvas, line, (55, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                        (80, 80, 80), 1, cv2.LINE_AA)
            y += 45
            line = word
        else:
            line = trial
    if line:
        cv2.putText(canvas, line, (55, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                    (80, 80, 80), 1, cv2.LINE_AA)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas)


def save_rotation_distribution_visualization(
    initial_selected_hypotheses,
    refined_selected_poses,
    scores,
    best_idx,
    consensus_info,
    R_cam_from_world,
    R_reference_world,
    frame_id,
    save_path,
    cone_deg,
):
    """
    Visualize the per-frame temporal rotation distribution in SO(3).

    Coordinate choice:
      - Convert every selected-camera rotation back to WORLD coordinates.
      - Express it relative to the previous accepted world rotation with the
        SO(3) log map (rotation vector = axis * geodesic angle, in degrees).

    Panels:
      1) dRx vs dRy
      2) dRx vs dRz
      3) dRy vs dRz
      4) geodesic angle from previous rotation vs FoundationPose score

    Encoding:
      - light gray tiny dots: original temporal-cone samples before refiner
      - score-colored dots: refined hypotheses that actually enter consensus
      - black ring: top-K by score
      - green ring: supporters of the selected consensus cluster
      - red star: final selected / most-supported pose
      - black cross at (0,0): previous accepted rotation (cone center)
    """
    initial_selected_hypotheses = to_numpy_pose_batch(initial_selected_hypotheses)
    refined_selected_poses = to_numpy_pose_batch(refined_selected_poses)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = len(refined_selected_poses)
    if n == 0:
        _draw_distribution_placeholder(save_path, frame_id, "No refined hypotheses were returned.")
        return
    if len(scores) != n:
        raise ValueError(f"distribution score count {len(scores)} != pose count {n}")

    R_cam_from_world = _project_rotation_so3(R_cam_from_world)
    R_world_from_cam = R_cam_from_world.T
    R_reference_world = _project_rotation_so3(R_reference_world)

    def batch_to_rotvecs_world(poses):
        vecs = []
        for pose in poses:
            R_selected = _project_rotation_so3(pose[:3, :3])
            R_world = _project_rotation_so3(R_world_from_cam @ R_selected)
            vecs.append(_relative_rotvec_deg(R_world, R_reference_world))
        return np.asarray(vecs, dtype=np.float64)

    initial_vecs = batch_to_rotvecs_world(initial_selected_hypotheses)
    refined_vecs = batch_to_rotvecs_world(refined_selected_poses)
    geodesic_deg = np.linalg.norm(refined_vecs, axis=1)

    colors, score_min, score_max = _score_colors_bgr(scores)
    finite_indices = np.where(np.isfinite(scores))[0]
    topk_n = min(int(consensus_info.get("topk", 0)), len(finite_indices))
    if topk_n > 0:
        sorted_finite = finite_indices[np.argsort(scores[finite_indices])[::-1]]
        top_indices = set(int(i) for i in sorted_finite[:topk_n])
    else:
        top_indices = set()
    supporter_indices = set(int(i) for i in consensus_info.get("supporter_indices", []))
    best_idx = int(best_idx)

    canvas_h, canvas_w = 1080, 1600
    canvas = np.full((canvas_h, canvas_w, 3), 248, dtype=np.uint8)
    dark = (35, 35, 35)
    muted = (105, 105, 105)
    grid = (218, 218, 218)
    panel_border = (180, 180, 180)
    supporter_color = (45, 170, 55)
    chosen_color = (20, 20, 230)
    initial_color = (205, 205, 205)

    mode = str(consensus_info.get("mode", ""))
    max_supporters = int(consensus_info.get("max_supporters", 0))
    chosen_score = float(scores[best_idx]) if 0 <= best_idx < len(scores) else float("nan")
    header1 = f"Frame {frame_id} | {n} refined rotation hypotheses | chosen #{best_idx} | mode={mode}"
    header2 = (
        f"chosen score={chosen_score:.6f} | top-K={topk_n} | max supporters={max_supporters} | "
        f"support threshold={float(consensus_info.get('angle_thresh_deg', np.nan)):.1f} deg | cone={float(cone_deg):.1f} deg"
    )
    cv2.putText(canvas, header1, (45, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.83, dark, 2, cv2.LINE_AA)
    cv2.putText(canvas, header2, (45, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.56, muted, 1, cv2.LINE_AA)
    cv2.putText(canvas,
                "SO(3) log-map is in WORLD coordinates relative to previous accepted rotation; values are axis*angle in degrees.",
                (45, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.49, muted, 1, cv2.LINE_AA)

    # Consistent symmetric scale for all component scatter panels.
    all_abs = np.max(np.abs(np.vstack([initial_vecs, refined_vecs]))) if len(initial_vecs) else np.max(np.abs(refined_vecs))
    limit = max(float(cone_deg) * 1.12, float(all_abs) * 1.10, 5.0)
    limit = min(max(limit, 5.0), 180.0)

    panels = [
        (45, 155, 755, 555, 0, 1, "dRx (deg)", "dRy (deg)"),
        (825, 155, 1535, 555, 0, 2, "dRx (deg)", "dRz (deg)"),
        (45, 620, 755, 1020, 1, 2, "dRy (deg)", "dRz (deg)"),
    ]

    def draw_scatter_panel(spec):
        x0, y0, x1, y1, xi, yi, xlabel, ylabel = spec
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), panel_border, 1)
        pad_l, pad_r, pad_t, pad_b = 72, 25, 32, 55
        px0, px1 = x0 + pad_l, x1 - pad_r
        py0, py1 = y0 + pad_t, y1 - pad_b
        pw, ph = px1 - px0, py1 - py0

        def map_pt(vx, vy):
            xx = px0 + int(round((float(vx) + limit) / (2.0 * limit) * pw))
            yy = py1 - int(round((float(vy) + limit) / (2.0 * limit) * ph))
            return xx, yy

        # Grid and tick labels.
        for frac in np.linspace(-1.0, 1.0, 5):
            val = frac * limit
            xx, _ = map_pt(val, 0.0)
            _, yy = map_pt(0.0, val)
            cv2.line(canvas, (xx, py0), (xx, py1), grid, 1, cv2.LINE_AA)
            cv2.line(canvas, (px0, yy), (px1, yy), grid, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{val:.0f}", (xx - 15, py1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, muted, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{val:.0f}", (x0 + 9, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, muted, 1, cv2.LINE_AA)

        # Projected nominal cone boundary.
        cx, cy = map_pt(0.0, 0.0)
        cone_px = int(round(float(cone_deg) / (2.0 * limit) * pw * 2.0))
        cone_py = int(round(float(cone_deg) / (2.0 * limit) * ph * 2.0))
        if cone_px > 1 and cone_py > 1:
            cv2.ellipse(canvas, (cx, cy), (max(1, cone_px // 2), max(1, cone_py // 2)), 0, 0, 360,
                        (190, 190, 190), 1, cv2.LINE_AA)

        # Initial pre-refiner samples first.
        for p in initial_vecs:
            pt = map_pt(p[xi], p[yi])
            cv2.circle(canvas, pt, 2, initial_color, -1, cv2.LINE_AA)

        # Refined samples, score-colored.
        for i, p in enumerate(refined_vecs):
            pt = map_pt(p[xi], p[yi])
            col = tuple(int(v) for v in colors[i])
            cv2.circle(canvas, pt, 4, col, -1, cv2.LINE_AA)
            if i in top_indices:
                cv2.circle(canvas, pt, 6, (25, 25, 25), 1, cv2.LINE_AA)
            if i in supporter_indices:
                cv2.circle(canvas, pt, 8, supporter_color, 2, cv2.LINE_AA)

        # Reference rotation and chosen pose on top.
        cv2.drawMarker(canvas, (cx, cy), (20, 20, 20), cv2.MARKER_CROSS, 15, 2, cv2.LINE_AA)
        if 0 <= best_idx < n:
            bp = refined_vecs[best_idx]
            _draw_star_bgr(canvas, map_pt(bp[xi], bp[yi]), 12, chosen_color)
            _draw_star_bgr(canvas, map_pt(bp[xi], bp[yi]), 12, (255, 255, 255), 1)

        cv2.putText(canvas, xlabel, (px0 + pw // 2 - 36, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, dark, 1, cv2.LINE_AA)
        cv2.putText(canvas, ylabel, (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, dark, 1, cv2.LINE_AA)

    for panel in panels:
        draw_scatter_panel(panel)

    # Bottom-right: geodesic angle versus score.
    x0, y0, x1, y1 = 825, 620, 1535, 1020
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), panel_border, 1)
    pad_l, pad_r, pad_t, pad_b = 72, 25, 32, 55
    px0, px1 = x0 + pad_l, x1 - pad_r
    py0, py1 = y0 + pad_t, y1 - pad_b
    pw, ph = px1 - px0, py1 - py0
    x_max = max(float(cone_deg) * 1.15, float(np.nanmax(geodesic_deg)) * 1.08, 1.0)
    finite_scores = scores[np.isfinite(scores)]
    if len(finite_scores) == 0:
        y_min, y_max = 0.0, 1.0
    else:
        y_min, y_max = float(np.min(finite_scores)), float(np.max(finite_scores))
        if abs(y_max - y_min) < 1e-12:
            y_min -= 0.5
            y_max += 0.5
        else:
            pad = 0.08 * (y_max - y_min)
            y_min -= pad
            y_max += pad

    def map_score_pt(angle, score):
        xx = px0 + int(round(float(angle) / x_max * pw))
        yy = py1 - int(round((float(score) - y_min) / (y_max - y_min) * ph))
        return xx, yy

    for frac in np.linspace(0.0, 1.0, 5):
        angle = frac * x_max
        xx, _ = map_score_pt(angle, y_min)
        cv2.line(canvas, (xx, py0), (xx, py1), grid, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{angle:.0f}", (xx - 12, py1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, muted, 1, cv2.LINE_AA)
    for frac in np.linspace(0.0, 1.0, 5):
        score = y_min + frac * (y_max - y_min)
        _, yy = map_score_pt(0.0, score)
        cv2.line(canvas, (px0, yy), (px1, yy), grid, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{score:.3f}", (x0 + 7, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, muted, 1, cv2.LINE_AA)

    if float(cone_deg) <= x_max:
        xx, _ = map_score_pt(float(cone_deg), y_min)
        cv2.line(canvas, (xx, py0), (xx, py1), (155, 155, 155), 1, cv2.LINE_AA)
        cv2.putText(canvas, "cone", (xx + 4, py0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, muted, 1, cv2.LINE_AA)

    for i, (ang, score) in enumerate(zip(geodesic_deg, scores)):
        if not np.isfinite(score):
            continue
        pt = map_score_pt(ang, score)
        col = tuple(int(v) for v in colors[i])
        cv2.circle(canvas, pt, 4, col, -1, cv2.LINE_AA)
        if i in top_indices:
            cv2.circle(canvas, pt, 6, (25, 25, 25), 1, cv2.LINE_AA)
        if i in supporter_indices:
            cv2.circle(canvas, pt, 8, supporter_color, 2, cv2.LINE_AA)
    if 0 <= best_idx < n and np.isfinite(scores[best_idx]):
        _draw_star_bgr(canvas, map_score_pt(geodesic_deg[best_idx], scores[best_idx]), 12, chosen_color)
        _draw_star_bgr(canvas, map_score_pt(geodesic_deg[best_idx], scores[best_idx]), 12, (255, 255, 255), 1)

    cv2.putText(canvas, "Geodesic angle from previous rotation (deg)", (px0 + 115, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, dark, 1, cv2.LINE_AA)
    cv2.putText(canvas, "FoundationPose score", (x0 + 8, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, dark, 1, cv2.LINE_AA)

    # Compact legend and score color bar between header and panels.
    legend_y = 137
    cv2.circle(canvas, (50, legend_y), 3, initial_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, "initial sample", (62, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, muted, 1, cv2.LINE_AA)
    cv2.circle(canvas, (195, legend_y), 5, (80, 140, 220), -1, cv2.LINE_AA)
    cv2.putText(canvas, "refined / score-colored", (208, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, muted, 1, cv2.LINE_AA)
    cv2.circle(canvas, (420, legend_y), 7, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(canvas, "top-K", (432, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, muted, 1, cv2.LINE_AA)
    cv2.circle(canvas, (505, legend_y), 8, supporter_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "supporter", (519, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, muted, 1, cv2.LINE_AA)
    _draw_star_bgr(canvas, (620, legend_y), 10, chosen_color)
    cv2.putText(canvas, "chosen", (635, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, muted, 1, cv2.LINE_AA)

    # Score colorbar.
    bar_x0, bar_x1 = 1180, 1515
    bar_y0, bar_y1 = 127, 142
    grad = np.arange(256, dtype=np.uint8).reshape(1, -1)
    grad = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
    grad = cv2.resize(grad, (bar_x1 - bar_x0, bar_y1 - bar_y0), interpolation=cv2.INTER_LINEAR)
    canvas[bar_y0:bar_y1, bar_x0:bar_x1] = grad
    cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x1, bar_y1), (90, 90, 90), 1)
    cv2.putText(canvas, f"score {score_min:.4f}" if np.isfinite(score_min) else "score n/a",
                (bar_x0 - 5, bar_y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, muted, 1, cv2.LINE_AA)
    if np.isfinite(score_max):
        cv2.putText(canvas, f"{score_max:.4f}", (bar_x1 - 65, bar_y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, muted, 1, cv2.LINE_AA)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(save_path), canvas)
    if not ok:
        raise RuntimeError(f"Failed to save rotation distribution visualization: {save_path}")
    logging.info(f"[rotation_distribution] saved frame={frame_id}: {save_path}")


def save_rotation_distribution_visualization_3d(
    initial_selected_hypotheses,
    refined_selected_poses,
    scores,
    best_idx,
    consensus_info,
    R_cam_from_world,
    R_reference_world,
    frame_id,
    save_path,
    cone_deg,
):
    """
    Save an additional true 3D rotation-distribution visualization.

    The coordinates are the SO(3) log-map vector components (dRx, dRy, dRz in degrees)
    relative to the previous accepted world rotation. This forms a genuine 3D "rotation
    ball" view, making it easier to see whether the 240 refined poses form one mode,
    multiple modes, or a diffuse cloud. The plot also writes the mean/variance of
    dRx/dRy/dRz directly on the figure.
    """
    if not _HAS_MATPLOTLIB:
        _draw_distribution_placeholder(
            save_path,
            frame_id,
            f"matplotlib is unavailable, so the 3D distribution figure could not be created: {_MATPLOTLIB_IMPORT_ERROR}",
        )
        logging.warning(f"[rotation_distribution_3d] matplotlib unavailable: {_MATPLOTLIB_IMPORT_ERROR}")
        return

    initial_selected_hypotheses = to_numpy_pose_batch(initial_selected_hypotheses)
    refined_selected_poses = to_numpy_pose_batch(refined_selected_poses)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = len(refined_selected_poses)
    if n == 0:
        _draw_distribution_placeholder(save_path, frame_id, "No refined hypotheses were returned.")
        return
    if len(scores) != n:
        raise ValueError(f"distribution score count {len(scores)} != pose count {n}")

    R_cam_from_world = _project_rotation_so3(R_cam_from_world)
    R_world_from_cam = R_cam_from_world.T
    R_reference_world = _project_rotation_so3(R_reference_world)

    def batch_to_rotvecs_world(poses):
        vecs = []
        for pose in poses:
            R_selected = _project_rotation_so3(pose[:3, :3])
            R_world = _project_rotation_so3(R_world_from_cam @ R_selected)
            vecs.append(_relative_rotvec_deg(R_world, R_reference_world))
        return np.asarray(vecs, dtype=np.float64)

    initial_vecs = batch_to_rotvecs_world(initial_selected_hypotheses)
    refined_vecs = batch_to_rotvecs_world(refined_selected_poses)
    geodesic_deg = np.linalg.norm(refined_vecs, axis=1)

    finite_indices = np.where(np.isfinite(scores))[0]
    topk_n = min(int(consensus_info.get("topk", 0)), len(finite_indices))
    if topk_n > 0:
        sorted_finite = finite_indices[np.argsort(scores[finite_indices])[::-1]]
        top_indices = set(int(i) for i in sorted_finite[:topk_n])
    else:
        top_indices = set()
    supporter_indices = set(int(i) for i in consensus_info.get("supporter_indices", []))
    best_idx = int(best_idx)

    all_abs = np.max(np.abs(np.vstack([initial_vecs, refined_vecs]))) if len(initial_vecs) else np.max(np.abs(refined_vecs))
    limit = max(float(cone_deg) * 1.12, float(all_abs) * 1.10, 5.0)
    limit = min(max(limit, 5.0), 180.0)

    mean_xyz = np.mean(refined_vecs, axis=0)
    var_xyz = np.var(refined_vecs, axis=0)
    std_xyz = np.std(refined_vecs, axis=0)
    avg_var_xyz = float(np.mean(var_xyz))
    if len(supporter_indices) > 0:
        supporter_arr = refined_vecs[sorted(supporter_indices)]
        supp_mean = np.mean(supporter_arr, axis=0)
        supp_var = np.var(supporter_arr, axis=0)
        supp_avg_var = float(np.mean(supp_var))
    else:
        supp_mean = None
        supp_var = None
        supp_avg_var = None

    finite_scores = scores[np.isfinite(scores)]
    if len(finite_scores) == 0:
        norm = mpl_colors.Normalize(vmin=0.0, vmax=1.0)
        score_min, score_max = np.nan, np.nan
    else:
        score_min = float(np.min(finite_scores))
        score_max = float(np.max(finite_scores))
        if abs(score_max - score_min) < 1e-12:
            score_min -= 0.5
            score_max += 0.5
        norm = mpl_colors.Normalize(vmin=score_min, vmax=score_max)
    cmap = mpl_cm.ColormapRegistry if False else __import__('matplotlib').colormaps['turbo']
    rgba = np.tile(np.asarray([[0.6, 0.6, 0.6, 1.0]], dtype=np.float64), (n, 1))
    if np.any(np.isfinite(scores)):
        rgba[np.isfinite(scores)] = cmap(norm(scores[np.isfinite(scores)]))

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 0.92])
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax_score = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis('off')

    # 3D rotation ball.
    if len(initial_vecs) > 0:
        ax3d.scatter(initial_vecs[:, 0], initial_vecs[:, 1], initial_vecs[:, 2],
                     s=16, c='lightgray', alpha=0.55, depthshade=False, label='initial sample')
    ax3d.scatter(refined_vecs[:, 0], refined_vecs[:, 1], refined_vecs[:, 2],
                 s=34, c=rgba, edgecolors='none', alpha=0.95, depthshade=False, label='refined')

    if len(top_indices) > 0:
        arr = refined_vecs[sorted(top_indices)]
        ax3d.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=90, facecolors='none',
                     edgecolors='black', linewidths=1.2, depthshade=False, label='top-K')
    if len(supporter_indices) > 0:
        arr = refined_vecs[sorted(supporter_indices)]
        ax3d.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=140, facecolors='none',
                     edgecolors='limegreen', linewidths=2.2, depthshade=False, label='supporter')
    if 0 <= best_idx < n:
        bp = refined_vecs[best_idx]
        ax3d.scatter([bp[0]], [bp[1]], [bp[2]], s=300, marker='*', c='red',
                     edgecolors='white', linewidths=0.8, depthshade=False, label='chosen')
    ax3d.scatter([0.0], [0.0], [0.0], s=130, marker='+', c='black', linewidths=2.0, depthshade=False, label='previous')

    # Cone sphere / ball boundary.
    uu = np.linspace(0.0, 2.0 * np.pi, 40)
    vv = np.linspace(0.0, np.pi, 22)
    r = float(cone_deg)
    xs = r * np.outer(np.cos(uu), np.sin(vv))
    ys = r * np.outer(np.sin(uu), np.sin(vv))
    zs = r * np.outer(np.ones_like(uu), np.cos(vv))
    ax3d.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, color=(0.75, 0.75, 0.75), linewidth=0.5, alpha=0.35)
    ax3d.plot([-limit, limit], [0, 0], [0, 0], color='gray', alpha=0.35, linewidth=1.0)
    ax3d.plot([0, 0], [-limit, limit], [0, 0], color='gray', alpha=0.35, linewidth=1.0)
    ax3d.plot([0, 0], [0, 0], [-limit, limit], color='gray', alpha=0.35, linewidth=1.0)

    ax3d.set_xlim(-limit, limit)
    ax3d.set_ylim(-limit, limit)
    ax3d.set_zlim(-limit, limit)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_xlabel('dRx (deg)')
    ax3d.set_ylabel('dRy (deg)')
    ax3d.set_zlabel('dRz (deg)')
    ax3d.view_init(elev=23, azim=38)
    ax3d.grid(True, alpha=0.25)
    ax3d.set_title(f'3D SO(3) distribution ball | frame {frame_id} | N={n} refined hypotheses', pad=16)

    # Right-top: angle vs score.
    finite_scores = scores[np.isfinite(scores)]
    if len(finite_scores) == 0:
        y_min, y_max = 0.0, 1.0
    else:
        y_min, y_max = float(np.min(finite_scores)), float(np.max(finite_scores))
        if abs(y_max - y_min) < 1e-12:
            y_min -= 0.5
            y_max += 0.5
        else:
            pad = 0.08 * (y_max - y_min)
            y_min -= pad
            y_max += pad
    ax_score.scatter(geodesic_deg, scores, s=30, c=rgba, alpha=0.92)
    if len(top_indices) > 0:
        arr_idx = np.asarray(sorted(top_indices), dtype=np.int64)
        ax_score.scatter(geodesic_deg[arr_idx], scores[arr_idx], s=84, facecolors='none', edgecolors='black', linewidths=1.1)
    if len(supporter_indices) > 0:
        arr_idx = np.asarray(sorted(supporter_indices), dtype=np.int64)
        ax_score.scatter(geodesic_deg[arr_idx], scores[arr_idx], s=120, facecolors='none', edgecolors='limegreen', linewidths=2.0)
    if 0 <= best_idx < n and np.isfinite(scores[best_idx]):
        ax_score.scatter([geodesic_deg[best_idx]], [scores[best_idx]], s=240, marker='*', c='red', edgecolors='white', linewidths=0.8)
    ax_score.axvline(float(cone_deg), color='gray', linestyle='--', linewidth=1.0, alpha=0.8, label='cone')
    ax_score.set_xlim(0.0, max(float(cone_deg) * 1.15, float(np.nanmax(geodesic_deg)) * 1.08, 1.0))
    ax_score.set_ylim(y_min, y_max)
    ax_score.set_xlabel('Geodesic angle from previous rotation (deg)')
    ax_score.set_ylabel('FoundationPose score')
    ax_score.set_title('Angle vs score')
    ax_score.grid(True, alpha=0.25)

    mode = str(consensus_info.get('mode', ''))
    max_supporters = int(consensus_info.get('max_supporters', 0))
    chosen_score = float(scores[best_idx]) if 0 <= best_idx < len(scores) and np.isfinite(scores[best_idx]) else float('nan')
    chosen_vec = refined_vecs[best_idx] if 0 <= best_idx < n else np.asarray([np.nan, np.nan, np.nan])
    support_thresh = float(consensus_info.get('angle_thresh_deg', np.nan))
    min_supporters = int(consensus_info.get('min_supporters', 0))

    def fmt_vec(v):
        return f"[{v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f}]"

    lines = [
        f"Frame: {frame_id}",
        f"Mode: {mode}",
        f"Refined hypotheses: {n}",
        f"Chosen idx: {best_idx}",
        f"Chosen score: {chosen_score:.6f}" if np.isfinite(chosen_score) else "Chosen score: n/a",
        f"Chosen dR (deg): {fmt_vec(chosen_vec)}",
        f"Top-K: {topk_n}",
        f"Max supporters: {max_supporters}   (threshold {support_thresh:.1f} deg, min {min_supporters})",
        "",
        "Refined cloud statistics (SO(3) log-map, deg):",
        f"mean[dRx,dRy,dRz] = {fmt_vec(mean_xyz)}",
        f"avg var(dRx,dRy,dRz) = {avg_var_xyz:.4f}",
        f"std [dRx,dRy,dRz] = {fmt_vec(std_xyz)}",
    ]
    if supp_mean is not None and supp_var is not None:
        lines += [
            "",
            f"Supporter cluster size: {len(supporter_indices)}",
            f"supporter mean = {fmt_vec(supp_mean)}",
            f"supporter avg var = {supp_avg_var:.4f}",
        ]
    lines += [
        "",
        f"Reference center: previous accepted rotation (0,0,0)",
        f"Cone radius: {float(cone_deg):.1f} deg",
        f"Axes: dRx, dRy, dRz are world-frame rotation-vector components",
        f"relative to the previous accepted rotation.",
    ]
    ax_text.text(0.02, 0.98, "\n".join(lines), va='top', ha='left', fontsize=10.5, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='#fafafa', edgecolor='#d0d0d0'))

    sm = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax3d, ax_score], shrink=0.82, pad=0.04)
    cbar.set_label('FoundationPose score')

    handles, labels = ax3d.get_legend_handles_labels()
    uniq = []
    seen = set()
    for h, lab in zip(handles, labels):
        if lab not in seen:
            uniq.append((h, lab))
            seen.add(lab)
    if uniq:
        ax3d.legend([u[0] for u in uniq], [u[1] for u in uniq], loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)

    fig.suptitle('3D temporal rotation distribution (refined 240 poses)', fontsize=15, y=0.995)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=170, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"[rotation_distribution_3d] saved frame={frame_id}: {save_path}")


def select_best_pose_by_largest_rotation_cluster_all(
    refined_all,
    scores_all,
    angle_thresh_deg=10.0,
    min_cluster_size=30,
):
    """
    Fast-fallback selector over ALL refined global-registration hypotheses.

    Rule requested for fast fallback:
      1. Use every finite-scored refined pose (normally exactly 240).
      2. For each pose i, define its rotation cluster as all poses j whose
         SO(3) geodesic rotation difference from i is STRICTLY < angle_thresh_deg.
      3. Pick the largest such cluster. For equal-size clusters, prefer the cluster
         whose highest score is larger, then the one with larger mean score.
      4. If the largest cluster has at least min_cluster_size poses, choose the
         HIGHEST-SCORE pose INSIDE that largest cluster.
      5. Otherwise choose the highest-score pose among ALL hypotheses.

    The cluster is therefore a 10-degree consensus neighborhood around a representative
    pose; it is not a transitive connected component that could chain far beyond 10 deg.
    """
    refined_all = np.asarray(refined_all)
    scores_all = np.asarray(scores_all, dtype=np.float64).reshape(-1)

    if refined_all.ndim != 3 or refined_all.shape[1:] != (4, 4):
        raise ValueError(f"refined_all must have shape (N,4,4), got {refined_all.shape}")
    if len(scores_all) != len(refined_all):
        raise RuntimeError(
            f"Score count {len(scores_all)} does not match pose count {len(refined_all)}"
        )

    finite_indices = np.where(np.isfinite(scores_all))[0]
    if len(finite_indices) == 0:
        raise RuntimeError("No finite scores for fast fallback cluster selection.")

    global_best_idx = int(finite_indices[np.argmax(scores_all[finite_indices])])
    global_best_score = float(scores_all[global_best_idx])
    angle_thresh_deg = float(angle_thresh_deg)
    min_cluster_size = int(min_cluster_size)

    R = np.asarray(
        [_project_rotation_so3(refined_all[k, :3, :3]) for k in finite_indices],
        dtype=np.float64,
    )

    best_anchor_local = -1
    best_support_locals = []
    best_count = -1
    best_cluster_max_score = -np.inf
    best_cluster_mean_score = -np.inf

    # N is 240 in the intended fallback, so O(N^2)=57,600 comparisons is cheap.
    for i in range(len(finite_indices)):
        supporters = []
        for j in range(len(finite_indices)):
            # User rule is strictly rotation difference < 10 degrees.
            if rotation_angle_deg(R[i], R[j]) < angle_thresh_deg:
                supporters.append(j)

        support_global = finite_indices[np.asarray(supporters, dtype=np.int64)]
        count = len(supporters)
        cluster_scores = scores_all[support_global]
        cluster_max_score = float(np.max(cluster_scores)) if count > 0 else -np.inf
        cluster_mean_score = float(np.mean(cluster_scores)) if count > 0 else -np.inf

        better = (
            count > best_count
            or (count == best_count and cluster_max_score > best_cluster_max_score)
            or (
                count == best_count
                and abs(cluster_max_score - best_cluster_max_score) < 1e-12
                and cluster_mean_score > best_cluster_mean_score
            )
        )
        if better:
            best_anchor_local = i
            best_support_locals = supporters
            best_count = count
            best_cluster_max_score = cluster_max_score
            best_cluster_mean_score = cluster_mean_score

    if best_count >= min_cluster_size:
        supporter_global_indices = finite_indices[np.asarray(best_support_locals, dtype=np.int64)]
        chosen_idx = int(
            supporter_global_indices[np.argmax(scores_all[supporter_global_indices])]
        )
        return chosen_idx, {
            "mode": "largest_rotation_cluster_then_best_score",
            "best_idx": chosen_idx,
            "global_best_idx": global_best_idx,
            "global_best_score": global_best_score,
            "num_finite": int(len(finite_indices)),
            "angle_thresh_deg": angle_thresh_deg,
            "min_cluster_size": min_cluster_size,
            "max_cluster_size": int(best_count),
            "cluster_anchor_idx": int(finite_indices[best_anchor_local]),
            "cluster_indices": [int(k) for k in supporter_global_indices],
            "cluster_best_score": float(scores_all[chosen_idx]),
            "cluster_mean_score": best_cluster_mean_score,
        }

    return global_best_idx, {
        "mode": "fallback_global_best_score_no_large_cluster",
        "best_idx": global_best_idx,
        "global_best_idx": global_best_idx,
        "global_best_score": global_best_score,
        "num_finite": int(len(finite_indices)),
        "angle_thresh_deg": angle_thresh_deg,
        "min_cluster_size": min_cluster_size,
        "max_cluster_size": int(best_count),
        "cluster_anchor_idx": (
            int(finite_indices[best_anchor_local]) if best_anchor_local >= 0 else -1
        ),
        "cluster_indices": [],
        "cluster_best_score": "",
        "cluster_mean_score": (
            float(best_cluster_mean_score) if np.isfinite(best_cluster_mean_score) else ""
        ),
    }


def select_best_pose_by_rotation_consensus(
    refined_all,
    scores_all,
    topk=20,
    angle_thresh_deg=10.0,
    min_supporters=6,
):
    """
    Select the best pose by rotation consensus among top-K scored poses.

    Logic:
      1. Sort all refined poses by score.
      2. Take top-K poses.
      3. For each top-K pose, count supporters whose rotation difference
         from it is <= angle_thresh_deg.
      4. If all top-K poses support each other:
            consensus has no extra information, so fallback to global best-score pose.
      5. Else if the largest supporter count >= min_supporters:
            choose the pose with the most supporters.
         Tie-break:
            a. more supporters
            b. higher own score
            c. higher supporter-group mean score
      6. If no valid consensus cluster exists:
            fallback to global best-score pose.

    Note:
      The pose itself is counted as one supporter.
      So min_supporters=6 means itself + 5 nearby poses.
    """
    refined_all = np.asarray(refined_all)
    scores_all = np.asarray(scores_all, dtype=np.float64).reshape(-1)

    if refined_all.ndim != 3 or refined_all.shape[1:] != (4, 4):
        raise ValueError(f"refined_all must have shape (N,4,4), got {refined_all.shape}")

    if len(scores_all) != len(refined_all):
        raise RuntimeError(
            f"Score count {len(scores_all)} does not match pose count {len(refined_all)}"
        )

    finite_mask = np.isfinite(scores_all)
    finite_indices = np.where(finite_mask)[0]

    if len(finite_indices) == 0:
        raise RuntimeError("No finite scores for consensus selection.")

    # Fallback: global best-score pose among all finite scores.
    fallback_idx = int(finite_indices[np.argmax(scores_all[finite_indices])])
    fallback_score = float(scores_all[fallback_idx])

    # Top-K by score.
    sorted_finite_indices = finite_indices[np.argsort(scores_all[finite_indices])[::-1]]
    topk = min(int(topk), len(sorted_finite_indices))
    top_indices = sorted_finite_indices[:topk]

    if topk <= 0:
        return fallback_idx, {
            "mode": "fallback_score_no_topk",
            "best_idx": fallback_idx,
            "fallback_idx": fallback_idx,
            "topk": 0,
            "angle_thresh_deg": float(angle_thresh_deg),
            "min_supporters": int(min_supporters),
            "max_supporters": 0,
            "best_score": fallback_score,
            "fallback_score": fallback_score,
            "supporter_indices": [],
            "supporter_scores": [],
        }

    R_top = refined_all[top_indices, :3, :3].astype(np.float64)

    best_local = None
    best_support_count = -1
    best_own_score = -np.inf
    best_support_mean_score = -np.inf
    best_support_local_indices = []

    # O(topK^2), topK=20, very cheap.
    for i in range(topk):
        supporter_locals = []

        for j in range(topk):
            angle_ij = rotation_angle_deg(R_top[i], R_top[j])
            if angle_ij <= float(angle_thresh_deg):
                supporter_locals.append(j)

        support_count = len(supporter_locals)
        supporter_global_indices = top_indices[supporter_locals]

        own_score = float(scores_all[top_indices[i]])
        support_mean_score = float(np.mean(scores_all[supporter_global_indices]))

        better = False

        if support_count > best_support_count:
            better = True

        elif support_count == best_support_count:
            # If same supporters, choose the pose with higher own score.
            if own_score > best_own_score:
                better = True

            elif abs(own_score - best_own_score) < 1e-12:
                # Final tie-break: choose the cluster with higher mean score.
                if support_mean_score > best_support_mean_score:
                    better = True

        if better:
            best_local = i
            best_support_count = support_count
            best_own_score = own_score
            best_support_mean_score = support_mean_score
            best_support_local_indices = supporter_locals

    # New logic:
    # If all top-K poses support each other, the consensus cluster is not informative.
    # In this case, directly choose the highest-score pose.
    if best_support_count >= topk:
        return fallback_idx, {
            "mode": "fallback_score_all_topk_same_cluster",
            "best_idx": fallback_idx,
            "fallback_idx": fallback_idx,
            "topk": int(topk),
            "angle_thresh_deg": float(angle_thresh_deg),
            "min_supporters": int(min_supporters),
            "max_supporters": int(best_support_count),
            "best_score": fallback_score,
            "fallback_score": fallback_score,
            "supporter_indices": [int(k) for k in top_indices],
            "supporter_scores": [float(scores_all[k]) for k in top_indices],
            "support_mean_score": float(best_support_mean_score),
        }

    if best_support_count >= int(min_supporters):
        best_idx = int(top_indices[best_local])
        supporter_global_indices = [int(top_indices[k]) for k in best_support_local_indices]

        return best_idx, {
            "mode": "rotation_consensus",
            "best_idx": best_idx,
            "fallback_idx": fallback_idx,
            "topk": int(topk),
            "angle_thresh_deg": float(angle_thresh_deg),
            "min_supporters": int(min_supporters),
            "max_supporters": int(best_support_count),
            "supporter_indices": supporter_global_indices,
            "supporter_scores": [float(scores_all[k]) for k in supporter_global_indices],
            "support_mean_score": float(best_support_mean_score),
            "best_score": float(scores_all[best_idx]),
            "fallback_score": fallback_score,
        }

    # No valid cluster, fallback to highest score.
    return fallback_idx, {
        "mode": "fallback_score_no_consensus",
        "best_idx": fallback_idx,
        "fallback_idx": fallback_idx,
        "topk": int(topk),
        "angle_thresh_deg": float(angle_thresh_deg),
        "min_supporters": int(min_supporters),
        "max_supporters": int(best_support_count),
        "best_score": fallback_score,
        "fallback_score": fallback_score,
        "supporter_indices": [],
        "supporter_scores": [],
    }


def refine_and_score_centered_hypotheses(
    est,
    obs,
    pose_hypotheses_centered,
    init_xyz_selected,
    args,
):
    """
    Refine/score temporal hypotheses that are already in selected-camera coordinates.

    New best-pose selection logic:
      1. Refine all hypotheses.
      2. Score all refined poses.
      3. Take top 20 scored poses.
      4. Among top 20, find the pose whose rotation has the most supporters.
         A supporter means rotation difference <= 10 degrees.
      5. The cluster must have at least 6 supporters.
      6. If no valid cluster exists, fallback to the highest-score pose.

    Tie-break inside valid consensus:
      1. More supporters.
      2. Same supporters -> higher own score.
      3. Same own score -> higher supporter-group mean score.

    Important convention:
      FoundationPose's refiner/scorer work on the internally centered mesh.
      Therefore pose_hypotheses_centered should have translation equal to the
      desired object/center position in the selected camera.

      Refinement outputs are passed to the scorer unchanged, including their refined xyz.
      The consensus/best-pose selection therefore operates on scores computed from the full
      refined 6D poses. By default, only AFTER the best refined pose has been selected is the
      CENTERED-MESH pose translation replaced by init_xyz_selected. With --no-force, this final
      replacement is skipped and the selected FoundationPose-refined translation is kept.
      The subsequent centered->original mesh transform is always kept intact.
    """
    pose_hypotheses_centered = to_numpy_pose_batch(pose_hypotheses_centered)
    init_xyz_selected = np.asarray(init_xyz_selected, dtype=np.float32).reshape(3)

    depth_used, depth_info = prepare_depth_for_foundationpose(
        obs=obs, init_xyz_selected=init_xyz_selected, args=args
    )
    xyz_map = depth2xyzmap(depth_used, obs["K"])
    normal_map = None
    logging.info(
        f"[temporal_depth] frame={obs.get('frame_id')} cam={obs.get('cam')} "
        f"mode={depth_info['mode']} rel_median={depth_info['relative_mask_median']:.6f} "
        f"z_ref={depth_info['z_reference']:.6f} scale={depth_info['scale']:.8f} "
        f"aligned_median={depth_info['aligned_mask_median']:.6f}"
    )

    batch_size = int(getattr(args, "temporal_hypothesis_batch_size", 0))
    if batch_size <= 0:
        batch_size = len(pose_hypotheses_centered)

    refined_batches = []
    score_batches = []

    for start in range(0, len(pose_hypotheses_centered), batch_size):
        end = min(start + batch_size, len(pose_hypotheses_centered))
        batch = pose_hypotheses_centered[start:end]

        refined, vis = est.refiner.predict(
            mesh=est.mesh,
            mesh_tensors=est.mesh_tensors,
            rgb=obs["color"],
            depth=depth_used,
            K=obs["K"],
            ob_in_cams=batch,
            normal_map=normal_map,
            xyz_map=xyz_map,
            glctx=est.glctx,
            mesh_diameter=est.diameter,
            iteration=args.est_refine_iter,
            get_vis=est.debug >= 2,
        )

        refined_np = to_numpy_pose_batch(refined)

        # IMPORTANT: do NOT overwrite refined xyz before scoring.
        # The scorer must evaluate the complete FoundationPose-refined 6D hypotheses.
        # Translation is replaced by triangulated xyz only AFTER best-pose selection.

        if vis is not None and est.debug >= 2 and not bool(getattr(args, "simp_saving", True)):
            imageio.imwrite(
                f"{est.debug_dir}/vis_refiner_temporal_cone_{obs['frame_id']}_{start:04d}.png",
                vis,
            )

        scores, vis = est.scorer.predict(
            mesh=est.mesh,
            rgb=obs["color"],
            depth=depth_used,
            K=obs["K"],
            ob_in_cams=refined_np,
            normal_map=normal_map,
            mesh_tensors=est.mesh_tensors,
            glctx=est.glctx,
            mesh_diameter=est.diameter,
            get_vis=est.debug >= 2,
        )

        scores_np = scores_to_numpy(scores)

        if vis is not None and est.debug >= 2 and not bool(getattr(args, "simp_saving", True)):
            imageio.imwrite(
                f"{est.debug_dir}/vis_score_temporal_cone_{obs['frame_id']}_{start:04d}.png",
                vis,
            )

        refined_batches.append(refined_np)
        score_batches.append(scores_np)

    refined_all = np.concatenate(refined_batches, axis=0).astype(np.float32)
    scores_all = np.concatenate(score_batches, axis=0).astype(np.float64)

    if len(scores_all) != len(refined_all):
        raise RuntimeError(
            f"Score count {len(scores_all)} does not match pose count {len(refined_all)}"
        )

    if len(scores_all) == 0 or not np.any(np.isfinite(scores_all)):
        raise RuntimeError("No finite temporal-cone scores were returned.")

    consensus_topk = int(getattr(args, "temporal_consensus_topk", 20))
    consensus_angle_deg = float(getattr(args, "temporal_consensus_angle_deg", 10.0))
    consensus_min_supporters = int(getattr(args, "temporal_consensus_min_supporters", 6))

    best_idx, consensus_info = select_best_pose_by_rotation_consensus(
        refined_all=refined_all,
        scores_all=scores_all,
        topk=consensus_topk,
        angle_thresh_deg=consensus_angle_deg,
        min_supporters=consensus_min_supporters,
    )

    logging.info(
        f"[temporal_consensus] frame={obs['frame_id']} "
        f"mode={consensus_info['mode']}, "
        f"best_idx={best_idx}, "
        f"score={float(scores_all[best_idx]):.8f}, "
        f"fallback_idx={consensus_info.get('fallback_idx', -1)}, "
        f"fallback_score={consensus_info.get('fallback_score', np.nan):.8f}, "
        f"topk={consensus_info.get('topk', consensus_topk)}, "
        f"max_supporters={consensus_info.get('max_supporters', 0)}, "
        f"angle_thresh={consensus_angle_deg}, "
        f"min_supporters={consensus_min_supporters}"
    )

    best_centered = refined_all[best_idx].copy()
    best_refined_xyz = best_centered[:3, 3].copy()

    # Best pose has already been selected using scores from the UNMODIFIED refined poses.
    # Default behavior: now discard the selected pose's refined xyz and replace it with
    # the RANSAC-triangulated object-center xyz in selected-camera coordinates.
    # With --no-force, keep the selected best pose's refined translation unchanged.
    tracking_cam_locked = bool(str(getattr(args, "tracking_cam", "") or "").strip())
    if bool(getattr(args, "no_force", False)) and not tracking_cam_locked:
        logging.info(
            f"[temporal_xyz_free] frame={obs['frame_id']} cam={obs.get('cam')}, "
            f"keeping_best_refined_xyz={best_refined_xyz.tolist()}, "
            f"triangulated_xyz_used_only_for_init={init_xyz_selected.tolist()}"
        )
    else:
        best_centered[:3, 3] = init_xyz_selected
        lock_reason = "tracking_cam" if tracking_cam_locked else "default_force"
        logging.info(
            f"[temporal_xyz_lock] frame={obs['frame_id']} cam={obs.get('cam')}, reason={lock_reason}, "
            f"best_refined_xyz_before_lock={best_refined_xyz.tolist()}, "
            f"triangulated_xyz_selected={init_xyz_selected.tolist()}"
        )

    best_centered_torch = torch.as_tensor(
        best_centered,
        device="cuda",
        dtype=torch.float,
    )

    # Convert the centered-mesh pose back to the original mesh coordinate frame.
    # IMPORTANT: do NOT overwrite best_original[:3, 3] after this multiplication.
    # init_xyz_selected is the triangulated GEOMETRIC CENTER, while best_original translation
    # is the ORIGINAL-MESH ORIGIN. If the mesh origin is offset from its center, these xyz values
    # must differ. This matches the working RGB reference implementation.
    best_original = best_centered_torch @ est.get_tf_to_centered_mesh()

    # Update estimator state.
    est.pose_last = best_centered_torch.detach().clone()
    est.poses = torch.as_tensor(refined_all, device="cuda", dtype=torch.float)
    est.scores = scores_all
    est.best_id = best_idx
    est.consensus_info = consensus_info

    return (
        best_original.detach().cpu().numpy().astype(np.float32),
        float(scores_all[best_idx]),
        refined_all,
        scores_all,
    )

def register_from_best_mask_camera_fast(est, observations, triangulation, cam_ext, last_pose_world, args):
    """
    FAST temporal registration for frames after the first one, using a top-view
    multi-camera 6D-pose consensus / RANSAC-style safety gate.

    Automatic-camera behavior (no --tracking_cam):
      1. Rank all valid observations ONLY by mask_area (foreground pixel count).
         Image brightness is not used in ranking or tie-breaking.
      2. Take the top --fast_consensus_topk views (default 10).
      3. For EACH selected camera, initialize ONE centered-mesh pose using:
             xyz = current multi-view RANSAC-triangulated center
             R   = previous accepted WORLD rotation transformed into that camera
         and run exactly 3 FoundationPose refinement iterations.
      4. Convert every refined rotation and refined centered-object XYZ to WORLD coordinates.
      5. Run a deterministic RANSAC-style consensus search over these candidates. A valid
         cluster has at least --fast_consensus_min_cluster views (default 3), and EVERY
         pair inside the cluster must satisfy BOTH:
             SO(3) geodesic rotation difference < --fast_rot_consistency_deg (default 30 deg)
             WORLD refined-translation distance <= --fast_xyz_consistency_m (default 0.05 m)
         With at most 10 views we exhaustively test subsets, which is deterministic and stricter
         than random RANSAC while implementing the same consensus idea.
      6. Among largest valid clusters, prefer the cluster with the largest SUM(mask_area),
         then the tighter normalized pairwise residual. Build the accepted pose by:
             translation = mask-area weighted mean of refined WORLD XYZ across the cluster
             rotation    = refined WORLD rotation from the largest-mask (best) camera in the cluster
         Rotation is deliberately NOT averaged/fused across cameras.
      7. If no satisfactory cluster exists, FALL BACK FOR THIS FRAME ONLY to the rank-1
         mask's existing full 360° registration (240 hypotheses), 5 refinement iterations.

    --tracking_cam compatibility:
      A user-requested locked camera remains strict. In that mode multi-view consensus is
      intentionally disabled and the previous single-camera fast path is kept.

    NOTE: last_pose_world follows the script's existing convention: it is the previous
    accepted centered-object WORLD pose after the script's online stage.
    """
    if len(observations) == 0:
        raise RuntimeError("No valid camera observation for this frame.")

    observations_sorted = sorted(
        observations,
        key=lambda x: x["mask_area"],
        reverse=True,
    )
    area_rank = {id(obs): i + 1 for i, obs in enumerate(observations_sorted)}

    xyz_world = np.asarray(triangulation["xyz_world"], dtype=np.float64).reshape(3)
    last_pose_world = np.asarray(last_pose_world, dtype=np.float64).reshape(4, 4)
    R_prev_world = _project_rotation_so3(last_pose_world[:3, :3])
    consistency_deg = float(getattr(args, "fast_rot_consistency_deg", 30.0))
    xyz_consistency_m = float(getattr(args, "fast_xyz_consistency_m", 0.05))
    consensus_topk = max(1, int(getattr(args, "fast_consensus_topk", 10)))
    consensus_min_cluster = max(3, int(getattr(args, "fast_consensus_min_cluster", 3)))
    tracking_cam = str(getattr(args, "tracking_cam", "") or "").strip()

    def _refine_one_fast_obs(obs, rank):
        """Refine exactly one previous-rotation hypothesis on one camera."""
        cam = str(obs["cam"])
        if cam not in cam_ext:
            raise KeyError(f"selected cam {cam} is missing in extri.yml")

        T_cam_from_world = np.asarray(cam_ext[cam]["ext"], dtype=np.float64).reshape(4, 4)
        R_cam_from_world = _project_rotation_so3(T_cam_from_world[:3, :3])
        R_world_from_cam = R_cam_from_world.T

        # xyz always comes from the current multi-view triangulated world point.
        init_xyz_selected = world_to_camera_xyz(xyz_world, T_cam_from_world)
        if init_xyz_selected[2] <= 0:
            raise ValueError(
                f"triangulated object is behind selected camera {cam}: "
                f"{init_xyz_selected.tolist()}"
            )

        # Previous accepted WORLD rotation -> current selected-camera coordinates.
        R_init_selected = _project_rotation_so3(R_cam_from_world @ R_prev_world)
        pose_init_centered = make_pose_from_R_t(
            R_init_selected,
            init_xyz_selected,
        ).astype(np.float32)[None, ...]

        depth_used, depth_info = prepare_depth_for_foundationpose(
            obs=obs,
            init_xyz_selected=init_xyz_selected,
            args=args,
        )
        xyz_map = depth2xyzmap(depth_used, obs["K"])

        logging.info(
            f"[fast3] frame={obs['frame_id']} rank={rank} cam={cam}, "
            f"area={obs['mask_area']}, "
            f"xyz_init={init_xyz_selected.tolist()}, refine_iter=3, hypotheses=1, "
            f"depth_mode={depth_info['mode']}"
        )

        refined, vis = est.refiner.predict(
            mesh=est.mesh,
            mesh_tensors=est.mesh_tensors,
            rgb=obs["color"],
            depth=depth_used,
            K=obs["K"],
            ob_in_cams=pose_init_centered,
            normal_map=None,
            xyz_map=xyz_map,
            glctx=est.glctx,
            mesh_diameter=est.diameter,
            iteration=3,
            get_vis=est.debug >= 2,
        )

        refined_all = to_numpy_pose_batch(refined)
        if len(refined_all) < 1:
            raise RuntimeError("Fast refiner returned no pose.")

        best_centered = refined_all[0].astype(np.float64).copy()
        refined_xyz_before_lock = best_centered[:3, 3].copy()
        best_centered[:3, :3] = _project_rotation_so3(best_centered[:3, :3])

        # Compare rotations in WORLD coordinates. Comparing camera-frame rotations
        # directly would incorrectly include the cameras' extrinsic orientation gap.
        R_refined_world = _project_rotation_so3(
            R_world_from_cam @ best_centered[:3, :3]
        )

        # Keep the actual refined CENTERED-object XYZ before any optional xyz lock and
        # express it in the common WORLD frame. This is the quantity averaged across
        # the three cameras when the fast rotation-consistency gate passes.
        refined_xyz_world = (
            np.linalg.inv(T_cam_from_world)
            @ np.asarray([
                refined_xyz_before_lock[0],
                refined_xyz_before_lock[1],
                refined_xyz_before_lock[2],
                1.0,
            ], dtype=np.float64)
        )[:3]

        # Fast mode always INITIALIZES xyz from the triangulated center, but the FINAL
        # accepted translation must respect --no-force exactly like the normal path.
        # A locked --tracking_cam intentionally keeps the historical xyz-lock behavior.
        tracking_cam_locked = bool(str(getattr(args, "tracking_cam", "") or "").strip())
        if bool(getattr(args, "no_force", False)) and not tracking_cam_locked:
            logging.info(
                f"[fast3_xyz_free] frame={obs['frame_id']} cam={cam}, "
                f"keeping_refined_centered_xyz={refined_xyz_before_lock.tolist()}, "
                f"triangulated_xyz_used_only_for_init={init_xyz_selected.tolist()}"
            )
        else:
            best_centered[:3, 3] = np.asarray(init_xyz_selected, dtype=np.float64)
            lock_reason = "tracking_cam" if tracking_cam_locked else "default_force"
            logging.info(
                f"[fast3_xyz_lock] frame={obs['frame_id']} cam={cam}, reason={lock_reason}, "
                f"centered_xyz_before_lock={refined_xyz_before_lock.tolist()}, "
                f"triangulated_xyz_measurement={init_xyz_selected.tolist()}"
            )

        if vis is not None and est.debug >= 2 and not bool(getattr(args, "simp_saving", True)):
            imageio.imwrite(
                f"{est.debug_dir}/vis_refiner_fast_{obs['frame_id']}_rank{rank}_cam{cam}.png",
                vis,
            )

        best_centered_torch = torch.as_tensor(
            best_centered,
            device="cuda",
            dtype=torch.float,
        )
        best_original = best_centered_torch @ est.get_tf_to_centered_mesh()

        return {
            "rank": int(rank),
            "cam": cam,
            "obs": obs,
            "depth_used": depth_used,
            "depth_info": depth_info,
            "init_xyz_selected": np.asarray(init_xyz_selected, dtype=np.float32),
            "best_centered": best_centered.astype(np.float32),
            "best_original": best_original.detach().cpu().numpy().astype(np.float32),
            "R_world": R_refined_world.astype(np.float64),
            "refined_xyz_before_lock": refined_xyz_before_lock.astype(np.float64),
            "refined_xyz_world": refined_xyz_world.astype(np.float64),
        }

    def _override_fast_trial_world_xyz(trial, xyz_world_override, xyz_source):
        """Return a copy of one fast trial with its CENTERED translation set from WORLD xyz."""
        trial = dict(trial)
        cam = str(trial["cam"])
        T_cam_from_world = np.asarray(cam_ext[cam]["ext"], dtype=np.float64).reshape(4, 4)
        xyz_cam = world_to_camera_xyz(xyz_world_override, T_cam_from_world)
        best_centered = np.asarray(trial["best_centered"], dtype=np.float64).reshape(4, 4).copy()
        best_centered[:3, 3] = xyz_cam
        best_centered_torch = torch.as_tensor(best_centered, device="cuda", dtype=torch.float)
        best_original = best_centered_torch @ est.get_tf_to_centered_mesh()
        trial["best_centered"] = best_centered.astype(np.float32)
        trial["best_original"] = best_original.detach().cpu().numpy().astype(np.float32)
        trial["accepted_xyz_world"] = np.asarray(xyz_world_override, dtype=np.float64).reshape(3)
        trial["xyz_source"] = str(xyz_source)
        return trial

    def _override_fast_trial_world_pose(trial, R_world_override, xyz_world_override, pose_source):
        """
        Return a copy of one representative fast trial with BOTH centered rotation and
        centered translation replaced by a fused WORLD-frame 6D pose.

        The representative camera is used only to package the fused pose back into the
        same object->camera convention expected by FoundationPose / downstream code.
        """
        trial = dict(trial)
        cam = str(trial["cam"])
        T_cam_from_world = np.asarray(cam_ext[cam]["ext"], dtype=np.float64).reshape(4, 4)
        R_cam_from_world = _project_rotation_so3(T_cam_from_world[:3, :3])
        R_world_override = _project_rotation_so3(
            np.asarray(R_world_override, dtype=np.float64).reshape(3, 3)
        )
        xyz_world_override = np.asarray(xyz_world_override, dtype=np.float64).reshape(3)

        best_centered = np.asarray(trial["best_centered"], dtype=np.float64).reshape(4, 4).copy()
        best_centered[:3, :3] = _project_rotation_so3(R_cam_from_world @ R_world_override)
        best_centered[:3, 3] = world_to_camera_xyz(xyz_world_override, T_cam_from_world)

        best_centered_torch = torch.as_tensor(best_centered, device="cuda", dtype=torch.float)
        best_original = best_centered_torch @ est.get_tf_to_centered_mesh()

        trial["best_centered"] = best_centered.astype(np.float32)
        trial["best_original"] = best_original.detach().cpu().numpy().astype(np.float32)
        trial["R_world"] = R_world_override.astype(np.float64)
        trial["refined_xyz_world"] = xyz_world_override.astype(np.float64)
        trial["accepted_xyz_world"] = xyz_world_override.astype(np.float64)
        trial["xyz_source"] = str(pose_source)
        trial["pose_source"] = str(pose_source)
        return trial

    def _install_fast_trial(trial, mode, selection_reason, trial_summary):
        """Install the accepted fast trial into FoundationPose state and return it."""
        best_centered_torch = torch.as_tensor(
            trial["best_centered"],
            device="cuda",
            dtype=torch.float,
        )
        est.pose_last = best_centered_torch.detach().clone()
        est.poses = best_centered_torch.detach().clone()[None, ...]
        est.scores = np.asarray([], dtype=np.float64)
        est.best_id = 0
        est.consensus_info = {"mode": mode}

        obs_used = dict(trial["obs"])
        obs_used["depth_used"] = trial["depth_used"]
        obs_used["depth_scale_info"] = trial["depth_info"]

        accepted_xyz = np.asarray(trial["best_centered"], dtype=np.float64).reshape(4, 4)[:3, 3]
        logging.info(
            f"[fast3] frame={trial['obs']['frame_id']} ACCEPT rank={trial['rank']} "
            f"cam={trial['cam']}: final_centered_xyz={accepted_xyz.tolist()}, "
            f"triangulated_xyz_init={trial['init_xyz_selected'].tolist()}, "
            f"no_force={int(bool(getattr(args, 'no_force', False)))}, "
            f"xyz_source={trial.get('xyz_source', 'single_refined_or_legacy')}, "
            f"reason={selection_reason}"
        )

        info = {
            "mode": mode,
            "score": "",
            "num_hypotheses": 1,
            "cone_deg": 0.0,
            "rotation_avg_var": "",
            "mask_area_rank": int(trial["rank"]),
            "mask_var_threshold": "",
            "mask_var_selection_reason": selection_reason,
            "mask_var_trials": trial_summary,
            "candidates_tried": trial["cam"],
            "xyz_source": trial.get("xyz_source", "single_refined_or_legacy"),
            "xyz_world_measurement": trial.get("accepted_xyz_world", ""),
        }
        return (
            obs_used,
            trial["best_original"],
            trial["init_xyz_selected"],
            info,
        )

    def _fallback_to_best_mask_full_register(reason, fast_trials=None, pairwise_text=""):
        """
        Fast-mode safety fallback for THIS FRAME ONLY:
          - use ONLY the rank-1 / largest-area mask camera;
          - run full all-direction FoundationPose registration (360-degree rotation coverage);
          - use exactly 240 global rotation hypotheses whenever est.rot_grid has >=240 entries;
          - refine all hypotheses for exactly 5 iterations and score all refined poses;
          - first find the largest refined-rotation consensus cluster using strict SO(3)
            geodesic distance < --fast_fallback_cluster_deg (default 10 deg);
          - require at least --fast_fallback_cluster_min poses (default 30);
          - if a valid cluster exists, choose the highest-score pose INSIDE the largest cluster;
          - otherwise choose the highest-score pose among ALL refined hypotheses;
          - final centered-mesh xyz follows --no-force: keep refined xyz when enabled,
            otherwise lock back to the current multi-view triangulated center.
        """
        fast_trials = fast_trials or []
        frame_id = observations_sorted[0].get("frame_id", "") if observations_sorted else ""
        cluster_deg = float(getattr(args, "fast_fallback_cluster_deg", 10.0))
        cluster_min = int(getattr(args, "fast_fallback_cluster_min", 30))

        logging.warning(
            f"[fast3_fallback] frame={frame_id}: {reason}. "
            f"Switching THIS FRAME to BEST MASK ONLY full 360deg registration, "
            f"240 hypotheses, 5 refinement iterations; largest rot cluster <{cluster_deg:.3f}deg "
            f"with min_size={cluster_min}, then best score inside cluster; "
            f"otherwise global best score. pairwise={pairwise_text or 'n/a'}"
        )

        if len(observations_sorted) < 1:
            raise RuntimeError("Fast fallback requested but no valid observations are available.")

        obs = observations_sorted[0]
        cam = str(obs["cam"])
        if cam not in cam_ext:
            raise KeyError(f"selected cam {cam} is missing in extri.yml")

        T_cam_from_world = np.asarray(cam_ext[cam]["ext"], dtype=np.float64).reshape(4, 4)
        init_xyz_selected = world_to_camera_xyz(xyz_world, T_cam_from_world)
        if init_xyz_selected[2] <= 0:
            raise ValueError(
                f"triangulated object is behind selected camera {cam}: {init_xyz_selected.tolist()}"
            )

        depth_used, depth_info = prepare_depth_for_foundationpose(
            obs=obs,
            init_xyz_selected=init_xyz_selected,
            args=args,
        )

        # FoundationPose's default global rotation grid is usually 252 before symmetry
        # clustering. Temporarily use exactly 240 evenly spread entries so this fallback
        # really refines/scores 240 hypotheses, then restore the estimator's original grid.
        original_rot_grid = getattr(est, "rot_grid", None)
        if original_rot_grid is None:
            raise RuntimeError(
                "FoundationPose estimator has no rot_grid; cannot enforce 240 full-rotation hypotheses."
            )

        n_grid = int(original_rot_grid.shape[0])
        if n_grid < 240:
            logging.warning(
                f"[fast3_fallback] FoundationPose rot_grid has only {n_grid} hypotheses after "
                "symmetry clustering; cannot create 240 unique grid entries without changing "
                "the object's symmetry handling. Using all available hypotheses instead."
            )
            fallback_rot_grid = original_rot_grid
        elif n_grid == 240:
            fallback_rot_grid = original_rot_grid
        else:
            # Evenly subsample across the precomputed all-direction grid rather than just
            # taking the first 240 entries, preserving broad 360-degree coverage.
            idx_np = np.linspace(0, n_grid - 1, 240, dtype=np.int64)
            if torch.is_tensor(original_rot_grid):
                idx_t = torch.as_tensor(idx_np, device=original_rot_grid.device, dtype=torch.long)
                fallback_rot_grid = original_rot_grid.index_select(0, idx_t)
            else:
                fallback_rot_grid = np.asarray(original_rot_grid)[idx_np]

        logging.info(
            f"[fast3_fallback] frame={obs['frame_id']} selected_cam={cam}, "
            f"area={obs['mask_area']}, "
            f"xyz_init={init_xyz_selected.tolist()}, full_register_refine_iter=5, "
            f"search=360deg, rot_grid_original={n_grid}, hypotheses_used={len(fallback_rot_grid)}, "
            f"cluster_deg={cluster_deg}, cluster_min={cluster_min}, depth_mode={depth_info['mode']}"
        )

        try:
            est.rot_grid = fallback_rot_grid

            # Run the normal global FoundationPose registration so the same all-direction
            # initialization/refiner/scorer implementation is used. We do NOT trust its
            # returned best-score pose yet; est.poses + est.scores contain all refined/scored
            # hypotheses and are re-selected below using the requested cluster rule.
            _ = est.register(
                K=obs["K"],
                rgb=obs["color"],
                depth=depth_used,
                ob_mask=obs["mask"].astype(bool),
                iteration=5,
                init_xyz=init_xyz_selected,
            )

            refined_all = to_numpy_pose_batch(est.poses)
            scores_all = scores_to_numpy(est.scores)
        finally:
            est.rot_grid = original_rot_grid

        if len(refined_all) != len(scores_all):
            raise RuntimeError(
                f"Fast fallback refined pose count {len(refined_all)} != score count {len(scores_all)}"
            )
        if len(refined_all) == 0:
            raise RuntimeError("Fast fallback returned zero refined poses.")

        best_idx, cluster_info = select_best_pose_by_largest_rotation_cluster_all(
            refined_all=refined_all,
            scores_all=scores_all,
            angle_thresh_deg=cluster_deg,
            min_cluster_size=cluster_min,
        )

        best_centered = refined_all[best_idx].astype(np.float64).copy()
        refined_xyz_before_lock = best_centered[:3, 3].copy()
        best_centered[:3, :3] = _project_rotation_so3(best_centered[:3, :3])

        # Final translation behavior in fast-fallback mode now matches the normal path:
        # use the triangulated center only as the initialization/measurement anchor, and
        # keep the refined xyz unchanged when --no-force is set (unless --tracking_cam is
        # locked, which still enforces translation locking to the triangulated center).
        tracking_cam_locked = bool(str(getattr(args, "tracking_cam", "") or "").strip())
        if bool(getattr(args, "no_force", False)) and not tracking_cam_locked:
            logging.info(
                f"[fast3_fallback_xyz_free] frame={obs['frame_id']} cam={cam}, "
                f"keeping_refined_centered_xyz={refined_xyz_before_lock.tolist()}, "
                f"triangulated_xyz_used_only_for_init={init_xyz_selected.tolist()}"
            )
        else:
            best_centered[:3, 3] = init_xyz_selected
            lock_reason = "tracking_cam" if tracking_cam_locked else "default_force"
            logging.info(
                f"[fast3_fallback_xyz_lock] frame={obs['frame_id']} cam={cam}, reason={lock_reason}, "
                f"centered_xyz_before_lock={refined_xyz_before_lock.tolist()}, "
                f"triangulated_xyz_measurement={init_xyz_selected.tolist()}"
            )

        best_centered_torch = torch.as_tensor(
            best_centered, device="cuda", dtype=torch.float
        )
        best_original = best_centered_torch @ est.get_tf_to_centered_mesh()

        # Keep FoundationPose state coherent for any downstream code. Rotation temporal
        # prior in this script comes from last_pose_world, but pose_last should still match
        # the actually accepted centered pose rather than est.register()'s score-only choice.
        est.pose_last = best_centered_torch.detach().clone()
        est.poses = torch.as_tensor(refined_all, device="cuda", dtype=torch.float)
        est.scores = scores_all.copy()
        est.best_id = int(best_idx)
        est.consensus_info = dict(cluster_info)

        obs_used = dict(obs)
        obs_used["depth_used"] = depth_used
        obs_used["depth_scale_info"] = depth_info

        prefix = "fast3_gate:" + reason.replace(";", ",")
        if pairwise_text:
            prefix += f":{pairwise_text}"

        chosen_score = float(scores_all[best_idx])
        global_best_idx = int(cluster_info.get("global_best_idx", best_idx))
        global_best_score = float(cluster_info.get("global_best_score", chosen_score))
        max_cluster_size = int(cluster_info.get("max_cluster_size", 0))
        selection_mode = str(cluster_info.get("mode", ""))

        info = {
            "mode": "fast3_fallback_best_mask_full_register_cluster",
            "score": chosen_score,
            "num_hypotheses": int(len(refined_all)),
            "cone_deg": 360.0,
            "rotation_avg_var": "",
            "mask_area_rank": 1,
            "mask_var_threshold": "",
            "mask_var_selection_reason": (
                f"{selection_mode};cluster_deg={cluster_deg:.3f};"
                f"min_cluster={cluster_min};max_cluster={max_cluster_size};"
                f"chosen_score={chosen_score:.8f};global_best_score={global_best_score:.8f}"
            ),
            "mask_var_trials": prefix,
            "candidates_tried": cam,
        }

        logging.info(
            f"[fast3_fallback_cluster] frame={obs['frame_id']} ACCEPT cam={cam}: "
            f"selection={selection_mode}, hypotheses={len(refined_all)}, "
            f"max_cluster_size={max_cluster_size}, cluster_min={cluster_min}, "
            f"cluster_deg={cluster_deg:.3f}, chosen_idx={best_idx}, chosen_score={chosen_score:.8f}, "
            f"global_best_idx={global_best_idx}, global_best_score={global_best_score:.8f}, "
            f"centered_xyz_refined={refined_xyz_before_lock.tolist()}, "
            f"triangulated_xyz_init={init_xyz_selected.tolist()}, "
            f"no_force={int(bool(getattr(args, 'no_force', False)))}"
        )

        return (
            obs_used,
            best_original.detach().cpu().numpy().astype(np.float32),
            np.asarray(init_xyz_selected, dtype=np.float32),
            info,
        )

    # Preserve the previous strict fixed-camera option. A locked camera and a
    # multi-camera consensus gate are mutually exclusive by definition.
    if tracking_cam:
        locked = [obs for obs in observations_sorted if str(obs["cam"]) == tracking_cam]
        if len(locked) == 0:
            available = [str(obs["cam"]) for obs in observations_sorted]
            raise RuntimeError(
                f"Locked tracking camera {tracking_cam} has no valid observation in this frame. "
                f"Valid cameras: {available}"
            )
        logging.info(
            f"[tracking_cam] FAST tracking locked to cam={tracking_cam}; "
            "multi-view top-k consensus is disabled because camera switching is explicitly locked; "
            "xyz still comes from multi-view triangulation"
        )
        rank = int(area_rank.get(id(locked[0]), 1))
        trial = _refine_one_fast_obs(locked[0], rank)
        return _install_fast_trial(
            trial,
            mode="fast_single_previous_rotation_locked_cam",
            selection_reason="locked_tracking_cam_no_multiview_consensus",
            trial_summary=f"rank{rank}:cam{tracking_cam}:fast_single_pose_locked",
        )

    # ------------------------------------------------------------------
    # Automatic FAST mode: top-K independent refinements -> deterministic
    # RANSAC-style 6D consensus -> fused XYZ + best-camera rotation.
    # ------------------------------------------------------------------
    top_obs = observations_sorted[:consensus_topk]
    if len(top_obs) < consensus_min_cluster:
        return _fallback_to_best_mask_full_register(
            reason=(
                f"only_{len(top_obs)}_valid_masks_available_need_at_least_"
                f"{consensus_min_cluster}_for_consensus"
            ),
        )

    fast_trials = []
    failures = []
    for rank, obs in enumerate(top_obs, start=1):
        try:
            fast_trials.append(_refine_one_fast_obs(obs, rank))
        except Exception as e:
            failures.append(f"rank{rank}:cam{obs.get('cam','?')}:{e}")
            logging.warning(
                f"[fast10] frame={obs.get('frame_id','')} rank={rank} cam={obs.get('cam','?')} "
                f"single-pose refinement failed: {e}"
            )

    if len(fast_trials) < consensus_min_cluster:
        return _fallback_to_best_mask_full_register(
            reason=(
                f"only_{len(fast_trials)}_successful_fast_poses_need_at_least_"
                f"{consensus_min_cluster};failures=" + "|".join(failures)
            ),
            fast_trials=fast_trials,
        )

    def _pair_metrics(trials):
        n = len(trials)
        rot_deg = np.full((n, n), np.inf, dtype=np.float64)
        xyz_m = np.full((n, n), np.inf, dtype=np.float64)
        ok = np.zeros((n, n), dtype=bool)
        for i in range(n):
            rot_deg[i, i] = 0.0
            xyz_m[i, i] = 0.0
            ok[i, i] = True
        for i in range(n):
            for j in range(i + 1, n):
                rd = float(rotation_angle_deg(trials[i]["R_world"], trials[j]["R_world"]))
                td = float(np.linalg.norm(
                    np.asarray(trials[i]["refined_xyz_world"], dtype=np.float64).reshape(3)
                    - np.asarray(trials[j]["refined_xyz_world"], dtype=np.float64).reshape(3)
                ))
                rot_deg[i, j] = rot_deg[j, i] = rd
                xyz_m[i, j] = xyz_m[j, i] = td
                # Requested rule: rotations strictly <30 deg, translations within 5 cm.
                pair_ok = (
                    np.isfinite(rd) and np.isfinite(td)
                    and rd < consistency_deg
                    and td <= xyz_consistency_m
                )
                ok[i, j] = ok[j, i] = bool(pair_ok)
        return rot_deg, xyz_m, ok

    def _cluster_compactness(indices, rot_deg, xyz_m):
        vals = []
        for a, b in itertools.combinations(indices, 2):
            # Normalize both residuals by their corresponding gates so rotation
            # and translation contribute on comparable scales.
            vals.append(
                (float(rot_deg[a, b]) / max(consistency_deg, 1e-9))
                + (float(xyz_m[a, b]) / max(xyz_consistency_m, 1e-9))
            )
        return float(np.mean(vals)) if vals else 0.0

    def _find_best_consensus_cluster(trials, rot_deg, xyz_m, pair_ok):
        """
        Deterministic RANSAC-style consensus over <=10 views.

        We search from largest subset down to min size and require a complete graph:
        EVERY pair in the chosen cluster passes BOTH the rotation and translation gate.
        This exactly implements "at least 3, each other <30 deg and within 5 cm".
        """
        n = len(trials)
        min_size = min(consensus_min_cluster, n)
        for size in range(n, min_size - 1, -1):
            valid = []
            for combo in itertools.combinations(range(n), size):
                mutual = True
                for a, b in itertools.combinations(combo, 2):
                    if not bool(pair_ok[a, b]):
                        mutual = False
                        break
                if not mutual:
                    continue
                area_sum = float(sum(max(float(trials[k]["obs"]["mask_area"]), 1.0) for k in combo))
                compact = _cluster_compactness(combo, rot_deg, xyz_m)
                # Larger area is better; smaller compactness is better.
                valid.append((combo, area_sum, compact))
            if valid:
                valid.sort(key=lambda item: (-item[1], item[2], tuple(item[0])))
                combo, area_sum, compact = valid[0]
                return list(combo), {
                    "size": int(size),
                    "area_sum": float(area_sum),
                    "compactness": float(compact),
                    "num_same_size_valid_clusters": int(len(valid)),
                }
        return None, {
            "size": 0,
            "area_sum": 0.0,
            "compactness": np.inf,
            "num_same_size_valid_clusters": 0,
        }

    def _weighted_rotation_mean_chordal(rotations, weights):
        """Legacy helper; FAST accepted-pose path no longer fuses rotations across cameras."""
        rotations = [
            _project_rotation_so3(np.asarray(R, dtype=np.float64).reshape(3, 3))
            for R in rotations
        ]
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        weights = np.maximum(weights, 1e-12)
        weights = weights / weights.sum()
        M = np.zeros((3, 3), dtype=np.float64)
        for w, R in zip(weights, rotations):
            M += float(w) * R
        U, _, Vt = np.linalg.svd(M)
        D = np.eye(3, dtype=np.float64)
        if np.linalg.det(U @ Vt) < 0.0:
            D[2, 2] = -1.0
        return _project_rotation_so3(U @ D @ Vt)

    rot_mat, xyz_mat, pair_ok = _pair_metrics(fast_trials)
    cluster_indices, cluster_info = _find_best_consensus_cluster(
        fast_trials, rot_mat, xyz_mat, pair_ok
    )

    # Compact debug summary of all successful top-K candidates.
    candidate_summary = ";".join(
        f"r{tr['rank']}:cam{tr['cam']}:area={int(tr['obs']['mask_area'])}"
        for tr in fast_trials
    )

    if cluster_indices is None or len(cluster_indices) < consensus_min_cluster:
        pair_debug = []
        for i in range(len(fast_trials)):
            for j in range(i + 1, len(fast_trials)):
                pair_debug.append(
                    f"cam{fast_trials[i]['cam']}-cam{fast_trials[j]['cam']}:"
                    f"R={rot_mat[i,j]:.2f}deg,T={xyz_mat[i,j]*100.0:.2f}cm,"
                    f"ok={int(pair_ok[i,j])}"
                )
        return _fallback_to_best_mask_full_register(
            reason=(
                f"no_pose_consensus_cluster_min{consensus_min_cluster}_"
                f"rotlt{consistency_deg:.3f}deg_xyzle{xyz_consistency_m*100.0:.2f}cm"
            ),
            fast_trials=fast_trials,
            pairwise_text="|".join(pair_debug),
        )

    cluster_trials = [fast_trials[i] for i in cluster_indices]
    weights = np.asarray(
        [max(float(tr["obs"]["mask_area"]), 1.0) for tr in cluster_trials],
        dtype=np.float64,
    )
    weights_norm = weights / (weights.sum() + 1e-12)

    # Fuse translation only. Each candidate contributes its actual refined centered-object
    # WORLD XYZ, weighted by visible mask area.
    fused_xyz_world = np.sum(
        np.stack([
            np.asarray(tr["refined_xyz_world"], dtype=np.float64).reshape(3)
            for tr in cluster_trials
        ], axis=0) * weights_norm[:, None],
        axis=0,
    )

    # IMPORTANT: do NOT fuse/average rotations across cameras.
    # The final rotation is taken directly from the best camera INSIDE the accepted
    # consensus cluster. "Best" follows the existing camera ranking: largest mask area,
    # with smaller area-rank as deterministic tie-break.
    representative = max(
        cluster_trials,
        key=lambda tr: (float(tr["obs"]["mask_area"]), -int(tr["rank"])),
    )
    best_cam_R_world = _project_rotation_so3(
        np.asarray(representative["R_world"], dtype=np.float64).reshape(3, 3)
    )

    chosen = _override_fast_trial_world_pose(
        representative,
        R_world_override=best_cam_R_world,
        xyz_world_override=fused_xyz_world,
        pose_source="best_cam_rotation_plus_maskarea_weighted_xyz",
    )

    # Verify/report the selected cluster's actual worst pair.
    max_cluster_rot_deg = 0.0
    max_cluster_xyz_m = 0.0
    selected_pair_text = []
    for ai, bi in itertools.combinations(cluster_indices, 2):
        rd = float(rot_mat[ai, bi])
        td = float(xyz_mat[ai, bi])
        max_cluster_rot_deg = max(max_cluster_rot_deg, rd)
        max_cluster_xyz_m = max(max_cluster_xyz_m, td)
        selected_pair_text.append(
            f"cam{fast_trials[ai]['cam']}-cam{fast_trials[bi]['cam']}="
            f"{rd:.2f}deg/{td*100.0:.2f}cm"
        )

    member_text = ",".join(
        f"cam{tr['cam']}(rank={tr['rank']},area={int(tr['obs']['mask_area'])},w={w:.4f})"
        for tr, w in zip(cluster_trials, weights_norm)
    )
    logging.info(
        f"[fast10_ransac] frame={top_obs[0]['frame_id']} ACCEPT "
        f"cluster_size={len(cluster_trials)}/{len(fast_trials)}, members=[{member_text}], "
        f"rot_gate_strict_lt={consistency_deg:.3f}deg, "
        f"xyz_gate_le={xyz_consistency_m*100.0:.2f}cm, "
        f"cluster_max_rot={max_cluster_rot_deg:.3f}deg, "
        f"cluster_max_xyz={max_cluster_xyz_m*100.0:.2f}cm, "
        f"area_sum={cluster_info['area_sum']:.1f}, compactness={cluster_info['compactness']:.6f}, "
        f"rot_source=best_cam{representative['cam']}(rank={representative['rank']},"
        f"area={int(representative['obs']['mask_area'])}), "
        f"fused_xyz_world={fused_xyz_world.tolist()}"
    )

    trial_summary = (
        f"topk={len(top_obs)};successful={len(fast_trials)};"
        f"min_cluster={consensus_min_cluster};cluster_size={len(cluster_trials)};"
        f"rot_lt={consistency_deg:.3f}deg;xyz_le={xyz_consistency_m*100.0:.2f}cm;"
        f"members={member_text};"
        f"rotation_source=best_cam{representative['cam']}_rank{representative['rank']};"
        f"xyz_source=maskarea_weighted_cluster_mean;"
        f"pairs={'|'.join(selected_pair_text)};"
        f"all_candidates={candidate_summary};failures={'|'.join(failures) if failures else 'none'}"
    )
    return _install_fast_trial(
        chosen,
        mode="fast_topk_ransac_bestcam_rotation_fused_xyz",
        selection_reason=(
            f"largest_mutual_6d_consensus_cluster_size_{len(cluster_trials)}_"
            f"of_{len(fast_trials)}_bestcam_rotation_maskarea_weighted_xyz"
        ),
        trial_summary=trial_summary,
    )

def register_from_best_mask_camera_temporal_cone(est, observations, triangulation, cam_ext, last_pose_world, args):
    """
    Temporal-cone registration with area-ranked mask retry using rotation variance.

    Camera/mask selection:
      1. Rank valid masks ONLY by mask area (foreground pixel count).
      2. Try rank-1 mask and compute the refined-240 rotation avg variance:
             avg_var = mean([var(dRx), var(dRy), var(dRz)])
         where dR is the WORLD-frame SO(3) log-map relative to last accepted rotation.
      3. If avg_var < --mask_rotation_var_threshold, accept immediately.
      4. Otherwise try rank-2, then rank-3 (or --mask_rotation_var_topn).
      5. If none gets below the threshold, choose the successful trial with the
         smallest avg_var.

    The original object-center xyz initialization and temporal rotation cone are
    unchanged. Only the selected mask/camera is quality-gated by the refined-pose
    distribution variance.
    """
    if len(observations) == 0:
        raise RuntimeError("No valid camera observation for this frame.")

    observations_sorted = sorted(
        observations,
        key=lambda x: x["mask_area"],
        reverse=True,
    )
    area_rank = {id(obs): i + 1 for i, obs in enumerate(observations_sorted)}

    var_gate_enabled = bool(int(getattr(args, "mask_rotation_var_gate", 1)))
    var_threshold = float(getattr(args, "mask_rotation_var_threshold", 6.0))
    topn = max(1, int(getattr(args, "mask_rotation_var_topn", 3)))
    tracking_cam = str(getattr(args, "tracking_cam", "") or "").strip()
    if tracking_cam:
        candidates = [obs for obs in observations_sorted if str(obs["cam"]) == tracking_cam]
        if len(candidates) == 0:
            available = [str(obs["cam"]) for obs in observations_sorted]
            raise RuntimeError(
                f"Locked tracking camera {tracking_cam} has no valid observation in this frame. "
                f"Valid cameras: {available}"
            )
        logging.info(
            f"[tracking_cam] temporal-cone tracking locked to cam={tracking_cam}; "
            "rotation-variance camera switching is disabled, xyz still comes from multi-view triangulation"
        )
    elif var_gate_enabled:
        candidates = observations_sorted[:min(topn, len(observations_sorted))]
    else:
        candidates = observations_sorted if args.fallback_next_best else observations_sorted[:1]

    xyz_world = np.asarray(triangulation["xyz_world"], dtype=np.float64).reshape(3)
    last_pose_world = np.asarray(last_pose_world, dtype=np.float64).reshape(4, 4)
    trials = []
    failed = []

    for candidate_index, obs in enumerate(candidates, start=1):
        rank = int(area_rank.get(id(obs), candidate_index))
        cam = str(obs["cam"])
        try:
            if cam not in cam_ext:
                raise KeyError(f"selected cam {cam} is missing in extri.yml")

            T_cam_from_world = np.asarray(cam_ext[cam]["ext"], dtype=np.float64).reshape(4, 4)
            R_cam_from_world = T_cam_from_world[:3, :3]

            init_xyz_selected = world_to_camera_xyz(xyz_world, T_cam_from_world)
            if init_xyz_selected[2] <= 0:
                raise ValueError(
                    f"triangulated object is behind selected camera {cam}: {init_xyz_selected.tolist()}"
                )

            # Use exactly the same world-space hypothesis distribution for every
            # candidate camera in the same frame, so camera variance comparison is fair.
            rng_seed = int(getattr(args, "temporal_seed", 0))
            frame_id_str = str(obs.get("frame_id", ""))
            if frame_id_str.isdigit():
                rng_seed += int(frame_id_str)
            rng = np.random.default_rng(rng_seed)

            R_world_hyps = sample_rotation_cone_around_last(
                last_pose_world[:3, :3],
                max_angle_deg=args.temporal_pose_cone_deg,
                num_hypotheses=args.temporal_num_hypotheses,
                rng=rng,
            )

            pose_selected_hyps_centered = []
            for R_world in R_world_hyps:
                R_selected = R_cam_from_world @ R_world
                pose_selected_hyps_centered.append(make_pose_from_R_t(R_selected, init_xyz_selected))
            pose_selected_hyps_centered = np.stack(pose_selected_hyps_centered, axis=0).astype(np.float32)

            logging.info(
                f"[mask_var_gate] frame={obs['frame_id']} try rank={rank} cam={cam}, "
                f"area={obs['mask_area']}, "
                f"threshold={var_threshold:.4f} deg^2"
            )

            pose_selected, best_score, refined_all, scores = refine_and_score_centered_hypotheses(
                est=est,
                obs=obs,
                pose_hypotheses_centered=pose_selected_hyps_centered,
                init_xyz_selected=init_xyz_selected,
                args=args,
            )

            avg_var, mean_xyz, var_xyz, _ = compute_refined_rotation_avg_var_world(
                refined_selected_poses=refined_all,
                R_cam_from_world=R_cam_from_world,
                R_reference_world=last_pose_world[:3, :3],
            )

            # Snapshot estimator state because we may continue to another mask and
            # later choose this trial again as the minimum-variance fallback.
            trial = {
                "rank": int(rank),
                "obs": obs,
                "cam": cam,
                "pose_selected": pose_to_numpy_4x4(pose_selected).astype(np.float32),
                "init_xyz_selected": np.asarray(init_xyz_selected, dtype=np.float32).copy(),
                "pose_selected_hyps_centered": pose_selected_hyps_centered.copy(),
                "refined_all": np.asarray(refined_all, dtype=np.float32).copy(),
                "scores": np.asarray(scores, dtype=np.float64).copy(),
                "best_score": float(best_score),
                "avg_var": float(avg_var),
                "mean_xyz": np.asarray(mean_xyz, dtype=np.float64).copy(),
                "var_xyz": np.asarray(var_xyz, dtype=np.float64).copy(),
                "R_cam_from_world": np.asarray(R_cam_from_world, dtype=np.float64).copy(),
                "best_id": int(getattr(est, "best_id", int(np.nanargmax(scores)))),
                "consensus_info": dict(getattr(est, "consensus_info", {})),
                "est_pose_last": est.pose_last.detach().clone() if torch.is_tensor(getattr(est, "pose_last", None)) else getattr(est, "pose_last", None),
                "est_poses": est.poses.detach().clone() if torch.is_tensor(getattr(est, "poses", None)) else getattr(est, "poses", None),
            }
            trials.append(trial)

            logging.info(
                f"[mask_var_gate] frame={obs['frame_id']} rank={rank} cam={cam}: "
                f"avg_var={avg_var:.6f} deg^2, "
                f"var_xyz=[{var_xyz[0]:.4f},{var_xyz[1]:.4f},{var_xyz[2]:.4f}], "
                f"best_score={best_score:.6f}"
            )

            # Accept the first area-ranked mask that gets below the variance threshold.
            if (not var_gate_enabled) or (np.isfinite(avg_var) and avg_var < var_threshold):
                logging.info(
                    f"[mask_var_gate] frame={obs['frame_id']} ACCEPT rank={rank} cam={cam}: "
                    f"avg_var={avg_var:.6f} < {var_threshold:.6f}"
                )
                break

            logging.info(
                f"[mask_var_gate] frame={obs['frame_id']} REJECT rank={rank} cam={cam}: "
                f"avg_var={avg_var:.6f} >= {var_threshold:.6f}; trying next mask if available"
            )

        except Exception as e:
            failed.append((rank, cam, str(e)))
            logging.warning(
                f"[mask_var_gate] frame={obs.get('frame_id', '')} rank={rank} cam={cam} failed: {e}; "
                f"trying next mask if available"
            )
            continue

    if len(trials) == 0:
        raise RuntimeError(
            "Temporal cone registration failed for all mask candidates: " +
            "; ".join([f"rank{r}:cam{c}:{msg}" for r, c, msg in failed])
        )

    passing = [tr for tr in trials if np.isfinite(tr["avg_var"]) and tr["avg_var"] < var_threshold]
    tracking_cam = str(getattr(args, "tracking_cam", "") or "").strip()
    if tracking_cam:
        chosen = trials[0]
        selection_reason = "locked_tracking_cam"
    elif var_gate_enabled and len(passing) > 0:
        # Because trials are performed in descending mask-area order and we stop on the first
        # passing one, this is exactly the brightest mask satisfying var < threshold.
        chosen = passing[0]
        selection_reason = "first_area_rank_below_var_threshold"
    elif var_gate_enabled:
        # None of top-N masks achieved var < threshold: user's requested fallback.
        chosen = min(trials, key=lambda tr: tr["avg_var"] if np.isfinite(tr["avg_var"]) else float("inf"))
        selection_reason = "all_top_masks_above_threshold_choose_min_var"
    else:
        chosen = trials[0]
        selection_reason = "variance_gate_disabled"

    # Restore estimator state corresponding to the actually chosen mask.
    est.pose_last = chosen["est_pose_last"]
    est.poses = chosen["est_poses"]
    est.scores = chosen["scores"].copy()
    est.best_id = int(chosen["best_id"])
    est.consensus_info = dict(chosen["consensus_info"])

    # Only the FINAL chosen mask owns the canonical per-frame distribution images.
    if not bool(getattr(args, "no_distribution_vis", False)):
        distribution_vis_dir = getattr(args, "distribution_vis_dir", None)
        if distribution_vis_dir is None or len(str(distribution_vis_dir).strip()) == 0:
            distribution_vis_dir = os.path.join(args.test_scene_dir, "distribution_vis")
        os.makedirs(distribution_vis_dir, exist_ok=True)
        distribution_path = os.path.join(distribution_vis_dir, f"{chosen['obs']['frame_id']}.png")
        distribution_path_3d = os.path.join(distribution_vis_dir, f"{chosen['obs']['frame_id']}_3d.png")
        save_rotation_distribution_visualization(
            initial_selected_hypotheses=chosen["pose_selected_hyps_centered"],
            refined_selected_poses=chosen["refined_all"],
            scores=chosen["scores"],
            best_idx=chosen["best_id"],
            consensus_info=chosen["consensus_info"],
            R_cam_from_world=chosen["R_cam_from_world"],
            R_reference_world=last_pose_world[:3, :3],
            frame_id=chosen["obs"]["frame_id"],
            save_path=distribution_path,
            cone_deg=float(args.temporal_pose_cone_deg),
        )
        save_rotation_distribution_visualization_3d(
            initial_selected_hypotheses=chosen["pose_selected_hyps_centered"],
            refined_selected_poses=chosen["refined_all"],
            scores=chosen["scores"],
            best_idx=chosen["best_id"],
            consensus_info=chosen["consensus_info"],
            R_cam_from_world=chosen["R_cam_from_world"],
            R_reference_world=last_pose_world[:3, :3],
            frame_id=chosen["obs"]["frame_id"],
            save_path=distribution_path_3d,
            cone_deg=float(args.temporal_pose_cone_deg),
        )

    trial_summary = ";".join(
        [f"rank{tr['rank']}:cam{tr['cam']}:var={tr['avg_var']:.6f}" for tr in trials]
    )
    if failed:
        fail_summary = ";".join([f"rank{r}:cam{c}:FAIL" for r, c, _ in failed])
        trial_summary = (trial_summary + ";" + fail_summary).strip(";")

    info = {
        "mode": "temporal_cone_mask_var_gate" if var_gate_enabled else "temporal_cone",
        "score": f"{chosen['best_score']:.8f}",
        "num_hypotheses": len(chosen["pose_selected_hyps_centered"]),
        "cone_deg": float(args.temporal_pose_cone_deg),
        "score_min": float(np.nanmin(chosen["scores"])),
        "score_max": float(np.nanmax(chosen["scores"])),
        "rotation_avg_var": float(chosen["avg_var"]),
        "rotation_var_x": float(chosen["var_xyz"][0]),
        "rotation_var_y": float(chosen["var_xyz"][1]),
        "rotation_var_z": float(chosen["var_xyz"][2]),
        "mask_area_rank": int(chosen["rank"]),
        "mask_var_threshold": float(var_threshold),
        "mask_var_selection_reason": selection_reason,
        "mask_var_trials": trial_summary,
        "candidates_tried": ";".join([tr["cam"] for tr in trials]),
    }

    logging.info(
        f"[mask_var_gate] frame={chosen['obs']['frame_id']} FINAL cam={chosen['cam']} "
        f"area_rank={chosen['rank']}, avg_var={chosen['avg_var']:.6f} deg^2, "
        f"reason={selection_reason}; trials={trial_summary}"
    )

    return (
        chosen["obs"],
        chosen["pose_selected"],
        chosen["init_xyz_selected"],
        info,
    )


def build_foundationpose_centered_bbox(mesh, tf_to_centered_mesh):
    """
    Build a visualization bbox whose ORIGIN is exactly the same centered-mesh
    origin used by FoundationPose.

    IMPORTANT:
        pose_original = pose_centered @ tf_to_centered_mesh

    Therefore:
        pose_centered = pose_original @ inv(tf_to_centered_mesh)

    The old code used trimesh.bounds.oriented_bounds(mesh), whose OBB center is
    generally different from FoundationPose's centered-mesh origin. That creates
    a fixed centimeter-scale offset between the forced triangulated xyz and the
    center of the rendered bbox.

    Here we transform the mesh into FoundationPose's centered frame, use its
    extents, and explicitly construct a bbox centered at [0, 0, 0]. Thus the bbox
    center and the pose_centered translation are exactly the same point.
    """
    tf_to_centered_mesh = np.asarray(tf_to_centered_mesh, dtype=np.float64).reshape(4, 4)
    vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)

    vertices_h = np.concatenate(
        [vertices, np.ones((len(vertices), 1), dtype=np.float64)], axis=1
    )
    vertices_centered = (tf_to_centered_mesh @ vertices_h.T).T[:, :3]

    vmin = vertices_centered.min(axis=0)
    vmax = vertices_centered.max(axis=0)
    extents = vmax - vmin
    actual_midpoint = 0.5 * (vmin + vmax)

    # Explicitly center the visualization box at the FoundationPose origin.
    bbox_centered = np.stack([-0.5 * extents, 0.5 * extents], axis=0).astype(np.float64)

    logging.info(
        "[bbox_center_fix] FoundationPose-centered mesh AABB: "
        f"vmin={vmin.tolist()}, vmax={vmax.tolist()}, "
        f"actual_midpoint={actual_midpoint.tolist()}, extents={extents.tolist()}"
    )
    logging.info(
        "[bbox_center_fix] visualization bbox center is explicitly [0,0,0] "
        "in the SAME centered-mesh frame used for xyz locking."
    )
    return bbox_centered


def draw_main_camera_visualization(
    reader, frame_idx, pose_main, tf_to_centered_mesh, bbox_centered, K_main
):
    """Draw bbox/XYZ axes using FoundationPose's exact centered-mesh frame."""
    color_main = reader.get_color(frame_idx)

    # pose_main is ORIGINAL-MESH -> main camera.
    # Convert it back to CENTERED-MESH -> main camera using the exact same
    # transform used during the xyz lock.
    pose_centered_main = (
        np.asarray(pose_main, dtype=np.float64).reshape(4, 4)
        @ np.linalg.inv(np.asarray(tf_to_centered_mesh, dtype=np.float64).reshape(4, 4))
    )

    vis = draw_posed_3d_box(
        K_main,
        img=color_main.copy(),
        ob_in_cam=pose_centered_main,
        bbox=bbox_centered,
    )
    vis = draw_xyz_axis(
        vis,
        ob_in_cam=pose_centered_main,
        scale=0.1,
        K=K_main,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )
    return vis




def _camera_sort_key(cam):
    """Sort camera ids numerically when possible, otherwise lexicographically."""
    cam = str(cam)
    try:
        return (0, int(cam))
    except ValueError:
        return (1, cam)


def put_label_rgb(img, lines, org=(10, 26)):
    """Draw readable labels on an RGB image."""
    out = img.copy()
    if isinstance(lines, str):
        lines = [lines]
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    line_h = 24
    for i, line in enumerate(lines):
        yy = y + i * line_h
        # Shadow/background for readability. OpenCV writes into RGB arrays fine;
        # the tuple is interpreted as channel values directly.
        cv2.putText(out, str(line), (x + 1, yy + 1), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(out, str(line), (x, yy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def resize_keep_aspect(img, target_width):
    """Resize RGB image by width while preserving aspect ratio."""
    if target_width is None or int(target_width) <= 0:
        return img
    H, W = img.shape[:2]
    target_width = int(target_width)
    if W == target_width:
        return img
    scale = target_width / float(W)
    new_h = max(1, int(round(H * scale)))
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)


def make_image_grid(images, cols=0, cell_width=360, pad=8, bg_value=0):
    """
    Fuse a list of RGB images into one grid image.

    Args:
        images: list of RGB uint8 images.
        cols: number of columns. <=0 means automatic sqrt layout.
        cell_width: resize each view to this width before grid. <=0 keeps original size.
        pad: padding between cells.
    """
    if len(images) == 0:
        return None

    resized = [resize_keep_aspect(img, cell_width) for img in images]
    if int(cols) <= 0:
        cols = int(np.ceil(np.sqrt(len(resized))))
    cols = max(1, int(cols))
    rows = int(np.ceil(len(resized) / float(cols)))

    cell_w = max(img.shape[1] for img in resized)
    cell_h = max(img.shape[0] for img in resized)
    pad = max(0, int(pad))

    grid_h = rows * cell_h + (rows + 1) * pad
    grid_w = cols * cell_w + (cols + 1) * pad
    grid = np.full((grid_h, grid_w, 3), int(bg_value), dtype=np.uint8)

    for idx, img in enumerate(resized):
        r = idx // cols
        c = idx % cols
        h, w = img.shape[:2]
        y0 = pad + r * (cell_h + pad) + (cell_h - h) // 2
        x0 = pad + c * (cell_w + pad) + (cell_w - w) // 2
        grid[y0:y0 + h, x0:x0 + w] = img

    return grid


def draw_all_camera_grid_visualization(
    observations, pose_world, cam_ext, tf_to_centered_mesh, bbox_centered,
    selected_cam=None, inlier_cams=None,
    grid_cols=0, grid_cell_width=360
):
    """
    Draw the registered object pose under every loaded camera view and fuse them
    into a single RGB grid image for this frame.

    pose_world convention:
        X_world = pose_world @ X_object

    cam_ext[cam]['ext'] convention:
        X_cam = T_cam_from_world @ X_world

    Therefore, for each camera:
        ob_in_cam = T_cam_from_world @ pose_world
    """
    if inlier_cams is None:
        inlier_cams = []
    inlier_cams = set(str(c) for c in inlier_cams)
    selected_cam = None if selected_cam is None else str(selected_cam)

    views = []
    for obs in sorted(observations, key=lambda o: _camera_sort_key(o["cam"])):
        cam = str(obs["cam"])
        if cam not in cam_ext:
            logging.warning(f"[grid_vis] cam={cam} missing in extri.yml, skip")
            continue

        color = obs["color"].copy()
        K = np.asarray(obs["K"], dtype=np.float64).reshape(3, 3)

        # pose_cam_original is ORIGINAL-MESH -> this camera.
        pose_cam_original = (
            np.asarray(cam_ext[cam]["ext"], dtype=np.float64).reshape(4, 4)
            @ np.asarray(pose_world, dtype=np.float64).reshape(4, 4)
        )
        # Use FoundationPose's exact CENTERED-MESH frame, not trimesh OBB center.
        pose_cam_centered = (
            pose_cam_original
            @ np.linalg.inv(np.asarray(tf_to_centered_mesh, dtype=np.float64).reshape(4, 4))
        )

        try:
            vis = draw_posed_3d_box(
                K, img=color.copy(), ob_in_cam=pose_cam_centered, bbox=bbox_centered
            )
            vis = draw_xyz_axis(
                vis,
                ob_in_cam=pose_cam_centered,
                scale=0.1,
                K=K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
        except Exception as e:
            logging.warning(f"[grid_vis] draw failed for cam={cam}: {e}")
            vis = color.copy()

        tags = []
        if cam == selected_cam:
            tags.append("REGISTER")
        if cam in inlier_cams:
            tags.append("INLIER")
        tag_text = (" | " + ",".join(tags)) if len(tags) > 0 else ""
        lines = [
            f"cam {cam}{tag_text}",
            f"mask brightness sum {float(obs.get('mask_total_brightness', 0.0)):.1f}",
            f"mask mean brightness {float(obs.get('mask_mean_brightness', 0.0)):.1f}",
            f"mask area {int(obs.get('mask_area', 0))}",
        ]
        vis = put_label_rgb(vis, lines)
        views.append(vis.astype(np.uint8))

    return make_image_grid(views, cols=grid_cols, cell_width=grid_cell_width, pad=8, bg_value=0)


def _draw_bbox_on_rgb_from_original_pose(
    color_rgb, K, pose_cam_original, tf_to_centered_mesh, bbox_centered,
    label_lines=None,
):
    """Draw the FoundationPose-centered 3D bbox/XYZ axes on one RGB view."""
    color_rgb = np.asarray(color_rgb).copy()
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    pose_cam_original = np.asarray(pose_cam_original, dtype=np.float64).reshape(4, 4)
    tf_to_centered_mesh = np.asarray(tf_to_centered_mesh, dtype=np.float64).reshape(4, 4)

    # ORIGINAL-MESH -> camera  ==>  CENTERED-MESH -> camera.
    pose_cam_centered = pose_cam_original @ np.linalg.inv(tf_to_centered_mesh)

    vis = draw_posed_3d_box(
        K,
        img=color_rgb,
        ob_in_cam=pose_cam_centered,
        bbox=bbox_centered,
    )
    vis = draw_xyz_axis(
        vis,
        ob_in_cam=pose_cam_centered,
        scale=0.1,
        K=K,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )
    if label_lines:
        vis = put_label_rgb(vis, label_lines)
    return vis.astype(np.uint8)


def _load_camera_rgb_for_frame(reader, cam, frame_id):
    """
    Load one RGB frame directly from images/<cam>/<frame_id>.*.

    Unlike the online grid visualization, this intentionally does NOT require a
    valid object mask. This lets the post-processed visualization contain every
    camera that exists in the capture, even if SAM/mask loading failed there.
    """
    cam = str(cam)
    frame_id = str(frame_id)
    color_file = find_existing_file(
        os.path.join(reader.rgb_root, cam), frame_id, IMAGE_EXTS
    )
    if color_file is None:
        return None
    return read_color_resized(color_file, (reader.H, reader.W))


def regenerate_postprocessed_bbox_videos(
    debug_dir,
    reader,
    cam_ext,
    K_by_cam,
    main_cam,
    tf_to_centered_mesh,
    bbox_centered,
    fps=10,
    grid_cols=0,
    grid_cell_width=360,
    save_frames=False,
    source_log_csv=None,
    source_world_dir=None,
    output_tag="rotfix",
    stage_label="POST-ROTATION-FIX",
):
    """
    Re-render FINAL 3D-bbox visualization after offline pose post-processing.

    Source of truth:
        debug_dir/ob_in_world_rotfix/<frame_id>.txt
        debug_dir/register_every_frame_log_rotfix.csv

    Outputs:
        debug_dir/register_every_frame_vis_rotfix.mp4  # main camera
        debug_dir/register_vis_grid_rotfix.mp4         # every camera

    Optional PNG frames:
        debug_dir/register_vis_frames_rotfix/<frame_id>.png
        debug_dir/register_vis_grid_rotfix/<frame_id>.png

    The all-camera grid loads RGB directly from images/<cam>/ and does not depend
    on masks, so "all cam" really means all available capture cameras.
    """
    fixed_log_csv = (
        str(source_log_csv)
        if source_log_csv is not None
        else os.path.join(debug_dir, "register_every_frame_log_rotfix.csv")
    )
    world_fixed_dir = (
        str(source_world_dir)
        if source_world_dir is not None
        else os.path.join(debug_dir, "ob_in_world_rotfix")
    )
    output_tag = str(output_tag).strip() or "rotfix"
    stage_label = str(stage_label).strip() or "POST-ROTATION-FIX"

    if not os.path.exists(fixed_log_csv):
        logging.warning(
            f"[postvis] fixed log does not exist, skip postprocessed video: {fixed_log_csv}"
        )
        return None
    if not os.path.isdir(world_fixed_dir):
        logging.warning(
            f"[postvis] fixed world-pose dir does not exist, skip postprocessed video: {world_fixed_dir}"
        )
        return None

    with open(fixed_log_csv, "r", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]

    records = []
    for row_idx, row in enumerate(rows):
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        frame_id = str(row.get("frame_id", "")).strip()
        if not frame_id:
            continue
        pose_path = os.path.join(world_fixed_dir, f"{frame_id}.txt")
        if not os.path.exists(pose_path):
            logging.warning(f"[postvis] missing postprocessed world pose: {pose_path}")
            continue
        try:
            frame_idx = int(row.get("frame_idx", row_idx))
        except Exception:
            frame_idx = row_idx
        records.append((frame_idx, frame_id, row, pose_path))

    records.sort(key=lambda x: (x[0], x[1]))
    if len(records) == 0:
        logging.warning("[postvis] no successful postprocessed poses found; skip video generation")
        return None

    main_cam = str(main_cam)
    all_cams = sorted(
        set(str(c) for c in getattr(reader, "all_cam_folders", [])),
        key=_camera_sort_key,
    )
    # If the reader somehow omitted a calibrated camera folder, still include
    # calibrated cameras that have an RGB folder.
    for cam in sorted(cam_ext.keys(), key=_camera_sort_key):
        if cam not in all_cams and os.path.isdir(os.path.join(reader.rgb_root, cam)):
            all_cams.append(cam)
    all_cams = sorted(set(all_cams), key=_camera_sort_key)

    if main_cam not in all_cams:
        all_cams.append(main_cam)
        all_cams = sorted(set(all_cams), key=_camera_sort_key)

    main_video_path = os.path.join(debug_dir, f"register_every_frame_vis_{output_tag}.mp4")
    grid_video_path = os.path.join(debug_dir, f"register_vis_grid_{output_tag}.mp4")
    main_frames_dir = os.path.join(debug_dir, f"register_vis_frames_{output_tag}")
    grid_frames_dir = os.path.join(debug_dir, f"register_vis_grid_{output_tag}")
    if bool(save_frames):
        os.makedirs(main_frames_dir, exist_ok=True)
        os.makedirs(grid_frames_dir, exist_ok=True)

    main_writer = None
    grid_writer = None
    main_written = 0
    grid_written = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = max(float(fps), 1e-6)

    # Fixed blank cell ensures a constant grid geometry even when one camera's
    # RGB file is missing for a particular frame.
    blank_template = np.zeros((reader.H, reader.W, 3), dtype=np.uint8)

    try:
        for seq_idx, (frame_idx, frame_id, row, pose_path) in enumerate(records):
            try:
                pose_world = np.loadtxt(pose_path).astype(np.float64).reshape(4, 4)
            except Exception as e:
                logging.warning(f"[postvis] failed to read pose frame={frame_id}: {e}")
                continue

            modified = str(row.get("rotfix_was_modified", "0")).strip() in {"1", "true", "True"}
            fix_tag = "SLERP-FIXED" if modified else "KEPT"

            # ----------------------- main camera -----------------------
            main_color = _load_camera_rgb_for_frame(reader, main_cam, frame_id)
            if main_color is None:
                main_vis = blank_template.copy()
                main_vis = put_label_rgb(
                    main_vis,
                    [f"cam {main_cam} | {stage_label} | {fix_tag}",
                     f"frame {frame_id}", "RGB MISSING"],
                )
            elif main_cam not in cam_ext:
                main_vis = main_color.copy()
                main_vis = put_label_rgb(
                    main_vis,
                    [f"cam {main_cam} | {stage_label} | {fix_tag}",
                     f"frame {frame_id}", "EXTRINSIC MISSING"],
                )
            else:
                try:
                    pose_main = (
                        np.asarray(cam_ext[main_cam]["ext"], dtype=np.float64).reshape(4, 4)
                        @ pose_world
                    )
                    K_main = np.asarray(K_by_cam.get(main_cam, reader.K), dtype=np.float64).reshape(3, 3)
                    main_vis = _draw_bbox_on_rgb_from_original_pose(
                        color_rgb=main_color,
                        K=K_main,
                        pose_cam_original=pose_main,
                        tf_to_centered_mesh=tf_to_centered_mesh,
                        bbox_centered=bbox_centered,
                        label_lines=[
                            f"cam {main_cam} | {stage_label} | {fix_tag}",
                            f"frame {frame_id}",
                        ],
                    )
                except Exception as e:
                    logging.warning(f"[postvis] main draw failed frame={frame_id}: {e}")
                    main_vis = put_label_rgb(
                        main_color.copy(),
                        [f"cam {main_cam} | {stage_label} | DRAW FAILED", f"frame {frame_id}"],
                    )

            if main_writer is None:
                Hm, Wm = main_vis.shape[:2]
                main_writer = cv2.VideoWriter(
                    main_video_path, fourcc, fps, (Wm, Hm)
                )
                if not main_writer.isOpened():
                    raise RuntimeError(f"Failed to open main postprocess video writer: {main_video_path}")
                logging.info(f"[postvis] writing FINAL main-camera bbox video: {main_video_path}")
            main_writer.write(main_vis[..., ::-1])
            main_written += 1
            if bool(save_frames):
                imageio.imwrite(os.path.join(main_frames_dir, f"{frame_id}.png"), main_vis)

            # ------------------------ all cameras ----------------------
            grid_views = []
            for cam in all_cams:
                color = _load_camera_rgb_for_frame(reader, cam, frame_id)
                if color is None:
                    vis_cam = blank_template.copy()
                    vis_cam = put_label_rgb(
                        vis_cam,
                        [f"cam {cam} | {stage_label} | {fix_tag}", "RGB MISSING"],
                    )
                    grid_views.append(vis_cam)
                    continue

                if cam not in cam_ext:
                    vis_cam = put_label_rgb(
                        color.copy(),
                        [f"cam {cam} | {stage_label} | {fix_tag}", "EXTRINSIC MISSING"],
                    )
                    grid_views.append(vis_cam)
                    continue

                try:
                    pose_cam = (
                        np.asarray(cam_ext[cam]["ext"], dtype=np.float64).reshape(4, 4)
                        @ pose_world
                    )
                    K = np.asarray(K_by_cam.get(cam, reader.K), dtype=np.float64).reshape(3, 3)
                    vis_cam = _draw_bbox_on_rgb_from_original_pose(
                        color_rgb=color,
                        K=K,
                        pose_cam_original=pose_cam,
                        tf_to_centered_mesh=tf_to_centered_mesh,
                        bbox_centered=bbox_centered,
                        label_lines=[f"cam {cam} | {stage_label} | {fix_tag}"],
                    )
                except Exception as e:
                    logging.warning(
                        f"[postvis] all-cam draw failed frame={frame_id} cam={cam}: {e}"
                    )
                    vis_cam = put_label_rgb(
                        color.copy(),
                        [f"cam {cam} | {stage_label} | DRAW FAILED"],
                    )
                grid_views.append(vis_cam.astype(np.uint8))

            grid_vis = make_image_grid(
                grid_views,
                cols=grid_cols,
                cell_width=grid_cell_width,
                pad=8,
                bg_value=0,
            )
            if grid_vis is not None:
                if grid_writer is None:
                    Hg, Wg = grid_vis.shape[:2]
                    grid_writer = cv2.VideoWriter(
                        grid_video_path, fourcc, fps, (Wg, Hg)
                    )
                    if not grid_writer.isOpened():
                        raise RuntimeError(f"Failed to open grid postprocess video writer: {grid_video_path}")
                    logging.info(f"[postvis] writing FINAL all-camera bbox video: {grid_video_path}")
                grid_writer.write(grid_vis[..., ::-1])
                grid_written += 1
                if bool(save_frames):
                    imageio.imwrite(os.path.join(grid_frames_dir, f"{frame_id}.png"), grid_vis)

            if (seq_idx + 1) % 25 == 0 or (seq_idx + 1) == len(records):
                logging.info(
                    f"[postvis] rendered {seq_idx + 1}/{len(records)} postprocessed frames"
                )
    finally:
        if main_writer is not None:
            main_writer.release()
        if grid_writer is not None:
            grid_writer.release()

    logging.info(
        f"[postvis] FINAL videos complete: main_frames={main_written}, grid_frames={grid_written}"
    )
    logging.info(f"[postvis] main-camera FINAL bbox video: {main_video_path}")
    logging.info(f"[postvis] all-camera FINAL bbox video: {grid_video_path}")
    if bool(save_frames):
        logging.info(f"[postvis] main FINAL frames: {main_frames_dir}")
        logging.info(f"[postvis] all-camera FINAL frames: {grid_frames_dir}")

    return {
        "main_video_path": main_video_path,
        "grid_video_path": grid_video_path,
        "main_frames_dir": main_frames_dir if bool(save_frames) else None,
        "grid_frames_dir": grid_frames_dir if bool(save_frames) else None,
        "num_records": len(records),
        "main_written": main_written,
        "grid_written": grid_written,
    }

def _str2bool(value):
    """Parse a flexible command-line boolean value."""
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "of"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value (true/false, 1/0, yes/no), got: {value}"
    )




# ---------------------------------------------------------------------------
# Offline post-FoundationPose rotation spike fix.
#
# Design goals for the current 10 FPS hand-object setting:
#   * NEVER affect online FoundationPose tracking/fallback decisions.
#   * Work in the CENTERED-object WORLD frame, because its XYZ is the physical
#     triangulated/Kalman object center. This avoids changing the center when the
#     original mesh origin is offset from the centered-mesh origin.
#   * Detect abrupt SO(3) angular-velocity edges using rolling median + MAD, with
#     a conservative floor/cap expressed in deg/s.
#   * Only repair a SHORT burst if a later spike edge closes the burst and the
#     two outside anchor rotations imply plausible average motion. A lone jump
#     that persists is preserved instead of being blindly smoothed away.
#   * Replace rotations inside accepted bursts with shortest-path SO(3) SLERP.
#     Translation is never smoothed/changed by this stage.
# ---------------------------------------------------------------------------

def _rotfix_geodesic_deg(R_a, R_b):
    return rotation_angle_deg(_project_rotation_so3(R_a), _project_rotation_so3(R_b))


def _rotfix_slerp_rotmat(R_a, R_b, alpha):
    """Shortest-path SO(3) interpolation using Rodrigues log/exp, no SciPy dependency."""
    R_a = _project_rotation_so3(R_a)
    R_b = _project_rotation_so3(R_b)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    R_rel = _project_rotation_so3(R_b @ R_a.T)
    rvec, _ = cv2.Rodrigues(R_rel)
    R_step, _ = cv2.Rodrigues(rvec.reshape(3, 1) * alpha)
    return _project_rotation_so3(R_step @ R_a)


def _rotfix_rolling_median_mad(values, window):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = len(values)
    med = np.zeros(n, dtype=np.float64)
    mad = np.zeros(n, dtype=np.float64)
    if n == 0:
        return med, mad
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    half = window // 2
    for i in range(n):
        w = values[max(0, i - half): min(n, i + half + 1)]
        m = float(np.median(w))
        med[i] = m
        mad[i] = float(np.median(np.abs(w - m)))
    return med, mad


def postprocess_rotation_spikes(
    debug_dir,
    log_csv,
    cam_ext,
    main_cam,
    tf_to_centered_mesh,
    fps=10.0,
    window=31,
    k_mad=8.0,
    min_threshold_deg_s=250.0,
    cap_threshold_deg_s=350.0,
    max_burst_frames=10,
    max_anchor_avg_deg_s=250.0,
):
    """
    Offline rotation repair after the complete FoundationPose pass.

    The raw outputs remain untouched:
        debug_dir/ob_in_world/*.txt
        debug_dir/ob_in_cam/*.txt
        debug_dir/ob_in_selected_cam/*.txt
        debug_dir/register_every_frame_log.csv

    Fixed outputs are written to:
        debug_dir/ob_in_world_rotfix/*.txt
        debug_dir/ob_in_cam_rotfix/*.txt
        debug_dir/ob_in_selected_cam_rotfix/*.txt
        debug_dir/register_every_frame_log_rotfix.csv

    Detection is performed on CENTERED-object WORLD rotations reconstructed from
    the saved original-mesh WORLD poses. For an edge i-1 -> i:

        omega_i = geodesic(R[i-1], R[i]) / dt    [deg/s]
        threshold_i = max(
            min(local_median + k_mad * local_MAD, cap_threshold_deg_s),
            min_threshold_deg_s,
        )

    A spike edge alone is NOT enough to modify data. We pair a spike-entry edge
    with a later spike-exit edge only when:
      1) the enclosed bad burst has <= max_burst_frames frames; and
      2) the outside anchor-to-anchor average angular speed is <=
         max_anchor_avg_deg_s.

    That makes the fix conservative: short wrong-rotation branches are repaired,
    while a persistent large real rotation without a nearby return edge is kept.
    """
    fps = max(float(fps), 1e-6)
    window = max(3, int(window))
    k_mad = max(float(k_mad), 0.0)
    min_threshold_deg_s = max(float(min_threshold_deg_s), 0.0)
    cap_threshold_deg_s = max(float(cap_threshold_deg_s), min_threshold_deg_s)
    max_burst_frames = max(int(max_burst_frames), 1)
    max_anchor_avg_deg_s = max(float(max_anchor_avg_deg_s), 0.0)

    fixed_log_csv = os.path.join(debug_dir, "register_every_frame_log_rotfix.csv")
    world_raw_dir = os.path.join(debug_dir, "ob_in_world")
    world_fixed_dir = os.path.join(debug_dir, "ob_in_world_rotfix")
    main_fixed_dir = os.path.join(debug_dir, "ob_in_cam_rotfix")
    selected_fixed_dir = os.path.join(debug_dir, "ob_in_selected_cam_rotfix")
    os.makedirs(world_fixed_dir, exist_ok=True)
    os.makedirs(main_fixed_dir, exist_ok=True)
    os.makedirs(selected_fixed_dir, exist_ok=True)

    with open(log_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        source_fields = list(reader.fieldnames or [])
        source_rows = [dict(r) for r in reader]

    # Keep only successfully saved pose rows for rotation analysis.
    valid_records = []
    for row_idx, row in enumerate(source_rows):
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        frame_id = str(row.get("frame_id", "")).strip()
        if frame_id == "":
            continue
        pose_path = os.path.join(world_raw_dir, f"{frame_id}.txt")
        if not os.path.exists(pose_path):
            logging.warning(f"[rotfix] missing raw world pose for frame={frame_id}: {pose_path}")
            continue
        try:
            pose_world = np.loadtxt(pose_path).astype(np.float64).reshape(4, 4)
            frame_idx = int(row.get("frame_idx", row_idx))
        except Exception as e:
            logging.warning(f"[rotfix] failed to read raw world pose frame={frame_id}: {e}")
            continue
        valid_records.append({
            "source_row_idx": row_idx,
            "frame_idx": frame_idx,
            "frame_id": frame_id,
            "selected_cam": str(row.get("selected_cam", "")).strip(),
            "registration_mode": str(row.get("registration_mode", "")).strip(),
            "pose_world_raw": pose_world,
        })

    n = len(valid_records)
    if n < 3:
        logging.warning(f"[rotfix] only {n} valid poses; copying raw poses without rotation repair")

    tf_to_centered_mesh = np.asarray(tf_to_centered_mesh, dtype=np.float64).reshape(4, 4)
    inv_tf_to_centered_mesh = np.linalg.inv(tf_to_centered_mesh)

    centered_raw = []
    for rec in valid_records:
        pose_centered = rec["pose_world_raw"] @ inv_tf_to_centered_mesh
        pose_centered[:3, :3] = _project_rotation_so3(pose_centered[:3, :3])
        centered_raw.append(pose_centered)
    centered_raw = np.asarray(centered_raw, dtype=np.float64) if n else np.zeros((0, 4, 4), dtype=np.float64)
    centered_fixed = centered_raw.copy()

    # Edge arrays use edge index i to mean transition record i-1 -> i.
    edge_angle_deg = np.full(n, np.nan, dtype=np.float64)
    edge_speed_deg_s = np.full(n, np.nan, dtype=np.float64)
    edge_gap_frames = np.ones(n, dtype=np.int64)
    for i in range(1, n):
        gap = max(valid_records[i]["frame_idx"] - valid_records[i - 1]["frame_idx"], 1)
        edge_gap_frames[i] = gap
        angle = _rotfix_geodesic_deg(centered_raw[i - 1, :3, :3], centered_raw[i, :3, :3])
        edge_angle_deg[i] = angle
        edge_speed_deg_s[i] = angle * fps / float(gap)

    if n >= 2:
        speeds = edge_speed_deg_s[1:].copy()
        local_med, local_mad = _rotfix_rolling_median_mad(speeds, window=window)
        adaptive = local_med + k_mad * np.maximum(local_mad, 1e-9)
        threshold = np.maximum(
            np.minimum(adaptive, cap_threshold_deg_s),
            min_threshold_deg_s,
        )
        edge_threshold_deg_s = np.full(n, np.nan, dtype=np.float64)
        edge_local_med_deg_s = np.full(n, np.nan, dtype=np.float64)
        edge_local_mad_deg_s = np.full(n, np.nan, dtype=np.float64)
        edge_threshold_deg_s[1:] = threshold
        edge_local_med_deg_s[1:] = local_med
        edge_local_mad_deg_s[1:] = local_mad
        spike_edges = [i for i in range(1, n) if np.isfinite(edge_speed_deg_s[i]) and edge_speed_deg_s[i] > edge_threshold_deg_s[i]]
    else:
        edge_threshold_deg_s = np.full(n, np.nan, dtype=np.float64)
        edge_local_med_deg_s = np.full(n, np.nan, dtype=np.float64)
        edge_local_mad_deg_s = np.full(n, np.nan, dtype=np.float64)
        spike_edges = []

    # Greedily pair entry/exit spike edges into short, physically plausible bursts.
    # A paired [s,e) burst replaces records s..e-1, anchored by s-1 and e.
    bursts = []
    p = 0
    burst_id = 0
    while p < len(spike_edges):
        s = int(spike_edges[p])
        chosen_q = None
        chosen_e = None
        chosen_avg = None
        for q in range(p + 1, len(spike_edges)):
            e = int(spike_edges[q])
            n_bad = e - s
            if n_bad <= 0:
                continue
            if n_bad > max_burst_frames:
                break
            anchor_before = s - 1
            anchor_after = e
            if anchor_before < 0 or anchor_after >= n:
                continue
            dt_frames = max(
                valid_records[anchor_after]["frame_idx"] - valid_records[anchor_before]["frame_idx"],
                1,
            )
            anchor_angle = _rotfix_geodesic_deg(
                centered_raw[anchor_before, :3, :3],
                centered_raw[anchor_after, :3, :3],
            )
            anchor_avg_deg_s = anchor_angle * fps / float(dt_frames)
            if anchor_avg_deg_s <= max_anchor_avg_deg_s:
                chosen_q = q
                chosen_e = e
                chosen_avg = anchor_avg_deg_s
                break

        if chosen_e is None:
            p += 1
            continue

        burst_id += 1
        bursts.append({
            "burst_id": burst_id,
            "s": s,
            "e": chosen_e,
            "anchor_before": s - 1,
            "anchor_after": chosen_e,
            "anchor_avg_deg_s": float(chosen_avg),
        })
        # The chosen exit spike is consumed as the boundary of this burst.
        p = int(chosen_q) + 1

    fixed_mask = np.zeros(n, dtype=bool)
    burst_for_index = {}
    alpha_for_index = {}
    for b in bursts:
        s = b["s"]
        e = b["e"]
        a0 = b["anchor_before"]
        a1 = b["anchor_after"]
        frame0 = valid_records[a0]["frame_idx"]
        frame1 = valid_records[a1]["frame_idx"]
        denom = max(frame1 - frame0, 1)
        R0 = centered_raw[a0, :3, :3]
        R1 = centered_raw[a1, :3, :3]
        for k in range(s, e):
            alpha = (valid_records[k]["frame_idx"] - frame0) / float(denom)
            centered_fixed[k, :3, :3] = _rotfix_slerp_rotmat(R0, R1, alpha)
            # IMPORTANT: centered XYZ stays exactly as produced by the existing XYZ Kalman filter.
            centered_fixed[k, :3, 3] = centered_raw[k, :3, 3]
            fixed_mask[k] = True
            burst_for_index[k] = b
            alpha_for_index[k] = float(alpha)

    # Fixed-edge diagnostics after interpolation.
    fixed_edge_angle_deg = np.full(n, np.nan, dtype=np.float64)
    fixed_edge_speed_deg_s = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        gap = max(valid_records[i]["frame_idx"] - valid_records[i - 1]["frame_idx"], 1)
        aa = _rotfix_geodesic_deg(centered_fixed[i - 1, :3, :3], centered_fixed[i, :3, :3])
        fixed_edge_angle_deg[i] = aa
        fixed_edge_speed_deg_s[i] = aa * fps / float(gap)

    # Map sequence index / source row index for writing the second full log.
    seq_by_source_row = {rec["source_row_idx"]: i for i, rec in enumerate(valid_records)}
    spike_edge_set = set(int(i) for i in spike_edges)
    paired_boundary_spikes = set()
    for b in bursts:
        paired_boundary_spikes.add(int(b["s"]))
        paired_boundary_spikes.add(int(b["e"]))

    extra_fields = [
        "rotfix_status",
        "rotfix_reason",
        "rotfix_fps",
        "rotfix_was_modified",
        "rotfix_burst_id",
        "rotfix_burst_start_frame",
        "rotfix_burst_end_frame",
        "rotfix_anchor_before_frame",
        "rotfix_anchor_after_frame",
        "rotfix_interp_alpha",
        "rotfix_raw_prev_angle_deg",
        "rotfix_raw_prev_speed_deg_s",
        "rotfix_local_median_deg_s",
        "rotfix_local_mad_deg_s",
        "rotfix_threshold_deg_s",
        "rotfix_raw_spike_edge",
        "rotfix_fixed_prev_angle_deg",
        "rotfix_fixed_prev_speed_deg_s",
        "rotfix_raw_to_fixed_deg",
        "pose_world_fixed_tx",
        "pose_world_fixed_ty",
        "pose_world_fixed_tz",
        "pose_world_fixed_qx",
        "pose_world_fixed_qy",
        "pose_world_fixed_qz",
        "pose_world_fixed_qw",
        "pose_world_fixed_rx_deg",
        "pose_world_fixed_ry_deg",
        "pose_world_fixed_rz_deg",
    ]

    fixed_rows = []
    for source_row_idx, row in enumerate(source_rows):
        out = dict(row)
        for key in extra_fields:
            out[key] = ""
        out["rotfix_fps"] = f"{fps:.6f}"

        if source_row_idx not in seq_by_source_row:
            if str(row.get("status", "")).strip().lower() != "ok":
                out["rotfix_status"] = "source_fail"
                out["rotfix_reason"] = "FoundationPose source row was not status=ok"
            else:
                out["rotfix_status"] = "source_pose_missing"
                out["rotfix_reason"] = "Raw ob_in_world pose was unavailable/unreadable"
            fixed_rows.append(out)
            continue

        i = seq_by_source_row[source_row_idx]
        rec = valid_records[i]
        pose_centered_fix = centered_fixed[i].copy()
        pose_world_fix = (pose_centered_fix @ tf_to_centered_mesh).astype(np.float64)
        pose_world_fix[:3, :3] = _project_rotation_so3(pose_world_fix[:3, :3])

        # Save fixed WORLD pose.
        np.savetxt(os.path.join(world_fixed_dir, f"{rec['frame_id']}.txt"), pose_world_fix)

        # Recompute fixed MAIN-camera and selected-camera poses from WORLD.
        if str(main_cam) in cam_ext:
            pose_main_fix = np.asarray(cam_ext[str(main_cam)]["ext"], dtype=np.float64).reshape(4, 4) @ pose_world_fix
            np.savetxt(os.path.join(main_fixed_dir, f"{rec['frame_id']}.txt"), pose_main_fix)
        selected_cam = str(rec["selected_cam"])
        if selected_cam in cam_ext:
            pose_selected_fix = np.asarray(cam_ext[selected_cam]["ext"], dtype=np.float64).reshape(4, 4) @ pose_world_fix
            np.savetxt(os.path.join(selected_fixed_dir, f"{rec['frame_id']}.txt"), pose_selected_fix)

        qfix = rotation_matrix_to_quaternion_xyzw(pose_world_fix[:3, :3])
        efix = rotation_matrix_to_euler_xyz_deg(pose_world_fix[:3, :3])
        raw_to_fixed = _rotfix_geodesic_deg(centered_raw[i, :3, :3], centered_fixed[i, :3, :3])

        out["rotfix_was_modified"] = int(bool(fixed_mask[i]))
        out["rotfix_raw_prev_angle_deg"] = "" if i == 0 else f"{edge_angle_deg[i]:.8f}"
        out["rotfix_raw_prev_speed_deg_s"] = "" if i == 0 else f"{edge_speed_deg_s[i]:.8f}"
        out["rotfix_local_median_deg_s"] = "" if i == 0 else f"{edge_local_med_deg_s[i]:.8f}"
        out["rotfix_local_mad_deg_s"] = "" if i == 0 else f"{edge_local_mad_deg_s[i]:.8f}"
        out["rotfix_threshold_deg_s"] = "" if i == 0 else f"{edge_threshold_deg_s[i]:.8f}"
        out["rotfix_raw_spike_edge"] = int(i in spike_edge_set)
        out["rotfix_fixed_prev_angle_deg"] = "" if i == 0 else f"{fixed_edge_angle_deg[i]:.8f}"
        out["rotfix_fixed_prev_speed_deg_s"] = "" if i == 0 else f"{fixed_edge_speed_deg_s[i]:.8f}"
        out["rotfix_raw_to_fixed_deg"] = f"{raw_to_fixed:.8f}"

        if fixed_mask[i]:
            b = burst_for_index[i]
            out["rotfix_status"] = "fixed_slerp"
            out["rotfix_reason"] = "short paired spike burst; SO(3) SLERP between outside anchors"
            out["rotfix_burst_id"] = b["burst_id"]
            out["rotfix_burst_start_frame"] = valid_records[b["s"]]["frame_id"]
            out["rotfix_burst_end_frame"] = valid_records[b["e"] - 1]["frame_id"]
            out["rotfix_anchor_before_frame"] = valid_records[b["anchor_before"]]["frame_id"]
            out["rotfix_anchor_after_frame"] = valid_records[b["anchor_after"]]["frame_id"]
            out["rotfix_interp_alpha"] = f"{alpha_for_index[i]:.8f}"
        else:
            out["rotfix_status"] = "kept_raw"
            if i in spike_edge_set and i not in paired_boundary_spikes:
                out["rotfix_reason"] = "spike edge detected but no short plausible closing edge; preserved as possible real/persistent motion"
            elif i in paired_boundary_spikes:
                out["rotfix_reason"] = "paired spike boundary/anchor; preserved"
            else:
                out["rotfix_reason"] = "no accepted short spike burst"

        out["pose_world_fixed_tx"] = f"{pose_world_fix[0, 3]:.8f}"
        out["pose_world_fixed_ty"] = f"{pose_world_fix[1, 3]:.8f}"
        out["pose_world_fixed_tz"] = f"{pose_world_fix[2, 3]:.8f}"
        out["pose_world_fixed_qx"] = f"{qfix[0]:.10f}"
        out["pose_world_fixed_qy"] = f"{qfix[1]:.10f}"
        out["pose_world_fixed_qz"] = f"{qfix[2]:.10f}"
        out["pose_world_fixed_qw"] = f"{qfix[3]:.10f}"
        out["pose_world_fixed_rx_deg"] = f"{efix[0]:.8f}"
        out["pose_world_fixed_ry_deg"] = f"{efix[1]:.8f}"
        out["pose_world_fixed_rz_deg"] = f"{efix[2]:.8f}"
        fixed_rows.append(out)

    with open(fixed_log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=source_fields + extra_fields)
        writer.writeheader()
        writer.writerows(fixed_rows)

    # Console summary: the two CSV logs are the requested persistent logs.
    max_raw_speed = float(np.nanmax(edge_speed_deg_s[1:])) if n > 1 else 0.0
    max_fixed_speed = float(np.nanmax(fixed_edge_speed_deg_s[1:])) if n > 1 else 0.0
    logging.info(
        f"[rotfix] complete: poses={n}, spike_edges={len(spike_edges)}, "
        f"accepted_bursts={len(bursts)}, fixed_frames={int(fixed_mask.sum())}, "
        f"max_raw_speed={max_raw_speed:.3f}deg/s, max_fixed_speed={max_fixed_speed:.3f}deg/s"
    )
    for b in bursts:
        logging.info(
            f"[rotfix] burst#{b['burst_id']}: "
            f"frames={valid_records[b['s']]['frame_id']}..{valid_records[b['e']-1]['frame_id']}, "
            f"anchors={valid_records[b['anchor_before']]['frame_id']}->{valid_records[b['anchor_after']]['frame_id']}, "
            f"anchor_avg={b['anchor_avg_deg_s']:.3f}deg/s"
        )
    unpaired = [i for i in spike_edges if i not in paired_boundary_spikes]
    if unpaired:
        logging.info(
            "[rotfix] unpaired spike-entry frames preserved: "
            + ",".join(valid_records[i]["frame_id"] for i in unpaired)
        )
    logging.info(f"[rotfix] raw log preserved: {log_csv}")
    logging.info(f"[rotfix] fixed log saved: {fixed_log_csv}")
    logging.info(f"[rotfix] fixed world poses: {world_fixed_dir}")
    logging.info(f"[rotfix] fixed main-camera poses: {main_fixed_dir}")
    logging.info(f"[rotfix] fixed selected-camera poses: {selected_fixed_dir}")

    return {
        "fixed_log_csv": fixed_log_csv,
        "num_valid_poses": n,
        "num_spike_edges": len(spike_edges),
        "num_bursts": len(bursts),
        "num_fixed_frames": int(fixed_mask.sum()),
        "bursts": bursts,
    }


# ---------------------------------------------------------------------------
# Optional offline POSTKAL rotation smoothing.
#
# This stage intentionally runs AFTER postprocess_rotation_spikes(). It is
# causal/forward-only and therefore matches the requested "from front to back"
# behavior. The state is a WORLD-frame orientation plus angular velocity:
#
#     nominal state: R, omega
#     local error:   [delta_theta, delta_omega] in R^6
#
# Prediction uses R_pred = Exp(omega * dt) @ R. Correction uses the left-
# invariant SO(3) residual Log(R_meas @ R_pred^T), then applies the Kalman
# correction multiplicatively to R. Translation is NEVER filtered here.
# ---------------------------------------------------------------------------

def _postkal_so3_exp(rotvec_rad):
    rotvec_rad = np.asarray(rotvec_rad, dtype=np.float64).reshape(3)
    R, _ = cv2.Rodrigues(rotvec_rad.reshape(3, 1))
    return _project_rotation_so3(R)


def _postkal_so3_log(R):
    R = _project_rotation_so3(R)
    rvec, _ = cv2.Rodrigues(R)
    return np.asarray(rvec, dtype=np.float64).reshape(3)


class RotationKalmanSO3CV:
    """
    Causal multiplicative SO(3) constant-angular-velocity Kalman filter.

    The 6D local error state is [delta_theta, delta_omega]. Orientation itself
    remains a proper 3x3 rotation matrix at every step, avoiding Euler-angle
    singularities and absolute Rodrigues branch jumps.
    """

    def __init__(
        self,
        measurement_std_deg=2.0,
        angular_accel_std_deg_s2=360.0,
        initial_omega_std_deg_s=180.0,
    ):
        self.measurement_std_rad = np.deg2rad(max(float(measurement_std_deg), 1e-6))
        self.angular_accel_std_rad_s2 = np.deg2rad(max(float(angular_accel_std_deg_s2), 1e-6))
        self.initial_omega_std_rad_s = np.deg2rad(max(float(initial_omega_std_deg_s), 1e-6))

        self.R = np.eye(3, dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64)
        self.initialized = False

        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[:, :3] = np.eye(3, dtype=np.float64)
        self.R_meas = (self.measurement_std_rad ** 2) * np.eye(3, dtype=np.float64)

    def initialize(self, R_measurement):
        self.R = _project_rotation_so3(R_measurement)
        self.omega[:] = 0.0
        self.P[:] = 0.0
        self.P[:3, :3] = (self.measurement_std_rad ** 2) * np.eye(3, dtype=np.float64)
        self.P[3:, 3:] = (self.initial_omega_std_rad_s ** 2) * np.eye(3, dtype=np.float64)
        self.initialized = True
        return self.R.copy(), self.omega.copy()

    def predict(self, dt):
        if not self.initialized:
            raise RuntimeError("RotationKalmanSO3CV must be initialized before predict().")

        dt = max(float(dt), 1e-6)
        self.R = _postkal_so3_exp(self.omega * dt) @ self.R
        self.R = _project_rotation_so3(self.R)

        F = np.eye(6, dtype=np.float64)
        F[:3, 3:] = dt * np.eye(3, dtype=np.float64)

        q = self.angular_accel_std_rad_s2 ** 2
        I3 = np.eye(3, dtype=np.float64)
        Q = q * np.block([
            [0.25 * (dt ** 4) * I3, 0.5 * (dt ** 3) * I3],
            [0.5 * (dt ** 3) * I3,        (dt ** 2) * I3],
        ])

        self.P = F @ self.P @ F.T + Q
        return self.R.copy(), self.omega.copy()

    def correct(self, R_measurement):
        if not self.initialized:
            return self.initialize(R_measurement)

        R_measurement = _project_rotation_so3(R_measurement)

        # Left-invariant orientation innovation in the tangent space at R_pred.
        innovation = _postkal_so3_log(R_measurement @ self.R.T)
        S = self.H @ self.P @ self.H.T + self.R_meas
        K = self.P @ self.H.T @ np.linalg.inv(S)
        delta = K @ innovation

        self.R = _postkal_so3_exp(delta[:3]) @ self.R
        self.R = _project_rotation_so3(self.R)
        self.omega = self.omega + delta[3:]

        I6 = np.eye(6, dtype=np.float64)
        A = I6 - K @ self.H
        self.P = A @ self.P @ A.T + K @ self.R_meas @ K.T

        return self.R.copy(), self.omega.copy()


def postprocess_rotation_kalman(
    debug_dir,
    cam_ext,
    main_cam,
    tf_to_centered_mesh,
    fps=10.0,
    measurement_std_deg=2.0,
    angular_accel_std_deg_s2=360.0,
    initial_omega_std_deg_s=180.0,
):
    """
    Run a forward-only SO(3) rotation Kalman filter AFTER SLERP rotfix.

    Input:
        ob_in_world_rotfix/*.txt
        register_every_frame_log_rotfix.csv

    Output:
        ob_in_world_rotfix_postkal/*.txt
        ob_in_cam_rotfix_postkal/*.txt
        ob_in_selected_cam_rotfix_postkal/*.txt
        register_every_frame_log_rotfix_postkal.csv

    CENTERED-object WORLD translation is copied exactly from the rotfix input.
    Only CENTERED-object WORLD rotation is filtered.
    """
    fps = max(float(fps), 1e-6)
    measurement_std_deg = max(float(measurement_std_deg), 1e-6)
    angular_accel_std_deg_s2 = max(float(angular_accel_std_deg_s2), 1e-6)
    initial_omega_std_deg_s = max(float(initial_omega_std_deg_s), 1e-6)

    input_log_csv = os.path.join(debug_dir, "register_every_frame_log_rotfix.csv")
    input_world_dir = os.path.join(debug_dir, "ob_in_world_rotfix")

    output_log_csv = os.path.join(debug_dir, "register_every_frame_log_rotfix_postkal.csv")
    output_world_dir = os.path.join(debug_dir, "ob_in_world_rotfix_postkal")
    output_main_dir = os.path.join(debug_dir, "ob_in_cam_rotfix_postkal")
    output_selected_dir = os.path.join(debug_dir, "ob_in_selected_cam_rotfix_postkal")

    if not os.path.exists(input_log_csv):
        raise FileNotFoundError(f"[postkal] rotfix log not found: {input_log_csv}")
    if not os.path.isdir(input_world_dir):
        raise FileNotFoundError(f"[postkal] rotfix WORLD pose dir not found: {input_world_dir}")

    os.makedirs(output_world_dir, exist_ok=True)
    os.makedirs(output_main_dir, exist_ok=True)
    os.makedirs(output_selected_dir, exist_ok=True)

    with open(input_log_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        source_fields = list(reader.fieldnames or [])
        source_rows = [dict(r) for r in reader]

    tf_to_centered_mesh = np.asarray(tf_to_centered_mesh, dtype=np.float64).reshape(4, 4)
    inv_tf_to_centered_mesh = np.linalg.inv(tf_to_centered_mesh)

    valid_records = []
    for row_idx, row in enumerate(source_rows):
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        frame_id = str(row.get("frame_id", "")).strip()
        if not frame_id:
            continue
        pose_path = os.path.join(input_world_dir, f"{frame_id}.txt")
        if not os.path.exists(pose_path):
            logging.warning(f"[postkal] missing rotfix WORLD pose frame={frame_id}: {pose_path}")
            continue
        try:
            pose_world = np.loadtxt(pose_path).astype(np.float64).reshape(4, 4)
            frame_idx = int(row.get("frame_idx", row_idx))
        except Exception as e:
            logging.warning(f"[postkal] failed to read rotfix pose frame={frame_id}: {e}")
            continue

        pose_centered = pose_world @ inv_tf_to_centered_mesh
        pose_centered[:3, :3] = _project_rotation_so3(pose_centered[:3, :3])
        valid_records.append({
            "source_row_idx": row_idx,
            "frame_idx": frame_idx,
            "frame_id": frame_id,
            "selected_cam": str(row.get("selected_cam", "")).strip(),
            "pose_centered_measurement": pose_centered,
        })

    valid_records.sort(key=lambda r: (int(r["frame_idx"]), str(r["frame_id"])))
    n = len(valid_records)
    if n == 0:
        raise RuntimeError("[postkal] no valid rotfix poses to filter")

    kf = RotationKalmanSO3CV(
        measurement_std_deg=measurement_std_deg,
        angular_accel_std_deg_s2=angular_accel_std_deg_s2,
        initial_omega_std_deg_s=initial_omega_std_deg_s,
    )

    filtered_by_source_row = {}
    last_frame_idx = None
    last_R_meas = None
    last_R_filt = None

    for seq_idx, rec in enumerate(valid_records):
        pose_meas = rec["pose_centered_measurement"]
        R_meas = _project_rotation_so3(pose_meas[:3, :3])
        frame_idx = int(rec["frame_idx"])

        if not kf.initialized:
            R_filt, omega = kf.initialize(R_meas)
            dt = 1.0 / fps
        else:
            gap = max(frame_idx - int(last_frame_idx), 1)
            dt = float(gap) / fps
            kf.predict(dt)
            R_filt, omega = kf.correct(R_meas)

        pose_centered_filt = pose_meas.copy()
        pose_centered_filt[:3, :3] = R_filt
        # Critical: preserve centered-object XYZ exactly from the SLERP rotfix input.
        pose_centered_filt[:3, 3] = pose_meas[:3, 3]

        pose_world_filt = pose_centered_filt @ tf_to_centered_mesh
        pose_world_filt[:3, :3] = _project_rotation_so3(pose_world_filt[:3, :3])

        frame_id = rec["frame_id"]
        np.savetxt(os.path.join(output_world_dir, f"{frame_id}.txt"), pose_world_filt)

        if str(main_cam) in cam_ext:
            pose_main = np.asarray(cam_ext[str(main_cam)]["ext"], dtype=np.float64).reshape(4, 4) @ pose_world_filt
            np.savetxt(os.path.join(output_main_dir, f"{frame_id}.txt"), pose_main)

        selected_cam = str(rec["selected_cam"])
        if selected_cam in cam_ext:
            pose_selected = np.asarray(cam_ext[selected_cam]["ext"], dtype=np.float64).reshape(4, 4) @ pose_world_filt
            np.savetxt(os.path.join(output_selected_dir, f"{frame_id}.txt"), pose_selected)

        meas_prev_deg = np.nan if last_R_meas is None else _rotfix_geodesic_deg(last_R_meas, R_meas)
        filt_prev_deg = np.nan if last_R_filt is None else _rotfix_geodesic_deg(last_R_filt, R_filt)
        meas_to_filt_deg = _rotfix_geodesic_deg(R_meas, R_filt)

        gap_for_speed = 1 if last_frame_idx is None else max(frame_idx - int(last_frame_idx), 1)
        meas_prev_speed = np.nan if not np.isfinite(meas_prev_deg) else meas_prev_deg * fps / float(gap_for_speed)
        filt_prev_speed = np.nan if not np.isfinite(filt_prev_deg) else filt_prev_deg * fps / float(gap_for_speed)

        q = rotation_matrix_to_quaternion_xyzw(pose_world_filt[:3, :3])
        euler = rotation_matrix_to_euler_xyz_deg(pose_world_filt[:3, :3])
        omega_deg_s = np.rad2deg(np.asarray(omega, dtype=np.float64).reshape(3))

        filtered_by_source_row[rec["source_row_idx"]] = {
            "pose_world": pose_world_filt,
            "dt": float(dt),
            "meas_prev_angle_deg": meas_prev_deg,
            "meas_prev_speed_deg_s": meas_prev_speed,
            "filt_prev_angle_deg": filt_prev_deg,
            "filt_prev_speed_deg_s": filt_prev_speed,
            "meas_to_filt_deg": meas_to_filt_deg,
            "omega_deg_s": omega_deg_s,
            "q": q,
            "euler": euler,
        }

        last_frame_idx = frame_idx
        last_R_meas = R_meas.copy()
        last_R_filt = R_filt.copy()

    extra_fields = [
        "postkal_status",
        "postkal_fps",
        "postkal_dt",
        "postkal_rot_meas_std_deg",
        "postkal_rot_accel_std_deg_s2",
        "postkal_initial_omega_std_deg_s",
        "postkal_meas_prev_angle_deg",
        "postkal_meas_prev_speed_deg_s",
        "postkal_filtered_prev_angle_deg",
        "postkal_filtered_prev_speed_deg_s",
        "postkal_meas_to_filtered_deg",
        "postkal_omega_x_deg_s",
        "postkal_omega_y_deg_s",
        "postkal_omega_z_deg_s",
        "pose_world_postkal_tx",
        "pose_world_postkal_ty",
        "pose_world_postkal_tz",
        "pose_world_postkal_qx",
        "pose_world_postkal_qy",
        "pose_world_postkal_qz",
        "pose_world_postkal_qw",
        "pose_world_postkal_rx_deg",
        "pose_world_postkal_ry_deg",
        "pose_world_postkal_rz_deg",
    ]

    out_rows = []
    for row_idx, row in enumerate(source_rows):
        out = dict(row)
        for key in extra_fields:
            out[key] = ""

        out["postkal_fps"] = f"{fps:.6f}"
        out["postkal_rot_meas_std_deg"] = f"{measurement_std_deg:.8f}"
        out["postkal_rot_accel_std_deg_s2"] = f"{angular_accel_std_deg_s2:.8f}"
        out["postkal_initial_omega_std_deg_s"] = f"{initial_omega_std_deg_s:.8f}"

        info = filtered_by_source_row.get(row_idx)
        if info is None:
            out["postkal_status"] = "not_filtered"
            out_rows.append(out)
            continue

        pose = info["pose_world"]
        q = info["q"]
        euler = info["euler"]
        omg = info["omega_deg_s"]

        out["postkal_status"] = "filtered"
        out["postkal_dt"] = f"{info['dt']:.8f}"
        out["postkal_meas_prev_angle_deg"] = "" if not np.isfinite(info["meas_prev_angle_deg"]) else f"{info['meas_prev_angle_deg']:.8f}"
        out["postkal_meas_prev_speed_deg_s"] = "" if not np.isfinite(info["meas_prev_speed_deg_s"]) else f"{info['meas_prev_speed_deg_s']:.8f}"
        out["postkal_filtered_prev_angle_deg"] = "" if not np.isfinite(info["filt_prev_angle_deg"]) else f"{info['filt_prev_angle_deg']:.8f}"
        out["postkal_filtered_prev_speed_deg_s"] = "" if not np.isfinite(info["filt_prev_speed_deg_s"]) else f"{info['filt_prev_speed_deg_s']:.8f}"
        out["postkal_meas_to_filtered_deg"] = f"{info['meas_to_filt_deg']:.8f}"
        out["postkal_omega_x_deg_s"] = f"{omg[0]:.8f}"
        out["postkal_omega_y_deg_s"] = f"{omg[1]:.8f}"
        out["postkal_omega_z_deg_s"] = f"{omg[2]:.8f}"
        out["pose_world_postkal_tx"] = f"{pose[0, 3]:.8f}"
        out["pose_world_postkal_ty"] = f"{pose[1, 3]:.8f}"
        out["pose_world_postkal_tz"] = f"{pose[2, 3]:.8f}"
        out["pose_world_postkal_qx"] = f"{q[0]:.10f}"
        out["pose_world_postkal_qy"] = f"{q[1]:.10f}"
        out["pose_world_postkal_qz"] = f"{q[2]:.10f}"
        out["pose_world_postkal_qw"] = f"{q[3]:.10f}"
        out["pose_world_postkal_rx_deg"] = f"{euler[0]:.8f}"
        out["pose_world_postkal_ry_deg"] = f"{euler[1]:.8f}"
        out["pose_world_postkal_rz_deg"] = f"{euler[2]:.8f}"
        out_rows.append(out)

    with open(output_log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=source_fields + extra_fields)
        writer.writeheader()
        writer.writerows(out_rows)

    changes = [v["meas_to_filt_deg"] for v in filtered_by_source_row.values()]
    max_change = float(np.max(changes)) if changes else 0.0
    mean_change = float(np.mean(changes)) if changes else 0.0

    logging.info(
        f"[postkal] complete: poses={n}, fps={fps:.3f}, "
        f"meas_std={measurement_std_deg:.3f}deg, "
        f"ang_accel_std={angular_accel_std_deg_s2:.3f}deg/s^2, "
        f"mean_meas_to_filtered={mean_change:.4f}deg, max={max_change:.4f}deg"
    )
    logging.info(f"[postkal] input SLERP-only poses preserved: {input_world_dir}")
    logging.info(f"[postkal] filtered WORLD poses: {output_world_dir}")
    logging.info(f"[postkal] filtered main-camera poses: {output_main_dir}")
    logging.info(f"[postkal] filtered selected-camera poses: {output_selected_dir}")
    logging.info(f"[postkal] filtered log: {output_log_csv}")

    return {
        "log_csv": output_log_csv,
        "world_dir": output_world_dir,
        "main_dir": output_main_dir,
        "selected_dir": output_selected_dir,
        "num_valid_poses": n,
        "mean_meas_to_filtered_deg": mean_change,
        "max_meas_to_filtered_deg": max_change,
    }



def _xyz_runs(mask):
    """Return [(start,end_exclusive), ...] for True runs in a 1-D bool mask."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    out = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i + 1
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _xyz_post_process_bad_mask(bad, gap_merge=1, max_burst_frames=10):
    """process_dataset.py-style short-gap merge and long-burst preservation."""
    out = np.asarray(bad, dtype=bool).copy()
    n = len(out)
    if n == 0 or not out.any():
        return out

    # Merge very short good gaps sandwiched by bad regions.
    i = 0
    while i < n:
        if not out[i]:
            j = i + 1
            while j < n and not out[j]:
                j += 1
            if (j - i) <= int(gap_merge):
                out[i:j] = True
            i = j
        else:
            i += 1

    # Match process_dataset.py: long bursts are presumed real motion and are left alone.
    if int(max_burst_frames) > 0:
        for s, e in _xyz_runs(out):
            if (e - s) > int(max_burst_frames):
                out[s:e] = False
    return out


def _interp_xyz_at_frame_indices(frame_indices, xyz, bad):
    """Linear XYZ interpolation at actual frame indices; boundary bad frames clamp."""
    frame_indices = np.asarray(frame_indices, dtype=np.float64).reshape(-1)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    bad = np.asarray(bad, dtype=bool).reshape(-1)
    if len(xyz) != len(frame_indices) or len(bad) != len(xyz):
        raise ValueError("frame_indices/xyz/bad length mismatch")
    if not bad.any():
        return xyz.copy()
    good = np.where(~bad)[0]
    if len(good) == 0:
        return xyz.copy()
    out = xyz.copy()
    for d in range(3):
        out[bad, d] = np.interp(
            frame_indices[bad],
            frame_indices[good],
            xyz[good, d],
        )
    return out


def _rolling_median_mad_1d(values, window):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = len(values)
    med = np.zeros(n, dtype=np.float64)
    mad = np.zeros(n, dtype=np.float64)
    half = max(int(window) // 2, 0)
    for i in range(n):
        w = values[max(0, i - half): min(n, i + half + 1)]
        m = float(np.median(w)) if len(w) else 0.0
        med[i] = m
        mad[i] = float(np.median(np.abs(w - m))) if len(w) else 0.0
    return med, mad


def _detect_xyz_velocity_bad_frames(
    frame_indices,
    xyz,
    fps=10.0,
    speed_cap_m_s=0.80,
    k_mad=8.0,
    window=31,
    gap_merge=1,
    max_burst_frames=10,
):
    """
    Detect short XYZ spikes using the same logic as process_dataset.py, but in m/s.

    For every edge i->i+1:
        speed = ||xyz[i+1]-xyz[i]|| / dt
        threshold = min(local_median + k_mad*local_MAD, speed_cap_m_s)

    A bad edge marks both endpoint frames. Detection is iterated twice, temporarily
    interpolating already-bad frames before recomputing velocities, then the shared
    short-gap / max-burst postprocess is applied.
    """
    frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n = len(xyz)
    if n < 3:
        return np.zeros(n, dtype=bool), {
            "raw_speed_m_s": np.zeros(max(n - 1, 0)),
            "threshold_m_s": np.zeros(max(n - 1, 0)),
        }

    bad = np.zeros(n, dtype=bool)
    last_speed = np.zeros(n - 1, dtype=np.float64)
    last_thresh = np.full(n - 1, float(speed_cap_m_s), dtype=np.float64)

    for _ in range(2):
        work = _interp_xyz_at_frame_indices(frame_indices, xyz, bad) if bad.any() else xyz
        gaps = np.maximum(np.diff(frame_indices).astype(np.float64), 1.0)
        dt = gaps / max(float(fps), 1e-6)
        speed = np.linalg.norm(np.diff(work, axis=0), axis=1) / dt
        med, mad = _rolling_median_mad_1d(speed, int(window))
        threshold = np.minimum(
            med + float(k_mad) * np.maximum(mad, 1e-10),
            float(speed_cap_m_s),
        )
        edge_bad = np.isfinite(speed) & (speed > threshold)
        frame_bad = np.zeros(n, dtype=bool)
        frame_bad[:-1] |= edge_bad
        frame_bad[1:] |= edge_bad
        new_bad = bad | frame_bad
        last_speed = speed
        last_thresh = threshold
        if np.array_equal(new_bad, bad):
            break
        bad = new_bad

    bad = _xyz_post_process_bad_mask(
        bad,
        gap_merge=int(gap_merge),
        max_burst_frames=int(max_burst_frames),
    )
    return bad, {
        "raw_speed_m_s": last_speed,
        "threshold_m_s": last_thresh,
    }


def _replace_dir_with_copy(src, dst):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    if os.path.isdir(src):
        shutil.copytree(src, dst)


def postprocess_xyz_velocity_interp_then_kalman(
    debug_dir,
    log_csv,
    cam_ext,
    main_cam,
    tf_to_centered_mesh,
    fps=10.0,
    speed_cap_m_s=0.80,
    k_mad=8.0,
    window=31,
    gap_merge=1,
    max_burst_frames=10,
    use_kalman=True,
    kalman_pos_meas_std=0.006,
    kalman_pos_accel_std=2.0,
):
    """
    Offline translation postprocess:
      raw refined/consensus-fused XYZ -> velocity spike interpolation -> forward XYZ Kalman.

    The standard ob_in_world / ob_in_cam / ob_in_selected_cam folders are replaced
    with the FINAL XYZ-postprocessed poses so all existing downstream rotation-fix code
    automatically starts from the repaired translation. Raw folders are backed up as
    *_xyzraw and the interpolation-only world poses are kept in ob_in_world_xyzinterp.
    """
    tf_to_centered_mesh = np.asarray(tf_to_centered_mesh, dtype=np.float64).reshape(4, 4)
    inv_tf = np.linalg.inv(tf_to_centered_mesh)
    world_dir = os.path.join(debug_dir, "ob_in_world")
    main_dir = os.path.join(debug_dir, "ob_in_cam")
    selected_dir = os.path.join(debug_dir, "ob_in_selected_cam")
    world_raw_backup = os.path.join(debug_dir, "ob_in_world_xyzraw")
    main_raw_backup = os.path.join(debug_dir, "ob_in_cam_xyzraw")
    selected_raw_backup = os.path.join(debug_dir, "ob_in_selected_cam_xyzraw")
    interp_world_dir = os.path.join(debug_dir, "ob_in_world_xyzinterp")
    interp_main_dir = os.path.join(debug_dir, "ob_in_cam_xyzinterp")
    interp_selected_dir = os.path.join(debug_dir, "ob_in_selected_cam_xyzinterp")
    raw_log_backup = os.path.join(debug_dir, "register_every_frame_log_xyzraw.csv")

    with open(log_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        source_fields = list(reader.fieldnames or [])

    records = []
    for row_idx, row in enumerate(rows):
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        frame_id = str(row.get("frame_id", "")).strip()
        if not frame_id:
            continue
        pose_path = os.path.join(world_dir, f"{frame_id}.txt")
        if not os.path.exists(pose_path):
            continue
        try:
            frame_idx = int(row.get("frame_idx", row_idx))
            pose_world = np.loadtxt(pose_path).astype(np.float64).reshape(4, 4)
            pose_centered = pose_world @ inv_tf
            pose_centered[:3, :3] = _project_rotation_so3(pose_centered[:3, :3])
        except Exception as e:
            logging.warning(f"[xyzpost] skip unreadable frame={frame_id}: {e}")
            continue
        records.append({
            "row_idx": row_idx,
            "frame_idx": frame_idx,
            "frame_id": frame_id,
            "selected_cam": str(row.get("selected_cam", "")).strip(),
            "centered_raw": pose_centered,
        })

    records.sort(key=lambda r: (r["frame_idx"], r["frame_id"]))
    if len(records) < 2:
        logging.warning(f"[xyzpost] only {len(records)} valid poses; skipping XYZ postprocess")
        return None

    frame_indices = np.asarray([r["frame_idx"] for r in records], dtype=np.int64)
    xyz_raw = np.stack([r["centered_raw"][:3, 3] for r in records], axis=0)

    bad, det = _detect_xyz_velocity_bad_frames(
        frame_indices=frame_indices,
        xyz=xyz_raw,
        fps=float(fps),
        speed_cap_m_s=float(speed_cap_m_s),
        k_mad=float(k_mad),
        window=int(window),
        gap_merge=int(gap_merge),
        max_burst_frames=int(max_burst_frames),
    )
    xyz_interp = _interp_xyz_at_frame_indices(frame_indices, xyz_raw, bad)

    # Preserve exactly what FoundationPose produced before any XYZ repair.
    _replace_dir_with_copy(world_dir, world_raw_backup)
    _replace_dir_with_copy(main_dir, main_raw_backup)
    _replace_dir_with_copy(selected_dir, selected_raw_backup)
    shutil.copy2(log_csv, raw_log_backup)

    for d in (interp_world_dir, interp_main_dir, interp_selected_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # Build interpolation-only poses first; rotations remain raw and untouched.
    interp_centered = []
    for rec, xyz_i in zip(records, xyz_interp):
        p = rec["centered_raw"].copy()
        p[:3, 3] = xyz_i
        interp_centered.append(p)
    interp_centered = np.asarray(interp_centered, dtype=np.float64)

    # Final forward XYZ Kalman happens ONLY after interpolation.
    final_centered = interp_centered.copy()
    kalman_infos = []
    if bool(use_kalman):
        kf = ObjectPoseKalmanFilter(
            fps=float(fps),
            pos_measurement_std=float(kalman_pos_meas_std),
            pos_acceleration_std=float(kalman_pos_accel_std),
        )
        for i, rec in enumerate(records):
            pf, info = kf.update(interp_centered[i], int(rec["frame_idx"]))
            # ObjectPoseKalmanFilter already passes current raw rotation through unchanged.
            final_centered[i] = pf
            kalman_infos.append(info)
    else:
        for i, rec in enumerate(records):
            rv = rotation_matrix_to_rotvec_rad(interp_centered[i, :3, :3])
            kalman_infos.append({
                "dt": 0.0,
                "measurement_xyz": xyz_interp[i].copy(),
                "filtered_xyz": xyz_interp[i].copy(),
                "filtered_velocity": np.zeros(3, dtype=np.float64),
                "measurement_rotvec_rad": rv.copy(),
                "filtered_rotvec_rad": rv.copy(),
                "filtered_omega_rad_s": np.zeros(3, dtype=np.float64),
            })

    # Clear standard final dirs, then rewrite them with final XYZ-postprocessed poses.
    for d in (world_dir, main_dir, selected_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    main_cam = str(main_cam)
    if main_cam not in cam_ext:
        raise KeyError(f"main cam {main_cam} missing in extri.yml during xyz postprocess")

    extra_fields = [
        "xyzpost_bad",
        "xyzpost_raw_center_x", "xyzpost_raw_center_y", "xyzpost_raw_center_z",
        "xyzpost_interp_center_x", "xyzpost_interp_center_y", "xyzpost_interp_center_z",
        "xyzpost_final_center_x", "xyzpost_final_center_y", "xyzpost_final_center_z",
        "xyzpost_final_vx", "xyzpost_final_vy", "xyzpost_final_vz",
        "xyzpost_speed_cap_m_s",
        "xyzpost_stage",
    ]
    final_by_row = {}

    for i, rec in enumerate(records):
        frame_id = rec["frame_id"]
        selected_cam = rec["selected_cam"]

        pose_world_interp = interp_centered[i] @ tf_to_centered_mesh
        pose_world_final = final_centered[i] @ tf_to_centered_mesh

        np.savetxt(os.path.join(interp_world_dir, f"{frame_id}.txt"), pose_world_interp)
        np.savetxt(os.path.join(world_dir, f"{frame_id}.txt"), pose_world_final)

        pose_main_interp = np.asarray(cam_ext[main_cam]["ext"], dtype=np.float64).reshape(4, 4) @ pose_world_interp
        pose_main_final = np.asarray(cam_ext[main_cam]["ext"], dtype=np.float64).reshape(4, 4) @ pose_world_final
        np.savetxt(os.path.join(interp_main_dir, f"{frame_id}.txt"), pose_main_interp)
        np.savetxt(os.path.join(main_dir, f"{frame_id}.txt"), pose_main_final)

        if selected_cam and selected_cam in cam_ext:
            ext_sel = np.asarray(cam_ext[selected_cam]["ext"], dtype=np.float64).reshape(4, 4)
            np.savetxt(os.path.join(interp_selected_dir, f"{frame_id}.txt"), ext_sel @ pose_world_interp)
            np.savetxt(os.path.join(selected_dir, f"{frame_id}.txt"), ext_sel @ pose_world_final)

        info = kalman_infos[i]
        final_by_row[rec["row_idx"]] = {
            "pose_world_final": pose_world_final,
            "raw": xyz_raw[i],
            "interp": xyz_interp[i],
            "final": np.asarray(info["filtered_xyz"], dtype=np.float64).reshape(3),
            "vel": np.asarray(info["filtered_velocity"], dtype=np.float64).reshape(3),
            "dt": float(info["dt"]),
            "bad": bool(bad[i]),
        }

    # Rewrite the standard log so it describes the standard FINAL pose folders.
    fields = source_fields + [f for f in extra_fields if f not in source_fields]
    for row_idx, row in enumerate(rows):
        rec = final_by_row.get(row_idx)
        if rec is None:
            continue
        pw = rec["pose_world_final"]
        row["pose_world_tx"] = f"{pw[0, 3]:.8f}"
        row["pose_world_ty"] = f"{pw[1, 3]:.8f}"
        row["pose_world_tz"] = f"{pw[2, 3]:.8f}"
        row["kalman_enabled"] = int(bool(use_kalman))
        row["kalman_dt"] = f"{rec['dt']:.8f}"
        row["kalman_center_x"] = f"{rec['final'][0]:.8f}"
        row["kalman_center_y"] = f"{rec['final'][1]:.8f}"
        row["kalman_center_z"] = f"{rec['final'][2]:.8f}"
        row["kalman_vx"] = f"{rec['vel'][0]:.8f}"
        row["kalman_vy"] = f"{rec['vel'][1]:.8f}"
        row["kalman_vz"] = f"{rec['vel'][2]:.8f}"
        row["xyzpost_bad"] = int(rec["bad"])
        row["xyzpost_raw_center_x"] = f"{rec['raw'][0]:.8f}"
        row["xyzpost_raw_center_y"] = f"{rec['raw'][1]:.8f}"
        row["xyzpost_raw_center_z"] = f"{rec['raw'][2]:.8f}"
        row["xyzpost_interp_center_x"] = f"{rec['interp'][0]:.8f}"
        row["xyzpost_interp_center_y"] = f"{rec['interp'][1]:.8f}"
        row["xyzpost_interp_center_z"] = f"{rec['interp'][2]:.8f}"
        row["xyzpost_final_center_x"] = f"{rec['final'][0]:.8f}"
        row["xyzpost_final_center_y"] = f"{rec['final'][1]:.8f}"
        row["xyzpost_final_center_z"] = f"{rec['final'][2]:.8f}"
        row["xyzpost_final_vx"] = f"{rec['vel'][0]:.8f}"
        row["xyzpost_final_vy"] = f"{rec['vel'][1]:.8f}"
        row["xyzpost_final_vz"] = f"{rec['vel'][2]:.8f}"
        row["xyzpost_speed_cap_m_s"] = f"{float(speed_cap_m_s):.6f}"
        row["xyzpost_stage"] = "interp_then_kalman" if bool(use_kalman) else "interp_only"

    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    raw_edge_speed = np.asarray(det.get("raw_speed_m_s", []), dtype=np.float64)
    logging.info(
        f"[xyzpost] complete: valid_poses={len(records)}, bad_frames={int(bad.sum())}, "
        f"fps={float(fps):.3f}, hard_speed_cap={float(speed_cap_m_s):.3f}m/s, "
        f"max_raw_edge_speed={float(np.max(raw_edge_speed)) if raw_edge_speed.size else 0.0:.4f}m/s, "
        f"final_kalman={int(bool(use_kalman))}"
    )
    logging.info(f"[xyzpost] raw WORLD backup: {world_raw_backup}")
    logging.info(f"[xyzpost] interpolation-only WORLD: {interp_world_dir}")
    logging.info(f"[xyzpost] FINAL WORLD (standard path): {world_dir}")
    logging.info(f"[xyzpost] raw log backup: {raw_log_backup}")

    return {
        "bad_mask": bad,
        "num_bad": int(bad.sum()),
        "world_raw_backup": world_raw_backup,
        "world_interp_dir": interp_world_dir,
        "world_final_dir": world_dir,
        "raw_log_backup": raw_log_backup,
        "log_csv": log_csv,
    }


def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument("--mesh_file", type=str, default=f"../my_sequence/meshes/compressed.ply")
    parser.add_argument("--test_scene_dir", type=str, default=f"../my_sequence")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug_dir", type=str, default=f"../my_sequence/debug_register_best_mask")
    parser.add_argument("--i_th_cam", type=str, default="1", help="Main camera id used for saved pose/visualization.")
    parser.add_argument(
        "--tracking_cam", "--tracking-cam",
        dest="tracking_cam",
        type=str,
        default=None,
        help=(
            "Optional fixed camera id used for ALL FoundationPose registration/tracking/refinement. "
            "When omitted, the default largest-mask-area camera selection is used. When set, the "
            "specified camera is used strictly (no camera fallback), while xyz is STILL triangulated "
            "from all valid multi-view masks and the centered-mesh translation is locked to that "
            "triangulated xyz. Example: --tracking_cam 12."
        ),
    )
    parser.add_argument("--save_fps", type=int, default=10)
    parser.add_argument(
        "--postprocess_vis", "--postprocess-vis",
        dest="postprocess_vis",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "After offline rotation-fix post-processing, regenerate FINAL 3D-bbox videos "
            "for both the main camera and an all-camera grid from ob_in_world_rotfix. "
            "This is independent of --simp-saving/--no_grid_vis. Default: true."
        ),
    )
    parser.add_argument(
        "--postprocess_vis_save_frames", "--postprocess-vis-save-frames",
        dest="postprocess_vis_save_frames",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Also save per-frame PNGs for the postprocessed main/all-camera bbox visualizations. "
            "Videos are always generated when --postprocess_vis true. Default: false."
        ),
    )
    parser.add_argument(
        "--simp-saving", "--simp_saving",
        dest="simp_saving",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "Simplified saving mode (default: true). When true, save only the main-camera "
            "visualization frames and the normal main-camera mp4 visualization; disable fused "
            "all-camera grid images/grid video, temporal rotation-distribution figures, and "
            "FoundationPose/refiner/scorer intermediate debug visualizations. Core pose txt/CSV "
            "outputs are still saved. Use --simp-saving false to restore the detailed saving behavior."
        ),
    )
    parser.add_argument(
        "--use-kalman", "--use_kalman", dest="use_kalman",
        type=_str2bool, nargs="?", const=True, default=True,
        help=(
            "FINAL OFFLINE XYZ-only constant-velocity Kalman filter (default: true). It runs only "
            "AFTER velocity-spike XYZ interpolation; there is no XYZ Kalman inside the FoundationPose "
            "tracking loop. Translation state: [x,y,z,vx,vy,vz]. Rotation is preserved raw."
        ),
    )
    parser.add_argument(
        "--kalman_fps", type=float, default=10.0,
        help="Frame rate used by Kalman dynamics. Default: 10 FPS (low-lag responsive preset).",
    )
    parser.add_argument(
        "--kalman_pos_meas_std", type=float, default=0.006,
        help=(
            "XYZ measurement std in meters. Larger values trust each raw XYZ measurement less "
            "and therefore suppress position jitter more strongly. Default: 0.006 (10 FPS low-lag preset)."
        ),
    )
    parser.add_argument(
        "--kalman_pos_accel_std", type=float, default=2.0,
        help=(
            "Translation acceleration-process std in m/s^2. Smaller values enforce a smoother "
            "constant-velocity trajectory and reduce XYZ shaking. Default: 2.0 (10 FPS low-lag preset)."
        ),
    )
    parser.add_argument(
        "--xyz_fix_fps", "--xyz-fix-fps", dest="xyz_fix_fps", type=float, default=10.0,
        help="FPS used by offline XYZ velocity-spike detection. Default: 10.",
    )
    parser.add_argument(
        "--xyz_speed_cap_m_s", "--xyz-speed-cap-m-s", dest="xyz_speed_cap_m_s",
        type=float, default=1.0,
        help="Hard XYZ speed cap for spike detection in m/s. Default: 1.2 m/s = 120 cm/s.",
    )
    parser.add_argument(
        "--xyz_fix_k_mad", "--xyz-fix-k-mad", dest="xyz_fix_k_mad", type=float, default=8.0,
        help="Local median+MAD multiplier for XYZ speed spike detection. Default: 8.0.",
    )
    parser.add_argument(
        "--xyz_fix_window", "--xyz-fix-window", dest="xyz_fix_window", type=int, default=31,
        help="Centered local speed window for median/MAD detection. Default: 31 edges.",
    )
    parser.add_argument(
        "--xyz_fix_gap_merge", "--xyz-fix-gap-merge", dest="xyz_fix_gap_merge", type=int, default=1,
        help="Merge good gaps this short between bad XYZ runs. Default: 1 frame.",
    )
    parser.add_argument(
        "--xyz_fix_max_burst_frames", "--xyz-fix-max-burst-frames", dest="xyz_fix_max_burst_frames",
        type=int, default=10,
        help=(
            "process_dataset.py-style maximum short bad burst to interpolate. Runs longer than this "
            "are treated as real motion and left unmasked. Default: 10; <=0 disables this long-run unmasking."
        ),
    )
    parser.add_argument(
        "--kalman_rot_meas_std_deg", type=float, default=1.5,
        help="Deprecated/ignored: rotation Kalman is disabled; kept only for CLI compatibility.",
    )
    parser.add_argument(
        "--kalman_rot_accel_std_deg", type=float, default=540.0,
        help="Deprecated/ignored: rotation Kalman is disabled; kept only for CLI compatibility.",
    )
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument(
        "--end_frame",
        type=int,
        default=-1,
        help="Exclusive end frame index. -1 means use all frames.",
    )
    parser.add_argument(
        "--min_mask_area",
        type=int,
        default=20,
        help="Skip cameras whose object mask area is smaller than this many pixels.",
    )
    parser.add_argument(
        "--mask_center_method",
        type=str,
        default="min_area_rect",
        choices=["min_area_rect", "pca_rotated_box", "centroid", "bbox"],
        help=(
            "How to compute each 2D mask center for triangulation. "
            "Default min_area_rect uses OpenCV rotated bounding-box center. "
            "pca_rotated_box uses PCA-oriented bbox center."
        ),
    )
    parser.add_argument(
        "--ransac_reproj_thresh",
        type=float,
        default=1000,
        help="RANSAC reprojection inlier threshold in pixels for mask-center triangulation.",
    )
    parser.add_argument(
        "--min_triangulation_inliers",
        type=int,
        default=6,
        help="Minimum number of camera observations used in final triangulation.",
    )
    parser.add_argument(
        "--min_pair_angle_deg",
        type=float,
        default=2.0,
        help="Skip camera pairs whose two viewing rays have smaller angle than this; small baseline gives unstable depth.",
    )
    parser.add_argument(
        "--final_triangulation_method",
        type=str,
        default="ray_lstsq",
        choices=["ray_lstsq", "dlt"],
        help="Final multi-view triangulation method after RANSAC inlier selection.",
    )
    parser.add_argument(
        "--fallback_next_best",
        action="store_true",
        help=(
            "If the brightest masked-region camera registration fails, try cameras "
            "in descending mask-area order."
        ),
    )
    parser.add_argument(
        "--use_depth",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "0 (default): preserve original working RGB-only behavior and pass zero depth. "
            "1: load uint16 depth and divide by 1000. By default it is used without "
            "triangulated-Z rescaling; set --do-not-rescale-depth false for the old alignment."
        ),
    )
    parser.add_argument(
        "--depth_dir",
        type=str,
        default=None,
        help="Depth root containing <cam>/<frame>.png. Default: <test_scene_dir>/depth.",
    )
    parser.add_argument(
        "--do-not-rescale-depth", "--do_not_rescale_depth",
        dest="do_not_rescale_depth",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "Do not rescale loaded depth according to the triangulated object-center camera-Z "
            "(default: true). With true, depth is only converted from uint16 by /1000 and then "
            "used directly. Set --do-not-rescale-depth false to restore the old masked-median "
            "to triangulated-Z scale alignment."
        ),
    )
    parser.add_argument(
        "--require_depth",
        action="store_true",
        help="When --use_depth 1, skip cameras with missing depth instead of zero-depth fallback.",
    )
    parser.add_argument(
        "--depth_mask_only",
        action="store_true",
        help="After scale alignment, zero depth outside the object mask. Default keeps full aligned depth map.",
    )
    parser.add_argument(
        "--no_temporal_cone",
        action="store_true",
        help="Disable temporal-cone mode after the first frame and run full all-direction registration every frame.",
    )
    parser.add_argument(
        "--fast",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Fast mode. First frame uses normal full registration with 3 refinement iterations. "
            "Later frames refine one previous-rotation pose on the top-K largest-area masks (default K=10; 3 iter each), "
            "find the largest >=3-view mutual 6D consensus cluster (rotation <30deg and translation <=5cm by default), "
            "then fuse only WORLD XYZ using mask area as weight; WORLD rotation comes directly from "
            "the largest-mask best camera inside the accepted cluster (no cross-camera rotation averaging). "
            "If no satisfactory cluster exists, that frame falls back to rank-1 mask full "
            "360deg/240-hyp registration with 5 refinement iterations. Fallback first finds the "
            "largest <10deg rotation cluster (minimum 30 poses) and chooses the highest score in "
            "that cluster; if no cluster reaches 30, it uses the global highest score. "
            "No XYZ Kalman runs online. After the full sequence, XYZ spikes are interpolated with a "
            "0.80 m/s default cap and then the final forward XYZ Kalman filter is applied."
        ),
    )
    parser.add_argument(
        "--fast_rot_consistency_deg", "--fast-rot-consistency-deg",
        dest="fast_rot_consistency_deg",
        type=float,
        default=30,
        help=(
            "Fast multi-view consensus WORLD-rotation threshold in degrees. Every pair INSIDE the "
            "accepted cluster must be strictly below this value. If no >=3-view cluster satisfies "
            "the rotation AND translation gates, the current frame falls back to rank-1 full registration. "
            "Fallback pose selection uses the largest <10deg rotation cluster (min 30) and takes "
            "the highest score inside it; if no such cluster exists, it uses the global best score. "
            "Default: 30.0."
        ),
    )
    parser.add_argument(
        "--fast_xyz_consistency_m", "--fast-xyz-consistency-m",
        dest="fast_xyz_consistency_m",
        type=float,
        default=0.05,
        help=(
            "FAST multi-view consensus translation gate in meters. Every pair INSIDE the accepted "
            "cluster must be within this distance (<= threshold) in refined WORLD centered-object XYZ, "
            "in addition to the rotation gate. Default: 0.05 m = 5 cm."
        ),
    )
    parser.add_argument(
        "--fast_consensus_topk", "--fast-consensus-topk",
        dest="fast_consensus_topk",
        type=int,
        default=10,
        help="Number of largest-mask views independently refined in FAST consensus mode. Default: 10.",
    )
    parser.add_argument(
        "--fast_consensus_min_cluster", "--fast-consensus-min-cluster",
        dest="fast_consensus_min_cluster",
        type=int,
        default=4,
        help=(
            "Minimum number of mutually consistent refined camera poses required to accept FAST consensus. "
            "All pairs in the cluster must satisfy the rotation and translation gates. Default: 3."
        ),
    )
    parser.add_argument(
        "--fast_fallback_cluster_deg", "--fast-fallback-cluster-deg",
        dest="fast_fallback_cluster_deg",
        type=float,
        default=30.0,
        help=(
            "Rotation geodesic threshold for the fast fallback's largest-cluster selector. "
            "A pose belongs to an anchor's cluster only when rotation difference is strictly less "
            "than this value. Default: 10.0 degrees."
        ),
    )
    parser.add_argument(
        "--fast_fallback_cluster_min", "--fast-fallback-cluster-min",
        dest="fast_fallback_cluster_min",
        type=int,
        default=30,
        help=(
            "Minimum size required for the fast fallback rotation cluster. If the largest cluster "
            "is smaller than this, choose the highest FoundationPose score over all hypotheses. "
            "Default: 30."
        ),
    )
    parser.add_argument(
        "--no-force",
        dest="no_force",
        action="store_true",
        help=(
            "Do not force/recenter the selected best pose translation back to the "
            "RANSAC-triangulated object center after FoundationPose refinement/selection. "
            "This also applies in --fast / fast-fallback mode. Triangulated xyz is still used for "
            "initialization. Default keeps the old forced-recenter behavior."
        ),
    )
    parser.add_argument(
        "--temporal_pose_cone_deg",
        type=float,
        default=180.0,
        help="For frames after the first one, sample/refine rotations within this SO(3) angle from the last world rotation.",
    )
    parser.add_argument(
        "--temporal_num_hypotheses",
        type=int,
        default=240,
        help="Total number of temporal-cone rotation hypotheses for each frame after the first one.",
    )
    parser.add_argument(
        "--mask_rotation_var_gate",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "1: quality-gate the area-ranked mask camera using refined rotation avg variance. "
            "Try up to --mask_rotation_var_topn masks and accept the first with avg var below threshold; "
            "if none passes, choose the smallest-var result. 0 disables this gate."
        ),
    )
    parser.add_argument(
        "--mask_rotation_var_threshold",
        type=float,
        default=5.0,
        help=(
            "Threshold in deg^2 for mean(var(dRx),var(dRy),var(dRz)) over the refined temporal hypotheses. "
            "Default 6.0. A mask passes only when avg var < threshold."
        ),
    )
    parser.add_argument(
        "--mask_rotation_var_topn",
        type=int,
        default=1,
        help="Maximum number of area-ranked masks to try for the rotation-variance gate. Default 3.",
    )
    parser.add_argument(
        "--distribution_vis_dir",
        type=str,
        default=None,
        help=(
            "Directory for per-frame 240-rotation distribution PNGs. "
            "Default: <test_scene_dir>/distribution_vis."
        ),
    )
    parser.add_argument(
        "--no_distribution_vis",
        action="store_true",
        help="Disable per-frame temporal rotation distribution visualization.",
    )
    parser.add_argument(
        "--temporal_seed",
        type=int,
        default=0,
        help="Base random seed for temporal-cone hypothesis sampling.",
    )
    parser.add_argument(
        "--temporal_hypothesis_batch_size",
        type=int,
        default=0,
        help="Process temporal hypotheses in chunks to reduce VRAM. 0 means process all hypotheses at once.",
    )
    parser.add_argument(
        "--no_grid_vis",
        default=True,
        help="Do not save fused all-camera grid visualization images.",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=0,
        help="Number of columns in the fused all-camera grid. 0 means automatic sqrt layout.",
    )
    parser.add_argument(
        "--grid_cell_width",
        type=int,
        default=360,
        help="Resize every camera view to this width before fusing into the grid. <=0 keeps original width.",
    )
    parser.add_argument(
        "--save_grid_video",
        default=False,
        help="Also save an mp4 video from the fused all-camera grid images.",
    )
    parser.add_argument(
        "--no_video",
        default=True,
        help="Do not save mp4 visualization video.",
    )
    parser.add_argument(
        "--no_window",
        action="store_true",
        help="Do not call cv2.imshow. Useful on headless servers.",
    )
    parser.add_argument(
        "--temporal_consensus_topk",
        type=int,
        default=60,
        help="Use only the top-K scored temporal hypotheses for rotation-consensus selection.",
    ) 

    parser.add_argument(
        "--temporal_consensus_angle_deg",
        type=float,
        default=10,
        help="Rotation inlier threshold in degrees for temporal consensus selection.",
    )

    parser.add_argument(
        "--temporal_consensus_min_supporters",
        type=int,
        default=240,
        help="Minimum number of rotation supporters required; otherwise fallback to best score.",
    )
    parser.add_argument(
        "--rotation_fix", "--rotation-fix",
        dest="rotation_fix",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "After the complete FoundationPose pass, detect short WORLD-frame SO(3) rotation spike bursts "
            "and repair only those rotations with SLERP. Online tracking remains unchanged. Default: true."
        ),
    )
    parser.add_argument(
        "--rotation_fix_fps", "--rotation-fix-fps",
        dest="rotation_fix_fps",
        type=float,
        default=10.0,
        help="FPS used by offline rotation-fix angular-speed thresholds. Current recommended setting: 10.",
    )
    parser.add_argument(
        "--rotation_fix_window", "--rotation-fix-window",
        dest="rotation_fix_window",
        type=int,
        default=31,
        help="Rolling median/MAD edge-speed window (odd frames/edges). Default: 31.",
    )
    parser.add_argument(
        "--rotation_fix_k_mad", "--rotation-fix-k-mad",
        dest="rotation_fix_k_mad",
        type=float,
        default=8.0,
        help="Adaptive spike threshold multiplier: local median + K*MAD. Default: 8.",
    )
    parser.add_argument(
        "--rotation_fix_min_threshold_deg_s", "--rotation-fix-min-threshold-deg-s",
        dest="rotation_fix_min_threshold_deg_s",
        type=float,
        default=250.0,
        help="Do not call an edge a spike below this angular speed. At 10 FPS, 250 deg/s = 25 deg/frame.",
    )
    parser.add_argument(
        "--rotation_fix_cap_threshold_deg_s", "--rotation-fix-cap-threshold-deg-s",
        dest="rotation_fix_cap_threshold_deg_s",
        type=float,
        default=350.0,
        help="Hard upper cap on adaptive spike threshold. At 10 FPS, 350 deg/s = 35 deg/frame.",
    )
    parser.add_argument(
        "--rotation_fix_max_burst_frames", "--rotation-fix-max-burst-frames",
        dest="rotation_fix_max_burst_frames",
        type=int,
        default=10,
        help="Maximum enclosed bad-rotation burst length to interpolate. At 10 FPS, 10 frames = 1 s.",
    )
    parser.add_argument(
        "--rotation_fix_max_anchor_avg_deg_s", "--rotation-fix-max-anchor-avg-deg-s",
        dest="rotation_fix_max_anchor_avg_deg_s",
        type=float,
        default=250.0,
        help=(
            "A spike pair is repairable only if the outside anchor rotations imply <= this average speed. "
            "At 10 FPS, 250 deg/s = 25 deg/frame."
        ),
    )
    parser.add_argument(
        "--postkal",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "After offline rotation spike interpolation/SLERP, run the complete repaired WORLD-frame "
            "rotation sequence from first to last through a causal SO(3) rotation Kalman filter. "
            "Translation remains unchanged. Default: false."
        ),
    )
    parser.add_argument(
        "--postkal_fps", "--postkal-fps",
        dest="postkal_fps",
        type=float,
        default=10.0,
        help="FPS for the post-SLERP rotation Kalman filter. Default: 10.",
    )
    parser.add_argument(
        "--postkal_rot_meas_std_deg", "--postkal-rot-meas-std-deg",
        dest="postkal_rot_meas_std_deg",
        type=float,
        default=2.0,
        help=(
            "Rotation measurement std for postkal in degrees. Larger = smoother / less trust in each pose. "
            "Default: 2.0 deg."
        ),
    )
    parser.add_argument(
        "--postkal_rot_accel_std_deg_s2", "--postkal-rot-accel-std-deg-s2",
        dest="postkal_rot_accel_std_deg_s2",
        type=float,
        default=360.0,
        help=(
            "Angular-acceleration process std for postkal in deg/s^2. Larger = more responsive / less smooth. "
            "Default: 360 at 10 FPS."
        ),
    )
    parser.add_argument(
        "--postkal_initial_omega_std_deg_s", "--postkal-initial-omega-std-deg-s",
        dest="postkal_initial_omega_std_deg_s",
        type=float,
        default=180.0,
        help="Initial angular-velocity uncertainty for postkal in deg/s. Default: 180.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    set_logging_format()
    set_seed(0)

    debug = int(args.debug)
    debug_dir = args.debug_dir
    main_cam = str(args.i_th_cam)

    # FAST-IO policy: tracking-pass visualization is disabled. The intended final
    # green-camera rendering is a separate downstream/final visualization stage and
    # is intentionally not handled here.
    args.no_grid_vis = True
    args.save_grid_video = False
    args.no_distribution_vis = True
    args.no_video = True
    args.no_window = True
    args.postprocess_vis = False

    # Simplified saving is ON by default. It changes only what visual debug artifacts
    # are written; pose estimation and core pose/CSV outputs are unchanged.
    if bool(args.simp_saving):
        args.no_grid_vis = True
        args.save_grid_video = False
        args.no_distribution_vis = True
        logging.info(
            "[simp-saving] enabled: saving main-camera visualization frames + normal mp4 only; "
            "grid/multi-camera visualizations, grid video, rotation distributions, and internal "
            "FoundationPose debug visualizations are disabled."
        )
    else:
        logging.info("[simp-saving] disabled: detailed visualization saving follows the other CLI flags.")

    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "ob_in_cam"), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "ob_in_selected_cam"), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "ob_in_world"), exist_ok=True)
    if not args.no_grid_vis:
        os.makedirs(os.path.join(debug_dir, "register_vis_grid"), exist_ok=True)

    if args.distribution_vis_dir is None or len(str(args.distribution_vis_dir).strip()) == 0:
        args.distribution_vis_dir = os.path.join(args.test_scene_dir, "distribution_vis")
    if not args.no_distribution_vis:
        os.makedirs(args.distribution_vis_dir, exist_ok=True)
        logging.info(f"[rotation_distribution] output dir: {args.distribution_vis_dir}")

    log_csv = os.path.join(debug_dir, "register_every_frame_log.csv")
    video_path = os.path.join(debug_dir, "register_every_frame_vis.mp4")
    grid_video_path = os.path.join(debug_dir, "register_vis_grid.mp4")
    video_writer = None
    grid_video_writer = None

    mesh = trimesh.load(args.mesh_file)

    # IMPORTANT: do NOT use trimesh.bounds.oriented_bounds(mesh) for the bbox
    # center. Its OBB origin can differ from FoundationPose's centered-mesh
    # origin by several centimeters.
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator_debug = 0 if bool(args.simp_saving) else debug
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=estimator_debug,
        glctx=glctx,
    )
    logging.info("estimator initialization done")

    # Use the EXACT centered-mesh transform from FoundationPose for visualization.
    # This is the same transform used by the registration xyz-lock code above.
    tf_to_centered_mesh_vis = pose_to_numpy_4x4(
        est.get_tf_to_centered_mesh()
    ).astype(np.float64)
    bbox_centered_vis = build_foundationpose_centered_bbox(
        mesh=mesh,
        tf_to_centered_mesh=tf_to_centered_mesh_vis,
    )

    reader = YcbineoatReader(
        video_dir=args.test_scene_dir,
        shorter_side=None,
        zfar=np.inf,
        maincam=main_cam,
    )

    cam_ext = readextfromyml(args.test_scene_dir)
    cam_pose_rest = reader.get_relat(cam_ext, main_cam)
    # Only read the shared intrinsic matrix from cam_K.txt.
    # K_by_cam must still be a dict because downstream code uses K_by_cam.get(cam, reader.K).
    K_shared = read_intrinsics_from_txt(args.test_scene_dir, downscale=reader.downscale)

    K_by_cam = {
        str(cam): K_shared.copy()
        for cam in getattr(reader, "all_cam_folders", [])
    }

    # If the reader did not expose camera folders for some reason, still keep the main camera valid.
    if main_cam not in K_by_cam:
        K_by_cam[main_cam] = K_shared.copy()

    K_main = K_by_cam.get(main_cam, reader.K)

    # Optional strict FoundationPose tracking-camera lock. This affects ONLY which camera
    # supplies RGB/mask/depth to register/refiner/scorer. Multi-view center triangulation
    # still uses the complete observations list from all valid cameras.
    tracking_cam = str(args.tracking_cam).strip() if args.tracking_cam is not None else ""
    args.tracking_cam = tracking_cam if tracking_cam else None
    if tracking_cam:
        available_reader_cams = [str(cam) for cam in getattr(reader, "all_cam_folders", [])]
        if tracking_cam not in available_reader_cams:
            raise ValueError(
                f"--tracking_cam {tracking_cam} is not present in scene camera folders: "
                f"{available_reader_cams}"
            )
        if tracking_cam not in cam_ext:
            raise ValueError(
                f"--tracking_cam {tracking_cam} is missing from extri.yml. "
                f"Available extrinsic cameras: {sorted(cam_ext.keys(), key=_camera_sort_key)}"
            )
        logging.info(
            f"[tracking_cam] LOCKED to cam={tracking_cam} for FoundationPose pose estimation/tracking; "
            "multi-view RANSAC triangulation remains unchanged and centered xyz is forced to the triangulated center."
        )
    else:
        logging.info(
            "[tracking_cam] AUTO: rank FoundationPose camera selection by mask area only."
        )

    use_depth = bool(int(args.use_depth))
    depth_root = args.depth_dir
    if depth_root is None or len(str(depth_root).strip()) == 0:
        depth_root = os.path.join(args.test_scene_dir, "depth")
    depth_root = os.path.abspath(depth_root)
    if use_depth and not os.path.isdir(depth_root):
        if args.require_depth:
            raise FileNotFoundError(f"Required depth directory does not exist: {depth_root}")
        logging.warning(f"[depth] missing depth directory {depth_root}; cameras will use zero-depth fallback")

    # Accepted RAW centered-object WORLD pose used as next-frame temporal prior.
    # IMPORTANT: online XYZ Kalman is intentionally disabled. The complete raw sequence
    # is first cleaned by velocity-spike interpolation, then (optionally) Kalman-filtered offline.
    last_pose_world = None
    kalman_fps = float(args.kalman_fps) if args.kalman_fps is not None else float(args.save_fps)
    pose_kalman = None
    logging.info(
        "[kalman] ONLINE XYZ Kalman DISABLED by design. FoundationPose first saves raw XYZ; "
        f"offline order is velocity interpolation -> final XYZ Kalman(enabled={int(bool(args.use_kalman))}), "
        f"xyz_fix_fps={args.xyz_fix_fps}, speed_cap={args.xyz_speed_cap_m_s}m/s, "
        f"kalman_fps={kalman_fps:.6g}."
    )

    start_frame = 0
    end_frame = len(reader.id_strs)
    if end_frame <= start_frame:
        raise ValueError(f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}")

    if bool(args.fast):
        logging.info(
            f"[mode] FAST enabled: first frame full all-direction registration with 3 refine iters; "
            f"following frames top{args.fast_consensus_topk} previous-rotation init + 3-iter refinement; "
            f"accept largest >= {args.fast_consensus_min_cluster}-view mutual 6D consensus cluster, then "
            f"mask-area weighted mean of WORLD rotation+translation; otherwise full-register fallback; "
            f"rot strict<{args.fast_rot_consistency_deg:.2f}deg, xyz <= {args.fast_xyz_consistency_m * 100.0:.2f}cm; "
            f"offline xyz interpolation cap={args.xyz_speed_cap_m_s}m/s then final Kalman; use_depth={int(use_depth)}"
        )
    else:
        logging.info(
            f"[mode] first frame full all-direction registration; following frames temporal cone; "
            f"ORIGINAL RANSAC mask-center xyz/pose convention preserved; use_depth={int(use_depth)}"
        )
    logging.info(
        f"[depth] use_depth={int(use_depth)}, root={depth_root}, uint16_divisor=1000, "
        f"do_not_rescale_depth={args.do_not_rescale_depth}, "
        f"scale_anchor={'disabled_raw_div1000' if args.do_not_rescale_depth else 'masked_median_to_triangulated_camera_Z'}, "
        f"mask_only={args.depth_mask_only}"
    )
    logging.info(f"[temporal] cone={args.temporal_pose_cone_deg}deg, hypotheses={args.temporal_num_hypotheses}, batch_size={args.temporal_hypothesis_batch_size}")
    logging.info(f"[fast] enabled={bool(args.fast)}, later_frame_hypotheses={1 if args.fast else args.temporal_num_hypotheses}, later_frame_refine_iter={3 if args.fast else args.est_refine_iter}")
    logging.info(f"[mask_var_gate] enabled={args.mask_rotation_var_gate}, threshold={args.mask_rotation_var_threshold} deg^2, topn={args.mask_rotation_var_topn}")
    logging.info(f"[frames] start={start_frame}, end={end_frame}, total={end_frame - start_frame}")
    logging.info(f"[cams] all cameras: {reader.all_cam_folders}")
    logging.info(f"[io2] all cameras load masks only; FoundationPose RGB/depth topK={args.fast_consensus_topk}")
    logging.info(f"[main_cam] {main_cam}")
    logging.info(f"[mask_center_method] {args.mask_center_method}")
    if tracking_cam:
        logging.info(
            f"[camera_selection] FoundationPose camera is strictly locked to cam={tracking_cam}; "
            "area ranking is bypassed by the lock. Triangulation still uses all valid cameras."
        )
    else:
        logging.info(
            "[camera_selection] rank views ONLY by object-mask foreground area (pixel count); "
            "brightness is not used"
        )

    with open(log_csv, "w", newline="") as f_log:
        writer = csv.DictWriter(
            f_log,
            fieldnames=[
                "frame_idx",
                "frame_id",
                "status",
                "selected_cam",
                "selected_mask_area",
                "selected_mask_total_brightness",
                "selected_mask_mean_brightness",
                "registration_mode",
                "temporal_score",
                "temporal_cone_deg",
                "temporal_num_hypotheses",
                "rotation_avg_var_deg2",
                "mask_area_rank",
                "mask_var_threshold_deg2",
                "mask_var_selection_reason",
                "mask_var_trials",
                "num_valid_cams",
                "all_cam_mask_areas",
                "all_cam_mask_total_brightness",
                "all_cam_mask_mean_brightness",
                "mask_center_method",
                "triangulation_best_pair",
                "triangulation_inlier_cams",
                "triangulation_num_inliers",
                "triangulation_mean_error",
                "object_xyz_world",
                "object_xyz_selected_cam",
                "pose_world_tx",
                "pose_world_ty",
                "pose_world_tz",
                "pose_world_qx",
                "pose_world_qy",
                "pose_world_qz",
                "pose_world_qw",
                "pose_world_rx_deg",
                "pose_world_ry_deg",
                "pose_world_rz_deg",
                "kalman_enabled",
                "kalman_dt",
                "kalman_center_x",
                "kalman_center_y",
                "kalman_center_z",
                "kalman_vx",
                "kalman_vy",
                "kalman_vz",
                "kalman_rx_deg",
                "kalman_ry_deg",
                "kalman_rz_deg",
                "kalman_wx_deg_s",
                "kalman_wy_deg_s",
                "kalman_wz_deg_s",
                "message",
            ],
        )
        writer.writeheader()

        for frame_idx in range(start_frame, end_frame):
            frame_id = reader.id_strs[frame_idx]
            logging.info(f"========== frame_idx={frame_idx}, frame_id={frame_id} ==========")

            try:
                # ----------------------------------------------------------
                # Stage 1: ALL cameras -> MASK ONLY -> geometry/RANSAC.
                # No RGB or depth is decoded here.
                # ----------------------------------------------------------
                geometry_observations = load_all_camera_observations(
                    reader=reader,
                    frame_idx=frame_idx,
                    K_by_cam=K_by_cam,
                    min_mask_area=args.min_mask_area,
                    mask_center_method=args.mask_center_method,
                    use_depth=False,
                    depth_root=None,
                    require_depth=False,
                )

                geometry_by_cam = sorted(
                    geometry_observations, key=lambda x: _camera_sort_key(x["cam"])
                )
                mask_area_summary = ";".join(
                    [f"{obs['cam']}:{obs['mask_area']}" for obs in geometry_by_cam]
                )

                triangulation = ransac_triangulate_object_center(
                    observations=geometry_observations,
                    cam_ext=cam_ext,
                    reproj_thresh=args.ransac_reproj_thresh,
                    min_inliers=args.min_triangulation_inliers,
                    min_pair_angle_deg=args.min_pair_angle_deg,
                    final_method=args.final_triangulation_method,
                )
                logging.info(
                    f"[triangulation] frame={frame_id}, xyz_world={triangulation['xyz_world'].tolist()}, "
                    f"best_pair={triangulation['best_pair_cams']}, "
                    f"pair_angle={triangulation.get('best_pair_angle_deg', np.nan):.3f}deg, "
                    f"inliers={triangulation['final_inlier_cams']}, "
                    f"method={triangulation.get('final_method', '')}, "
                    f"mean_error={triangulation['final_mean_inlier_error']:.3f}"
                )

                # ----------------------------------------------------------
                # Stage 2: ONLY top-K mask-area cameras -> decode RGB/depth.
                # Keep existing FAST defaults unchanged:
                #   fast_consensus_topk=10
                #   fast_consensus_min_cluster=4
                # ----------------------------------------------------------
                io_topk = int(getattr(args, "fast_consensus_topk", 10))
                observations = hydrate_topk_foundationpose_observations(
                    observations=geometry_observations,
                    reader=reader,
                    frame_idx=frame_idx,
                    topk=io_topk,
                    use_depth=use_depth,
                    depth_root=depth_root,
                    require_depth=(args.require_depth and use_depth),
                    tracking_cam=getattr(args, "tracking_cam", None),
                )
                if len(observations) == 0:
                    raise RuntimeError(
                        f"No top-{io_topk} RGB/depth observation could be loaded for FoundationPose."
                    )

                # Brightness is now intentionally computed ONLY for the hydrated top-K
                # views. Non-top-K cameras are represented as NA in these debug CSV fields.
                loaded_by_cam = {str(obs["cam"]): obs for obs in observations}
                mask_total_brightness_summary = ";".join(
                    [
                        (
                            f"{obs['cam']}:{loaded_by_cam[str(obs['cam'])]['mask_total_brightness']:.3f}"
                            if str(obs["cam"]) in loaded_by_cam
                            else f"{obs['cam']}:NA"
                        )
                        for obs in geometry_by_cam
                    ]
                )
                mask_mean_brightness_summary = ";".join(
                    [
                        (
                            f"{obs['cam']}:{loaded_by_cam[str(obs['cam'])]['mask_mean_brightness']:.3f}"
                            if str(obs["cam"]) in loaded_by_cam
                            else f"{obs['cam']}:NA"
                        )
                        for obs in geometry_by_cam
                    ]
                )

                if last_pose_world is None:
                    # First processed frame is ALWAYS the normal full FoundationPose register,
                    # including when --fast is enabled.
                    selected_obs, pose_selected, init_xyz_selected = register_from_best_mask_camera(
                        est=est,
                        observations=observations,
                        triangulation=triangulation,
                        cam_ext=cam_ext,
                        args=args,
                    )
                    registration_info = {
                        "mode": "first_full_all_direction",
                        "score": "",
                        "cone_deg": "",
                        "num_hypotheses": "",
                    }
                    if not args.no_distribution_vis:
                        _draw_distribution_placeholder(
                            os.path.join(args.distribution_vis_dir, f"{frame_id}.png"),
                            frame_id,
                            "The first frame uses normal full all-direction FoundationPose registration.",
                        )
                        _draw_distribution_placeholder(
                            os.path.join(args.distribution_vis_dir, f"{frame_id}_3d.png"),
                            frame_id,
                            "The first frame uses normal full all-direction FoundationPose registration.",
                        )

                elif bool(args.fast):
                    selected_obs, pose_selected, init_xyz_selected, registration_info = register_from_best_mask_camera_fast(
                        est=est,
                        observations=observations,
                        triangulation=triangulation,
                        cam_ext=cam_ext,
                        last_pose_world=last_pose_world,
                        args=args,
                    )
                    # Fast mode intentionally has no rotation-distribution plot because it
                    # contains exactly one initialized/refined rotation, not a hypothesis cloud.

                elif args.no_temporal_cone:
                    selected_obs, pose_selected, init_xyz_selected = register_from_best_mask_camera(
                        est=est,
                        observations=observations,
                        triangulation=triangulation,
                        cam_ext=cam_ext,
                        args=args,
                    )
                    registration_info = {
                        "mode": "full_all_direction_temporal_disabled",
                        "score": "",
                        "cone_deg": "",
                        "num_hypotheses": "",
                    }
                    if not args.no_distribution_vis:
                        _draw_distribution_placeholder(
                            os.path.join(args.distribution_vis_dir, f"{frame_id}.png"),
                            frame_id,
                            "Temporal-cone mode was disabled for this frame (--no_temporal_cone).",
                        )
                        _draw_distribution_placeholder(
                            os.path.join(args.distribution_vis_dir, f"{frame_id}_3d.png"),
                            frame_id,
                            "Temporal-cone mode was disabled for this frame (--no_temporal_cone).",
                        )

                else:
                    selected_obs, pose_selected, init_xyz_selected, registration_info = register_from_best_mask_camera_temporal_cone(
                        est=est,
                        observations=observations,
                        triangulation=triangulation,
                        cam_ext=cam_ext,
                        last_pose_world=last_pose_world,
                        args=args,
                    )

                pose_selected_raw = pose_to_numpy_4x4(pose_selected).astype(np.float64)

                # Raw ORIGINAL-MESH pose -> WORLD.
                pose_world_raw = convert_pose_to_world(
                    pose_cam=pose_selected_raw,
                    cam=selected_obs["cam"],
                    cam_ext=cam_ext,
                ).astype(np.float64)

                # Convert to the FoundationPose CENTERED object pose. There is deliberately
                # NO online XYZ Kalman here; raw/consensus-fused refined XYZ is saved first.
                # pose_original = pose_centered @ tf_to_centered_mesh.
                pose_centered_world_raw = (
                    pose_world_raw
                    @ np.linalg.inv(np.asarray(tf_to_centered_mesh_vis, dtype=np.float64).reshape(4, 4))
                )

                if pose_kalman is not None:
                    pose_centered_world, kalman_info = pose_kalman.update(
                        pose_centered_world_measurement=pose_centered_world_raw,
                        frame_idx=frame_idx,
                    )
                else:
                    pose_centered_world = pose_centered_world_raw.copy()
                    raw_rvec = rotation_matrix_to_rotvec_rad(pose_centered_world_raw[:3, :3])
                    kalman_info = {
                        "dt": 0.0,
                        "measurement_xyz": pose_centered_world_raw[:3, 3].copy(),
                        "filtered_xyz": pose_centered_world_raw[:3, 3].copy(),
                        "filtered_velocity": np.zeros(3, dtype=np.float64),
                        "measurement_rotvec_rad": raw_rvec.copy(),
                        "filtered_rotvec_rad": raw_rvec.copy(),
                        "filtered_omega_rad_s": np.zeros(3, dtype=np.float64),
                    }

                # RAW centered pose -> ORIGINAL-MESH WORLD pose for the first-pass output.
                # It will be XYZ-interpolated and Kalman-filtered only after the full sequence.
                pose_world = (
                    pose_centered_world
                    @ np.asarray(tf_to_centered_mesh_vis, dtype=np.float64).reshape(4, 4)
                ).astype(np.float64)

                selected_cam = str(selected_obs["cam"])
                if selected_cam not in cam_ext:
                    raise KeyError(f"selected cam {selected_cam} is missing in extri.yml")
                if main_cam not in cam_ext:
                    raise KeyError(f"main cam {main_cam} is missing in extri.yml")

                pose_selected_np = (
                    np.asarray(cam_ext[selected_cam]["ext"], dtype=np.float64).reshape(4, 4)
                    @ pose_world
                ).astype(np.float32)
                pose_main = (
                    np.asarray(cam_ext[main_cam]["ext"], dtype=np.float64).reshape(4, 4)
                    @ pose_world
                ).astype(np.float32)
                pose_world = pose_world.astype(np.float32)

                q_world_xyzw = rotation_matrix_to_quaternion_xyzw(pose_world[:3, :3])
                euler_world_xyz_deg = rotation_matrix_to_euler_xyz_deg(pose_world[:3, :3])

                k_xyz = np.asarray(kalman_info["filtered_xyz"], dtype=np.float64).reshape(3)
                k_vel = np.asarray(kalman_info["filtered_velocity"], dtype=np.float64).reshape(3)
                k_rvec_deg = np.degrees(np.asarray(kalman_info["filtered_rotvec_rad"], dtype=np.float64).reshape(3))
                k_omega_deg_s = np.degrees(np.asarray(kalman_info["filtered_omega_rad_s"], dtype=np.float64).reshape(3))

                logging.info(
                    f"[xyz_raw_online_no_kalman] frame={frame_id}, online_kalman=0, "
                    f"xyz_raw={pose_centered_world_raw[:3, 3].tolist()}, xyz_firstpass={k_xyz.tolist()}, "
                    f"vxyz={k_vel.tolist()}, rotation_raw_unfiltered_rvec_deg={k_rvec_deg.tolist()}"
                )

                np.savetxt(
                    os.path.join(debug_dir, "ob_in_selected_cam", f"{frame_id}.txt"),
                    pose_selected_np.reshape(4, 4),
                )
                np.savetxt(
                    os.path.join(debug_dir, "ob_in_cam", f"{frame_id}.txt"),
                    pose_main.reshape(4, 4),
                )
                np.savetxt(
                    os.path.join(debug_dir, "ob_in_world", f"{frame_id}.txt"),
                    pose_world.reshape(4, 4),
                )

                writer.writerow({
                    "frame_idx": frame_idx,
                    "frame_id": frame_id,
                    "status": "ok",
                    "selected_cam": selected_obs["cam"],
                    "selected_mask_area": selected_obs["mask_area"],
                    "selected_mask_total_brightness": f"{selected_obs['mask_total_brightness']:.6f}",
                    "selected_mask_mean_brightness": f"{selected_obs['mask_mean_brightness']:.6f}",
                    "registration_mode": registration_info.get("mode", ""),
                    "temporal_score": registration_info.get("score", ""),
                    "temporal_cone_deg": registration_info.get("cone_deg", ""),
                    "temporal_num_hypotheses": registration_info.get("num_hypotheses", ""),
                    "rotation_avg_var_deg2": registration_info.get("rotation_avg_var", ""),
                    "mask_area_rank": registration_info.get("mask_area_rank", ""),
                    "mask_var_threshold_deg2": registration_info.get("mask_var_threshold", ""),
                    "mask_var_selection_reason": registration_info.get("mask_var_selection_reason", ""),
                    "mask_var_trials": registration_info.get("mask_var_trials", ""),
                    "num_valid_cams": len(geometry_observations),
                    "all_cam_mask_areas": mask_area_summary,
                    "all_cam_mask_total_brightness": mask_total_brightness_summary,
                    "all_cam_mask_mean_brightness": mask_mean_brightness_summary,
                    "mask_center_method": args.mask_center_method,
                    "triangulation_best_pair": "-".join(triangulation["best_pair_cams"]),
                    "triangulation_inlier_cams": ";".join(triangulation["final_inlier_cams"]),
                    "triangulation_num_inliers": triangulation["final_num_inliers"],
                    "triangulation_mean_error": triangulation["final_mean_inlier_error"],
                    "object_xyz_world": ",".join([f"{x:.8f}" for x in triangulation["xyz_world"]]),
                    "object_xyz_selected_cam": ",".join([f"{x:.8f}" for x in init_xyz_selected]),
                    "pose_world_tx": f"{pose_world[0, 3]:.8f}",
                    "pose_world_ty": f"{pose_world[1, 3]:.8f}",
                    "pose_world_tz": f"{pose_world[2, 3]:.8f}",
                    "pose_world_qx": f"{q_world_xyzw[0]:.10f}",
                    "pose_world_qy": f"{q_world_xyzw[1]:.10f}",
                    "pose_world_qz": f"{q_world_xyzw[2]:.10f}",
                    "pose_world_qw": f"{q_world_xyzw[3]:.10f}",
                    "pose_world_rx_deg": f"{euler_world_xyz_deg[0]:.8f}",
                    "pose_world_ry_deg": f"{euler_world_xyz_deg[1]:.8f}",
                    "pose_world_rz_deg": f"{euler_world_xyz_deg[2]:.8f}",
                    "kalman_enabled": int(pose_kalman is not None),
                    "kalman_dt": f"{float(kalman_info['dt']):.8f}",
                    "kalman_center_x": f"{k_xyz[0]:.8f}",
                    "kalman_center_y": f"{k_xyz[1]:.8f}",
                    "kalman_center_z": f"{k_xyz[2]:.8f}",
                    "kalman_vx": f"{k_vel[0]:.8f}",
                    "kalman_vy": f"{k_vel[1]:.8f}",
                    "kalman_vz": f"{k_vel[2]:.8f}",
                    "kalman_rx_deg": f"{k_rvec_deg[0]:.8f}",
                    "kalman_ry_deg": f"{k_rvec_deg[1]:.8f}",
                    "kalman_rz_deg": f"{k_rvec_deg[2]:.8f}",
                    "kalman_wx_deg_s": f"{k_omega_deg_s[0]:.8f}",
                    "kalman_wy_deg_s": f"{k_omega_deg_s[1]:.8f}",
                    "kalman_wz_deg_s": f"{k_omega_deg_s[2]:.8f}",
                    "message": "",
                })
                f_log.flush()

                logging.info(
                    f"[ok] frame={frame_id}, selected_cam={selected_obs['cam']}, "
                    f"mask_total_brightness={selected_obs['mask_total_brightness']:.3f}, "
                    f"mask_mean_brightness={selected_obs['mask_mean_brightness']:.3f}, "
                    f"mask_area={selected_obs['mask_area']}, "
                    f"mode={registration_info.get('mode', '')}, "
                    f"avg_var={registration_info.get('rotation_avg_var', 'n/a')}, "
                    f"area_rank={registration_info.get('mask_area_rank', 'n/a')}, "
                    f"tri_inliers={triangulation['final_num_inliers']}, saved main-cam/world pose"
                )

                # Use the accepted RAW centered-object WORLD pose as the next-frame temporal
                # rotation prior. XYZ is irrelevant to the rotation prior, and no online Kalman exists.
                last_pose_world = pose_centered_world.astype(np.float64).copy()

                # Tracking-pass visualization intentionally disabled for speed.
                # No per-frame main PNG, grid PNG/video, main MP4, GUI window, or
                # intermediate render is produced here. Pose txt/CSV outputs are unchanged.

            except Exception as e:
                logging.exception(f"[fail] frame={frame_id}: {e}")
                writer.writerow({
                    "frame_idx": frame_idx,
                    "frame_id": frame_id,
                    "status": "fail",
                    "selected_cam": "",
                    "selected_mask_area": "",
                    "selected_mask_total_brightness": "",
                    "selected_mask_mean_brightness": "",
                    "registration_mode": "",
                    "temporal_score": "",
                    "temporal_cone_deg": "",
                    "temporal_num_hypotheses": "",
                    "rotation_avg_var_deg2": "",
                    "mask_area_rank": "",
                    "mask_var_threshold_deg2": "",
                    "mask_var_selection_reason": "",
                    "mask_var_trials": "",
                    "num_valid_cams": 0,
                    "all_cam_mask_areas": "",
                    "all_cam_mask_total_brightness": "",
                    "all_cam_mask_mean_brightness": "",
                    "mask_center_method": args.mask_center_method,
                    "triangulation_best_pair": "",
                    "triangulation_inlier_cams": "",
                    "triangulation_num_inliers": "",
                    "triangulation_mean_error": "",
                    "object_xyz_world": "",
                    "object_xyz_selected_cam": "",
                    "pose_world_tx": "",
                    "pose_world_ty": "",
                    "pose_world_tz": "",
                    "pose_world_qx": "",
                    "pose_world_qy": "",
                    "pose_world_qz": "",
                    "pose_world_qw": "",
                    "pose_world_rx_deg": "",
                    "pose_world_ry_deg": "",
                    "pose_world_rz_deg": "",
                    "message": str(e),
                })
                f_log.flush()
                continue

    if video_writer is not None:
        video_writer.release()
        logging.info(f"[video] saved video: {video_path}")

    if grid_video_writer is not None:
        grid_video_writer.release()
        logging.info(f"[video] saved fused grid video: {grid_video_path}")

    if not args.no_window:
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # OFFLINE XYZ postprocess. There was NO XYZ Kalman in the tracking loop.
    # Strict order: raw refined/consensus-fused XYZ -> velocity interpolation -> forward XYZ Kalman.
    # The standard ob_in_* folders are rewritten with this FINAL translation so all existing
    # downstream rotation repair operates on the repaired XYZ without any extra wiring.
    # ------------------------------------------------------------------
    xyzpost_result = postprocess_xyz_velocity_interp_then_kalman(
        debug_dir=debug_dir,
        log_csv=log_csv,
        cam_ext=cam_ext,
        main_cam=main_cam,
        tf_to_centered_mesh=tf_to_centered_mesh_vis,
        fps=args.xyz_fix_fps,
        speed_cap_m_s=args.xyz_speed_cap_m_s,
        k_mad=args.xyz_fix_k_mad,
        window=args.xyz_fix_window,
        gap_merge=args.xyz_fix_gap_merge,
        max_burst_frames=args.xyz_fix_max_burst_frames,
        use_kalman=bool(args.use_kalman),
        kalman_pos_meas_std=args.kalman_pos_meas_std,
        kalman_pos_accel_std=args.kalman_pos_accel_std,
    )

    # IMPORTANT: all rotation post-processing runs only AFTER the complete FoundationPose pass
    # AND AFTER the XYZ interpolation+Kalman stage, so rotation repair never feeds back into
    # fast tracking/fallback decisions.
    # Order is strictly: raw poses -> XYZ interpolation -> XYZ Kalman -> SLERP rotfix
    # -> optional forward SO(3) postkal -> visualization.
    rotfix_result = None
    postkal_result = None
    postvis_result = None
    if bool(args.rotation_fix):
        rotfix_result = postprocess_rotation_spikes(
            debug_dir=debug_dir,
            log_csv=log_csv,
            cam_ext=cam_ext,
            main_cam=main_cam,
            tf_to_centered_mesh=tf_to_centered_mesh_vis,
            fps=args.rotation_fix_fps,
            window=args.rotation_fix_window,
            k_mad=args.rotation_fix_k_mad,
            min_threshold_deg_s=args.rotation_fix_min_threshold_deg_s,
            cap_threshold_deg_s=args.rotation_fix_cap_threshold_deg_s,
            max_burst_frames=args.rotation_fix_max_burst_frames,
            max_anchor_avg_deg_s=args.rotation_fix_max_anchor_avg_deg_s,
        )

        final_post_log = os.path.join(debug_dir, "register_every_frame_log_rotfix.csv")
        final_post_world_dir = os.path.join(debug_dir, "ob_in_world_rotfix")
        final_post_tag = "rotfix"
        final_stage_label = "POST-ROTATION-FIX"

        if bool(args.postkal):
            postkal_result = postprocess_rotation_kalman(
                debug_dir=debug_dir,
                cam_ext=cam_ext,
                main_cam=main_cam,
                tf_to_centered_mesh=tf_to_centered_mesh_vis,
                fps=args.postkal_fps,
                measurement_std_deg=args.postkal_rot_meas_std_deg,
                angular_accel_std_deg_s2=args.postkal_rot_accel_std_deg_s2,
                initial_omega_std_deg_s=args.postkal_initial_omega_std_deg_s,
            )
            final_post_log = postkal_result["log_csv"]
            final_post_world_dir = postkal_result["world_dir"]
            final_post_tag = "rotfix_postkal"
            final_stage_label = "POST-ROTATION-KALMAN"
        else:
            logging.info("[postkal] disabled; final rotation poses are SLERP rotfix outputs")

        # Re-render visualization ONLY AFTER the final requested post-processing stage.
        if bool(args.postprocess_vis):
            postvis_result = regenerate_postprocessed_bbox_videos(
                debug_dir=debug_dir,
                reader=reader,
                cam_ext=cam_ext,
                K_by_cam=K_by_cam,
                main_cam=main_cam,
                tf_to_centered_mesh=tf_to_centered_mesh_vis,
                bbox_centered=bbox_centered_vis,
                fps=args.save_fps,
                grid_cols=args.grid_cols,
                grid_cell_width=args.grid_cell_width,
                save_frames=args.postprocess_vis_save_frames,
                source_log_csv=final_post_log,
                source_world_dir=final_post_world_dir,
                output_tag=final_post_tag,
                stage_label=final_stage_label,
            )
        else:
            logging.info("[postvis] disabled by --postprocess_vis false")
    else:
        logging.info("[rotfix] disabled by --rotation_fix false")
        if bool(args.postkal):
            logging.warning("[postkal] requested but skipped because --rotation_fix is false; postkal is defined after SLERP rotfix")
        if bool(args.postprocess_vis):
            logging.info("[postvis] skipped because --rotation_fix is false and no rotfix/postkal poses exist")

    logging.info(f"[done] main-camera poses saved to: {os.path.join(debug_dir, 'ob_in_cam')}")
    logging.info(f"[done] selected-camera poses saved to: {os.path.join(debug_dir, 'ob_in_selected_cam')}")
    logging.info(f"[done] world-coordinate poses saved to: {os.path.join(debug_dir, 'ob_in_world')}")
    if not args.no_grid_vis:
        logging.info(f"[done] fused all-camera grid frames saved to: {os.path.join(debug_dir, 'register_vis_grid')}")
    if not args.no_distribution_vis:
        logging.info(f"[done] per-frame rotation distributions saved to: {args.distribution_vis_dir}")
        logging.info(f"[done] additional 3D rotation-ball figures saved as: <frame_id>_3d.png under {args.distribution_vis_dir}")
    logging.info(f"[done] raw log saved to: {log_csv}")
    if bool(args.rotation_fix):
        logging.info(f"[done] rotation-fixed log saved to: {os.path.join(debug_dir, 'register_every_frame_log_rotfix.csv')}")
        logging.info(f"[done] rotation-fixed WORLD poses saved to: {os.path.join(debug_dir, 'ob_in_world_rotfix')}")
        logging.info(f"[done] rotation-fixed main-camera poses saved to: {os.path.join(debug_dir, 'ob_in_cam_rotfix')}")
        logging.info(f"[done] rotation-fixed selected-camera poses saved to: {os.path.join(debug_dir, 'ob_in_selected_cam_rotfix')}")
        if bool(args.postkal) and postkal_result is not None:
            logging.info(f"[done] POSTKAL log saved to: {postkal_result['log_csv']}")
            logging.info(f"[done] POSTKAL WORLD poses saved to: {postkal_result['world_dir']}")
            logging.info(f"[done] POSTKAL main-camera poses saved to: {postkal_result['main_dir']}")
            logging.info(f"[done] POSTKAL selected-camera poses saved to: {postkal_result['selected_dir']}")
        if postvis_result is not None:
            logging.info(f"[done] FINAL postprocessed main-camera bbox video: {postvis_result['main_video_path']}")
            logging.info(f"[done] FINAL postprocessed all-camera bbox video: {postvis_result['grid_video_path']}")


if __name__ == "__main__":
    main()
