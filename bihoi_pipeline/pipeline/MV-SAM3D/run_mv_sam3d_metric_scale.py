#!/usr/bin/env python3
"""
MV-SAM3D -> metric-scale mesh using 8-view mesh generation + 31-camera depth + calibrated cameras.

Expected dataset layout:
    ../<setting_name>/
        images1/
            1.png (or .jpg/.jpeg/...)
            ...
            31.png
        mask/
            1.png (or 1_mask.png, ...)
            ...
            31.png
        depth/
            1/
                <first_depth_file>
            ...
            31/
                <first_depth_file>
        intri.yml
        extri.yml
        meshes/
            compressed.ply   <-- output

Run from the MV-SAM3D repository root, e.g.
    CUDA_VISIBLE_DEVICES=0 python run_mv_sam3d_metric_scale.py \
        --setting_name my_sequence

Notes
-----
1) The output vertex coordinates are in metric units (meters) only if the input
   depth is metric (or --depth_scale correctly converts it to meters) and the
   extrinsic translation uses the same metric convention.
2) This script restores SCALE only. It deliberately keeps SAM3D's canonical
   object orientation/centering rather than forcing an ambiguous PCA rotation
   into the world frame.
3) It uses all valid masked first-frame depth observations, backprojects them
   with K, transforms them with world_to_cam R/T, and estimates an isotropic
   scale from robust PCA extents.
4) The final PLY is binary and is simplified until it is below --max_size_mb.
"""

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import trimesh
from PIL import Image
from loguru import logger


# -----------------------------------------------------------------------------
# Repository imports
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent

# Reuse torch.compile/Triton artifacts across repeated dataset runs.
_COMPILE_CACHE = REPO_ROOT / ".compile_cache_mv_sam3d"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_COMPILE_CACHE / "torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(_COMPILE_CACHE / "triton"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "notebook"))

from inference import Inference  # noqa: E402


IMAGE_EXTS = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
)
DEPTH_EXTS = (
    ".npy", ".npz", ".png", ".tif", ".tiff", ".exr"
)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def parse_cam_ids(text: str) -> List[str]:
    """
    Examples:
        "1-31" -> ["1", ..., "31"]
        "1,4,7" -> ["1","4","7"]
        "1-5,9,12-14"
    """
    out: List[str] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            a_i, b_i = int(a), int(b)
            step = 1 if b_i >= a_i else -1
            out.extend(str(i) for i in range(a_i, b_i + step, step))
        else:
            out.append(str(int(token)))
    # preserve order, remove duplicates
    return list(dict.fromkeys(out))


def first_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def find_cam_file(folder: Path, cam_id: str, is_mask: bool = False) -> Path:
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    stems = [
        cam_id,
        f"{int(cam_id):02d}",
        f"{int(cam_id):03d}",
        f"cam{cam_id}",
        f"cam_{cam_id}",
    ]
    if is_mask:
        stems += [
            f"{cam_id}_mask",
            f"{int(cam_id):02d}_mask",
            f"cam{cam_id}_mask",
            f"cam_{cam_id}_mask",
        ]

    candidates = [folder / f"{stem}{ext}" for stem in stems for ext in IMAGE_EXTS]
    found = first_existing(candidates)
    if found is not None:
        return found

    # Fallback: inspect all files and compare the first integer in the stem.
    matches = []
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        nums = re.findall(r"\d+", p.stem)
        if nums and int(nums[0]) == int(cam_id):
            if is_mask or "mask" not in p.stem.lower():
                matches.append(p)

    if matches:
        matches = sorted(matches, key=natural_key)
        return matches[0]

    raise FileNotFoundError(
        f"Cannot find {'mask' if is_mask else 'image'} for cam {cam_id} in {folder}"
    )


def find_first_depth_file(depth_cam_dir: Path) -> Path:
    if not depth_cam_dir.exists():
        raise FileNotFoundError(f"Missing depth camera folder: {depth_cam_dir}")

    files = [
        p for p in depth_cam_dir.iterdir()
        if p.is_file() and p.suffix.lower() in DEPTH_EXTS
    ]
    files = sorted(files, key=natural_key)
    if not files:
        raise FileNotFoundError(f"No supported depth file in {depth_cam_dir}")
    return files[0]


# -----------------------------------------------------------------------------
# Image / mask / depth IO
# -----------------------------------------------------------------------------
def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(path: Path, target_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    im = Image.open(path)
    arr = np.asarray(im)

    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] >= 4:
        # Prefer alpha for RGBA masks.
        alpha = arr[..., 3]
        if np.any(alpha > 0):
            gray = alpha
        else:
            gray = np.max(arr[..., :3], axis=-1)
    elif arr.ndim == 3:
        gray = np.max(arr[..., :3], axis=-1)
    else:
        raise ValueError(f"Unsupported mask shape {arr.shape}: {path}")

    if np.issubdtype(gray.dtype, np.floating):
        threshold = 0.5 if np.nanmax(gray) <= 1.0 else 127.0
    else:
        threshold = 0 if np.max(gray) <= 1 else 127

    mask = gray > threshold

    if target_hw is not None and mask.shape != target_hw:
        h, w = target_hw
        mask = cv2.resize(
            mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    return mask


def load_depth_raw(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        preferred = ["depth", "depth_map", "pred_depth", "arr_0"]
        key = next((k for k in preferred if k in data), None)
        if key is None:
            # Pick first 2D array.
            key = next(
                (k for k in data.files if np.asarray(data[k]).squeeze().ndim == 2),
                data.files[0],
            )
        depth = data[key]
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"cv2.imread failed for depth: {path}")

    depth = np.asarray(depth).squeeze()
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D after squeeze; got {depth.shape}: {path}")

    return depth


def infer_depth_multiplier_to_meters(
    depth_raw: np.ndarray,
    depth_scale: Optional[float],
    depth_unit: str,
) -> float:
    """
    Return multiplier such that depth_m = raw * multiplier.
    """
    if depth_scale is not None:
        return float(depth_scale)

    if depth_unit == "m":
        return 1.0
    if depth_unit == "mm":
        return 1e-3
    if depth_unit == "cm":
        return 1e-2

    valid = depth_raw[np.isfinite(depth_raw) & (depth_raw > 0)]
    if valid.size == 0:
        return 1.0

    med = float(np.median(valid))

    # Practical heuristic:
    # - float metric depth usually around 0.x ~ several meters
    # - uint16 depth is commonly millimeters (hundreds/thousands)
    if np.issubdtype(depth_raw.dtype, np.integer):
        if med > 20.0:
            return 1e-3
    if med > 20.0:
        logger.warning(
            f"Auto depth unit: median raw depth={med:.3f} > 20; assuming millimeters."
        )
        return 1e-3

    return 1.0


# -----------------------------------------------------------------------------
# OpenCV / EasyMocap calibration readers
# -----------------------------------------------------------------------------
def _read_fs_mat(fs: cv2.FileStorage, keys: Sequence[str]) -> Optional[np.ndarray]:
    for key in keys:
        node = fs.getNode(key)
        if node.empty():
            continue
        try:
            mat = node.mat()
        except cv2.error:
            mat = None
        if mat is not None:
            return np.asarray(mat, dtype=np.float64)
    return None


def load_camera_calibration(
    intri_path: Path,
    extri_path: Path,
    cam_id: str,
    extrinsic_translation_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fs_i = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_READ)
    if not fs_i.isOpened():
        raise RuntimeError(f"Cannot open intrinsics file: {intri_path}")

    K = _read_fs_mat(
        fs_i,
        [
            f"K_{cam_id}",
            f"K{cam_id}",
            f"intri_{cam_id}",
            f"intrinsic_{cam_id}",
        ],
    )
    fs_i.release()

    if K is None:
        raise KeyError(f"Cannot find K for cam {cam_id} in {intri_path}")
    K = K.reshape(3, 3)

    fs_e = cv2.FileStorage(str(extri_path), cv2.FILE_STORAGE_READ)
    if not fs_e.isOpened():
        raise RuntimeError(f"Cannot open extrinsics file: {extri_path}")

    R = _read_fs_mat(
        fs_e,
        [
            f"R_{cam_id}",
            f"R{cam_id}",
            f"Rot_{cam_id}",
            f"rotation_{cam_id}",
        ],
    )
    T = _read_fs_mat(
        fs_e,
        [
            f"T_{cam_id}",
            f"T{cam_id}",
            f"trans_{cam_id}",
            f"translation_{cam_id}",
        ],
    )
    fs_e.release()

    if R is None:
        raise KeyError(f"Cannot find R for cam {cam_id} in {extri_path}")
    if T is None:
        raise KeyError(f"Cannot find T for cam {cam_id} in {extri_path}")

    R = np.asarray(R, dtype=np.float64).squeeze()
    if R.size == 3:
        R, _ = cv2.Rodrigues(R.reshape(3))
    R = R.reshape(3, 3)

    T = np.asarray(T, dtype=np.float64).reshape(3)
    T = T * float(extrinsic_translation_scale)

    return K, R, T


def scaled_intrinsics_for_depth(
    K: np.ndarray,
    depth_wh: Tuple[int, int],
    image_wh: Tuple[int, int],
    calib_width: Optional[int],
    calib_height: Optional[int],
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Scale K from its calibration/reference image size to depth resolution.

    Priority:
      1. --calib_width/--calib_height
      2. If K principal point is compatible with images1 dimensions, use images1 size
      3. If K principal point is compatible with depth dimensions, assume already matched
      4. Estimate reference size as roughly (2*cx, 2*cy)
    """
    target_w, target_h = depth_wh
    image_w, image_h = image_wh

    K2 = K.copy()
    cx, cy = float(K[0, 2]), float(K[1, 2])

    if calib_width is not None and calib_height is not None:
        src_w, src_h = float(calib_width), float(calib_height)
        source_reason = "explicit --calib_width/--calib_height"
    else:
        image_matches = (
            0.20 * image_w <= cx <= 0.80 * image_w
            and 0.20 * image_h <= cy <= 0.80 * image_h
        )
        depth_matches = (
            0.20 * target_w <= cx <= 0.80 * target_w
            and 0.20 * target_h <= cy <= 0.80 * target_h
        )

        if image_matches:
            src_w, src_h = float(image_w), float(image_h)
            source_reason = "K principal point matches images1 resolution"
        elif depth_matches:
            src_w, src_h = float(target_w), float(target_h)
            source_reason = "K already appears to match depth resolution"
        else:
            src_w = max(1.0, 2.0 * cx)
            src_h = max(1.0, 2.0 * cy)
            source_reason = "estimated from ~2*principal_point"
            logger.warning(
                f"K does not clearly match image/depth size. "
                f"Inferring calibration size ~= ({src_w:.1f}, {src_h:.1f}). "
                f"For exact behavior pass --calib_width/--calib_height."
            )

    sx = float(target_w) / src_w
    sy = float(target_h) / src_h

    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy

    logger.debug(
        f"Intrinsic scaling: source=({src_w:.1f},{src_h:.1f}) "
        f"-> depth=({target_w},{target_h}), "
        f"sx={sx:.6f}, sy={sy:.6f}, reason={source_reason}"
    )

    return K2, (sx, sy), (src_w, src_h)


# -----------------------------------------------------------------------------
# SAM3D inference
# -----------------------------------------------------------------------------
def load_multiview_inputs(
    images_dir: Path,
    masks_dir: Path,
    cam_ids: Sequence[str],
    io_workers: int = 8,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Path], Dict[str, Path]]:
    """Load camera RGB/masks in parallel while preserving camera order."""
    def load_one(cam_id: str):
        ip = find_cam_file(images_dir, cam_id, is_mask=False)
        mp = find_cam_file(masks_dir, cam_id, is_mask=True)
        rgb = load_rgb(ip)
        mask = load_mask(mp, target_hw=rgb.shape[:2])
        if mask.sum() == 0:
            raise ValueError(f"Mask is empty for cam {cam_id}: {mp}")
        return cam_id, rgb, mask, ip, mp

    workers = max(1, min(int(io_workers), len(cam_ids)))
    if workers == 1:
        rows = [load_one(c) for c in cam_ids]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mv-input") as pool:
            rows = list(pool.map(load_one, cam_ids))

    images: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    image_paths: Dict[str, Path] = {}
    mask_paths: Dict[str, Path] = {}

    for cam_id, rgb, mask, ip, mp in rows:
        images.append(rgb)
        masks.append(mask)
        image_paths[cam_id] = ip
        mask_paths[cam_id] = mp
        logger.info(
            f"[Input cam {cam_id}] image={ip.name} {rgb.shape[1]}x{rgb.shape[0]}, "
            f"mask={mp.name}, foreground={mask.mean()*100:.2f}%"
        )
    return images, masks, image_paths, mask_paths


def run_mv_sam3d(
    view_images: List[np.ndarray],
    view_masks: List[np.ndarray],
    model_tag: str,
    seed: int,
    stage1_steps: int,
    stage2_steps: int,
    generate_texture: bool = False,
    compile_model: bool = False,
):
    config_path = REPO_ROOT / "checkpoints" / model_tag / "pipeline.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"MV-SAM3D config not found: {config_path}")

    logger.info(f"Loading MV-SAM3D model from {config_path}")
    logger.info(f"torch.compile model path: {bool(compile_model)}")

    try:
        import torch
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    # Some checkpoint/config references in the repo are relative to repository root.
    # Dataset/output paths were already resolved to absolute paths in main().
    os.chdir(REPO_ROOT)
    if compile_model:
        try:
            inference = Inference(str(config_path), compile=True)
        except Exception as exc:
            # This MV-SAM3D PointMap revision has an incompatible compile warmup:
            # InferencePipelinePointMap._warmup() may call a missing run_layout_model().
            # Fall back automatically so a compile-path mismatch does not kill the job.
            msg = str(exc)
            if "run_layout_model" in msg or "InferencePipelinePointMap" in msg:
                logger.warning(
                    "MV-SAM3D torch.compile path is incompatible with this PointMap "
                    "pipeline revision (%s). Retrying with compile=False.",
                    msg,
                )
                try:
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                inference = Inference(str(config_path), compile=False)
            else:
                raise
    else:
        inference = Inference(str(config_path), compile=False)

    if hasattr(inference._pipeline, "rendering_engine"):
        if inference._pipeline.rendering_engine != "pytorch3d":
            logger.warning(
                f"rendering_engine={inference._pipeline.rendering_engine}; "
                f"switching to pytorch3d as in the repo's run_inference.py"
            )
            inference._pipeline.rendering_engine = "pytorch3d"

    logger.info(f"Running MV-SAM3D on {len(view_images)} views...")
    result = inference._pipeline.run_multi_view(
        view_images=view_images,
        view_masks=view_masks,
        seed=seed,
        mode="multidiffusion",
        stage1_inference_steps=stage1_steps,
        stage2_inference_steps=stage2_steps,
        # Repo postprocess_slat_output builds GLB from BOTH gaussian + mesh.
        decode_formats=["gaussian", "mesh"],
        with_mesh_postprocess=False,
        with_texture_baking=False,
        use_vertex_color=True,
    )

    if result.get("glb", None) is None:
        raise RuntimeError("MV-SAM3D returned no GLB mesh.")

    if generate_texture:
        logger.info("Baking optional 1024px UV texture from reconstructed Gaussian appearance...")
        textured = inference._pipeline.postprocess_slat_output(
            dict(result), with_mesh_postprocess=False,
            with_texture_baking=True, use_vertex_color=False,
        )
        result["textured_glb"] = textured["glb"]
    return result


def scene_or_mesh_to_trimesh(obj) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Trimesh):
        return obj.copy()

    if isinstance(obj, trimesh.Scene):
        # Newer trimesh
        if hasattr(obj, "to_mesh"):
            try:
                m = obj.to_mesh()
                if isinstance(m, trimesh.Trimesh):
                    return m
            except Exception:
                pass

        # Older trimesh; dump(concatenate=True) applies node transforms.
        dumped = obj.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh):
            return dumped
        if isinstance(dumped, (list, tuple)):
            meshes = [g for g in dumped if isinstance(g, trimesh.Trimesh)]
            if meshes:
                return trimesh.util.concatenate(meshes)

    raise TypeError(f"Cannot convert object of type {type(obj)} to trimesh.Trimesh")


def extract_sam3d_mesh(result: dict, work_dir: Path) -> trimesh.Trimesh:
    raw_glb = work_dir / "sam3d_raw.glb"
    result["glb"].export(str(raw_glb))

    loaded = trimesh.load(str(raw_glb), force="scene", process=False)
    mesh = scene_or_mesh_to_trimesh(loaded)

    if mesh.faces is None or len(mesh.faces) == 0:
        raise RuntimeError("Extracted SAM3D mesh has no faces.")

    # Version-safe cleanup.
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    logger.info(
        f"SAM3D canonical mesh: vertices={len(mesh.vertices):,}, "
        f"faces={len(mesh.faces):,}, bounds={mesh.extents}"
    )
    return mesh


# -----------------------------------------------------------------------------
# Depth -> fused metric point cloud
# -----------------------------------------------------------------------------
def backproject_to_world(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    R_w2c: np.ndarray,
    T_w2c: np.ndarray,
    pixel_stride: int,
    min_depth: float,
    max_depth: float,
    mask_erode: int,
    max_points_per_cam: int,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = depth_m.shape

    if mask.shape != (h, w):
        mask = cv2.resize(
            mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    if mask_erode > 0:
        k = 2 * mask_erode + 1
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    valid = (
        mask
        & np.isfinite(depth_m)
        & (depth_m > min_depth)
        & (depth_m < max_depth)
    )

    # Remove the most obvious depth spikes inside the object mask.
    z_valid = depth_m[valid]
    if z_valid.size >= 100:
        q_low, q_high = np.quantile(z_valid, [0.005, 0.995])
        valid &= (depth_m >= q_low) & (depth_m <= q_high)

    vv, uu = np.nonzero(valid)

    if pixel_stride > 1 and len(uu) > 0:
        stride_keep = ((uu % pixel_stride) == 0) & ((vv % pixel_stride) == 0)
        uu, vv = uu[stride_keep], vv[stride_keep]

    if len(uu) == 0:
        return np.empty((0, 3), dtype=np.float32)

    if len(uu) > max_points_per_cam:
        idx = rng.choice(len(uu), size=max_points_per_cam, replace=False)
        uu, vv = uu[idx], vv[idx]

    z = depth_m[vv, uu].astype(np.float64)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    x = (uu.astype(np.float64) - cx) * z / fx
    y = (vv.astype(np.float64) - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)

    # EasyMocap/OpenCV convention:
    # X_cam = R_w2c @ X_world + T_w2c
    # => X_world = R_w2c.T @ (X_cam - T_w2c)
    pts_world = (R_w2c.T @ (pts_cam - T_w2c[None, :]).T).T
    return pts_world.astype(np.float32)


def robust_point_extent(points: np.ndarray, q: float = 0.01):
    """
    Rotation-invariant-ish robust object extent:
    median-center -> PCA axes -> percentile extent.
    Returns sorted extents descending.
    """
    if len(points) < 20:
        raise ValueError("Too few points for extent estimation.")

    pts = np.asarray(points, dtype=np.float64)

    # Limit covariance cost while retaining robust coverage.
    if len(pts) > 300_000:
        rng = np.random.default_rng(12345)
        pts = pts[rng.choice(len(pts), 300_000, replace=False)]

    center = np.median(pts, axis=0)
    X = pts - center

    # Remove extreme radial outliers.
    r = np.linalg.norm(X, axis=1)
    cutoff = np.quantile(r, 0.995)
    X = X[r <= cutoff]
    pts2 = X + center

    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]

    proj = (pts2 - center) @ axes
    lo = np.quantile(proj, q, axis=0)
    hi = np.quantile(proj, 1.0 - q, axis=0)
    extents = np.maximum(hi - lo, 1e-9)

    extents_sorted = np.sort(extents)[::-1]
    return extents_sorted, center, axes


def build_metric_object_cloud(
    cam_ids: Sequence[str],
    image_paths: Dict[str, Path],
    mask_paths: Dict[str, Path],
    depth_root: Path,
    intri_path: Path,
    extri_path: Path,
    depth_scale: Optional[float],
    depth_unit: str,
    extrinsic_translation_scale: float,
    calib_width: Optional[int],
    calib_height: Optional[int],
    pixel_stride: int,
    min_depth: float,
    max_depth: float,
    mask_erode: int,
    max_points_per_cam: int,
    seed: int,
    workers: int = 8,
):
    """Fuse first-frame metric depth; camera I/O/backprojection runs in parallel."""

    def process_cam(item):
        order_idx, cam_id = item
        depth_path = find_first_depth_file(depth_root / cam_id)
        depth_raw = load_depth_raw(depth_path)
        multiplier = infer_depth_multiplier_to_meters(
            depth_raw, depth_scale=depth_scale, depth_unit=depth_unit
        )
        depth_m = depth_raw.astype(np.float32) * float(multiplier)

        # We need only the RGB dimensions here; reading the image header avoids
        # a full RGB decode for every camera.
        with Image.open(image_paths[cam_id]) as im:
            image_wh = im.size
        depth_wh = (depth_m.shape[1], depth_m.shape[0])

        mask = load_mask(mask_paths[cam_id], target_hw=depth_m.shape)

        K, R, T = load_camera_calibration(
            intri_path,
            extri_path,
            cam_id,
            extrinsic_translation_scale=extrinsic_translation_scale,
        )
        Kd, (sx, sy), calib_wh = scaled_intrinsics_for_depth(
            K,
            depth_wh=depth_wh,
            image_wh=image_wh,
            calib_width=calib_width,
            calib_height=calib_height,
        )

        # Deterministic independent RNG per camera, safe under threading.
        cam_rng = np.random.default_rng(int(seed) + 10007 * int(order_idx + 1))
        pts = backproject_to_world(
            depth_m=depth_m,
            mask=mask,
            K=Kd,
            R_w2c=R,
            T_w2c=T,
            pixel_stride=pixel_stride,
            min_depth=min_depth,
            max_depth=max_depth,
            mask_erode=mask_erode,
            max_points_per_cam=max_points_per_cam,
            rng=cam_rng,
        )

        if len(pts) < 50:
            return cam_id, None, None, (
                f"[Depth cam {cam_id}] only {len(pts)} valid object points; skipping."
            )

        center = np.median(pts, axis=0)
        local_extent, _, _ = robust_point_extent(pts, q=0.02)
        info = {
            "depth_file": str(depth_path),
            "depth_raw_dtype": str(depth_raw.dtype),
            "depth_multiplier_to_m": float(multiplier),
            "depth_resolution": [int(depth_wh[0]), int(depth_wh[1])],
            "image_resolution": [int(image_wh[0]), int(image_wh[1])],
            "calibration_reference_resolution": [
                float(calib_wh[0]), float(calib_wh[1])
            ],
            "intrinsic_scale_xy": [float(sx), float(sy)],
            "num_points": int(len(pts)),
            "world_median": center.tolist(),
            "robust_extent_m": local_extent.tolist(),
        }
        msg = (
            f"[Depth cam {cam_id}] {depth_path.name}, "
            f"points={len(pts):,}, multiplier={multiplier:g}, "
            f"center={np.round(center, 4)}, "
            f"extent~={np.round(local_extent, 4)} m"
        )
        return cam_id, pts, info, msg

    items = list(enumerate(cam_ids))
    n_workers = max(1, min(int(workers), len(items)))
    if n_workers == 1:
        rows = [process_cam(x) for x in items]
    else:
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="metric-depth") as pool:
            rows = list(pool.map(process_cam, items))

    per_cam_points: Dict[str, np.ndarray] = {}
    per_cam_info: Dict[str, dict] = {}
    for cam_id, pts, info, msg in rows:
        if pts is None:
            logger.warning(msg)
            continue
        per_cam_points[cam_id] = pts
        per_cam_info[cam_id] = info
        logger.info(msg)

    if len(per_cam_points) < 2:
        raise RuntimeError(
            f"Only {len(per_cam_points)} cameras produced usable depth. "
            f"Need at least 2; preferably many views."
        )

    valid_ids = list(per_cam_points.keys())
    centers = np.stack(
        [np.median(per_cam_points[c], axis=0) for c in valid_ids], axis=0
    )
    med_center = np.median(centers, axis=0)
    center_dist = np.linalg.norm(centers - med_center[None, :], axis=1)

    if len(valid_ids) >= 5:
        med_d = np.median(center_dist)
        mad_d = np.median(np.abs(center_dist - med_d)) + 1e-9
        robust_thr = med_d + 4.0 * 1.4826 * mad_d
        p80 = np.quantile(center_dist, 0.80)
        threshold = max(robust_thr, p80, 1e-4)
        accepted = [
            c for c, d in zip(valid_ids, center_dist) if d <= threshold
        ]
        if len(accepted) < max(3, int(0.7 * len(valid_ids))):
            accepted = valid_ids
    else:
        threshold = None
        accepted = valid_ids

    rejected = [c for c in valid_ids if c not in accepted]
    if rejected:
        logger.warning(
            f"Rejecting calibration/depth outlier camera(s) by object-center consistency: "
            f"{rejected}"
        )

    fused = np.concatenate([per_cam_points[c] for c in accepted], axis=0)

    rng = np.random.default_rng(seed)
    if len(fused) > 1_000_000:
        fused = fused[rng.choice(len(fused), 1_000_000, replace=False)]

    extent_m, center_m, axes = robust_point_extent(fused, q=0.01)

    logger.info(
        f"Fused metric point cloud: cameras={len(accepted)}/{len(cam_ids)}, "
        f"points={len(fused):,}"
    )
    logger.info(f"Fused robust PCA extent = {np.round(extent_m, 6)} meters")
    logger.info(f"Fused robust center     = {np.round(center_m, 6)} world units/meters")

    info = {
        "accepted_cameras": accepted,
        "rejected_cameras": rejected,
        "camera_center_rejection_threshold": (
            None if threshold is None else float(threshold)
        ),
        "fused_num_points": int(len(fused)),
        "fused_robust_extent_m": extent_m.tolist(),
        "fused_robust_center_world": center_m.tolist(),
        "per_camera": per_cam_info,
    }
    return fused, extent_m, info


# -----------------------------------------------------------------------------
# Metric scaling
# -----------------------------------------------------------------------------
def sample_mesh_surface_points(
    mesh: trimesh.Trimesh,
    count: int,
    seed: int,
) -> np.ndarray:
    np.random.seed(seed)
    n = max(20_000, min(int(count), max(20_000, len(mesh.faces) * 3)))
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts, dtype=np.float64)


def estimate_metric_scale(
    mesh: trimesh.Trimesh,
    real_extent_m: np.ndarray,
    mesh_sample_points: int,
    seed: int,
):
    mesh_pts = sample_mesh_surface_points(mesh, mesh_sample_points, seed)
    mesh_extent, mesh_center_robust, _ = robust_point_extent(mesh_pts, q=0.01)

    ratios = np.asarray(real_extent_m, dtype=np.float64) / np.asarray(
        mesh_extent, dtype=np.float64
    )

    # Sorted PCA extents are orientation-independent up to axis permutation.
    # Median across the three dimensions is robust to one imperfect/occluded axis.
    scale = float(np.median(ratios))

    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(
            f"Invalid scale={scale}; real_extent={real_extent_m}, mesh_extent={mesh_extent}"
        )

    logger.info(f"SAM3D robust canonical extent = {np.round(mesh_extent, 6)}")
    logger.info(f"Metric robust extent          = {np.round(real_extent_m, 6)} m")
    logger.info(f"Per-axis scale ratios         = {np.round(ratios, 6)}")
    logger.info(f"Chosen isotropic metric scale = {scale:.9f}")

    return scale, mesh_extent, ratios


def apply_isotropic_scale_about_center(
    mesh: trimesh.Trimesh,
    scale: float,
) -> trimesh.Trimesh:
    """
    Convert the complete canonical coordinate system to metric scale.

    We scale about the canonical origin, not about the mesh centroid.  This way
    every coordinate component (including any canonical offset from the origin)
    is expressed in the same metric scale after the transform.
    """
    out = mesh.copy()
    v = np.asarray(out.vertices, dtype=np.float64)
    out.vertices = v * float(scale)
    return out


# -----------------------------------------------------------------------------
# PLY export + strict file-size reduction
# -----------------------------------------------------------------------------
def get_vertex_colors(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
    try:
        colors = np.asarray(mesh.visual.vertex_colors)
        if len(colors) == len(mesh.vertices):
            return colors.copy()
    except Exception:
        pass
    return None


def transfer_vertex_colors(
    src: trimesh.Trimesh,
    dst: trimesh.Trimesh,
) -> trimesh.Trimesh:
    colors = get_vertex_colors(src)
    if colors is None or len(dst.vertices) == 0:
        return dst

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.asarray(src.vertices))
        _, idx = tree.query(np.asarray(dst.vertices), k=1)
        dst.visual.vertex_colors = colors[idx]
    except Exception as exc:
        logger.warning(f"Could not transfer vertex colors after simplification: {exc}")

    return dst


def export_binary_ply(mesh: trimesh.Trimesh, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    # Avoid carrying unreferenced vertices.
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        blob = trimesh.exchange.ply.export_ply(
            mesh,
            encoding="binary_little_endian",
            include_attributes=True,
        )
    except Exception:
        blob = trimesh.exchange.ply.export_ply(
            mesh,
            encoding="binary",
        )

    path.write_bytes(blob)


def simplify_qem(
    mesh: trimesh.Trimesh,
    target_faces: int,
) -> Optional[trimesh.Trimesh]:
    target_faces = int(max(4, target_faces))
    if target_faces >= len(mesh.faces):
        return mesh.copy()

    # First try trimesh wrapper.
    try:
        out = mesh.simplify_quadric_decimation(face_count=target_faces)
        if out is not None and len(out.faces) > 0:
            return transfer_vertex_colors(mesh, out)
    except Exception as exc:
        logger.debug(f"trimesh simplify_quadric_decimation unavailable: {exc}")

    # Direct fast-simplification fallback.
    try:
        import fast_simplification

        points, faces = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32),
            target_count=target_faces,
        )
        out = trimesh.Trimesh(
            vertices=np.asarray(points),
            faces=np.asarray(faces),
            process=False,
        )
        return transfer_vertex_colors(mesh, out)
    except Exception as exc:
        logger.debug(f"fast_simplification unavailable: {exc}")

    return None


def vertex_cluster_simplify(
    mesh: trimesh.Trimesh,
    target_vertices: int,
) -> trimesh.Trimesh:
    """
    Pure NumPy fallback. Less shape-faithful than QEM, but ensures we can still
    reduce file size when optional simplification packages are unavailable.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    colors = get_vertex_colors(mesh)

    if len(v) <= target_vertices:
        return mesh.copy()

    vmin = v.min(axis=0)
    ext = np.maximum(v.max(axis=0) - vmin, 1e-12)
    max_ext = float(np.max(ext))

    # Start from roughly cube-root grid resolution.
    cells_per_axis = max(2.0, float(target_vertices) ** (1.0 / 3.0))
    cell = max_ext / cells_per_axis

    best = None
    for _ in range(12):
        q = np.floor((v - vmin[None, :]) / max(cell, 1e-12)).astype(np.int64)
        _, inverse = np.unique(q, axis=0, return_inverse=True)
        n_new = int(inverse.max()) + 1

        counts = np.bincount(inverse, minlength=n_new).astype(np.float64)
        new_v = np.zeros((n_new, 3), dtype=np.float64)
        for d in range(3):
            new_v[:, d] = np.bincount(
                inverse, weights=v[:, d], minlength=n_new
            ) / np.maximum(counts, 1.0)

        new_f = inverse[f]
        nondeg = (
            (new_f[:, 0] != new_f[:, 1])
            & (new_f[:, 1] != new_f[:, 2])
            & (new_f[:, 0] != new_f[:, 2])
        )
        new_f = new_f[nondeg]

        if len(new_f) == 0:
            cell *= 0.8
            continue

        # Remove duplicate triangles ignoring winding for the duplicate test.
        canon = np.sort(new_f, axis=1)
        _, unique_idx = np.unique(canon, axis=0, return_index=True)
        new_f = new_f[np.sort(unique_idx)]

        out = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)

        if colors is not None:
            c = colors.astype(np.float64)
            new_c = np.zeros((n_new, c.shape[1]), dtype=np.float64)
            for d in range(c.shape[1]):
                new_c[:, d] = np.bincount(
                    inverse, weights=c[:, d], minlength=n_new
                ) / np.maximum(counts, 1.0)
            out.visual.vertex_colors = np.clip(new_c, 0, 255).astype(np.uint8)

        best = out
        if n_new <= target_vertices:
            break

        cell *= 1.25

    if best is None:
        raise RuntimeError("Vertex clustering simplification failed.")
    return best


def compress_mesh_below_size(
    mesh: trimesh.Trimesh,
    output_path: Path,
    max_size_mb: float,
    min_faces: int = 500,
) -> trimesh.Trimesh:
    target_bytes = int(float(max_size_mb) * 1_000_000)
    current = mesh.copy()

    export_binary_ply(current, output_path)
    size = output_path.stat().st_size

    logger.info(
        f"Initial binary PLY: {size/1e6:.3f} MB, "
        f"V={len(current.vertices):,}, F={len(current.faces):,}"
    )

    iteration = 0
    while size > target_bytes and len(current.faces) > min_faces:
        iteration += 1

        # Aim a bit below target to account for format overhead.
        ratio = max(0.05, min(0.90, (target_bytes / size) * 0.82))
        target_faces = max(min_faces, int(len(current.faces) * ratio))

        logger.info(
            f"[Compress {iteration}] target_faces={target_faces:,} "
            f"(ratio={ratio:.3f})"
        )

        simplified = simplify_qem(current, target_faces)

        if simplified is None or len(simplified.faces) >= len(current.faces):
            # Portable fallback.
            target_vertices = max(
                300,
                int(len(current.vertices) * ratio * 0.8),
            )
            logger.warning(
                f"QEM simplifier unavailable/ineffective; "
                f"using vertex clustering target_vertices={target_vertices:,}"
            )
            simplified = vertex_cluster_simplify(
                current, target_vertices=target_vertices
            )

        current = simplified

        try:
            current.remove_unreferenced_vertices()
        except Exception:
            pass

        export_binary_ply(current, output_path)
        new_size = output_path.stat().st_size

        logger.info(
            f"[Compress {iteration}] {new_size/1e6:.3f} MB, "
            f"V={len(current.vertices):,}, F={len(current.faces):,}"
        )

        if new_size >= size and len(current.faces) > min_faces:
            # Force more aggressive fallback on the next iteration.
            current = vertex_cluster_simplify(
                current,
                target_vertices=max(300, int(len(current.vertices) * 0.65)),
            )
            export_binary_ply(current, output_path)
            new_size = output_path.stat().st_size

        size = new_size

        if iteration >= 12:
            break

    if size > target_bytes:
        raise RuntimeError(
            f"Could not get mesh below {max_size_mb:.3f} MB. "
            f"Final={size/1e6:.3f} MB, V={len(current.vertices)}, "
            f"F={len(current.faces)}. Install 'fast-simplification' "
            f"for stronger QEM reduction."
        )

    logger.info(
        f"Final compressed mesh: {output_path} "
        f"({size/1e6:.3f} MB, V={len(current.vertices):,}, "
        f"F={len(current.faces):,})"
    )
    return current


def save_debug_pointcloud(points: np.ndarray, path: Path):
    pc = trimesh.points.PointCloud(np.asarray(points, dtype=np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    pc.export(str(path))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run selected-view MV-SAM3D reconstruction, recover metric object scale "
            "from calibrated first-frame depth, and save compressed.ply < 10 MB."
        )
    )

    parser.add_argument("--setting_name", required=True, type=str)
    parser.add_argument(
        "--data_root",
        type=str,
        default="..",
        help="Dataset parent. Default: .., so dataset is ../<setting_name>/",
    )

    parser.add_argument("--images_dir_name", default="images1")
    parser.add_argument("--masks_dir_name", default="mask")
    parser.add_argument("--depth_dir_name", default="depth")
    parser.add_argument("--intri_name", default="intri.yml")
    parser.add_argument("--extri_name", default="extri.yml")

    parser.add_argument(
        "--mesh_cam_ids",
        type=str,
        default="1,5,9,13,17,21,25,29",
        help=(
            'Camera IDs used ONLY for MV-SAM3D mesh generation. '
            'Default: "1,5,9,13,17,21,25,29".'
        ),
    )
    parser.add_argument(
        "--depth_cam_ids",
        type=str,
        default="1-31",
        help='Camera IDs used ONLY for metric depth point-cloud fusion. Default: "1-31".',
    )

    # MV-SAM3D
    parser.add_argument("--model_tag", default="hf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1_steps", type=int, default=50)
    parser.add_argument("--stage2_steps", type=int, default=25)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable MV-SAM3D's built-in torch.compile path. Default: disabled because "
            "this PointMap pipeline revision has an incompatible compile warmup "
            "(missing run_layout_model). If explicitly enabled and that incompatibility "
            "is detected, the script automatically retries with compile=False."
        ),
    )
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Aggressive speed profile: cap diffusion at 25/12 steps, use pixel_stride>=4, "
            "max_points_per_cam<=20000, mesh_sample_points<=60000. Faster, with some quality tradeoff."
        ),
    )
    parser.add_argument("--io_workers", type=int, default=8)
    parser.add_argument("--depth_workers", type=int, default=8)

    # Depth/calibration
    parser.add_argument(
        "--depth_unit",
        choices=["auto", "m", "cm", "mm"],
        default="auto",
        help="Stored depth unit. auto assumes mm when raw median >20.",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=None,
        help=(
            "Explicit multiplier from stored depth values to meters. "
            "Overrides --depth_unit. Example: uint16 millimeters -> 0.001."
        ),
    )
    parser.add_argument(
        "--extrinsic_translation_scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to T from extri.yml before fusion. "
            "Use 0.001 if T is stored in millimeters while depth is meters."
        ),
    )
    parser.add_argument("--calib_width", type=int, default=None)
    parser.add_argument("--calib_height", type=int, default=None)

    parser.add_argument("--min_depth", type=float, default=0.05)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument(
        "--pixel_stride",
        type=int,
        default=2,
        help="Use every Nth masked depth pixel in x/y. Default: 2.",
    )
    parser.add_argument(
        "--mask_erode",
        type=int,
        default=1,
        help="Erode mask by this pixel radius before depth backprojection.",
    )
    parser.add_argument(
        "--max_points_per_cam",
        type=int,
        default=50000,
    )

    # Scale/mesh
    parser.add_argument("--mesh_sample_points", type=int, default=120000)
    parser.add_argument(
        "--max_size_mb",
        type=float,
        default=9.8,
        help="Strict decimal MB limit for final PLY. Default: 9.8 MB.",
    )
    parser.add_argument(
        "--save_debug_pointcloud",
        action="store_true",
        help="Also save meshes/metric_depth_cloud.ply.",
    )

    parser.add_argument("--generate_texture", action="store_true", help="Also bake a 1024px UV texture and export metric meshes/textured.glb; extra GPU time, no PLY size limit on GLB.")
    args = parser.parse_args()

    if args.fast:
        args.stage1_steps = min(int(args.stage1_steps), 25)
        args.stage2_steps = min(int(args.stage2_steps), 12)
        args.pixel_stride = max(int(args.pixel_stride), 4)
        args.max_points_per_cam = min(int(args.max_points_per_cam), 20000)
        args.mesh_sample_points = min(int(args.mesh_sample_points), 60000)
        logger.info(
            "[FAST PROFILE] stage1=%d stage2=%d pixel_stride=%d "
            "max_points_per_cam=%d mesh_sample_points=%d",
            args.stage1_steps, args.stage2_steps, args.pixel_stride,
            args.max_points_per_cam, args.mesh_sample_points,
        )

    mesh_cam_ids = parse_cam_ids(args.mesh_cam_ids)
    depth_cam_ids = parse_cam_ids(args.depth_cam_ids)
    if not mesh_cam_ids:
        raise ValueError("--mesh_cam_ids resolved to zero cameras")
    if not depth_cam_ids:
        raise ValueError("--depth_cam_ids resolved to zero cameras")

    dataset_dir = (Path(args.data_root) / args.setting_name).resolve()
    images_dir = dataset_dir / args.images_dir_name
    masks_dir = dataset_dir / args.masks_dir_name
    depth_root = dataset_dir / args.depth_dir_name
    intri_path = dataset_dir / args.intri_name
    extri_path = dataset_dir / args.extri_name
    meshes_dir = dataset_dir / "meshes"
    output_path = meshes_dir / "compressed.ply"
    info_path = meshes_dir / "metric_scale_info.json"

    for p in [images_dir, masks_dir, depth_root]:
        if not p.exists():
            raise FileNotFoundError(p)
    for p in [intri_path, extri_path]:
        if not p.exists():
            raise FileNotFoundError(p)
    meshes_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_dir}")
    logger.info(f"Mesh views ({len(mesh_cam_ids)}):  {mesh_cam_ids}")
    logger.info(f"Depth cams ({len(depth_cam_ids)}): {depth_cam_ids}")
    logger.info(f"Output:  {output_path}")
    logger.info("=" * 80)

    t_total = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Read exactly the requested multi-view images + masks.
    # ------------------------------------------------------------------
    # Load the union once. MV-SAM3D receives only mesh_cam_ids, while depth
    # fusion can still use all depth_cam_ids (normally all 31 cameras).
    all_needed_cam_ids = list(dict.fromkeys(mesh_cam_ids + depth_cam_ids))
    all_images, all_masks, image_paths, mask_paths = load_multiview_inputs(
        images_dir=images_dir,
        masks_dir=masks_dir,
        cam_ids=all_needed_cam_ids,
        io_workers=args.io_workers,
    )
    cam_to_index = {cam_id: i for i, cam_id in enumerate(all_needed_cam_ids)}
    view_images = [all_images[cam_to_index[c]] for c in mesh_cam_ids]
    view_masks = [all_masks[cam_to_index[c]] for c in mesh_cam_ids]
    logger.info(
        f"MV-SAM3D uses {len(mesh_cam_ids)} views {mesh_cam_ids}; "
        f"metric depth fusion uses {len(depth_cam_ids)} cameras."
    )
    logger.info(f"[Timing] input load: {time.perf_counter() - t_total:.2f}s")
    t_mesh = time.perf_counter()

    # ------------------------------------------------------------------
    # 2. Generate canonical object with MV-SAM3D.
    # ------------------------------------------------------------------
    result = run_mv_sam3d(
        view_images=view_images,
        view_masks=view_masks,
        model_tag=args.model_tag,
        seed=args.seed,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        generate_texture=args.generate_texture,
        compile_model=args.compile,
    )

    with tempfile.TemporaryDirectory(prefix="mv_sam3d_metric_") as td:
        work_dir = Path(td)
        sam_mesh = extract_sam3d_mesh(result, work_dir)

    logger.info(f"[Timing] MV-SAM3D + mesh extraction: {time.perf_counter() - t_mesh:.2f}s")
    t_depth = time.perf_counter()

    # ------------------------------------------------------------------
    # 3. Read FIRST depth frame for every camera and fuse a metric cloud.
    # ------------------------------------------------------------------
    fused_cloud, real_extent_m, depth_info = build_metric_object_cloud(
        cam_ids=depth_cam_ids,
        image_paths=image_paths,
        mask_paths=mask_paths,
        depth_root=depth_root,
        intri_path=intri_path,
        extri_path=extri_path,
        depth_scale=args.depth_scale,
        depth_unit=args.depth_unit,
        extrinsic_translation_scale=args.extrinsic_translation_scale,
        calib_width=args.calib_width,
        calib_height=args.calib_height,
        pixel_stride=args.pixel_stride,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        mask_erode=args.mask_erode,
        max_points_per_cam=args.max_points_per_cam,
        seed=args.seed,
        workers=args.depth_workers,
    )

    logger.info(f"[Timing] metric depth fusion: {time.perf_counter() - t_depth:.2f}s")

    if args.save_debug_pointcloud:
        debug_cloud_path = meshes_dir / "metric_depth_cloud.ply"
        save_debug_pointcloud(fused_cloud, debug_cloud_path)
        logger.info(f"Saved debug metric cloud: {debug_cloud_path}")

    # ------------------------------------------------------------------
    # 4. Estimate isotropic scale and rescale canonical SAM3D mesh.
    # ------------------------------------------------------------------
    scale, canonical_extent, axis_ratios = estimate_metric_scale(
        mesh=sam_mesh,
        real_extent_m=real_extent_m,
        mesh_sample_points=args.mesh_sample_points,
        seed=args.seed,
    )

    metric_mesh = apply_isotropic_scale_about_center(sam_mesh, scale)

    # ------------------------------------------------------------------
    # 5. Binary PLY + simplification until strict < max_size_mb.
    # ------------------------------------------------------------------
    final_mesh = compress_mesh_below_size(
        metric_mesh,
        output_path=output_path,
        max_size_mb=args.max_size_mb,
    )

    texture_info = None
    if args.generate_texture:
        from texture_export import export_textured_metric
        texture_info = export_textured_metric(result["textured_glb"], scale, output_path)
        logger.info(f"Textured metric model: {texture_info['path']}")

    # Re-check final physical extent for logging.
    final_pts = sample_mesh_surface_points(
        final_mesh,
        min(args.mesh_sample_points, 80000),
        args.seed,
    )
    final_extent, _, _ = robust_point_extent(final_pts, q=0.01)

    model_pose_info = {}
    for key in ["scale", "rotation", "translation"]:
        if key in result:
            value = result[key]
            try:
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                value = np.asarray(value).tolist()
            except Exception:
                value = str(value)
            model_pose_info[key] = value

    info = {
        "setting_name": args.setting_name,
        "mesh_cam_ids": mesh_cam_ids,
        "depth_cam_ids": depth_cam_ids,
        "convention": {
            "extrinsics": "world_to_cam: X_cam = R @ X_world + T",
            "output_scale_unit": "meters if depth/extrinsic translations are metric",
            "output_pose": (
                "SAM3D canonical orientation/centering retained; metric scale only"
            ),
        },
        "sam3d": {
            "model_tag": args.model_tag,
            "stage1_steps": args.stage1_steps,
            "stage2_steps": args.stage2_steps,
            "predicted_pose_not_used_for_metric_scale": model_pose_info,
            "canonical_robust_extent": canonical_extent.tolist(),
        },
        "metric_depth": depth_info,
        "scale_estimation": {
            "real_robust_extent_m": real_extent_m.tolist(),
            "canonical_robust_extent": canonical_extent.tolist(),
            "per_axis_extent_ratios": axis_ratios.tolist(),
            "isotropic_scale_applied": float(scale),
            "final_mesh_robust_extent_m": final_extent.tolist(),
        },
        "output": {
            "path": str(output_path),
            "size_bytes": int(output_path.stat().st_size),
            "size_mb_decimal": float(output_path.stat().st_size / 1e6),
            "vertices": int(len(final_mesh.vertices)),
            "faces": int(len(final_mesh.faces)),
        },
        "texture": texture_info,
        "args": vars(args),
    }

    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("DONE")
    print(f"Metric scale factor : {scale:.9f}")
    print(f"Depth object extent : {np.round(real_extent_m, 6)} m")
    print(f"Final mesh extent   : {np.round(final_extent, 6)} m")
    print(
        f"Final mesh          : {output_path} "
        f"({output_path.stat().st_size/1e6:.3f} MB)"
    )
    print(f"Scale metadata      : {info_path}")
    print(f"Total elapsed        : {time.perf_counter() - t_total:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
