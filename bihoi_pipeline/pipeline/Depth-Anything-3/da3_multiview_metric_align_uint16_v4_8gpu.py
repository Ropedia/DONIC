import re
import json
import shutil
import argparse
import queue
import multiprocessing as mp
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


# ============================================================
# Basic utilities
# ============================================================

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def list_image_files(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: natural_key(p.name),
    )


def list_depth_files(folder: Path) -> List[Path]:
    exts = {".npy", ".npz", ".png", ".tif", ".tiff", ".exr"}
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: natural_key(p.name),
    )


def read_gray_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def load_depth(path: Path, uint16_input_scale: float = 1.0) -> np.ndarray:
    """
    Load a depth map as float32.

    For uint16 PNG/TIFF:
        returned_depth = uint16_value / uint16_input_scale

    The default scale=1 preserves raw relative-depth values.
    """
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(str(path))

    elif suffix == ".npz":
        obj = np.load(str(path))
        if "depth" in obj:
            arr = obj["depth"]
        elif len(obj.files) > 0:
            arr = obj[obj.files[0]]
        else:
            raise ValueError(f"Empty npz depth file: {path}")

    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"Failed to read depth: {path}")

        if arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / float(uint16_input_scale)

    arr = np.asarray(arr).astype(np.float32)

    if arr.ndim == 3:
        arr = arr[..., 0]

    return arr


def save_depth_npy(path: Path, depth: np.ndarray):
    ensure_dir(path.parent)
    np.save(str(path), depth.astype(np.float32))


def save_depth_uint16_png(
    path: Path,
    depth: np.ndarray,
    scale: float = 1000.0,
    compression: int = 0,
):
    """
    Save float depth to uint16 PNG.

    encoded_uint16 = round(depth * scale)

    Typical metric convention:
      depth in meters + scale=1000  -> millimeters in uint16 PNG.

    Invalid / non-positive values become 0.
    Values > 65535 are clipped.
    """
    ensure_dir(path.parent)

    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)

    encoded = np.zeros(d.shape, dtype=np.uint16)

    vals = np.rint(d[valid] * float(scale))
    vals = np.clip(vals, 1, 65535)
    encoded[valid] = vals.astype(np.uint16)

    ok = cv2.imwrite(
        str(path),
        encoded,
        [cv2.IMWRITE_PNG_COMPRESSION, int(compression)],
    )
    if not ok:
        raise RuntimeError(f"Failed to write uint16 depth PNG: {path}")


def depth_valid_mask(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > 1e-8)


def resize_depth_to_match(depth: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    if depth.shape[:2] == (h, w):
        return depth
    return cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)


def resize_mask_to_match(mask: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    if mask.shape[:2] == (h, w):
        return mask.astype(bool)
    out = cv2.resize(
        mask.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    return out > 0


def overlay_mask_vis(gray01: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape != gray01.shape:
        mask = resize_mask_to_match(mask, gray01.shape)

    img = np.clip(gray01 * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis[..., 1] = np.where(mask, 255, vis[..., 1])
    vis[..., 2] = np.where(mask, 0, vis[..., 2])
    return vis


def sample_evenly(items: List[Path], max_n: int) -> List[Path]:
    if len(items) <= max_n:
        return items
    idx = np.linspace(0, len(items) - 1, max_n).round().astype(int)
    return [items[i] for i in idx]


# ============================================================
# EasyMocap intri.yml / extri.yml
# ============================================================

def _try_node_mat(fs: cv2.FileStorage, key: str) -> Optional[np.ndarray]:
    node = fs.getNode(key)
    if node.empty():
        return None

    mat = node.mat()
    if mat is None:
        return None

    return np.asarray(mat, dtype=np.float32)


def _find_first_existing_mat(
    fs: cv2.FileStorage,
    keys: List[str],
) -> Optional[np.ndarray]:
    for key in keys:
        mat = _try_node_mat(fs, key)
        if mat is not None:
            return mat
    return None


def _rotation_to_matrix(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float32)

    if rot.shape == (3, 3):
        return rot

    rot = rot.reshape(-1)

    if rot.size == 3:
        R, _ = cv2.Rodrigues(rot.reshape(3, 1))
        return R.astype(np.float32)

    raise ValueError(f"Unsupported rotation format: {rot.shape}")


def load_intrinsics_extrinsics(
    folder: Path,
    cam_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      intrinsics: [N, 3, 3]
      extrinsics: [N, 4, 4], world-to-camera
    """
    intri_path = folder / "intri.yml"
    extri_path = folder / "extri.yml"

    if not intri_path.exists():
        raise FileNotFoundError(intri_path)

    if not extri_path.exists():
        raise FileNotFoundError(extri_path)

    fs_i = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_READ)
    fs_e = cv2.FileStorage(str(extri_path), cv2.FILE_STORAGE_READ)

    if not fs_i.isOpened() or not fs_e.isOpened():
        raise RuntimeError("Failed to open intri.yml/extri.yml with OpenCV FileStorage")

    intrinsics = []
    extrinsics = []

    for cam in cam_ids:
        aliases = [str(cam)]

        try:
            aliases.append(str(int(cam)))
        except Exception:
            pass

        K = None
        R_raw = None
        T_raw = None

        for alias in aliases:
            if K is None:
                K = _find_first_existing_mat(
                    fs_i,
                    [f"K_{alias}", f"k_{alias}", alias],
                )

            if R_raw is None:
                R_raw = _find_first_existing_mat(
                    fs_e,
                    [
                        f"Rot_{alias}",
                        f"R_{alias}",
                        f"rot_{alias}",
                        f"r_{alias}",
                    ],
                )

            if T_raw is None:
                T_raw = _find_first_existing_mat(
                    fs_e,
                    [
                        f"T_{alias}",
                        f"t_{alias}",
                        f"Trans_{alias}",
                        f"trans_{alias}",
                    ],
                )

        if K is None:
            raise KeyError(f"Cannot find K for camera {cam}")

        if R_raw is None or T_raw is None:
            raise KeyError(f"Cannot find extrinsics for camera {cam}")

        R = _rotation_to_matrix(R_raw)
        T = np.asarray(T_raw, dtype=np.float32).reshape(3, 1)

        ext = np.eye(4, dtype=np.float32)
        ext[:3, :3] = R
        ext[:3, 3:4] = T

        intrinsics.append(np.asarray(K, dtype=np.float32))
        extrinsics.append(ext)

    fs_i.release()
    fs_e.release()

    return (
        np.stack(intrinsics, axis=0),
        np.stack(extrinsics, axis=0),
    )


# ============================================================
# Static-region detection
# ============================================================

def build_static_mask(
    image_files: List[Path],
    max_samples: int = 80,
    static_percentile: float = 35.0,
    grad_percentile: float = 55.0,
    min_keep_ratio: float = 0.03,
) -> np.ndarray:
    """
    Detect regions likely to remain static over the whole sequence.

    Criteria:
      1. low temporal MAD
      2. sufficient image gradient / texture
    """
    # By default use ALL frames for static-region estimation.
    # If max_samples > 0, uniformly subsample to that many frames.
    sampled = (
        image_files
        if max_samples <= 0
        else sample_evenly(image_files, max_samples)
    )

    print(
        f"    static-region frames used: "
        f"{len(sampled)}/{len(image_files)}"
    )

    stack = np.stack(
        [read_gray_image(p) for p in sampled],
        axis=0,
    )

    median_img = np.median(stack, axis=0)
    mad = np.median(
        np.abs(stack - median_img[None]),
        axis=0,
    )

    gx = cv2.Sobel(
        median_img,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        median_img,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    grad = np.sqrt(gx * gx + gy * gy)

    static_thr = np.percentile(
        mad,
        static_percentile,
    )

    grad_thr = np.percentile(
        grad,
        grad_percentile,
    )

    mask = (
        (mad <= static_thr)
        & (grad >= grad_thr)
    )

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)

    mask_u8 = mask.astype(np.uint8) * 255

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_OPEN,
        kernel3,
    )

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        kernel5,
    )

    mask = mask_u8 > 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    if num_labels > 1:
        components = [
            (i, stats[i, cv2.CC_STAT_AREA])
            for i in range(1, num_labels)
        ]

        components.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        keep = np.zeros_like(
            mask,
            dtype=bool,
        )

        running = 0

        for idx, area in components:
            keep |= labels == idx
            running += area

            if running / mask.size >= min_keep_ratio:
                break

        if keep.any():
            mask = keep

    # Fallback when the texture criterion was too strict.
    if mask.mean() < 0.01:
        static_thr = np.percentile(
            mad,
            min(static_percentile + 15, 70),
        )

        mask = mad <= static_thr

        mask_u8 = (
            mask.astype(np.uint8)
            * 255
        )

        mask_u8 = cv2.morphologyEx(
            mask_u8,
            cv2.MORPH_OPEN,
            kernel3,
        )

        mask_u8 = cv2.morphologyEx(
            mask_u8,
            cv2.MORPH_CLOSE,
            kernel5,
        )

        mask = mask_u8 > 0

    return mask


# ============================================================
# Robust scale / affine alignment
# ============================================================

def robust_fit_depth_map(
    src: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    mode: str = "affine",
    max_points: int = 50000,
    iters: int = 4,
    sigma_thresh: float = 3.0,
):
    """
    Fit

        target ~= a * src + b

    on robust inliers.

    mode:
      affine      -> solve a and b
      scale_only  -> b = 0
    """
    mask = (
        valid_mask
        & np.isfinite(src)
        & np.isfinite(target)
    )

    x = src[mask].reshape(-1).astype(np.float64)
    y = target[mask].reshape(-1).astype(np.float64)

    if x.size < 32:
        return 1.0, 0.0, int(x.size), float("inf")

    if x.size > max_points:
        ids = np.linspace(
            0,
            x.size - 1,
            max_points,
        ).round().astype(int)

        x = x[ids]
        y = y[ids]

    inliers = np.ones(
        x.shape,
        dtype=bool,
    )

    a = 1.0
    b = 0.0

    for _ in range(iters):
        xi = x[inliers]
        yi = y[inliers]

        if xi.size < 16:
            break

        if mode == "scale_only":
            a = float(
                np.dot(xi, yi)
                /
                (np.dot(xi, xi) + 1e-12)
            )
            b = 0.0

        else:
            A = np.stack(
                [
                    xi,
                    np.ones_like(xi),
                ],
                axis=1,
            )

            sol, *_ = np.linalg.lstsq(
                A,
                yi,
                rcond=None,
            )

            a = float(sol[0])
            b = float(sol[1])

        residual = (
            a * x
            + b
            - y
        )

        med = float(
            np.median(residual)
        )

        mad = float(
            np.median(
                np.abs(
                    residual
                    - med
                )
            )
        )

        if mad < 1e-12:
            break

        sigma = 1.4826 * mad

        inliers = (
            np.abs(
                residual
                - med
            )
            <= sigma_thresh * sigma
        )

    final_residual = (
        a * x
        + b
        - y
    )

    rmse = float(
        np.sqrt(
            np.mean(
                final_residual
                * final_residual
            )
        )
    )

    return (
        a,
        b,
        int(inliers.sum()),
        rmse,
    )


def apply_transform(
    depth: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    output = np.asarray(
        depth,
        dtype=np.float32,
    ).copy()

    valid = depth_valid_mask(output)

    output[valid] = (
        a * output[valid]
        + b
    )

    output[valid] = np.maximum(
        output[valid],
        1e-6,
    )

    output[~valid] = 0

    return output


# ============================================================
# Camera / frame handling
# ============================================================

def collect_cam_ids(
    images_root: Path,
) -> List[str]:
    cams = sorted(
        [
            p.name
            for p in images_root.iterdir()
            if p.is_dir()
        ],
        key=natural_key,
    )

    if not cams:
        raise RuntimeError(
            f"No camera folders under {images_root}"
        )

    return cams


def get_first_frame_paths(
    images_root: Path,
    cam_ids: List[str],
) -> List[Path]:
    paths = []

    for cam in cam_ids:
        files = list_image_files(
            images_root / cam
        )

        if not files:
            raise RuntimeError(
                f"No images in {images_root / cam}"
            )

        paths.append(files[0])

    return paths


def match_image_depth_pairs(
    cam_img_dir: Path,
    cam_depth_dir: Path,
):
    images = list_image_files(
        cam_img_dir
    )

    depths = list_depth_files(
        cam_depth_dir
    )

    if not images:
        raise RuntimeError(
            f"No images in {cam_img_dir}"
        )

    if not depths:
        raise RuntimeError(
            f"No depth maps in {cam_depth_dir}"
        )

    depth_by_stem = {
        p.stem: p
        for p in depths
    }

    matched = [
        (img, depth_by_stem[img.stem])
        for img in images
        if img.stem in depth_by_stem
    ]

    if len(matched) == len(images):
        return matched

    if len(images) != len(depths):
        raise RuntimeError(
            f"Image/depth count mismatch for {cam_img_dir.name}: "
            f"{len(images)} images vs {len(depths)} depths"
        )

    # Fallback to natural-sort order.
    return list(zip(images, depths))


# ============================================================
# DA3 first-frame multi-view metric inference
# ============================================================

def run_da3_first_frames(
    folder: Path,
    cam_ids: List[str],
    model_dir: str,
    device: str,
    process_res: int,
    process_res_method: str,
    da3_out_dir: Path,
):
    from depth_anything_3.api import DepthAnything3

    images_root = folder / "images"

    first_paths = get_first_frame_paths(
        images_root,
        cam_ids,
    )

    intrinsics, extrinsics = load_intrinsics_extrinsics(
        folder,
        cam_ids,
    )

    print(
        f"[DA3] Loading: {model_dir}"
    )

    model = (
        DepthAnything3
        .from_pretrained(model_dir)
        .to(device)
        .eval()
    )

    print(
        f"[DA3] Joint multi-view inference: "
        f"{len(first_paths)} cameras"
    )

    pred = model.inference(
        image=[
            str(p)
            for p in first_paths
        ],
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        align_to_input_ext_scale=True,
        process_res=process_res,
        process_res_method=process_res_method,
    )

    ensure_dir(
        da3_out_dir / "npy"
    )

    ensure_dir(
        da3_out_dir / "conf"
    )

    ensure_dir(
        da3_out_dir / "vis"
    )

    save_json(
        da3_out_dir / "meta.json",
        {
            "cam_ids": cam_ids,
            "first_frames": [
                p.name
                for p in first_paths
            ],
        },
    )

    for i, cam in enumerate(cam_ids):
        depth = np.asarray(
            pred.depth[i],
            dtype=np.float32,
        )

        conf = np.asarray(
            pred.conf[i],
            dtype=np.float32,
        )

        np.save(
            str(
                da3_out_dir
                / "npy"
                / f"{cam}.npy"
            ),
            depth,
        )

        np.save(
            str(
                da3_out_dir
                / "conf"
                / f"{cam}.npy"
            ),
            conf,
        )

        valid = depth_valid_mask(depth)

        if valid.any():
            lo = np.percentile(
                depth[valid],
                2,
            )

            hi = np.percentile(
                depth[valid],
                98,
            )

            vis = np.clip(
                (depth - lo)
                /
                max(hi - lo, 1e-6),
                0,
                1,
            )

            vis = (
                vis * 255
            ).astype(np.uint8)

            vis = cv2.applyColorMap(
                vis,
                cv2.COLORMAP_INFERNO,
            )

            cv2.imwrite(
                str(
                    da3_out_dir
                    / "vis"
                    / f"{cam}.png"
                ),
                vis,
            )



# ============================================================
# Direct DA3 modes: first frame only / all frames
# ============================================================

def prepare_direct_depth_output(folder: Path) -> Tuple[Path, Optional[Path]]:
    """
    Prepare a clean <folder>/depth directory for direct DA3 output.

    On the first direct-DA3 run:
        depth/ -> depth_before_da3_direct/

    On later reruns, the original backup is kept and the current depth/
    (which is assumed to be a previous DA3-direct result) is removed.
    """
    depth_root = folder / "depth"
    backup_root = folder / "depth_before_da3_direct"

    if depth_root.exists():
        if backup_root.exists():
            print(
                f"[DA3 direct] Keeping existing original-depth backup: {backup_root}"
            )
            print(
                f"[DA3 direct] Removing previous direct-DA3 output: {depth_root}"
            )
            shutil.rmtree(depth_root)
        else:
            print(
                f"[DA3 direct] Backing up existing depth folder:\n"
                f"  {depth_root}\n"
                f"-> {backup_root}"
            )
            shutil.move(str(depth_root), str(backup_root))

    ensure_dir(depth_root)
    return depth_root, (backup_root if backup_root.exists() else None)


def _image_hw(path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image for output size: {path}")
    return img.shape[:2]


def _validate_all_cam_frame_counts(
    images_root: Path,
    cam_ids: List[str],
) -> dict:
    """
    Build naturally sorted frame lists for every camera and require equal counts.
    Multi-view inference is then performed by frame INDEX: frame i from every camera.
    """
    files_by_cam = {}
    counts = {}

    for cam in cam_ids:
        files = list_image_files(images_root / cam)
        if not files:
            raise RuntimeError(f"No images found for camera {cam}: {images_root / cam}")
        files_by_cam[cam] = files
        counts[cam] = len(files)

    unique_counts = sorted(set(counts.values()))
    if len(unique_counts) != 1:
        details = ", ".join(f"cam{c}={n}" for c, n in counts.items())
        raise RuntimeError(
            "--all_da3 requires every camera folder to contain the same number "
            f"of synchronized frames. Counts: {details}"
        )

    return files_by_cam


def run_da3_direct(
    folder: Path,
    cam_ids: List[str],
    model_dir: str,
    device: str,
    process_res: int,
    process_res_method: str,
    png_scale: float,
    mode: str,
    png_compression: int = 0,
):
    """
    Direct DA3 metric-depth generation.

    mode == "first_only":
        Jointly infer the first image from every camera once.

    mode == "all_da3":
        For each synchronized frame index, jointly infer that frame from ALL cameras.

    Output:
        <folder>/depth/<cam_id>/<image_stem>.png

    The PNG is uint16 with:
        uint16 = round(metric_depth * png_scale)
    """
    from depth_anything_3.api import DepthAnything3

    images_root = folder / "images"
    output_root, backup_root = prepare_direct_depth_output(folder)

    intrinsics, extrinsics = load_intrinsics_extrinsics(folder, cam_ids)

    print(f"[DA3 direct] Loading model: {model_dir}")
    model = DepthAnything3.from_pretrained(model_dir).to(device).eval()

    if mode == "first_only":
        files_by_cam = {
            cam: [get_first_frame_paths(images_root, [cam])[0]]
            for cam in cam_ids
        }
        num_frames = 1
    elif mode == "all_da3":
        files_by_cam = _validate_all_cam_frame_counts(images_root, cam_ids)
        num_frames = len(files_by_cam[cam_ids[0]])
    else:
        raise ValueError(f"Unknown direct DA3 mode: {mode}")

    for cam in cam_ids:
        ensure_dir(output_root / cam)

    save_json(
        folder / "da3_direct_meta.json",
        {
            "mode": mode,
            "model_dir": model_dir,
            "num_cameras": len(cam_ids),
            "num_frames": num_frames,
            "cam_ids": cam_ids,
            "process_res": process_res,
            "process_res_method": process_res_method,
            "png_scale": png_scale,
            "depth_output": str(output_root),
            "previous_depth_backup": str(backup_root) if backup_root else None,
            "note": (
                "Each time step is inferred jointly across all cameras using the "
                "provided intri.yml/extri.yml with align_to_input_ext_scale=True."
            ),
        },
    )

    print(
        f"[DA3 direct] Mode={mode}; cameras={len(cam_ids)}; "
        f"multi-view groups={num_frames}"
    )

    for frame_idx in range(num_frames):
        image_paths = [files_by_cam[cam][frame_idx] for cam in cam_ids]

        label = image_paths[0].stem
        print(
            f"[DA3 direct] MV frame {frame_idx + 1}/{num_frames} "
            f"(first-cam stem={label})"
        )

        pred = model.inference(
            image=[str(p) for p in image_paths],
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            align_to_input_ext_scale=True,
            process_res=process_res,
            process_res_method=process_res_method,
        )

        if len(pred.depth) != len(cam_ids):
            raise RuntimeError(
                f"DA3 returned {len(pred.depth)} depth maps for "
                f"{len(cam_ids)} input cameras"
            )

        for view_idx, cam in enumerate(cam_ids):
            depth = np.asarray(pred.depth[view_idx], dtype=np.float32)

            # DA3 may infer at its processing resolution. Resize the depth map
            # spatially back to the corresponding original RGB resolution so
            # downstream image/depth geometry stays pixel-aligned. Metric depth
            # values themselves are not scaled by this image resize.
            orig_hw = _image_hw(image_paths[view_idx])
            depth = resize_depth_to_match(depth, orig_hw)

            output_path = output_root / cam / f"{image_paths[view_idx].stem}.png"
            save_depth_uint16_png(
                output_path,
                depth,
                scale=png_scale,
                compression=png_compression,
            )

    print()
    print("====================================================")
    print("Direct DA3 generation complete")
    print("====================================================")
    print(f"Mode:               {mode}")
    print(f"Metric depth root:  {output_root}")
    print(f"PNG encoding:       uint16 = round(metric_depth * {png_scale})")
    if backup_root is not None:
        print(f"Previous depth backup: {backup_root}")



def _visible_gpu_free_mb() -> List[Tuple[int, int]]:
    """Return [(visible_cuda_index, free_MiB), ...], respecting CUDA_VISIBLE_DEVICES."""
    import torch

    visible = torch.cuda.device_count()
    if visible <= 0:
        raise RuntimeError("No CUDA GPU is visible to PyTorch.")

    rows = []
    for gid in range(visible):
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(gid)
            free_mb = int(free_bytes // (1024 * 1024))
            rows.append((gid, free_mb))
        except Exception as exc:
            print(f"[GPU auto] cuda:{gid}: free-memory query failed -> skip ({exc})")
    return rows


def _resolve_single_device(device_spec: str, min_free_vram_mb: int = 32768) -> str:
    """Resolve a single CUDA device; auto/cuda picks the most-free qualifying GPU."""
    spec = str(device_spec).strip().lower()
    if spec == "cpu" or spec.startswith("mps"):
        return str(device_spec)

    mem_rows = _visible_gpu_free_mb()
    free_by_gpu = dict(mem_rows)
    threshold = int(min_free_vram_mb)

    if spec in {"auto", "cuda"}:
        eligible = [(gid, free_mb) for gid, free_mb in mem_rows if free_mb > threshold]
        eligible.sort(key=lambda x: (-x[1], x[0]))
        for gid, free_mb in mem_rows:
            state = "eligible" if free_mb > threshold else "skip"
            print(f"[GPU auto][single] cuda:{gid}: {free_mb} MiB free -> {state}")
        if not eligible:
            raise RuntimeError(f"No visible GPU has free VRAM > {threshold} MiB for single-GPU DA3 work.")
        gid, free_mb = eligible[0]
        print(f"[GPU auto][single] selected cuda:{gid} with {free_mb} MiB free")
        return f"cuda:{gid}"

    if spec.startswith("cuda:"):
        try:
            gid = int(spec.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid --device value: {device_spec!r}") from exc
        if gid not in free_by_gpu:
            raise RuntimeError(f"Requested {device_spec}, but that CUDA-visible GPU does not exist.")
        free_mb = free_by_gpu[gid]
        if free_mb <= threshold:
            raise RuntimeError(
                f"Requested {device_spec} has only {free_mb} MiB free; strictly > {threshold} MiB is required."
            )
        return f"cuda:{gid}"

    return str(device_spec)


def _parse_gpu_ids(gpus: str, min_free_vram_mb: int = 32768) -> List[int]:
    """Resolve GPUs. auto/all means every visible GPU with STRICTLY > threshold free VRAM."""
    import torch

    visible = torch.cuda.device_count()
    if visible <= 0:
        raise RuntimeError("No CUDA GPU is visible to PyTorch.")
    if int(min_free_vram_mb) < 0:
        raise ValueError("--min_free_vram_mb must be >= 0")

    mem_rows = _visible_gpu_free_mb()
    free_by_gpu = dict(mem_rows)
    spec = gpus.strip().lower()

    if spec in {"auto", "all", "*"}:
        eligible = [(gid, free_mb) for gid, free_mb in mem_rows if free_mb > int(min_free_vram_mb)]
        eligible.sort(key=lambda x: (-x[1], x[0]))
        for gid, free_mb in mem_rows:
            state = "eligible" if free_mb > int(min_free_vram_mb) else "skip"
            print(f"[GPU auto] cuda:{gid}: {free_mb} MiB free -> {state}")
        ids = [gid for gid, _ in eligible]
        if not ids:
            raise RuntimeError(
                f"No visible GPU has free VRAM > {int(min_free_vram_mb)} MiB. "
                "Free GPU memory or lower --min_free_vram_mb."
            )
        print(f"[GPU auto] selected all qualifying GPUs: {ids}")
        return ids

    try:
        ids = [int(x.strip()) for x in gpus.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise ValueError(f"Invalid --gpus value: {gpus}") from exc

    if not ids:
        raise ValueError("--gpus resolved to an empty GPU list")
    if len(set(ids)) != len(ids):
        raise ValueError(f"--gpus contains duplicates: {ids}")
    bad = [gid for gid in ids if gid < 0 or gid >= visible]
    if bad:
        raise RuntimeError(
            f"Requested GPU ids {bad}, but PyTorch sees only {visible} CUDA device(s) "
            f"with local ids 0..{visible - 1}."
        )
    low = [(gid, free_by_gpu.get(gid, -1)) for gid in ids if free_by_gpu.get(gid, -1) <= int(min_free_vram_mb)]
    if low:
        raise RuntimeError(
            "Explicit --gpus includes GPU(s) without enough free VRAM (> "
            f"{int(min_free_vram_mb)} MiB required): {low}"
        )
    return ids


def _da3_all_worker(
    worker_rank: int,
    gpu_id: int,
    frame_indices: List[int],
    folder_str: str,
    cam_ids: List[str],
    model_dir: str,
    process_res: int,
    process_res_method: str,
    png_scale: float,
    writer_threads: int,
    png_compression: int,
    max_pending_writes: int,
    result_queue,
):
    """
    One persistent DA3 worker on one GPU.

    Each assigned frame_idx is still processed as ONE joint multi-view inference
    containing all cameras. Parallelism is only across time/frame groups.
    """
    try:
        import torch
        from depth_anything_3.api import DepthAnything3

        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        folder = Path(folder_str)
        images_root = folder / "images"
        output_root = folder / "depth"

        files_by_cam = _validate_all_cam_frame_counts(images_root, cam_ids)
        intrinsics, extrinsics = load_intrinsics_extrinsics(folder, cam_ids)

        # All preprocessed frames of one camera are fixed-resolution. Read each
        # camera's dimensions ONCE instead of decoding 31 RGB images per MV frame.
        orig_hw_by_cam = {
            cam: _image_hw(files_by_cam[cam][0])
            for cam in cam_ids
        }

        print(
            f"[GPU {gpu_id}] worker {worker_rank}: loading {model_dir}; "
            f"assigned {len(frame_indices)} MV frame groups; "
            f"writers={writer_threads}, png_compression={png_compression}",
            flush=True,
        )

        model = DepthAnything3.from_pretrained(model_dir).to(device).eval()

        writer_pool = ThreadPoolExecutor(
            max_workers=max(1, int(writer_threads)),
            thread_name_prefix=f"depth-writer-gpu{gpu_id}",
        )
        pending = deque()

        def submit_depth(path, arr):
            # Own the array independently of `pred`; PNG encoding/writing can then
            # overlap the next GPU inference safely.
            future = writer_pool.submit(
                save_depth_uint16_png,
                path,
                np.ascontiguousarray(arr, dtype=np.float32),
                png_scale,
                png_compression,
            )
            pending.append(future)
            limit = max(1, int(max_pending_writes))
            while len(pending) > limit:
                pending.popleft().result()

        total_local = len(frame_indices)
        try:
            with torch.inference_mode():
                for local_idx, frame_idx in enumerate(frame_indices, start=1):
                    image_paths = [files_by_cam[cam][frame_idx] for cam in cam_ids]
                    label = image_paths[0].stem

                    print(
                        f"[GPU {gpu_id}] MV frame {frame_idx + 1} "
                        f"(local {local_idx}/{total_local}, first-cam stem={label})",
                        flush=True,
                    )

                    pred = model.inference(
                        image=[str(p) for p in image_paths],
                        intrinsics=intrinsics,
                        extrinsics=extrinsics,
                        align_to_input_ext_scale=True,
                        process_res=process_res,
                        process_res_method=process_res_method,
                    )

                    if len(pred.depth) != len(cam_ids):
                        raise RuntimeError(
                            f"GPU {gpu_id}: DA3 returned {len(pred.depth)} depths for "
                            f"{len(cam_ids)} camera inputs at frame index {frame_idx}"
                        )

                    for view_idx, cam in enumerate(cam_ids):
                        depth = np.asarray(pred.depth[view_idx], dtype=np.float32)
                        depth = resize_depth_to_match(depth, orig_hw_by_cam[cam])
                        output_path = output_root / cam / f"{image_paths[view_idx].stem}.png"
                        # copy only when resize returned an array backed by predictor output
                        submit_depth(output_path, np.array(depth, copy=True))

                    del pred

            while pending:
                pending.popleft().result()
        finally:
            writer_pool.shutdown(wait=True)

        del model
        torch.cuda.empty_cache()
        result_queue.put((worker_rank, gpu_id, True, len(frame_indices), ""))

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        try:
            result_queue.put((worker_rank, gpu_id, False, 0, f"{exc}\n{tb}"))
        finally:
            raise


def run_da3_all_multigpu(
    folder: Path,
    cam_ids: List[str],
    model_dir: str,
    process_res: int,
    process_res_method: str,
    png_scale: float,
    gpus: str,
    min_free_vram_mb: int = 32768,
    writer_threads: int = 4,
    png_compression: int = 0,
    max_pending_writes: int = 64,
):
    """
    Multi-GPU --all_da3.

    With 8 GPUs, eight persistent workers are spawned. Each worker owns one GPU
    and receives a round-robin subset of synchronized frame indices. A frame
    group always contains ALL camera views and is never split across GPUs.
    """
    images_root = folder / "images"
    files_by_cam = _validate_all_cam_frame_counts(images_root, cam_ids)
    num_frames = len(files_by_cam[cam_ids[0]])

    gpu_ids = _parse_gpu_ids(gpus, min_free_vram_mb=min_free_vram_mb)
    if len(gpu_ids) > num_frames:
        gpu_ids = gpu_ids[:num_frames]

    output_root, backup_root = prepare_direct_depth_output(folder)
    for cam in cam_ids:
        ensure_dir(output_root / cam)

    frame_splits = [
        list(range(rank, num_frames, len(gpu_ids)))
        for rank in range(len(gpu_ids))
    ]

    print("[DA3 all multi-GPU]")
    print(f"  cameras:          {len(cam_ids)}")
    print(f"  MV frame groups:  {num_frames}")
    print(f"  GPUs:             {gpu_ids}")
    print(f"  groups/GPU:       {[len(x) for x in frame_splits]}")
    print("  IMPORTANT: every frame group still contains ALL camera views.")

    save_json(
        folder / "da3_direct_meta.json",
        {
            "mode": "all_da3",
            "multigpu": True,
            "gpu_ids": gpu_ids,
            "model_dir": model_dir,
            "num_cameras": len(cam_ids),
            "num_frames": num_frames,
            "cam_ids": cam_ids,
            "frame_groups_per_gpu": {
                str(gpu): len(frame_splits[rank])
                for rank, gpu in enumerate(gpu_ids)
            },
            "process_res": process_res,
            "process_res_method": process_res_method,
            "png_scale": png_scale,
            "depth_output": str(output_root),
            "previous_depth_backup": str(backup_root) if backup_root else None,
            "note": (
                "Temporal frame groups are distributed across GPUs. Each individual "
                "frame group is a joint multi-view inference over all cameras using "
                "intri.yml/extri.yml and align_to_input_ext_scale=True."
            ),
        },
    )

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for rank, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=_da3_all_worker,
            args=(
                rank,
                gpu_id,
                frame_splits[rank],
                str(folder),
                cam_ids,
                model_dir,
                process_res,
                process_res_method,
                png_scale,
                writer_threads,
                png_compression,
                max_pending_writes,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = []
    for _ in processes:
        try:
            results.append(result_queue.get(timeout=2.0))
        except queue.Empty:
            break

    failed_processes = [
        (rank, gpu_ids[rank], p.exitcode)
        for rank, p in enumerate(processes)
        if p.exitcode != 0
    ]
    worker_errors = [r for r in results if not r[2]]

    if failed_processes or worker_errors:
        msg = ["One or more DA3 GPU workers failed."]
        if failed_processes:
            msg.append(f"Process failures: {failed_processes}")
        for rank, gpu, _, _, err in worker_errors:
            msg.append(f"--- worker {rank}, GPU {gpu} ---\n{err}")
        raise RuntimeError("\n".join(msg))

    completed = sum(r[3] for r in results if r[2])
    if completed != num_frames:
        raise RuntimeError(
            f"Workers reported {completed}/{num_frames} completed frame groups."
        )

    print()
    print("====================================================")
    print("Multi-GPU all-frame DA3 generation complete")
    print("====================================================")
    print(f"GPUs used:          {gpu_ids}")
    print(f"MV frame groups:    {num_frames}")
    print(f"Metric depth root:  {output_root}")
    print(f"PNG encoding:       uint16 = round(metric_depth * {png_scale})")
    if backup_root is not None:
        print(f"Previous depth backup: {backup_root}")


# ============================================================
# Per-camera temporal + metric alignment
# ============================================================

def process_one_camera(
    folder: Path,
    source_depth_root: Path,
    cam: str,
    da3_depth_path: Path,
    consistent_root: Path,
    final_output_root: Path,
    static_mask_root: Path,
    meta_root: Path,
    max_static_samples: int,
    temporal_fit: str,
    metric_fit: str,
    final_png_scale: float,
    uint16_input_scale: float,
):
    image_dir = (
        folder
        / "images"
        / cam
    )

    source_depth_dir = (
        source_depth_root
        / cam
    )

    pairs = match_image_depth_pairs(
        image_dir,
        source_depth_dir,
    )

    image_files = [
        x[0]
        for x in pairs
    ]

    depth_files = [
        x[1]
        for x in pairs
    ]

    print(
        f"[Cam {cam}] "
        f"static-region analysis "
        f"from {len(image_files)} total frames"
    )

    static_mask_image_res = build_static_mask(
        image_files,
        max_samples=max_static_samples,
    )

    first_gray = read_gray_image(
        image_files[0]
    )

    ensure_dir(static_mask_root)

    cv2.imwrite(
        str(
            static_mask_root
            / f"{cam}.png"
        ),
        static_mask_image_res.astype(np.uint8)
        * 255,
    )

    cv2.imwrite(
        str(
            static_mask_root
            / f"{cam}_overlay.png"
        ),
        overlay_mask_vis(
            first_gray,
            static_mask_image_res,
        ),
    )

    first_relative = load_depth(
        depth_files[0],
        uint16_input_scale=uint16_input_scale,
    )

    static_mask = resize_mask_to_match(
        static_mask_image_res,
        first_relative.shape[:2],
    )

    valid0 = (
        static_mask
        & depth_valid_mask(first_relative)
    )

    if valid0.sum() < 64:
        raise RuntimeError(
            f"[Cam {cam}] insufficient static valid depth pixels: "
            f"{int(valid0.sum())}"
        )

    print(
        f"[Cam {cam}] temporal alignment "
        f"on static regions"
    )

    temporal_records = []

    for image_path, depth_path in pairs:
        depth = load_depth(
            depth_path,
            uint16_input_scale=uint16_input_scale,
        )

        depth = resize_depth_to_match(
            depth,
            first_relative.shape[:2],
        )

        valid = (
            static_mask
            & depth_valid_mask(depth)
            & depth_valid_mask(first_relative)
        )

        a, b, ninliers, rmse = robust_fit_depth_map(
            src=depth,
            target=first_relative,
            valid_mask=valid,
            mode=temporal_fit,
        )

        aligned = apply_transform(
            depth,
            a,
            b,
        )

        # Keep a float copy internally so metric fitting does not suffer
        # uint16 quantization.
        save_depth_npy(
            consistent_root
            / cam
            / f"{image_path.stem}.npy",
            aligned,
        )

        temporal_records.append(
            {
                "frame": image_path.name,
                "source_depth": depth_path.name,
                "a": a,
                "b": b,
                "num_inliers": ninliers,
                "rmse": rmse,
            }
        )

    first_consistent = load_depth(
        consistent_root
        / cam
        / f"{image_files[0].stem}.npy"
    )

    da3_metric = load_depth(
        da3_depth_path
    )

    da3_metric = resize_depth_to_match(
        da3_metric,
        first_consistent.shape[:2],
    )

    conf_path = (
        da3_depth_path.parent.parent
        / "conf"
        / f"{cam}.npy"
    )

    metric_mask = (
        static_mask
        & depth_valid_mask(first_consistent)
        & depth_valid_mask(da3_metric)
    )

    if conf_path.exists():
        conf = load_depth(
            conf_path
        )

        conf = resize_depth_to_match(
            conf,
            first_consistent.shape[:2],
        )

        finite = np.isfinite(conf)

        if finite.any():
            conf_thr = np.percentile(
                conf[finite],
                30,
            )

            high_conf_metric_mask = (
                metric_mask
                & (conf >= conf_thr)
            )

            if high_conf_metric_mask.sum() >= 64:
                metric_mask = high_conf_metric_mask

    if metric_mask.sum() < 64:
        raise RuntimeError(
            f"[Cam {cam}] insufficient pixels for DA3 metric fitting"
        )

    print(
        f"[Cam {cam}] metric calibration "
        f"against DA3 first frame"
    )

    metric_a, metric_b, metric_n, metric_rmse = robust_fit_depth_map(
        src=first_consistent,
        target=da3_metric,
        valid_mask=metric_mask,
        mode=metric_fit,
    )

    print(
        f"[Cam {cam}] metric mapping: "
        f"depth_metric = "
        f"{metric_a:.8f} * depth_consistent "
        f"+ {metric_b:.8f}"
    )

    if metric_a <= 0:
        raise RuntimeError(
            f"[Cam {cam}] invalid non-positive metric scale: {metric_a}"
        )

    if metric_fit == "scale_only" and abs(metric_b) > 1e-12:
        raise RuntimeError(
            f"[Cam {cam}] scale_only unexpectedly produced non-zero offset: {metric_b}"
        )

    print(
        f"[Cam {cam}] writing uint16 PNG metric depth"
    )

    for image_path, _ in pairs:
        aligned = load_depth(
            consistent_root
            / cam
            / f"{image_path.stem}.npy"
        )

        metric_depth = apply_transform(
            aligned,
            metric_a,
            metric_b,
        )

        save_depth_uint16_png(
            final_output_root
            / cam
            / f"{image_path.stem}.png",
            metric_depth,
            scale=final_png_scale,
        )

    save_json(
        meta_root
        / f"{cam}.json",
        {
            "cam": cam,
            "first_frame": image_files[0].name,
            "source_depth_root": str(source_depth_root),
            "final_output_root": str(final_output_root),
            "static_mask_pixels": int(static_mask.sum()),
            "temporal_fit_mode": temporal_fit,
            "metric_fit_mode": metric_fit,
            "final_png_scale": final_png_scale,
            "uint16_input_scale": uint16_input_scale,
            "metric_mapping": {
                "a": metric_a,
                "b": metric_b,
                "num_inliers": metric_n,
                "rmse": metric_rmse,
            },
            "temporal_alignment": temporal_records,
        },
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        required=True,
        help="Dataset root, e.g. ../biscuit",
    )

    parser.add_argument(
        "--model-dir",
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="Single-GPU DA3 device. auto/cuda selects the most-free GPU with >32 GiB free; cuda:N is memory-checked.",
    )

    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
    )

    parser.add_argument(
        "--process-res-method",
        default="upper_bound_resize",
        choices=[
            "upper_bound_resize",
            "lower_bound_resize",
        ],
    )

    parser.add_argument(
        "--max-static-samples",
        type=int,
        default=0,
        help=(
            "Frames used for static-region estimation. "
            "0 (default) = use ALL frames. "
            "Positive N = uniformly sample at most N frames."
        ),
    )

    parser.add_argument(
        "--temporal-fit",
        default="affine",
        choices=[
            "affine",
            "scale_only",
        ],
    )

    parser.add_argument(
        "--metric-fit",
        default="scale_only",
        choices=[
            "affine",
            "scale_only",
        ],
        help=(
            "Mapping from temporally-consistent relative depth to DA3 metric depth. "
            "Default scale_only avoids unstable large offsets."
        ),
    )

    parser.add_argument(
        "--uint16-input-scale",
        type=float,
        default=1.0,
        help=(
            "Only affects pre-generated uint16 source depths. "
            "Loaded float = uint16 / this value. "
            "Keep 1 for arbitrary relative-depth uint16."
        ),
    )

    parser.add_argument(
        "--png-scale",
        type=float,
        default=1000.0,
        help=(
            "Final uint16 PNG encoding scale. "
            "Default 1000 means metric depth in meters is stored as millimeters."
        ),
    )

    parser.add_argument(
        "--writer-threads",
        type=int,
        default=4,
        help="Background uint16 PNG writer threads per DA3 GPU worker. Default: 4.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=0,
        choices=range(10),
        help="PNG compression level. 0 is fastest and lossless, but larger. Default: 0.",
    )
    parser.add_argument(
        "--max-pending-writes",
        type=int,
        default=64,
        help="Bound async depth writes per GPU worker. Default: 64.",
    )

    parser.add_argument(
        "--overwrite-da3",
        action="store_true",
    )

    parser.add_argument(
        "--replace-depth",
        action="store_true",
        help=(
            "Move the original <folder>/depth to "
            "<folder>/depth_relative_original and write final metric uint16 PNGs "
            "back to <folder>/depth. Useful when downstream code is hard-coded "
            "to read <folder>/depth/<cam>/<frame>.png."
        ),
    )

    direct_group = parser.add_mutually_exclusive_group()
    direct_group.add_argument(
        "--first_only",
        action="store_true",
        help=(
            "Direct DA3 mode: jointly process only the first frame from every "
            "camera using intri.yml/extri.yml, then write uint16 metric depth "
            "directly to <folder>/depth/<cam>/<first_frame>.png. Skips all "
            "relative-depth temporal alignment."
        ),
    )
    direct_group.add_argument(
        "--all_da3",
        action="store_true",
        help=(
            "Direct DA3 mode: for every synchronized frame index, jointly process "
            "all camera views using intri.yml/extri.yml, then write every uint16 "
            "metric depth directly to <folder>/depth/<cam>/<frame>.png. Skips "
            "all relative-depth temporal alignment. This is the DEFAULT when "
            "neither --first_only nor --all_da3 is explicitly provided."
        ),
    )

    parser.add_argument(
        "--gpus",
        default="auto",
        help=(
            "GPU ids for --all_da3 multi-GPU inference. Default: auto. "
            "auto/all uses every visible GPU whose current free VRAM is strictly above "
            "--min_free_vram_mb. Explicit comma-separated ids are also memory-checked."
        ),
    )
    parser.add_argument(
        "--min_free_vram_mb",
        "--min-free-vram-mb",
        type=int,
        default=32768,
        help="Minimum free VRAM in MiB; selected GPUs must have STRICTLY more than this. Default: 32768 (32 GiB).",
    )

    args = parser.parse_args()

    # Default direct mode: ALL frames with DA3.
    # Explicit --first_only still works because all_da3 is enabled only when
    # neither mutually-exclusive mode flag was supplied.
    if not args.first_only and not args.all_da3:
        args.all_da3 = True
        print("[DA3 mode] No direct-mode flag provided -> defaulting to --all_da3")

    # Single-GPU modes choose the most-free qualifying device. For --all_da3 this
    # value is not used for workers, but resolving it keeps the CLI policy consistent.
    args.device = _resolve_single_device(args.device, args.min_free_vram_mb)

    folder = Path(args.folder)

    images_root = (
        folder
        / "images"
    )

    original_depth_root = (
        folder
        / "depth"
    )

    backup_depth_root = (
        folder
        / "depth_relative_original"
    )

    if not images_root.exists():
        raise FileNotFoundError(
            images_root
        )

    # --------------------------------------------------------
    # Direct DA3 modes bypass all pre-generated-depth alignment.
    # --------------------------------------------------------
    if args.first_only or args.all_da3:
        if args.replace_depth:
            print(
                "[warning] --replace-depth is ignored in --first_only/--all_da3 "
                "because direct DA3 modes already write directly to <folder>/depth."
            )

        cam_ids = collect_cam_ids(images_root)
        print("Cameras:", cam_ids)

        if args.all_da3:
            run_da3_all_multigpu(
                folder=folder,
                cam_ids=cam_ids,
                model_dir=args.model_dir,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                png_scale=args.png_scale,
                gpus=args.gpus,
                min_free_vram_mb=args.min_free_vram_mb,
                writer_threads=args.writer_threads,
                png_compression=args.png_compression,
                max_pending_writes=args.max_pending_writes,
            )
        else:
            # --first_only is one multi-view group, so it stays on one GPU.
            run_da3_direct(
                folder=folder,
                cam_ids=cam_ids,
                model_dir=args.model_dir,
                device=args.device,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                png_scale=args.png_scale,
                mode="first_only",
                png_compression=args.png_compression,
            )
        return

    # --------------------------------------------------------
    # Decide where relative source depth comes from and where
    # the final metric uint16 PNG depth will be written.
    # --------------------------------------------------------
    if args.replace_depth:
        if backup_depth_root.exists():
            source_depth_root = backup_depth_root

            # depth/ may already be a previous output. Remove it so we can
            # deterministically regenerate it.
            if original_depth_root.exists():
                print(
                    f"[replace-depth] Removing previous output: "
                    f"{original_depth_root}"
                )
                shutil.rmtree(
                    original_depth_root
                )

        else:
            if not original_depth_root.exists():
                raise FileNotFoundError(
                    original_depth_root
                )

            print(
                f"[replace-depth] Moving source relative depths:\n"
                f"  {original_depth_root}\n"
                f"-> {backup_depth_root}"
            )

            shutil.move(
                str(original_depth_root),
                str(backup_depth_root),
            )

            source_depth_root = backup_depth_root

        final_output_root = original_depth_root

    else:
        if original_depth_root.exists():
            source_depth_root = original_depth_root
        elif backup_depth_root.exists():
            source_depth_root = backup_depth_root
        else:
            raise FileNotFoundError(
                f"Neither {original_depth_root} nor {backup_depth_root} exists"
            )

        final_output_root = (
            folder
            / "depth_metric"
        )

    cam_ids = collect_cam_ids(
        images_root
    )

    print(
        "Cameras:",
        cam_ids,
    )

    print(
        "Relative source depth:",
        source_depth_root,
    )

    print(
        "Final uint16 metric depth:",
        final_output_root,
    )

    # --------------------------------------------------------
    # DA3 first-frame metric reference
    # --------------------------------------------------------
    da3_root = (
        folder
        / "da3_firstframe_metric"
    )

    da3_npy_root = (
        da3_root
        / "npy"
    )

    need_da3 = (
        args.overwrite_da3
        or not da3_npy_root.exists()
        or any(
            not (
                da3_npy_root
                / f"{cam}.npy"
            ).exists()
            for cam in cam_ids
        )
    )

    if need_da3:
        run_da3_first_frames(
            folder=folder,
            cam_ids=cam_ids,
            model_dir=args.model_dir,
            device=args.device,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            da3_out_dir=da3_root,
        )
    else:
        print(
            f"[DA3] Reusing {da3_root}"
        )

    # Internal float alignment output.
    consistent_root = (
        folder
        / "depth_consistent_float"
    )

    debug_root = (
        folder
        / "depth_alignment_debug"
    )

    static_mask_root = (
        debug_root
        / "static_masks"
    )

    meta_root = (
        debug_root
        / "meta"
    )

    ensure_dir(
        consistent_root
    )

    ensure_dir(
        final_output_root
    )

    ensure_dir(
        static_mask_root
    )

    ensure_dir(
        meta_root
    )

    # --------------------------------------------------------
    # Per-camera temporal alignment and metric scaling
    # --------------------------------------------------------
    for cam in cam_ids:
        process_one_camera(
            folder=folder,
            source_depth_root=source_depth_root,
            cam=cam,
            da3_depth_path=(
                da3_npy_root
                / f"{cam}.npy"
            ),
            consistent_root=consistent_root,
            final_output_root=final_output_root,
            static_mask_root=static_mask_root,
            meta_root=meta_root,
            max_static_samples=args.max_static_samples,
            temporal_fit=args.temporal_fit,
            metric_fit=args.metric_fit,
            final_png_scale=args.png_scale,
            uint16_input_scale=args.uint16_input_scale,
        )

    print()
    print("====================================================")
    print("Done")
    print("====================================================")
    print(
        "DA3 first-frame metric reference:",
        da3_root,
    )
    print(
        "Internal float consistent depths:",
        consistent_root,
    )
    print(
        "Final uint16 PNG metric depths:",
        final_output_root,
    )
    print(
        "Debug:",
        debug_root,
    )

    if args.replace_depth:
        print(
            "Original relative depths backed up at:",
            backup_depth_root,
        )
        print(
            "Downstream-compatible metric depth root:",
            folder / "depth",
        )
    else:
        print(
            "NOTE: downstream code hard-coded to <folder>/depth will still read "
            "the original relative depths. Use --replace-depth if you want the "
            "final metric uint16 PNGs written back to <folder>/depth."
        )

    print()
    print(
        f"PNG encoding: uint16 = round(metric_depth * {args.png_scale})"
    )


if __name__ == "__main__":
    main()
