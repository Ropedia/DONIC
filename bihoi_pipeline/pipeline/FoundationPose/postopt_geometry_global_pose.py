#!/usr/bin/env python3
"""
Geometry-aware global object-pose optimization for the user's custom FoundationPose pipeline.

Purpose
-------
Run AFTER the existing FoundationPose tracker.  This script does NOT re-run
FoundationPose.  It reads the final per-frame object->world poses and jointly
optimizes the full trajectory using:

    L = w_geom * L_geometry
      + w_reg_t * L_pose_prior_translation
      + w_reg_r * L_pose_prior_rotation
      + w_vel_t * L_translation_velocity
      + w_acc_t * L_translation_acceleration
      + w_vel_r * L_rotation_velocity
      + w_acc_r * L_rotation_acceleration

This mirrors the role of HOCap's 06_object_pose_solver.py, but is standalone and
compatible with the user's EasyMocap/FoundationPose directory layout.

Geometry term
-------------
For every frame:
  1. Use all calibrated cameras.
  2. Back-project masked metric-depth pixels into camera-space 3D.
  3. Transform them to WORLD using extri.yml (world_to_cam convention).
  4. Aggregate/downsample a multi-view object point cloud.
  5. Build data->mesh nearest-surface correspondences against sampled mesh points.
  6. Optimize pose with a robust point-to-surface loss (point-to-plane + point-to-point).

The nearest-surface correspondences are rebuilt between outer iterations, i.e.
an alternating ICP-style optimization.  This avoids requiring the HOCap-specific
MeshSDFLoss / SequenceLoader package while retaining geometry-aware global pose
refinement.

Expected default scene layout
-----------------------------
../<data_name>/
    intri.yml
    extri.yml                       # world_to_cam
    masks/<cam>/<frame>.png         # fallback: mask/
    depth/<cam>/<frame>.png|npy     # metric; uint16 default millimetres
    meshes/compressed.ply
    debug_depth/
        ob_in_world/*.txt
        ob_in_world_rotfix/*.txt
        ob_in_world_rotfix_postkal/*.txt

Pose convention
---------------
Each input pose is a 4x4 ORIGINAL-MESH -> WORLD transform:

    X_world = T_world_from_object @ X_object

The script auto-selects the newest/final logical pose stage in this order:
    ob_in_world_rotfix_postkal
    ob_in_world_rotfix
    ob_in_world

Output
------
<debug_dir>/geometry_global_opt/
    ob_in_world/<frame>.txt
    poses_world_initial.npy
    poses_world_optimized.npy
    pose_delta.csv
    optimization_log.csv
    run_config.json
    cloud_cache_<hash>/<frame>.npz

Example
-------
CUDA_VISIBLE_DEVICES=0 python postopt_geometry_global_pose.py \
    --data_name my_sequence \
    --data_root .. \
    --debug_dir ../my_sequence/debug_depth \
    --depth_scale 1000 \
    --fps 10

If intri.yml was calibrated at a different image resolution, pass e.g.:
    --calib_width 1920 --calib_height 1080
so K is scaled to each actual depth-map resolution.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import time
import csv
import glob
import hashlib
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEPTH_EXTS = (".png", ".tif", ".tiff", ".npy")


def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(s))]


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {v}")


def find_existing_file(folder: Path, stem: str, exts: Sequence[str]) -> Optional[Path]:
    for ext in exts:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    candidates = []
    for p in folder.glob(f"{stem}.*"):
        if p.suffix.lower() in exts:
            candidates.append(p)
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: natural_key(x.name))[0]


def load_names(fs: cv2.FileStorage) -> List[str]:
    node = fs.getNode("names")
    if node.empty():
        return []
    names = []
    for i in range(node.size()):
        s = node.at(i).string()
        if s:
            names.append(str(s))
    return names


def load_extrinsics_world_to_cam(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open extrinsics: {path}")
    names = load_names(fs)
    if not names:
        fs.release()
        raise ValueError(f"No names list in {path}")

    out: Dict[str, np.ndarray] = {}
    for cam in names:
        rot_node = fs.getNode(f"Rot_{cam}")
        t_node = fs.getNode(f"T_{cam}")
        R = rot_node.mat() if not rot_node.empty() else None
        t = t_node.mat() if not t_node.empty() else None

        if R is None:
            rvec_node = fs.getNode(f"R_{cam}")
            rvec = rvec_node.mat() if not rvec_node.empty() else None
            if rvec is not None:
                R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))

        if R is None or t is None:
            logging.warning("Missing extrinsic for cam %s; skipping", cam)
            continue

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(R, np.float64).reshape(3, 3)
        T[:3, 3] = np.asarray(t, np.float64).reshape(3)
        out[str(cam)] = T

    fs.release()
    if not out:
        raise RuntimeError(f"No usable camera extrinsics in {path}")
    return out


def _read_matrix_node(fs: cv2.FileStorage, keys: Sequence[str]) -> Optional[np.ndarray]:
    for key in keys:
        node = fs.getNode(key)
        if not node.empty():
            mat = node.mat()
            if mat is not None:
                return np.asarray(mat, np.float64)
    return None


def load_intrinsics(path: Path, cams: Sequence[str], fallback_txt: Optional[Path] = None) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if path.exists():
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            for cam in cams:
                K = _read_matrix_node(fs, [f"K_{cam}", f"k_{cam}"])
                if K is not None and K.size == 9:
                    out[str(cam)] = K.reshape(3, 3)
            fs.release()

    if len(out) < len(cams) and fallback_txt is not None and fallback_txt.exists():
        vals = []
        for line in fallback_txt.read_text().splitlines():
            line = line.strip().replace(",", " ")
            if not line or line.startswith("#"):
                continue
            vals.extend(float(x) for x in line.split())
        if len(vals) == 9:
            K_shared = np.asarray(vals, np.float64).reshape(3, 3)
            for cam in cams:
                out.setdefault(str(cam), K_shared.copy())

    missing = [c for c in cams if str(c) not in out]
    if missing:
        raise RuntimeError(
            f"Missing intrinsics for cameras {missing}. Tried {path}"
            + (f" and {fallback_txt}" if fallback_txt is not None else "")
        )
    return out


def scaled_K(K: np.ndarray, width: int, height: int, calib_width: int, calib_height: int) -> np.ndarray:
    K2 = np.asarray(K, np.float64).reshape(3, 3).copy()
    if calib_width > 0 and calib_height > 0:
        sx = float(width) / float(calib_width)
        sy = float(height) / float(calib_height)
        K2[0, 0] *= sx
        K2[0, 2] *= sx
        K2[1, 1] *= sy
        K2[1, 2] *= sy
    return K2


def read_mask(path: Path, erode_px: int) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    if m.ndim == 3:
        # Any foreground channel counts as object foreground.
        m = np.max(m, axis=2)
    m = (m > 0).astype(np.uint8)
    if erode_px > 0:
        k = 2 * int(erode_px) + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        m = cv2.erode(m, kernel, iterations=1)
    return m.astype(bool)


def read_depth_m(path: Path, integer_scale: float, float_scale: float) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        d = np.load(path)
    else:
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise RuntimeError(f"Failed to read depth: {path}")
        if d.ndim == 3:
            d = d[..., 0]

    if np.issubdtype(d.dtype, np.integer):
        d = d.astype(np.float32) / float(integer_scale)
    else:
        d = d.astype(np.float32) * float(float_scale)
    d[~np.isfinite(d)] = 0.0
    d[d <= 0] = 0.0
    return d


def voxel_downsample_np(points: np.ndarray, voxel: float) -> np.ndarray:
    if len(points) == 0 or voxel <= 0:
        return points
    keys = np.floor(points / float(voxel)).astype(np.int64)
    # Keep one deterministic representative per voxel.
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    return points[idx]


def camera_points_to_world(points_cam: np.ndarray, T_cam_from_world: np.ndarray) -> np.ndarray:
    Rcw = T_cam_from_world[:3, :3]
    tcw = T_cam_from_world[:3, 3]
    # p_cam = Rcw p_world + tcw => p_world = Rcw^T (p_cam - tcw)
    return (Rcw.T @ (points_cam - tcw[None]).T).T


def backproject_masked_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    T_cam_from_world: np.ndarray,
    min_depth: float,
    max_depth: float,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if mask.shape != depth.shape:
        mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    valid = mask & np.isfinite(depth) & (depth > float(min_depth)) & (depth < float(max_depth))
    vv, uu = np.where(valid)
    if len(uu) == 0:
        return np.empty((0, 3), np.float32)
    if max_points > 0 and len(uu) > max_points:
        sel = rng.choice(len(uu), size=max_points, replace=False)
        uu = uu[sel]
        vv = vv[sel]
    z = depth[vv, uu].astype(np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (uu.astype(np.float64) - cx) * z / fx
    y = (vv.astype(np.float64) - cy) * z / fy
    pc = np.stack([x, y, z], axis=1)
    pw = camera_points_to_world(pc, T_cam_from_world)
    return pw.astype(np.float32)


def resolve_pose_dir(debug_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"Pose dir does not exist: {p}")
        return p
    for name in ["ob_in_world_rotfix_postkal", "ob_in_world_rotfix", "ob_in_world"]:
        p = debug_dir / name
        if p.is_dir() and any(p.glob("*.txt")):
            logging.info("Auto-selected pose directory: %s", p)
            return p
    raise FileNotFoundError(
        f"No pose folder found under {debug_dir}; expected ob_in_world_rotfix_postkal, ob_in_world_rotfix, or ob_in_world"
    )


def load_pose_sequence(pose_dir: Path) -> Tuple[List[str], np.ndarray]:
    files = sorted(pose_dir.glob("*.txt"), key=lambda p: natural_key(p.stem))
    if not files:
        raise RuntimeError(f"No .txt poses in {pose_dir}")
    frame_ids: List[str] = []
    poses: List[np.ndarray] = []
    for f in files:
        T = np.loadtxt(f, dtype=np.float64)
        if T.size != 16:
            logging.warning("Skipping malformed pose %s shape=%s", f, T.shape)
            continue
        T = T.reshape(4, 4)
        if not np.all(np.isfinite(T)):
            logging.warning("Skipping non-finite pose %s", f)
            continue
        frame_ids.append(f.stem)
        poses.append(T)
    if not poses:
        raise RuntimeError(f"No valid poses in {pose_dir}")
    return frame_ids, np.stack(poses, axis=0)


def project_to_so3_np(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0:
        U[:, -1] *= -1
        Rp = U @ Vt
    return Rp


def rotation_angle_deg_np(Ra: np.ndarray, Rb: np.ndarray) -> float:
    Rrel = project_to_so3_np(Ra).T @ project_to_so3_np(Rb)
    c = np.clip((np.trace(Rrel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def skew_torch(w: torch.Tensor) -> torch.Tensor:
    wx, wy, wz = w.unbind(-1)
    O = torch.zeros_like(wx)
    return torch.stack(
        [O, -wz, wy, wz, O, -wx, -wy, wx, O], dim=-1
    ).reshape(w.shape[:-1] + (3, 3))


def so3_exp_torch(w: torch.Tensor) -> torch.Tensor:
    """Differentiable axis-angle exponential map."""
    theta2 = torch.sum(w * w, dim=-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-16)
    W = skew_torch(w)
    W2 = W @ W

    # Stable Taylor expansions around zero.
    small = theta < 1e-4
    A = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta,
    )
    B = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / (theta2 + 1e-16),
    )
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand(w.shape[:-1] + (3, 3))
    return I + A[..., None] * W + B[..., None] * W2


def so3_log_torch(R: torch.Tensor) -> torch.Tensor:
    """Stable-enough SO(3) logarithm for adjacent trajectory rotations."""
    v = 0.5 * torch.stack(
        [R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]],
        dim=-1,
    )
    sin_theta = torch.linalg.norm(v, dim=-1, keepdim=True)
    cos_theta = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).unsqueeze(-1)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.atan2(sin_theta, cos_theta)
    scale = torch.where(sin_theta < 1e-5, torch.ones_like(sin_theta), theta / (sin_theta + 1e-12))
    return scale * v


def pseudo_huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    d = float(delta)
    return d * d * (torch.sqrt(1.0 + (x / d) ** 2) - 1.0)


@dataclass
class Correspondences:
    frame_idx: np.ndarray
    model_idx: np.ndarray
    target_world: np.ndarray
    nearest_dist: np.ndarray
    active_frames: int


class GeometryGlobalOptimizer:
    def __init__(self, args):
        self.args = args
        self.scene = self._resolve_scene()
        self.debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else self.scene / "debug_depth"
        self.pose_dir = resolve_pose_dir(self.debug_dir, args.pose_dir)
        self.out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else self.debug_dir / "geometry_global_opt"
        self.out_world = self.out_dir / "ob_in_world"
        self.out_world.mkdir(parents=True, exist_ok=True)

        self.depth_root = Path(args.depth_dir).expanduser().resolve() if args.depth_dir else self.scene / "depth"
        if args.mask_dir:
            self.mask_root = Path(args.mask_dir).expanduser().resolve()
        else:
            self.mask_root = self.scene / "masks"
            if not self.mask_root.is_dir():
                self.mask_root = self.scene / "mask"
        self.mesh_path = Path(args.mesh_file).expanduser().resolve() if args.mesh_file else self.scene / "meshes" / "compressed.ply"
        self.intri_path = Path(args.intri).expanduser().resolve() if args.intri else self.scene / "intri.yml"
        self.extri_path = Path(args.extri).expanduser().resolve() if args.extri else self.scene / "extri.yml"

        self._check_inputs()
        self.frame_ids, self.poses0 = load_pose_sequence(self.pose_dir)
        self.R0 = np.stack([project_to_so3_np(T[:3, :3]) for T in self.poses0], axis=0)
        self.t0 = self.poses0[:, :3, 3].copy()

        self.extri = load_extrinsics_world_to_cam(self.extri_path)
        cams = sorted(self.extri.keys(), key=natural_key)
        if args.cams:
            wanted = [x.strip() for x in args.cams.split(",") if x.strip()]
            cams = [c for c in cams if c in wanted]
        cams = [c for c in cams if (self.depth_root / c).is_dir() and (self.mask_root / c).is_dir()]
        if not cams:
            raise RuntimeError("No cameras have both depth and mask folders")
        self.cams = cams
        self.Ks = load_intrinsics(self.intri_path, self.cams, fallback_txt=self.scene / "cam_K.txt")

        mesh = trimesh.load(str(self.mesh_path), force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected Trimesh, got {type(mesh)}")
        if len(mesh.faces) == 0:
            raise ValueError("Mesh has no faces")
        self.mesh = mesh

        # Sample a fixed object-space surface point set.  Each sample carries its face normal.
        rng_state = np.random.get_state()
        np.random.seed(int(args.seed))
        pts, face_idx = trimesh.sample.sample_surface(mesh, int(args.mesh_samples))
        np.random.set_state(rng_state)
        self.model_points_obj = np.asarray(pts, np.float32)
        self.model_normals_obj = np.asarray(mesh.face_normals[np.asarray(face_idx, np.int64)], np.float32)

        self.device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
        self.dtype = torch.float32
        self.R0_t = torch.as_tensor(self.R0, device=self.device, dtype=self.dtype)
        self.t0_t = torch.as_tensor(self.t0, device=self.device, dtype=self.dtype)
        self.model_points_t = torch.as_tensor(self.model_points_obj, device=self.device, dtype=self.dtype)
        self.model_normals_t = torch.as_tensor(self.model_normals_obj, device=self.device, dtype=self.dtype)

        self.delta_r = torch.nn.Parameter(torch.zeros((len(self.frame_ids), 3), device=self.device, dtype=self.dtype))
        self.delta_t = torch.nn.Parameter(torch.zeros((len(self.frame_ids), 3), device=self.device, dtype=self.dtype))

        groups = []
        if not args.freeze_rotation:
            groups.append({"params": [self.delta_r], "lr": float(args.lr_rot)})
        else:
            self.delta_r.requires_grad_(False)
        if not args.freeze_translation:
            groups.append({"params": [self.delta_t], "lr": float(args.lr_trans)})
        else:
            self.delta_t.requires_grad_(False)
        if not groups:
            raise ValueError("Both rotation and translation are frozen; nothing to optimize")
        self.optimizer = torch.optim.Adam(groups)

        self.cache_dir = self._make_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._write_config()

    def _resolve_scene(self) -> Path:
        if self.args.scene_dir:
            return Path(self.args.scene_dir).expanduser().resolve()
        if not self.args.data_name:
            raise ValueError("Provide --data_name or --scene_dir")
        return (Path(self.args.data_root).expanduser() / self.args.data_name).resolve()

    def _check_inputs(self):
        for p, desc in [
            (self.depth_root, "depth dir"),
            (self.mask_root, "mask dir"),
            (self.mesh_path, "mesh"),
            (self.extri_path, "extri.yml"),
        ]:
            if not p.exists():
                raise FileNotFoundError(f"Missing {desc}: {p}")
        if not self.intri_path.exists() and not (self.scene / "cam_K.txt").exists():
            raise FileNotFoundError(f"Need {self.intri_path} or {self.scene / 'cam_K.txt'}")

    def _make_cache_dir(self) -> Path:
        payload = {
            "depth": str(self.depth_root), "mask": str(self.mask_root), "cams": self.cams,
            "depth_scale": self.args.depth_scale, "float_depth_scale": self.args.float_depth_scale,
            "min_depth": self.args.min_depth, "max_depth": self.args.max_depth,
            "erode_px": self.args.erode_px, "sample_per_cam": self.args.sample_per_cam,
            "voxel": self.args.voxel_size, "max_obs": self.args.max_obs_per_frame,
            "calib_width": self.args.calib_width, "calib_height": self.args.calib_height,
        }
        h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf8")).hexdigest()[:10]
        return self.out_dir / f"cloud_cache_{h}"

    def _write_config(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        cfg = vars(self.args).copy()
        cfg.update({
            "resolved_scene": str(self.scene), "resolved_debug_dir": str(self.debug_dir),
            "resolved_pose_dir": str(self.pose_dir), "resolved_depth_root": str(self.depth_root),
            "resolved_mask_root": str(self.mask_root), "resolved_mesh": str(self.mesh_path),
            "resolved_intri": str(self.intri_path), "resolved_extri": str(self.extri_path),
            "resolved_cams": self.cams, "num_frames": len(self.frame_ids),
            "cache_dir": str(self.cache_dir),
        })
        (self.out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))

    def current_pose_torch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dR = so3_exp_torch(self.delta_r)
        R = dR @ self.R0_t
        t = self.t0_t + self.delta_t
        return R, t

    @torch.no_grad()
    def current_pose_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        R, t = self.current_pose_torch()
        return R.detach().cpu().double().numpy(), t.detach().cpu().double().numpy()

    def build_frame_cloud(self, frame_idx: int) -> np.ndarray:
        frame_id = self.frame_ids[frame_idx]
        cache_file = self.cache_dir / f"{frame_id}.npz"
        if self.args.reuse_cloud_cache and cache_file.exists():
            try:
                return np.load(cache_file, allow_pickle=False)["points"].astype(np.float32, copy=False)
            except Exception:
                pass

        rng = np.random.default_rng(int(self.args.seed) + frame_idx * 1009)
        clouds = []
        used = 0
        for cam in self.cams:
            dfile = find_existing_file(self.depth_root / cam, frame_id, DEPTH_EXTS)
            mfile = find_existing_file(self.mask_root / cam, frame_id, IMAGE_EXTS)
            if dfile is None or mfile is None:
                continue
            try:
                depth = read_depth_m(dfile, self.args.depth_scale, self.args.float_depth_scale)
                mask = read_mask(mfile, self.args.erode_px)
                H, W = depth.shape[:2]
                K = scaled_K(self.Ks[cam], W, H, self.args.calib_width, self.args.calib_height)
                pts = backproject_masked_depth(
                    depth, mask, K, self.extri[cam], self.args.min_depth, self.args.max_depth,
                    self.args.sample_per_cam, rng,
                )
                if len(pts):
                    clouds.append(pts)
                    used += 1
            except Exception as exc:
                logging.warning("frame=%s cam=%s depth-cloud failed: %s", frame_id, cam, exc)

        if not clouds:
            pts = np.empty((0, 3), np.float32)
        else:
            pts = np.concatenate(clouds, axis=0).astype(np.float32)
            pts = voxel_downsample_np(pts, self.args.voxel_size).astype(np.float32)
            if self.args.max_obs_per_frame > 0 and len(pts) > self.args.max_obs_per_frame:
                sel = rng.choice(len(pts), self.args.max_obs_per_frame, replace=False)
                pts = pts[sel]

        # Compression is surprisingly expensive when hundreds of frames are cached.
        # The fast default uses an uncompressed .npz: more disk, much less CPU time.
        if bool(self.args.cache_compress):
            np.savez_compressed(cache_file, points=pts, cameras_used=np.int32(used))
        else:
            np.savez(cache_file, points=pts, cameras_used=np.int32(used))
        return pts

    def preload_clouds(self) -> List[np.ndarray]:
        total = len(self.frame_ids)
        workers = max(1, int(self.args.cloud_workers))
        logging.info(
            "Preparing multi-view metric-depth clouds for %d frames using %d cameras | cloud_workers=%d",
            total, len(self.cams), workers,
        )
        t0 = time.perf_counter()

        if workers <= 1:
            clouds = []
            for i, fid in enumerate(self.frame_ids):
                pts = self.build_frame_cloud(i)
                clouds.append(pts)
                if (i + 1) % 25 == 0 or i + 1 == total:
                    logging.info("clouds %d/%d | frame=%s points=%d", i + 1, total, fid, len(pts))
        else:
            # build_frame_cloud() is frame-local and deterministic (per-frame RNG),
            # so frames can safely be prepared concurrently.  executor.map preserves
            # input/output ordering, which keeps frame indexing unchanged.
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                clouds = []
                for i, pts in enumerate(ex.map(self.build_frame_cloud, range(total))):
                    clouds.append(pts)
                    if (i + 1) % 25 == 0 or i + 1 == total:
                        logging.info(
                            "clouds %d/%d | frame=%s points=%d",
                            i + 1, total, self.frame_ids[i], len(pts),
                        )

        logging.info(
            "[timing] cloud preparation %.2fs (%.4fs/frame)",
            time.perf_counter() - t0,
            (time.perf_counter() - t0) / max(total, 1),
        )
        return clouds

    def build_correspondences(self, clouds: Sequence[np.ndarray], outer_idx: int) -> Correspondences:
        R, t = self.current_pose_numpy()
        all_f, all_m, all_y, all_d = [], [], [], []
        active_frames = 0
        rng = np.random.default_rng(int(self.args.seed) + 500000 + outer_idx)
        workers = max(1, int(self.args.corr_workers))
        t0 = time.perf_counter()

        def _query_one(item):
            i, obs = item
            if len(obs) < self.args.min_corr_points:
                return i, obs, None, None
            model_world = self.model_points_obj.astype(np.float64) @ R[i].T + t[i][None]
            tree = cKDTree(model_world)

            # When several frames are processed concurrently, keep each individual
            # query single-threaded to avoid workers x workers CPU oversubscription.
            query_workers = 1 if workers > 1 else -1
            try:
                dist, midx = tree.query(obs.astype(np.float64), k=1, workers=query_workers)
            except TypeError:
                dist, midx = tree.query(obs.astype(np.float64), k=1)
            return i, obs, np.asarray(dist), np.asarray(midx)

        indexed_clouds = list(enumerate(clouds))
        if workers <= 1:
            queried = map(_query_one, indexed_clouds)
        else:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
            queried = executor.map(_query_one, indexed_clouds)

        try:
            # Keep filtering/downsampling in deterministic frame order. This preserves
            # one shared RNG stream for corr_per_frame sampling just like the original.
            for i, obs, dist, midx in queried:
                if dist is None:
                    continue
                keep = np.isfinite(dist) & (dist < float(self.args.corr_dist))
                ids = np.where(keep)[0]
                if len(ids) < self.args.min_corr_points:
                    continue
                if self.args.corr_per_frame > 0 and len(ids) > self.args.corr_per_frame:
                    ids = rng.choice(ids, size=self.args.corr_per_frame, replace=False)
                active_frames += 1
                all_f.append(np.full((len(ids),), i, np.int64))
                all_m.append(np.asarray(midx[ids], np.int64))
                all_y.append(obs[ids].astype(np.float32))
                all_d.append(np.asarray(dist[ids], np.float32))
        finally:
            if workers > 1:
                executor.shutdown(wait=True)

        elapsed = time.perf_counter() - t0
        if not all_f:
            logging.info(
                "[timing] correspondence rebuild outer=%d %.2fs | active_frames=0",
                outer_idx + 1, elapsed,
            )
            return Correspondences(
                frame_idx=np.empty((0,), np.int64),
                model_idx=np.empty((0,), np.int64),
                target_world=np.empty((0, 3), np.float32),
                nearest_dist=np.empty((0,), np.float32),
                active_frames=0,
            )

        corr = Correspondences(
            frame_idx=np.concatenate(all_f),
            model_idx=np.concatenate(all_m),
            target_world=np.concatenate(all_y),
            nearest_dist=np.concatenate(all_d),
            active_frames=active_frames,
        )
        logging.info(
            "[timing] correspondence rebuild outer=%d %.2fs | workers=%d active_frames=%d corr=%d",
            outer_idx + 1, elapsed, workers, active_frames, len(corr.frame_idx),
        )
        return corr

    def geometry_loss(self, corr: Correspondences, R: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = len(corr.frame_idx)
        if n == 0:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        if self.args.geom_batch_size > 0 and n > self.args.geom_batch_size:
            ids_np = np.random.choice(n, size=self.args.geom_batch_size, replace=False)
        else:
            ids_np = np.arange(n)

        fi = torch.as_tensor(corr.frame_idx[ids_np], device=self.device, dtype=torch.long)
        mi = torch.as_tensor(corr.model_idx[ids_np], device=self.device, dtype=torch.long)
        y = torch.as_tensor(corr.target_world[ids_np], device=self.device, dtype=self.dtype)
        x_obj = self.model_points_t[mi]
        n_obj = self.model_normals_t[mi]

        if R is None or t is None:
            R, t = self.current_pose_torch()
        Rb = R[fi]
        tb = t[fi]
        xw = torch.bmm(Rb, x_obj.unsqueeze(-1)).squeeze(-1) + tb
        nw = torch.bmm(Rb, n_obj.unsqueeze(-1)).squeeze(-1)
        nw = nw / (torch.linalg.norm(nw, dim=-1, keepdim=True) + 1e-12)

        diff = xw - y
        p2p = torch.linalg.norm(diff, dim=-1) / float(self.args.geom_sigma)
        p2plane = torch.abs(torch.sum(diff * nw, dim=-1)) / float(self.args.geom_sigma)
        a = float(self.args.point_plane_weight)
        return (a * pseudo_huber(p2plane, 1.0) + (1.0 - a) * pseudo_huber(p2p, 1.0)).mean()

    def regularization_losses(self, R: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if R is None or t is None:
            R, t = self.current_pose_torch()
        losses: Dict[str, torch.Tensor] = {}
        losses["reg_t"] = ((self.delta_t / float(self.args.prior_trans_sigma)) ** 2).sum(-1).mean()
        losses["reg_r"] = ((self.delta_r / math.radians(float(self.args.prior_rot_sigma_deg))) ** 2).sum(-1).mean()

        if len(self.frame_ids) >= 2:
            vt = t[1:] - t[:-1]
            relR = R[:-1].transpose(-1, -2) @ R[1:]
            vr = so3_log_torch(relR)
            losses["vel_t"] = ((vt / float(self.args.trans_vel_sigma)) ** 2).sum(-1).mean()
            losses["vel_r"] = ((vr / math.radians(float(self.args.rot_vel_sigma_deg))) ** 2).sum(-1).mean()
        else:
            z = torch.zeros((), device=self.device, dtype=self.dtype)
            losses["vel_t"] = z
            losses["vel_r"] = z

        if len(self.frame_ids) >= 3:
            at = (t[2:] - t[1:-1]) - (t[1:-1] - t[:-2])
            relR1 = R[:-1].transpose(-1, -2) @ R[1:]
            vr = so3_log_torch(relR1)
            ar = vr[1:] - vr[:-1]
            losses["acc_t"] = ((at / float(self.args.trans_acc_sigma)) ** 2).sum(-1).mean()
            losses["acc_r"] = ((ar / math.radians(float(self.args.rot_acc_sigma_deg))) ** 2).sum(-1).mean()
        else:
            z = torch.zeros((), device=self.device, dtype=self.dtype)
            losses["acc_t"] = z
            losses["acc_r"] = z
        return losses

    @torch.no_grad()
    def clamp_updates(self):
        if self.delta_r.requires_grad:
            max_r = math.radians(float(self.args.max_rot_update_deg))
            n = torch.linalg.norm(self.delta_r, dim=-1, keepdim=True)
            scale = torch.clamp(max_r / (n + 1e-12), max=1.0)
            self.delta_r.mul_(scale)
        if self.delta_t.requires_grad:
            max_t = float(self.args.max_trans_update_m)
            n = torch.linalg.norm(self.delta_t, dim=-1, keepdim=True)
            scale = torch.clamp(max_t / (n + 1e-12), max=1.0)
            self.delta_t.mul_(scale)

    def total_loss(self, corr: Correspondences) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Avoid computing the SO(3) exponential twice per Adam step.
        R, t = self.current_pose_torch()
        geom = self.geometry_loss(corr, R=R, t=t)
        ls = self.regularization_losses(R=R, t=t)
        total = (
            float(self.args.w_geom) * geom
            + float(self.args.w_reg_t) * ls["reg_t"]
            + float(self.args.w_reg_r) * ls["reg_r"]
            + float(self.args.w_vel_t) * ls["vel_t"]
            + float(self.args.w_acc_t) * ls["acc_t"]
            + float(self.args.w_vel_r) * ls["vel_r"]
            + float(self.args.w_acc_r) * ls["acc_r"]
        )
        vals = {"geom": float(geom.detach().cpu())}
        vals.update({k: float(v.detach().cpu()) for k, v in ls.items()})
        vals["total"] = float(total.detach().cpu())
        return total, vals

    def optimize(self, clouds: Sequence[np.ndarray]):
        log_rows = []
        global_step = 0
        for outer in range(int(self.args.outer_iters)):
            outer_t0 = time.perf_counter()
            corr = self.build_correspondences(clouds, outer)
            if len(corr.frame_idx) == 0:
                raise RuntimeError(
                    "No geometry correspondences. Check metric depth scale, intrinsics resolution, extri.yml convention, masks, and --corr_dist."
                )
            mean_nn = float(np.mean(corr.nearest_dist))
            med_nn = float(np.median(corr.nearest_dist))
            logging.info(
                "OUTER %d/%d | active_frames=%d/%d corr=%d meanNN=%.4fm medianNN=%.4fm",
                outer + 1, self.args.outer_iters, corr.active_frames, len(self.frame_ids),
                len(corr.frame_idx), mean_nn, med_nn,
            )

            for inner in range(int(self.args.inner_steps)):
                self.optimizer.zero_grad(set_to_none=True)
                total, vals = self.total_loss(corr)
                total.backward()
                if self.args.grad_clip > 0:
                    params = [p for p in [self.delta_r, self.delta_t] if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(params, float(self.args.grad_clip))
                self.optimizer.step()
                self.clamp_updates()
                global_step += 1

                row = {
                    "outer": outer, "inner": inner, "global_step": global_step,
                    "active_frames": corr.active_frames, "num_corr": len(corr.frame_idx),
                    "mean_nn_m": mean_nn, "median_nn_m": med_nn,
                    **vals,
                }
                log_rows.append(row)
                if inner == 0 or (inner + 1) % max(1, int(self.args.log_every)) == 0 or inner + 1 == int(self.args.inner_steps):
                    logging.info(
                        "  step=%d total=%.6f geom=%.6f reg_t=%.6f reg_r=%.6f acc_t=%.6f acc_r=%.6f",
                        global_step, vals["total"], vals["geom"], vals["reg_t"], vals["reg_r"], vals["acc_t"], vals["acc_r"],
                    )

            logging.info(
                "[timing] outer %d/%d total %.2fs",
                outer + 1, self.args.outer_iters, time.perf_counter() - outer_t0,
            )

        if log_rows:
            with open(self.out_dir / "optimization_log.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
                w.writeheader()
                w.writerows(log_rows)

    def save(self):
        R, t = self.current_pose_numpy()
        poses = np.tile(np.eye(4, dtype=np.float64), (len(self.frame_ids), 1, 1))
        poses[:, :3, :3] = R
        poses[:, :3, 3] = t
        np.save(self.out_dir / "poses_world_initial.npy", self.poses0.astype(np.float64))
        np.save(self.out_dir / "poses_world_optimized.npy", poses.astype(np.float64))

        rows = []
        for i, fid in enumerate(self.frame_ids):
            np.savetxt(self.out_world / f"{fid}.txt", poses[i])
            trans_shift = float(np.linalg.norm(t[i] - self.t0[i]))
            rot_shift = rotation_angle_deg_np(self.R0[i], R[i])
            rows.append({
                "frame_idx": i, "frame_id": fid,
                "trans_update_m": trans_shift, "trans_update_cm": trans_shift * 100.0,
                "rot_update_deg": rot_shift,
                "tx_init": self.t0[i, 0], "ty_init": self.t0[i, 1], "tz_init": self.t0[i, 2],
                "tx_opt": t[i, 0], "ty_opt": t[i, 1], "tz_opt": t[i, 2],
            })
        with open(self.out_dir / "pose_delta.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        shifts_t = np.asarray([r["trans_update_m"] for r in rows])
        shifts_r = np.asarray([r["rot_update_deg"] for r in rows])
        logging.info("Saved optimized WORLD poses: %s", self.out_world)
        logging.info(
            "Pose update summary: translation median=%.3f cm max=%.3f cm | rotation median=%.3f deg max=%.3f deg",
            np.median(shifts_t) * 100.0, np.max(shifts_t) * 100.0,
            np.median(shifts_r), np.max(shifts_r),
        )

    def run(self):
        logging.info("Scene: %s", self.scene)
        logging.info("Input pose dir: %s", self.pose_dir)
        logging.info("Output dir: %s", self.out_dir)
        logging.info("Frames=%d cameras=%d device=%s", len(self.frame_ids), len(self.cams), self.device)
        logging.info("Pose convention: ORIGINAL MESH -> WORLD; extrinsics: WORLD -> CAMERA")
        clouds = self.preload_clouds()
        n_nonempty = sum(len(x) >= self.args.min_corr_points for x in clouds)
        logging.info("Usable depth-cloud frames: %d/%d", n_nonempty, len(clouds))
        if n_nonempty == 0:
            raise RuntimeError("All masked depth clouds are empty. Check depth scale / masks / frame IDs.")
        self.optimize(clouds)
        self.save()


def build_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Geometry-aware global object pose optimization after custom FoundationPose",
    )
    # Paths / layout
    p.add_argument("--data_name", default=None, help="Dataset name, e.g. my_sequence")
    p.add_argument("--data_root", default="..", help="Dataset parent; scene=<data_root>/<data_name>")
    p.add_argument("--scene_dir", default=None, help="Explicit scene dir; overrides data_name/data_root")
    p.add_argument("--debug_dir", default=None, help="FoundationPose debug dir; default <scene>/debug_depth")
    p.add_argument("--pose_dir", default=None, help="Explicit object->world pose folder; otherwise auto-detect final stage")
    p.add_argument("--output_dir", default=None, help="Default <debug_dir>/geometry_global_opt")
    p.add_argument("--mesh_file", default=None, help="Default <scene>/meshes/compressed.ply")
    p.add_argument("--depth_dir", default=None, help="Default <scene>/depth")
    p.add_argument("--mask_dir", default=None, help="Default <scene>/masks, fallback <scene>/mask")
    p.add_argument("--intri", default=None, help="Default <scene>/intri.yml")
    p.add_argument("--extri", default=None, help="Default <scene>/extri.yml; interpreted as world_to_cam")
    p.add_argument("--cams", default=None, help="Optional comma-separated camera IDs; default all calibrated cams with depth+mask")

    # Depth / point clouds
    p.add_argument("--depth_scale", type=float, default=1000.0, help="Integer depth divisor; uint16 millimetres => 1000")
    p.add_argument("--float_depth_scale", type=float, default=1.0, help="Multiplier for float/npy depth already in metres")
    p.add_argument("--min_depth", type=float, default=0.05)
    p.add_argument("--max_depth", type=float, default=5.0)
    p.add_argument("--erode_px", type=int, default=2, help="Erode object mask to avoid hand/boundary depth contamination")
    p.add_argument("--sample_per_cam", type=int, default=1200, help="Max masked depth pixels sampled from each camera per frame")
    p.add_argument("--voxel_size", type=float, default=0.003, help="WORLD cloud voxel size in metres")
    p.add_argument("--max_obs_per_frame", type=int, default=6000, help="Cap aggregated multi-view points per frame")
    p.add_argument("--calib_width", type=int, default=0, help="If >0, intri.yml calibration image width; K is scaled to actual depth width")
    p.add_argument("--calib_height", type=int, default=0, help="If >0, intri.yml calibration image height; K is scaled to actual depth height")
    p.add_argument("--reuse_cloud_cache", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument(
        "--cache_compress",
        type=str2bool, nargs="?", const=True, default=False,
        help="Compress cloud cache .npz files. False is faster and is the new default.",
    )
    p.add_argument(
        "--cloud_workers",
        type=int, default=8,
        help="Parallel frame workers for depth+mask cloud preparation. 1 restores serial behavior.",
    )

    # Surface correspondences / geometry
    p.add_argument("--mesh_samples", type=int, default=12000, help="Fixed surface samples from input mesh")
    p.add_argument("--corr_dist", type=float, default=0.03, help="Reject data->mesh correspondence beyond this many metres")
    p.add_argument("--corr_per_frame", type=int, default=1500, help="Max geometry correspondences per frame per outer iteration")
    p.add_argument("--min_corr_points", type=int, default=100, help="Frame has no geometry term if fewer correspondences survive")
    p.add_argument("--geom_sigma", type=float, default=0.005, help="Geometry residual normalization, metres")
    p.add_argument("--point_plane_weight", type=float, default=0.7, help="0=p2p only, 1=p2plane only")
    p.add_argument("--geom_batch_size", type=int, default=120000, help="Random correspondences used per Adam step; 0=all")
    p.add_argument(
        "--corr_workers",
        type=int, default=8,
        help="Parallel frame workers for cKDTree build/query during each correspondence rebuild. 1 uses the original serial-frame path.",
    )

    # Optimization
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--outer_iters", type=int, default=6, help="Rebuild nearest-surface correspondences this many times")
    p.add_argument("--inner_steps", type=int, default=20, help="Adam steps per correspondence rebuild")
    p.add_argument("--lr_rot", type=float, default=0.003, help="Learning rate for axis-angle pose correction")
    p.add_argument("--lr_trans", type=float, default=0.001, help="Learning rate for translation correction")
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--max_rot_update_deg", type=float, default=20.0, help="Hard radius around initial rotation")
    p.add_argument("--max_trans_update_m", type=float, default=0.03, help="Hard radius around initial translation")
    p.add_argument("--freeze_rotation", action="store_true")
    p.add_argument("--freeze_translation", action="store_true")

    # Normalized prior/smoothness scales
    p.add_argument("--prior_trans_sigma", type=float, default=0.015, help="Translation prior scale in metres")
    p.add_argument("--prior_rot_sigma_deg", type=float, default=8.0, help="Rotation prior scale in degrees")
    p.add_argument("--trans_vel_sigma", type=float, default=0.03, help="Expected translation per frame scale, metres/frame")
    p.add_argument("--trans_acc_sigma", type=float, default=0.008, help="Expected translation acceleration scale, metres/frame^2")
    p.add_argument("--rot_vel_sigma_deg", type=float, default=15.0, help="Expected rotation per frame scale, deg/frame")
    p.add_argument("--rot_acc_sigma_deg", type=float, default=5.0, help="Expected angular acceleration scale, deg/frame^2")

    # Loss weights. Conservative defaults: geometry corrects, but cannot freely rewrite a good FP trajectory.
    p.add_argument("--w_geom", type=float, default=1.0)
    p.add_argument("--w_reg_t", type=float, default=0.25)
    p.add_argument("--w_reg_r", type=float, default=0.25)
    p.add_argument("--w_vel_t", type=float, default=0.005)
    p.add_argument("--w_acc_t", type=float, default=0.05)
    p.add_argument("--w_vel_r", type=float, default=0.005)
    p.add_argument("--w_acc_r", type=float, default=0.05)

    p.add_argument("--fps", type=float, default=10.0, help="Recorded for provenance; smoothness scales above are per-frame")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=5)
    return p


def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    if not (0.0 <= args.point_plane_weight <= 1.0):
        raise ValueError("--point_plane_weight must be in [0,1]")
    if args.depth_scale <= 0 or args.geom_sigma <= 0:
        raise ValueError("depth_scale and geom_sigma must be > 0")
    solver = GeometryGlobalOptimizer(args)
    solver.run()


if __name__ == "__main__":
    main()
