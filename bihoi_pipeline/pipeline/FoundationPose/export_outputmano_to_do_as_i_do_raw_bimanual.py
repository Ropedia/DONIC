#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert an already-generated outputmano folder into the raw format expected by
Do-as-I-Do retargeting, without re-running contact refinement/rendering.

Expected input produced by render_object_both_hands_contact_mano_fit_save48*.py:
  outputmano/contact_mano/right/smpl/*.json
  outputmano/contact_mano/left/smpl/*.json   optional
  outputmano/smoothed_ob_in_world/*.txt      optional, preferred if available

Expected HOI dataset folder:
  test_scene_dir/extri.yml
  mesh_file, e.g. test_scene_dir/meshes/compressed.ply

Output Do-as-I-Do raw folder:
  <dai_raw_dir>/config.json
  <dai_raw_dir>/gravity.json
  <dai_raw_dir>/<dai_task>/all_hand_meshes.npz
  <dai_raw_dir>/obj_tracking_out/<object>/combined_visualization/layout_camera_frame_optimized.json
  <dai_raw_dir>/video_segmentation/masks/frame_000000_masks/<object>/<object>.obj
"""

import argparse
import json
import logging
import math
import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import smplx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------- basic utilities -----------------------------


def set_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def natural_sort_key(path):
    name = Path(path).stem if not isinstance(path, Path) else path.stem
    parts = re.split(r"(\d+)", str(name))
    return [int(s) if s.isdigit() else s.lower() for s in parts]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def flatten_np(x, dtype=np.float64):
    return np.asarray(x, dtype=dtype).reshape(-1)


def normalize_vec3(v, name="vector"):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} contains non-finite values: {v.tolist()}")
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        raise ValueError(f"{name} has near-zero norm: {v.tolist()}")
    return v / n


def parse_vec3(s, name="vec3"):
    if isinstance(s, (list, tuple, np.ndarray)):
        arr = np.asarray(s, dtype=np.float64).reshape(-1)
    else:
        arr = np.fromstring(str(s).replace(",", " "), sep=" ", dtype=np.float64)
    if arr.size != 3:
        raise ValueError(f"{name} must contain exactly 3 numbers, got: {s!r}")
    return arr.astype(np.float64)


def rodrigues_to_matrix(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    return R.astype(np.float64)


def transform_points(T, points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return points @ T[:3, :3].T + T[:3, 3].reshape(1, 3)


def build_pose_from_Rh_Th(Rh, Th):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rodrigues_to_matrix(Rh)
    T[:3, 3] = np.asarray(Th, dtype=np.float64).reshape(3)
    return T


def project_rot_to_so3(R):
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def rotmat_to_quat_xyzw(R):
    """Dependency-free rotation matrix -> quaternion [x,y,z,w]."""
    R = project_rot_to_so3(R)
    tr = float(np.trace(R))
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12)) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12)) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12)) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return q


# ------------------------------ camera / gravity -----------------------------


def read_extri_yml(test_scene_dir):
    extri_path = Path(test_scene_dir) / "extri.yml"
    if not extri_path.exists():
        raise FileNotFoundError(f"Cannot find extri.yml: {extri_path}")

    fs = cv2.FileStorage(str(extri_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open extri.yml: {extri_path}")

    names_node = fs.getNode("names")
    if names_node.empty():
        fs.release()
        raise ValueError("No 'names' field found in extri.yml")

    cam_ext = {}
    for i in range(names_node.size()):
        cam_name = str(names_node.at(i).string())
        rot_node = fs.getNode(f"Rot_{cam_name}")
        t_node = fs.getNode(f"T_{cam_name}")
        if rot_node.empty() or t_node.empty():
            logging.warning("Missing extrinsic for camera %s, skipped", cam_name)
            continue
        R = np.asarray(rot_node.mat(), dtype=np.float64).reshape(3, 3)
        t = np.asarray(t_node.mat(), dtype=np.float64).reshape(3, 1)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3:] = t
        # T is camera_from_world in EasyMocap/OpenCV convention.
        cam_ext[cam_name] = {"Rot": R, "T": t, "ext": T}

    fs.release()
    if not cam_ext:
        raise RuntimeError(f"No valid cameras loaded from {extri_path}")
    return cam_ext


def kmeans_1d(values, n_clusters, max_iter=100):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    centers = np.quantile(values, (np.arange(n_clusters) + 0.5) / n_clusters)
    labels = None
    for _ in range(max_iter):
        new_labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        new_centers = centers.copy()
        for k in range(n_clusters):
            members = values[new_labels == k]
            if len(members) == 0:
                raise RuntimeError("auto_three_ring gravity got an empty ring; use manual mode.")
            new_centers[k] = float(np.mean(members))
        if labels is not None and np.array_equal(labels, new_labels):
            centers, labels = new_centers, new_labels
            break
        centers, labels = new_centers, new_labels
    order = np.argsort(centers)
    remap = np.empty(n_clusters, dtype=np.int64)
    remap[order] = np.arange(n_clusters, dtype=np.int64)
    return remap[labels], centers[order]


def fit_plane(points, orient_like=None):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    center = points.mean(axis=0)
    _, s, vt = np.linalg.svd(points - center[None], full_matrices=False)
    normal = normalize_vec3(vt[-1], "ring plane normal")
    if orient_like is not None and float(np.dot(normal, orient_like)) < 0:
        normal = -normal
    signed = (points - center[None]) @ normal
    rms = float(np.sqrt(np.mean(signed ** 2)))
    return center, normal, s, rms


def estimate_up_world_auto_three_ring(cam_ext, num_rings=3):
    """Estimate physical world-up from a 3-ring EasyMocap camera rig."""
    cam_ids = sorted([str(x) for x in cam_ext.keys()], key=natural_sort_key)
    if len(cam_ids) < num_rings * 3:
        raise ValueError(f"Need at least {num_rings * 3} cameras for auto_three_ring gravity.")

    centers = []
    for cam in cam_ids:
        T_cam_from_world = np.asarray(cam_ext[cam]["ext"], dtype=np.float64).reshape(4, 4)
        R = T_cam_from_world[:3, :3]
        t = T_cam_from_world[:3, 3]
        centers.append(-R.T @ t)
    centers = np.asarray(centers, dtype=np.float64)

    rig_center = centers.mean(axis=0)
    _, singular_all, vt_all = np.linalg.svd(centers - rig_center[None], full_matrices=False)
    initial_axis = normalize_vec3(vt_all[-1], "initial camera-rig axis")
    heights = (centers - rig_center[None]) @ initial_axis
    labels, _ = kmeans_1d(heights, num_rings)

    ring_fits = []
    normals = []
    for ring_id in range(num_rings):
        idx = np.where(labels == ring_id)[0]
        center, normal, s, rms = fit_plane(centers[idx], orient_like=initial_axis)
        radial = centers[idx] - center[None]
        radial -= (radial @ normal)[:, None] * normal[None]
        radius = np.linalg.norm(radial, axis=1)
        ring_fits.append({
            "indices": idx,
            "center": center,
            "normal": normal,
            "singular_values": s,
            "plane_planarity_rms": rms,
            "radius_mean": float(np.mean(radius)),
            "radius_std": float(np.std(radius)),
        })
        normals.append(normal)

    axis = normalize_vec3(np.sum(normals, axis=0), "refined camera-rig axis")
    low_idx = int(np.argmax([r["radius_mean"] for r in ring_fits]))
    high_idx = int(np.argmin([r["radius_mean"] for r in ring_fits]))
    up_hint = normalize_vec3(ring_fits[high_idx]["center"] - ring_fits[low_idx]["center"], "ring up hint")
    alignment = float(np.dot(axis, up_hint))
    if abs(alignment) < 0.80:
        raise RuntimeError(
            f"auto_three_ring gravity sign is uncertain, alignment={alignment:.3f}; use manual mode."
        )
    if alignment < 0:
        axis = -axis

    for fit in ring_fits:
        fit["height_along_up"] = float(np.dot(fit["center"] - rig_center, axis))
    ring_fits.sort(key=lambda r: r["height_along_up"])

    diagnostics = {
        "method": "auto_three_ring_from_extri",
        "num_cameras": len(cam_ids),
        "num_rings": int(num_rings),
        "camera_ids": cam_ids,
        "camera_center_mean_world": rig_center,
        "all_camera_center_singular_values": singular_all,
        "estimated_up_world": axis,
        "sign_rule": "largest_radius_ring_to_smallest_radius_ring",
        "sign_alignment_cosine": abs(alignment),
        "rings_low_to_high": [
            {
                "camera_ids": [cam_ids[int(i)] for i in r["indices"]],
                "center_world": r["center"],
                "height_along_up": r["height_along_up"],
                "radius_mean": r["radius_mean"],
                "radius_std": r["radius_std"],
                "plane_normal_world": r["normal"],
                "plane_planarity_rms": r["plane_planarity_rms"],
                "plane_singular_values": r["singular_values"],
            }
            for r in ring_fits
        ],
    }
    return axis, diagnostics


# ------------------------------ MANO utilities ------------------------------


def resolve_mano_model_path(mano_dir, side):
    mano_dir = Path(os.path.expanduser(str(mano_dir))).resolve()
    model_file = "MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl"

    candidates = []
    if mano_dir.is_file():
        candidates.append(mano_dir)
    candidates += [
        mano_dir / model_file,
        mano_dir / "mano" / model_file,
        mano_dir / "MANO" / model_file,
        mano_dir / "models" / "mano" / model_file,
        mano_dir / "models" / "MANO" / model_file,
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    raise FileNotFoundError(
        f"Cannot find {model_file}. Tried:\n" + "\n".join(f"  {p}" for p in candidates)
    )


def create_mano_layer(mano_dir, side, use_pca, num_pca_comps, flat_hand_mean, device):
    model_path = resolve_mano_model_path(mano_dir, side)
    logging.info(
        "[MANO] loading %s hand from %s, use_pca=%s, flat_hand_mean=%s",
        side, model_path, int(use_pca), int(flat_hand_mean),
    )
    kwargs = dict(
        model_path=model_path,
        is_rhand=(side == "right"),
        use_pca=bool(use_pca),
        flat_hand_mean=bool(flat_hand_mean),
    )
    if use_pca:
        kwargs["num_pca_comps"] = int(num_pca_comps)
    try:
        layer = smplx.MANO(**kwargs, batch_size=1)
    except TypeError:
        layer = smplx.MANO(**kwargs)
    return layer.to(device)


class ManoExporter:
    """Build MANO vertices/joints from saved EasyMocap-style MANO JSON records.

    The accelerated path batches many frames into one MANO forward call. Models
    are loaded lazily so a sequence that only contains PCA poses does not also
    pay the cost of constructing the full-axis-angle MANO layer (and vice versa).
    """

    def __init__(self, mano_dir, side, num_pca_comps=20, device="cpu", pose_dims=None):
        self.side = side
        self.device = torch.device(device)
        self.num_pca_comps = int(num_pca_comps)
        self.mano_dir = mano_dir
        self.model_full = None
        self.model_pca = None

        pose_dims = set(int(x) for x in (pose_dims or []))
        full_dim = 48
        pca_dim = 3 + self.num_pca_comps
        if not pose_dims:
            pose_dims = {full_dim, pca_dim}

        if full_dim in pose_dims:
            self.model_full = create_mano_layer(
                mano_dir, side, use_pca=False, num_pca_comps=num_pca_comps,
                flat_hand_mean=True, device=self.device,
            )
        if pca_dim in pose_dims:
            self.model_pca = create_mano_layer(
                mano_dir, side, use_pca=True, num_pca_comps=num_pca_comps,
                flat_hand_mean=False, device=self.device,
            )

        face_model = self.model_full if self.model_full is not None else self.model_pca
        if face_model is None:
            raise ValueError(
                f"No supported MANO pose dimensions found: {sorted(pose_dims)}; "
                f"expected 48 or {pca_dim}."
            )
        self.faces = np.asarray(face_model.faces, dtype=np.int32)
        self._pose_mean_cache = None
        self._components_cache = None

    def _ensure_full_model(self):
        if self.model_full is None:
            self.model_full = create_mano_layer(
                self.mano_dir, self.side, use_pca=False,
                num_pca_comps=self.num_pca_comps,
                flat_hand_mean=True, device=self.device,
            )
        return self.model_full

    def _ensure_pca_model(self):
        if self.model_pca is None:
            self.model_pca = create_mano_layer(
                self.mano_dir, self.side, use_pca=True,
                num_pca_comps=self.num_pca_comps,
                flat_hand_mean=False, device=self.device,
            )
        return self.model_pca

    def pose_mean_48(self):
        if self._pose_mean_cache is not None:
            return self._pose_mean_cache
        model_pca = self._ensure_pca_model()
        pm = getattr(model_pca, "pose_mean", None)
        if pm is None:
            pm = torch.zeros(48, dtype=torch.float32, device=self.device)
            if hasattr(model_pca, "hand_mean"):
                pm[3:] = model_pca.hand_mean.reshape(-1)
        if not torch.is_tensor(pm):
            pm = torch.tensor(pm, dtype=torch.float32, device=self.device)
        self._pose_mean_cache = pm.to(self.device).float().reshape(1, 48)
        return self._pose_mean_cache

    def hand_components(self):
        if self._components_cache is not None:
            return self._components_cache
        model_pca = self._ensure_pca_model()
        comps = model_pca.hand_components[:self.num_pca_comps]
        if not torch.is_tensor(comps):
            comps = torch.tensor(comps, dtype=torch.float32, device=self.device)
        self._components_cache = comps.to(self.device).float()
        return self._components_cache

    @staticmethod
    def _betas10(betas):
        betas = np.asarray(betas, dtype=np.float32).reshape(-1)
        if betas.size < 10:
            betas = np.pad(betas, (0, 10 - betas.size))
        return betas[:10].astype(np.float32, copy=False)

    def restore_full48_from_pca23(self, pose23):
        full = self.restore_full48_from_pca_batch(
            np.asarray(pose23, dtype=np.float32).reshape(1, -1)
        )
        return full[0]

    def restore_full48_from_pca_batch(self, poses_pca):
        poses_pca = np.asarray(poses_pca, dtype=np.float32)
        if poses_pca.ndim != 2 or poses_pca.shape[1] != 3 + self.num_pca_comps:
            raise ValueError(
                f"Expected PCA poses (B,{3 + self.num_pca_comps}), got {poses_pca.shape}."
            )
        pose_t = torch.as_tensor(poses_pca, dtype=torch.float32, device=self.device)
        pca = pose_t[:, 3:3 + self.num_pca_comps]
        hand45 = torch.matmul(pca, self.hand_components()).reshape(-1, 45)
        full48_t = torch.cat([pose_t[:, :3], hand45], dim=1) + self.pose_mean_48()
        return full48_t.detach().cpu().numpy().astype(np.float32)

    def _extract_batch_output(self, out, full48_t, betas10_np):
        verts = out.vertices.detach().cpu().numpy().astype(np.float32, copy=False)
        joints16 = out.joints[:, :16].detach().cpu().numpy().astype(np.float32, copy=False)
        if verts.ndim != 3 or verts.shape[1:] != (778, 3):
            raise ValueError(f"Unexpected batched verts shape {verts.shape}")
        if joints16.ndim != 3 or joints16.shape[1:] != (16, 3):
            raise ValueError(f"Unexpected batched joints16 shape {joints16.shape}")

        tip_ids = np.asarray(
            [745, 317, 444, 556, 673] if self.side == "right" else [745, 317, 445, 556, 673],
            dtype=np.int64,
        )
        tips = verts[:, tip_ids, :]
        joints21_native = np.concatenate([joints16, tips], axis=1)
        reorder = np.asarray(
            [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
            dtype=np.int64,
        )
        joints21 = joints21_native[:, reorder, :]
        full48 = full48_t.detach().cpu().numpy().astype(np.float32, copy=False)
        return verts, joints21, full48, betas10_np.astype(np.float32, copy=False)

    def build_local_batch(self, poses_batch, betas_batch):
        """Batch MANO inference while preserving the old per-frame outputs."""
        if len(poses_batch) != len(betas_batch):
            raise ValueError("poses_batch and betas_batch must have the same length")
        B = len(poses_batch)
        if B == 0:
            return (
                np.zeros((0, 778, 3), dtype=np.float32),
                np.zeros((0, 21, 3), dtype=np.float32),
                np.zeros((0, 48), dtype=np.float32),
                np.zeros((0, 10), dtype=np.float32),
            )

        pose_arrays = [np.asarray(p, dtype=np.float32).reshape(-1) for p in poses_batch]
        betas10 = np.stack([self._betas10(b) for b in betas_batch], axis=0)
        dims = np.asarray([p.size for p in pose_arrays], dtype=np.int64)
        allowed = {48, 3 + self.num_pca_comps}
        bad = sorted(set(int(x) for x in dims.tolist()) - allowed)
        if bad:
            raise ValueError(
                f"Unsupported MANO pose dims {bad}; expected 48 or {3 + self.num_pca_comps}."
            )

        verts_all = np.zeros((B, 778, 3), dtype=np.float32)
        joints_all = np.zeros((B, 21, 3), dtype=np.float32)
        full48_all = np.zeros((B, 48), dtype=np.float32)
        betas_all = betas10.astype(np.float32, copy=False)

        # Mixed pose formats are uncommon, but grouping preserves full compatibility.
        for pose_dim in sorted(set(int(x) for x in dims.tolist())):
            ids = np.where(dims == pose_dim)[0]
            poses_np = np.stack([pose_arrays[int(i)] for i in ids], axis=0)
            betas_np = betas10[ids]
            pose_t = torch.as_tensor(poses_np, dtype=torch.float32, device=self.device)
            betas_t = torch.as_tensor(betas_np, dtype=torch.float32, device=self.device)

            with torch.inference_mode():
                if pose_dim == 48:
                    model = self._ensure_full_model()
                    full48_t = pose_t
                    out = model(
                        global_orient=pose_t[:, :3],
                        hand_pose=pose_t[:, 3:48],
                        betas=betas_t,
                        return_verts=True,
                    )
                else:
                    model = self._ensure_pca_model()
                    hand45 = torch.matmul(
                        pose_t[:, 3:3 + self.num_pca_comps], self.hand_components()
                    ).reshape(-1, 45)
                    full48_t = torch.cat([pose_t[:, :3], hand45], dim=1) + self.pose_mean_48()
                    out = model(
                        global_orient=pose_t[:, :3],
                        hand_pose=pose_t[:, 3:3 + self.num_pca_comps],
                        betas=betas_t,
                        return_verts=True,
                    )

            verts, joints, full48, betas_group = self._extract_batch_output(
                out, full48_t, betas_np
            )
            verts_all[ids] = verts
            joints_all[ids] = joints
            full48_all[ids] = full48
            betas_all[ids] = betas_group

        return verts_all, joints_all, full48_all, betas_all

    def build_local(self, poses, betas):
        # Compatibility wrapper used by the batch-failure fallback path.
        verts, joints, full48, betas10 = self.build_local_batch([poses], [betas])
        return (
            verts[0].astype(np.float64),
            joints[0].astype(np.float64),
            full48[0],
            betas10[0],
        )


# ------------------------------ input indexing ------------------------------


def read_pose_txt(path):
    arr = np.loadtxt(path)
    return np.asarray(arr, dtype=np.float64).reshape(4, 4)


def collect_object_pose_files(args):
    outputmano_dir = Path(args.outputmano_dir).expanduser().resolve()

    candidates = []
    if args.ob_in_world_dir:
        candidates.append(Path(args.ob_in_world_dir).expanduser().resolve())
    candidates += [
        outputmano_dir / "smoothed_ob_in_world",
        outputmano_dir / "ob_in_world",
    ]

    for d in candidates:
        if d.is_dir():
            files = sorted(d.glob("*.txt"), key=natural_sort_key)
            if files:
                logging.info("[object] using object pose dir: %s", d)
                return files, d

    raise FileNotFoundError(
        "Cannot find object pose txt files. Pass --ob_in_world_dir explicitly, "
        "or put poses under outputmano/smoothed_ob_in_world."
    )


def default_hand_dir(outputmano_dir, side):
    outputmano_dir = Path(outputmano_dir).expanduser().resolve()
    candidates = [
        outputmano_dir / "contact_mano" / side / "smpl",
        outputmano_dir / side / "smpl",
        outputmano_dir / f"mano_{side}" / "smpl",
    ]
    for d in candidates:
        if d.is_dir() and any(d.glob("*.json")):
            return d
    return candidates[0]


def make_json_index(json_dir):
    json_dir = Path(json_dir).expanduser().resolve()
    index_exact = {}
    index_int = {}
    for p in sorted(json_dir.glob("*.json"), key=natural_sort_key):
        stem = p.stem
        index_exact[stem] = p
        try:
            index_int[int(stem)] = p
        except Exception:
            pass
    return index_exact, index_int


def lookup_json(index_pair, frame_id):
    exact, by_int = index_pair
    fid = str(frame_id)
    if fid in exact:
        return exact[fid]
    try:
        return by_int.get(int(fid))
    except Exception:
        return None


def read_hand_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            raise ValueError(f"Empty MANO json: {path}")
        rec = data[0]
    elif isinstance(data, dict):
        rec = data
    else:
        raise ValueError(f"Unsupported MANO json type: {path}")

    for key in ["poses", "shapes", "Rh", "Th"]:
        if key not in rec:
            raise KeyError(f"{path} misses key {key!r}")
    return {
        "poses": flatten_np(rec["poses"], np.float32),
        "betas": flatten_np(rec["shapes"], np.float32),
        "Rh": flatten_np(rec["Rh"], np.float64)[:3],
        "Th": flatten_np(rec["Th"], np.float64)[:3],
        "pose_format": rec.get("contact_mano_pose_format", "unknown"),
    }


# ------------------------------- export logic -------------------------------


def export_object_layout(raw_dir, object_name, frame_ids, object_pose_by_frame, T_ref_from_world, mesh_scale):
    objects = []
    for frame_idx, fid in enumerate(frame_ids):
        pose_world = object_pose_by_frame.get(str(fid))
        if pose_world is None:
            continue
        pose_ref = T_ref_from_world @ np.asarray(pose_world, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(pose_ref)) or float(pose_ref[2, 3]) <= 0.0:
            logging.warning(
                "[object] frame %s has invalid/non-positive ref-camera depth %.6f; skipped in layout",
                fid, float(pose_ref[2, 3]),
            )
            continue
        q = rotmat_to_quat_xyzw(pose_ref[:3, :3])
        objects.append({
            "frame_idx": int(frame_idx),
            "frame_id": str(fid),
            "local_to_scene": {
                "translation_camera_frame": [float(x) for x in pose_ref[:3, 3]],
                "quat_wxyz_camera_frame": [float(q[3]), float(q[0]), float(q[1]), float(q[2])],
            },
        })

    if len(objects) < 2:
        raise RuntimeError(f"Do-as-I-Do export has only {len(objects)} valid object poses.")

    layout = {
        "objects": objects,
        "translation_scale_optimization": {"mesh_scale": float(mesh_scale)},
    }
    layout_dir = raw_dir / "obj_tracking_out" / object_name / "combined_visualization"
    ensure_dir(layout_dir)
    with open(layout_dir / "layout_camera_frame_optimized.json", "w") as f:
        json.dump(jsonable(layout), f, indent=2)
    return layout_dir / "layout_camera_frame_optimized.json", len(objects)


def export_object_mesh(raw_dir, object_name, mesh_file):
    mesh_file = Path(mesh_file).expanduser().resolve()
    if not mesh_file.exists():
        raise FileNotFoundError(f"Cannot find mesh_file: {mesh_file}")
    out_dir = raw_dir / "video_segmentation" / "masks" / "frame_000000_masks" / object_name
    ensure_dir(out_dir)
    out_path = out_dir / f"{object_name}.obj"

    # Always write .obj at the canonical path. trimesh handles .ply/.obj input.
    mesh = trimesh.load(str(mesh_file), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh.export(str(out_path))
    return out_path


def preflight_hand_inputs(frame_ids, sides, hand_dirs):
    """Fail early if the requested hand streams cannot form a retargeting input."""
    counts = {}
    for side in sides:
        json_dir = Path(hand_dirs[side]).expanduser().resolve()
        if not json_dir.is_dir():
            raise FileNotFoundError(
                f"Cannot find {side} hand json dir required for {sides}: {json_dir}"
            )
        idx = make_json_index(json_dir)
        matched = sum(lookup_json(idx, fid) is not None for fid in frame_ids)
        counts[side] = int(matched)
        if matched < 2:
            raise RuntimeError(
                f"Do-as-I-Do needs at least 2 frame-aligned {side} MANO JSONs; "
                f"found {matched}/{len(frame_ids)} in {json_dir}."
            )
        logging.info(
            "[preflight][%s] frame-aligned MANO jsons: %d/%d from %s",
            side, matched, len(frame_ids), json_dir,
        )
    if set(sides) == {"right", "left"}:
        logging.info("[preflight] bimanual input enabled: right + left share one object trajectory")
    return counts


def _validate_side_arrays(packed, side, N):
    """Validate exactly the fields consumed by retargeting/process_dataset.py."""
    expected = {
        f"{side}_joints": (N, 21, 3),
        f"{side}_valid": (N,),
        f"{side}_rot": (N, 3),
        f"{side}_vertices": (N, 778, 3),
        f"{side}_hand_pose": (N, 45),
        f"{side}_betas": (N, 10),
    }
    for key, shape in expected.items():
        if key not in packed.files:
            raise RuntimeError(f"Do-as-I-Do NPZ contract failed: missing key {key!r}")
        if tuple(packed[key].shape) != tuple(shape):
            raise RuntimeError(
                f"Do-as-I-Do NPZ contract failed: {key} has shape "
                f"{packed[key].shape}, expected {shape}"
            )

    faces_key = f"{side}_faces"
    if faces_key not in packed.files:
        raise RuntimeError(f"Do-as-I-Do NPZ contract failed: missing key {faces_key!r}")
    faces = np.asarray(packed[faces_key])
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise RuntimeError(
            f"Do-as-I-Do NPZ contract failed: {faces_key} must be (F,3), got {faces.shape}"
        )

    valid = np.asarray(packed[f"{side}_valid"], dtype=bool)
    n_valid = int(valid.sum())
    if n_valid < 2:
        raise RuntimeError(
            f"Do-as-I-Do NPZ contract failed: {side} has only {n_valid}/{N} valid frames"
        )

    # Only valid rows need to be finite. Invalid rows are intentionally zero-filled
    # and retargeting/process_dataset.py interpolates through them using *_valid.
    finite_keys = [
        f"{side}_joints", f"{side}_rot", f"{side}_vertices",
        f"{side}_hand_pose", f"{side}_betas",
    ]
    for key in finite_keys:
        arr = np.asarray(packed[key])
        if not np.all(np.isfinite(arr[valid])):
            raise RuntimeError(
                f"Do-as-I-Do NPZ contract failed: {key} contains non-finite values on valid frames"
            )
    return n_valid


def resolve_compute_device(device):
    requested = str(device).strip().lower()
    if requested in ("", "auto"):
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = requested
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested ({device!r}) but torch.cuda.is_available() is False."
        )
    logging.info("[speed] MANO device: requested=%s resolved=%s", device, resolved)
    return resolved


def export_hands_npz(raw_dir, task, frame_ids, sides, hand_dirs, mano_dir,
                     T_ref_from_world, num_pca_comps, device,
                     mano_batch_size=256, compress_npz=False):
    """Export MANO hands using batched inference for substantially higher throughput.

    The output arrays are unchanged. The only performance-oriented differences are:
      * MANO frames are forwarded in batches instead of one frame at a time;
      * ``device=auto`` uses CUDA when available;
      * NPZ compression is optional because deflate can become CPU-bound.
    """
    N = len(frame_ids)
    if N < 2:
        raise RuntimeError(f"Need at least 2 frames, got {N}")
    if not sides or any(s not in ("right", "left") for s in sides):
        raise ValueError(f"Invalid sides: {sides}")
    if len(set(sides)) != len(sides):
        raise ValueError(f"Duplicate hand side in {sides}")

    mano_batch_size = max(1, int(mano_batch_size))
    device = resolve_compute_device(device)
    if device.startswith("cuda"):
        # Avoid repeatedly creating CUDA workspaces; inference_mode is used below.
        torch.cuda.empty_cache()

    hand_data = {
        "frame_ids": np.asarray([str(x) for x in frame_ids], dtype=str),
    }
    valid_counts = {}
    R_ref_from_world = np.asarray(T_ref_from_world[:3, :3], dtype=np.float64)

    for side in sides:
        json_dir = Path(hand_dirs[side]).expanduser().resolve()
        if not json_dir.is_dir():
            raise FileNotFoundError(f"Cannot find {side} hand json dir: {json_dir}")
        idx = make_json_index(json_dir)

        # Read lightweight JSON first. This also tells us which MANO model format
        # actually exists, so unnecessary full/PCA layers are not constructed.
        records = []
        for i, fid in enumerate(frame_ids):
            p = lookup_json(idx, fid)
            if p is None:
                continue
            try:
                rec = read_hand_json(p)
                records.append((i, str(fid), p, rec))
            except Exception as exc:
                logging.warning("[hand][%s] invalid json frame %s from %s: %s", side, fid, p, exc)

        if len(records) < 2:
            raise RuntimeError(
                f"Do-as-I-Do needs at least 2 readable {side} frames; got {len(records)}/{N}."
            )

        pose_dims = sorted(set(int(rec["poses"].size) for _, _, _, rec in records))
        builder = ManoExporter(
            mano_dir=mano_dir,
            side=side,
            num_pca_comps=num_pca_comps,
            device=device,
            pose_dims=pose_dims,
        )
        logging.info(
            "[speed][%s] batched MANO: %d readable frames, batch_size=%d, pose_dims=%s",
            side, len(records), mano_batch_size, pose_dims,
        )

        joints = np.zeros((N, 21, 3), dtype=np.float32)
        vertices = np.zeros((N, 778, 3), dtype=np.float32)
        valid = np.zeros((N,), dtype=bool)
        global_rot = np.zeros((N, 3), dtype=np.float32)
        hand_pose = np.zeros((N, 45), dtype=np.float32)
        betas = np.zeros((N, 10), dtype=np.float32)

        for chunk_start in range(0, len(records), mano_batch_size):
            chunk = records[chunk_start:chunk_start + mano_batch_size]
            chunk_poses = [rec["poses"] for _, _, _, rec in chunk]
            chunk_betas = [rec["betas"] for _, _, _, rec in chunk]

            try:
                v_local, j_local, full48_batch, betas10_batch = builder.build_local_batch(
                    chunk_poses, chunk_betas
                )
                batch_ok = True
            except RuntimeError as exc:
                # A too-large CUDA batch should not kill the export. Fall back to
                # per-frame inference and tell the user how to tune it next run.
                if device.startswith("cuda") and "out of memory" in str(exc).lower():
                    logging.warning(
                        "[speed][%s] CUDA OOM at MANO batch_size=%d; falling back to "
                        "per-frame inference for this chunk. Re-run with a smaller "
                        "--mano_batch_size for best speed.",
                        side, mano_batch_size,
                    )
                    torch.cuda.empty_cache()
                    batch_ok = False
                else:
                    raise

            if batch_ok:
                chunk_results = [
                    (v_local[k], j_local[k], full48_batch[k], betas10_batch[k])
                    for k in range(len(chunk))
                ]
            else:
                chunk_results = []
                for _, _, _, rec in chunk:
                    chunk_results.append(builder.build_local(rec["poses"], rec["betas"]))

            for (i, fid, p, rec), result in zip(chunk, chunk_results):
                try:
                    verts_local, joints_local, full48, betas10 = result

                    T_hand_from_local_world = build_pose_from_Rh_Th(rec["Rh"], rec["Th"])
                    T_ref_from_local = T_ref_from_world @ T_hand_from_local_world
                    vertices_i = transform_points(T_ref_from_local, verts_local)
                    joints_i = transform_points(T_ref_from_local, joints_local)

                    R_effective = (
                        R_ref_from_world
                        @ rodrigues_to_matrix(rec["Rh"])
                        @ rodrigues_to_matrix(full48[:3])
                    )
                    rot_i = cv2.Rodrigues(project_rot_to_so3(R_effective))[0].reshape(3)

                    if not (
                        np.all(np.isfinite(vertices_i))
                        and np.all(np.isfinite(joints_i))
                        and np.all(np.isfinite(rot_i))
                        and np.all(np.isfinite(full48[3:48]))
                        and np.all(np.isfinite(betas10))
                    ):
                        raise ValueError("non-finite MANO data after reference-camera transform")

                    vertices[i] = vertices_i.astype(np.float32)
                    joints[i] = joints_i.astype(np.float32)
                    global_rot[i] = rot_i.astype(np.float32)
                    hand_pose[i] = full48[3:48].astype(np.float32)
                    betas[i] = betas10.astype(np.float32)
                    valid[i] = True
                except Exception as exc:
                    logging.warning(
                        "[hand][%s] invalid frame %s from %s: %s", side, fid, p, exc
                    )

        n_valid = int(np.count_nonzero(valid))
        if n_valid < 2:
            raise RuntimeError(
                f"Do-as-I-Do needs at least 2 valid {side} frames; got {n_valid}/{N}."
            )

        hand_data.update({
            f"{side}_joints": joints,
            f"{side}_valid": valid,
            f"{side}_rot": global_rot,
            f"{side}_vertices": vertices,
            f"{side}_faces": np.asarray(builder.faces, dtype=np.int32),
            f"{side}_hand_pose": hand_pose,
            f"{side}_betas": betas,
        })
        valid_counts[side] = n_valid
        logging.info("[hand][%s] valid frames: %d/%d", side, n_valid, N)

        # Release one hand's models before constructing the next hand on GPU.
        del builder
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    hand_dir = raw_dir / task
    ensure_dir(hand_dir)
    npz_path = hand_dir / "all_hand_meshes.npz"
    if compress_npz:
        logging.info("[speed] writing compressed NPZ (smaller, slower): %s", npz_path)
        np.savez_compressed(npz_path, **hand_data)
    else:
        logging.info("[speed] writing uncompressed NPZ (faster): %s", npz_path)
        np.savez(npz_path, **hand_data)

    with np.load(npz_path, allow_pickle=False) as packed:
        if "frame_ids" in packed.files and len(packed["frame_ids"]) != N:
            raise RuntimeError("NPZ self-check failed: frame_ids length mismatch")
        for side in sides:
            checked = _validate_side_arrays(packed, side, N)
            if checked != valid_counts[side]:
                raise RuntimeError(
                    f"NPZ self-check failed: {side} valid count changed "
                    f"{valid_counts[side]} -> {checked}"
                )

    if set(sides) == {"right", "left"}:
        logging.info(
            "[hand] bimanual NPZ ready: both right_* and left_* streams saved to %s",
            npz_path,
        )
    return npz_path, valid_counts, hand_data


def validate_dai_raw_contract(raw_dir, task, object_name, frame_ids, sides):
    """Validate the completed raw directory against current retargeting Stage-1 inputs."""
    N = len(frame_ids)
    anchor_hand = "bimanual" if set(sides) == {"right", "left"} else sides[0]

    config_path = raw_dir / "config.json"
    gravity_path = raw_dir / "gravity.json"
    npz_path = raw_dir / task / "all_hand_meshes.npz"
    layout_path = (
        raw_dir / "obj_tracking_out" / object_name /
        "combined_visualization" / "layout_camera_frame_optimized.json"
    )
    mesh_path = (
        raw_dir / "video_segmentation" / "masks" /
        "frame_000000_masks" / object_name / f"{object_name}.obj"
    )

    for p in (config_path, gravity_path, npz_path, layout_path, mesh_path):
        if not p.exists():
            raise RuntimeError(f"Do-as-I-Do raw contract failed: missing {p}")

    cfg = json.load(open(config_path))
    if cfg.get("anchor_hand") != anchor_hand:
        raise RuntimeError(
            f"config.json anchor_hand={cfg.get('anchor_hand')!r}, expected {anchor_hand!r}"
        )
    if not cfg.get("object_names") or cfg["object_names"][0] != object_name:
        raise RuntimeError(
            f"config.json object_names does not start with {object_name!r}: "
            f"{cfg.get('object_names')!r}"
        )

    gravity = json.load(open(gravity_path))
    up = np.asarray(gravity.get("vec3d", []), dtype=np.float64).reshape(-1)
    if up.size != 3 or not np.all(np.isfinite(up)) or np.linalg.norm(up) < 1e-8:
        raise RuntimeError(f"gravity.json vec3d invalid: {gravity.get('vec3d')!r}")

    valid_counts = {}
    with np.load(npz_path, allow_pickle=False) as packed:
        for side in sides:
            valid_counts[side] = _validate_side_arrays(packed, side, N)

    layout = json.load(open(layout_path))
    objects = layout.get("objects", [])
    valid_object_records = 0
    for obj in objects:
        fi = obj.get("frame_idx", obj.get("frame_index"))
        if fi is None or not (0 <= int(fi) < N):
            raise RuntimeError(f"Object layout has out-of-range frame index: {fi}")
        ls = obj.get("local_to_scene", {})
        t = np.asarray(ls.get("translation_camera_frame", []), dtype=np.float64).reshape(-1)
        q = np.asarray(ls.get("quat_wxyz_camera_frame", []), dtype=np.float64).reshape(-1)
        if t.size != 3 or q.size != 4 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(q)):
            raise RuntimeError(f"Object layout has invalid pose at frame_idx={fi}")
        if float(t[2]) > 0:
            valid_object_records += 1
    if valid_object_records < 2:
        raise RuntimeError(
            f"Object layout has only {valid_object_records} positive-depth poses; need >=2"
        )

    report = {
        "status": "PASS",
        "embodiment_type": anchor_hand,
        "num_frames": int(N),
        "valid_hand_frames": valid_counts,
        "valid_object_poses": int(valid_object_records),
        "hands_npz": str(npz_path),
        "layout": str(layout_path),
        "mesh": str(mesh_path),
    }
    logging.info(
        "[contract] PASS: embodiment=%s, N=%d, right=%s, left=%s, object=%d",
        anchor_hand,
        N,
        valid_counts.get("right", "n/a"),
        valid_counts.get("left", "n/a"),
        valid_object_records,
    )
    return report



# --------------------------- 3D-only visualization ---------------------------

def set_axes_equal_3d(ax, points, pad_ratio=0.08):
    """Make x/y/z axes have the same scale for honest 3D visualization."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius * (1.0 + float(pad_ratio)), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def hand_bones_21():
    return [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
    ]


def make_object_centers_ref(frame_ids, object_pose_by_frame, T_ref_from_world, mesh_file):
    mesh = trimesh.load(str(mesh_file), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    bounds = np.asarray(mesh.bounds, dtype=np.float64).reshape(2, 3)
    pivot_obj = 0.5 * (bounds[0] + bounds[1])

    centers = []
    valid = []
    for fid in frame_ids:
        pose_world = object_pose_by_frame.get(str(fid))
        if pose_world is None:
            centers.append([np.nan, np.nan, np.nan])
            valid.append(False)
            continue
        pose_ref = T_ref_from_world @ np.asarray(pose_world, dtype=np.float64).reshape(4, 4)
        c = pose_ref[:3, :3] @ pivot_obj + pose_ref[:3, 3]
        centers.append(c.tolist())
        valid.append(np.isfinite(c).all())
    return np.asarray(centers, dtype=np.float64), np.asarray(valid, dtype=bool)


def downsample_mesh_vertices(mesh, max_points=1200):
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) <= int(max_points):
        return verts
    idx = np.linspace(0, len(verts) - 1, int(max_points)).astype(np.int64)
    return verts[idx]


def load_object_mesh_for_vis(mesh_file, max_points=1200):
    mesh = trimesh.load(str(mesh_file), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = downsample_mesh_vertices(mesh, max_points=max_points)
    bounds = np.asarray(mesh.bounds, dtype=np.float64).reshape(2, 3)
    bbox_center = 0.5 * (bounds[0] + bounds[1])
    return mesh, vertices, bbox_center


def draw_camera_frame_axes(ax, length=0.08):
    """Draw reference camera frame axes at origin."""
    origin = np.zeros(3, dtype=np.float64)
    axes = np.eye(3, dtype=np.float64) * float(length)
    ax.plot([origin[0], axes[0, 0]], [origin[1], axes[0, 1]], [origin[2], axes[0, 2]], linewidth=2, label="ref X")
    ax.plot([origin[0], axes[1, 0]], [origin[1], axes[1, 1]], [origin[2], axes[1, 2]], linewidth=2, label="ref Y")
    ax.plot([origin[0], axes[2, 0]], [origin[1], axes[2, 1]], [origin[2], axes[2, 2]], linewidth=2, label="ref Z")


def draw_object_axes(ax, pose_ref, axis_len=0.06):
    origin = np.asarray(pose_ref[:3, 3], dtype=np.float64)
    R = np.asarray(pose_ref[:3, :3], dtype=np.float64)
    for k in range(3):
        end = origin + R[:, k] * float(axis_len)
        ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], linewidth=2)


def draw_hand_skeleton_3d(ax, joints, label=None):
    joints = np.asarray(joints, dtype=np.float64).reshape(21, 3)
    for chain in hand_bones_21():
        pts = joints[chain]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=10, label=label)


def save_scene_overview_plot(vis_dir, frame_ids, object_centers_ref, hand_data, sides):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    all_points = []
    obj_mask = np.isfinite(object_centers_ref).all(axis=1)
    obj_pts = object_centers_ref[obj_mask]
    if len(obj_pts) > 0:
        ax.plot(obj_pts[:, 0], obj_pts[:, 1], obj_pts[:, 2], linewidth=2, label="object center")
        ax.scatter(obj_pts[0, 0], obj_pts[0, 1], obj_pts[0, 2], s=40, label="object start")
        ax.scatter(obj_pts[-1, 0], obj_pts[-1, 1], obj_pts[-1, 2], s=40, label="object end")
        all_points.append(obj_pts)

    for side in sides:
        joints = hand_data.get(f"{side}_joints")
        valid = hand_data.get(f"{side}_valid")
        if joints is not None and valid is not None and np.any(valid):
            wrist = joints[valid, 0, :]
            ax.plot(wrist[:, 0], wrist[:, 1], wrist[:, 2], linewidth=2, label=f"{side} wrist")
            all_points.append(wrist)

    draw_camera_frame_axes(ax, length=0.08)

    if all_points:
        set_axes_equal_3d(ax, np.concatenate(all_points, axis=0))

    ax.set_xlabel("X in ref camera (m)")
    ax.set_ylabel("Y in ref camera (m)")
    ax.set_zlabel("Z in ref camera (m)")
    ax.set_title("Object / hand trajectories in reference camera frame")
    ax.legend(loc="best")
    fig.tight_layout()

    out = vis_dir / "scene_overview_ref_cam.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def save_trajectory_plot(vis_dir, frame_ids, object_centers_ref, hand_data, sides):
    N = len(frame_ids)
    x = np.arange(N)
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    for d, label in enumerate(["x", "y", "z"]):
        axes[d].plot(x, object_centers_ref[:, d], label="object center")
        for side in sides:
            joints = hand_data.get(f"{side}_joints")
            valid = hand_data.get(f"{side}_valid")
            if joints is not None and valid is not None:
                wrist = np.full((N,), np.nan, dtype=np.float64)
                wrist[valid] = joints[valid, 0, d]
                axes[d].plot(x, wrist, label=f"{side} wrist")
        axes[d].set_ylabel(f"{label} (m)")
        axes[d].grid(True, alpha=0.3)
        axes[d].legend(loc="best")

    axes[-1].set_xlabel("frame index")
    fig.tight_layout()

    out = vis_dir / "trajectories_ref_cam.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def create_image_grid(image_paths, out_path, cell_size=(360, 360), cols=4):
    image_paths = [Path(p) for p in image_paths if Path(p).exists()]
    if len(image_paths) == 0:
        return None

    cell_w, cell_h = int(cell_size[0]), int(cell_size[1])
    cols = max(1, int(cols))
    rows = int(math.ceil(len(image_paths) / cols))
    canvas = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 255

    for idx, p in enumerate(image_paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        r, c = divmod(idx, cols)
        canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = img

    cv2.imwrite(str(out_path), canvas)
    return out_path


def save_3d_frame_visualizations(
    vis_dir,
    frame_ids,
    object_pose_by_frame,
    T_ref_from_world,
    mesh_file,
    hand_data,
    sides,
    vis_stride=10,
    vis_max_frames=24,
    vis_object_points=1200,
):
    frames_dir = vis_dir / "frames_3d"
    ensure_dir(frames_dir)

    _, object_vertices_obj, bbox_center_obj = load_object_mesh_for_vis(
        mesh_file=mesh_file,
        max_points=vis_object_points,
    )

    stride = max(1, int(vis_stride))
    chosen = list(range(0, len(frame_ids), stride))
    if int(vis_max_frames) > 0:
        chosen = chosen[:int(vis_max_frames)]

    saved = []
    for i in chosen:
        fid = str(frame_ids[i])
        pose_world = object_pose_by_frame.get(fid)
        if pose_world is None:
            continue

        pose_ref = T_ref_from_world @ np.asarray(pose_world, dtype=np.float64).reshape(4, 4)
        object_vertices_ref = transform_points(pose_ref, object_vertices_obj)
        object_center_ref = pose_ref[:3, :3] @ bbox_center_obj + pose_ref[:3, 3]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        all_points = [object_vertices_ref]
        ax.scatter(
            object_vertices_ref[:, 0],
            object_vertices_ref[:, 1],
            object_vertices_ref[:, 2],
            s=1,
            alpha=0.35,
            label="object mesh",
        )
        ax.scatter([object_center_ref[0]], [object_center_ref[1]], [object_center_ref[2]], s=40, label="object center")
        draw_object_axes(ax, pose_ref, axis_len=0.08)

        for side in sides:
            joints = hand_data.get(f"{side}_joints")
            valid = hand_data.get(f"{side}_valid")
            if joints is not None and valid is not None and i < len(valid) and bool(valid[i]):
                draw_hand_skeleton_3d(ax, joints[i], label=f"{side} hand")
                all_points.append(joints[i])

        draw_camera_frame_axes(ax, length=0.08)
        set_axes_equal_3d(ax, np.concatenate(all_points, axis=0))

        ax.set_xlabel("X ref cam (m)")
        ax.set_ylabel("Y ref cam (m)")
        ax.set_zlabel("Z ref cam (m)")
        ax.set_title(f"3D check in ref camera frame: frame {fid}")
        ax.view_init(elev=18, azim=-65)
        ax.legend(loc="best")
        fig.tight_layout()

        out = frames_dir / f"{fid}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        saved.append(out)

    if saved:
        create_image_grid(saved[:16], vis_dir / "frames_3d_grid.jpg", cell_size=(360, 360), cols=4)
    else:
        logging.warning("[vis] no 3D frame visualizations were saved.")

    return saved


def generate_visualizations(
    raw_dir,
    frame_ids,
    object_pose_by_frame,
    T_ref_from_world,
    mesh_file,
    hand_data,
    sides,
    vis_stride=10,
    vis_max_frames=24,
    vis_object_points=1200,
):
    """
    Export visualization without using original RGB images.

    Output:
      raw_dir/visualization/scene_overview_ref_cam.png
      raw_dir/visualization/trajectories_ref_cam.png
      raw_dir/visualization/frames_3d/*.png
      raw_dir/visualization/frames_3d_grid.jpg
    """
    vis_dir = raw_dir / "visualization"
    ensure_dir(vis_dir)

    object_centers_ref, object_center_valid = make_object_centers_ref(
        frame_ids=frame_ids,
        object_pose_by_frame=object_pose_by_frame,
        T_ref_from_world=T_ref_from_world,
        mesh_file=mesh_file,
    )

    vis_files = {}
    vis_files["scene_overview_ref_cam"] = str(save_scene_overview_plot(vis_dir, frame_ids, object_centers_ref, hand_data, sides))
    vis_files["trajectories_ref_cam"] = str(save_trajectory_plot(vis_dir, frame_ids, object_centers_ref, hand_data, sides))

    frame_images = save_3d_frame_visualizations(
        vis_dir=vis_dir,
        frame_ids=frame_ids,
        object_pose_by_frame=object_pose_by_frame,
        T_ref_from_world=T_ref_from_world,
        mesh_file=mesh_file,
        hand_data=hand_data,
        sides=sides,
        vis_stride=vis_stride,
        vis_max_frames=vis_max_frames,
        vis_object_points=vis_object_points,
    )
    if frame_images:
        vis_files["frames_3d_dir"] = str(vis_dir / "frames_3d")
        grid_path = vis_dir / "frames_3d_grid.jpg"
        if grid_path.exists():
            vis_files["frames_3d_grid"] = str(grid_path)

    summary = {
        "type": "3d_only_no_original_images",
        "num_frames": int(len(frame_ids)),
        "object_center_valid_frames": int(np.count_nonzero(object_center_valid)),
        "visualization_files": vis_files,
    }
    with open(vis_dir / "visualization_summary.json", "w") as f:
        json.dump(jsonable(summary), f, indent=2)

    logging.info("[vis] saved 3D-only visualization outputs to: %s", vis_dir)
    return vis_dir, vis_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert saved outputmano/contact_mano + object poses into Do-as-I-Do raw format."
    )
    parser.add_argument("--outputmano_dir", type=str, required=True,
                        help="Existing outputmano folder, e.g. <data-root>/my_sequence/outputmano")
    parser.add_argument("--test_scene_dir", type=str, required=True,
                        help="HOI/EasyMocap scene folder containing extri.yml")
    parser.add_argument("--mesh_file", type=str, required=True,
                        help="Object mesh path, e.g. <data-root>/my_sequence/meshes/compressed.ply")
    parser.add_argument("--mano_dir", type=str, required=True,
                        help="Folder containing MANO_LEFT.pkl / MANO_RIGHT.pkl")
    parser.add_argument("--dai_raw_dir", type=str, required=True,
                        help="Target raw dir passed later to Do-as-I-Do --raw-dir")
    parser.add_argument("--dai_task", type=str, required=True,
                        help="Task subfolder name, e.g. my_sequence")
    parser.add_argument("--dai_object_name", type=str, default="object")
    parser.add_argument(
        "--dai_hand_side",
        choices=["right", "left", "bimanual"],
        default="bimanual",
        help=(
            "Hand embodiment exported for Do-as-I-Do. Default: bimanual. "
            "Bimanual writes both right_* and left_* streams sharing one object trajectory."
        ),
    )
    parser.add_argument("--dai_ref_cam", type=str, default="1")
    parser.add_argument("--dai_mesh_scale", type=float, default=1.0)

    parser.add_argument("--ob_in_world_dir", type=str, default="",
                        help="Object 4x4 pose txt dir. If omitted, tries outputmano/smoothed_ob_in_world then outputmano/ob_in_world.")
    parser.add_argument("--right_hand_json_dir", type=str, default="",
                        help="Defaults to outputmano/contact_mano/right/smpl")
    parser.add_argument("--left_hand_json_dir", type=str, default="",
                        help="Defaults to outputmano/contact_mano/left/smpl")

    parser.add_argument("--num_pca_comps", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:N; auto uses CUDA when available")
    parser.add_argument("--mano_batch_size", type=int, default=256,
                        help="MANO inference batch size. Larger is faster on GPU; reduce if CUDA OOM.")
    parser.add_argument("--compress_npz", action="store_true",
                        help="Use np.savez_compressed. Default is uncompressed NPZ for faster export.")

    parser.add_argument("--dai_gravity_mode", choices=["auto_three_ring", "manual"], default="auto_three_ring")
    parser.add_argument("--dai_gravity_num_rings", type=int, default=3)
    parser.add_argument("--dai_gravity_up_world", type=str, default="",
                        help='Required for manual mode, e.g. "0,0,1" in EasyMocap world coordinates')

    parser.add_argument("--overwrite", action="store_true", help="Remove existing dai_raw_dir before writing")
    parser.add_argument("--save_vis", action="store_true",
                        help="Also export 3D-only visualization. This does not require original images or cam_K.txt.")
    parser.add_argument("--vis_stride", type=int, default=10,
                        help="For 3D frame visualization, save one frame every N frames")
    parser.add_argument("--vis_max_frames", type=int, default=24,
                        help="Maximum number of 3D frame visualization images to save; <=0 means no limit")
    parser.add_argument("--vis_object_points", type=int, default=1200,
                        help="Number of object mesh vertices to sample for each 3D visualization frame")
    args = parser.parse_args()
    set_logging()

    outputmano_dir = Path(args.outputmano_dir).expanduser().resolve()
    raw_dir = Path(args.dai_raw_dir).expanduser().resolve()
    object_name = str(args.dai_object_name).strip()
    task = str(args.dai_task).strip()
    if not object_name:
        raise ValueError("--dai_object_name cannot be empty")
    if not task:
        raise ValueError("--dai_task cannot be empty")

    if raw_dir.exists() and args.overwrite:
        logging.info("[output] removing existing raw dir: %s", raw_dir)
        shutil.rmtree(raw_dir)
    ensure_dir(raw_dir)

    cam_ext = read_extri_yml(args.test_scene_dir)
    ref_cam = str(args.dai_ref_cam)
    if ref_cam not in cam_ext:
        raise KeyError(f"--dai_ref_cam {ref_cam!r} not found. Available: {sorted(cam_ext.keys(), key=natural_sort_key)}")
    T_ref_from_world = np.asarray(cam_ext[ref_cam]["ext"], dtype=np.float64).reshape(4, 4)

    # Gravity: write the physical up direction in the exported reference-camera frame.
    if args.dai_gravity_mode == "auto_three_ring":
        up_world, gravity_diag = estimate_up_world_auto_three_ring(cam_ext, args.dai_gravity_num_rings)
        gravity_source = "auto-estimated from three camera rings in extri.yml"
    else:
        up_world = normalize_vec3(parse_vec3(args.dai_gravity_up_world, "--dai_gravity_up_world"), "up_world")
        gravity_diag = {
            "method": "manual",
            "estimated_up_world": up_world,
            "note": "User-supplied physical up direction in EasyMocap world coordinates.",
        }
        gravity_source = "manual --dai_gravity_up_world"
    up_ref = normalize_vec3(T_ref_from_world[:3, :3] @ up_world, "up_ref")

    # Object poses.
    pose_files, object_pose_dir = collect_object_pose_files(args)
    frame_ids = [p.stem for p in pose_files]
    object_pose_by_frame = {p.stem: read_pose_txt(p) for p in pose_files}
    logging.info("[object] frames: %d", len(frame_ids))

    # Hand dirs / embodiment. In bimanual mode both streams use the exact same
    # object-derived frame timeline so downstream Stage-1 can build one shared mask.
    sides = (
        ["right"] if args.dai_hand_side == "right"
        else ["left"] if args.dai_hand_side == "left"
        else ["right", "left"]
    )
    anchor_hand = "bimanual" if len(sides) == 2 else sides[0]
    hand_dirs = {
        "right": args.right_hand_json_dir or str(default_hand_dir(outputmano_dir, "right")),
        "left": args.left_hand_json_dir or str(default_hand_dir(outputmano_dir, "left")),
    }
    preflight_hand_inputs(frame_ids, sides, hand_dirs)

    # config/gravity files. process_dataset.py resolves embodiment_type=auto
    # directly from this anchor_hand field.
    config = {
        "anchor_hand": anchor_hand,
        "object_names": [object_name],
        "source": "converted from existing outputmano/contact_mano folder",
        "outputmano_dir": str(outputmano_dir),
        "object_pose_dir": str(object_pose_dir),
        "reference_camera": ref_cam,
        "frame_ids": [str(x) for x in frame_ids],
    }
    with open(raw_dir / "config.json", "w") as f:
        json.dump(jsonable(config), f, indent=2)

    gravity = {
        "vec3d": [float(x) for x in up_ref],
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "source": gravity_source,
        "mode": args.dai_gravity_mode,
        "reference_camera": ref_cam,
        "up_world": [float(x) for x in up_world],
    }
    with open(raw_dir / "gravity.json", "w") as f:
        json.dump(jsonable(gravity), f, indent=2)
    with open(raw_dir / "gravity_estimation_from_extri.json", "w") as f:
        json.dump(jsonable(gravity_diag), f, indent=2)

    layout_path, n_obj = export_object_layout(
        raw_dir=raw_dir,
        object_name=object_name,
        frame_ids=frame_ids,
        object_pose_by_frame=object_pose_by_frame,
        T_ref_from_world=T_ref_from_world,
        mesh_scale=args.dai_mesh_scale,
    )
    object_mesh_path = export_object_mesh(raw_dir, object_name, args.mesh_file)
    npz_path, valid_counts, hand_data = export_hands_npz(
        raw_dir=raw_dir,
        task=task,
        frame_ids=frame_ids,
        sides=sides,
        hand_dirs=hand_dirs,
        mano_dir=args.mano_dir,
        T_ref_from_world=T_ref_from_world,
        num_pca_comps=args.num_pca_comps,
        device=args.device,
        mano_batch_size=args.mano_batch_size,
        compress_npz=args.compress_npz,
    )

    vis_files = {}
    if args.save_vis:
        _, vis_files = generate_visualizations(
            raw_dir=raw_dir,
            frame_ids=frame_ids,
            object_pose_by_frame=object_pose_by_frame,
            T_ref_from_world=T_ref_from_world,
            mesh_file=args.mesh_file,
            hand_data=hand_data,
            sides=sides,
            vis_stride=args.vis_stride,
            vis_max_frames=args.vis_max_frames,
            vis_object_points=args.vis_object_points,
        )

    contract_report = validate_dai_raw_contract(
        raw_dir=raw_dir,
        task=task,
        object_name=object_name,
        frame_ids=frame_ids,
        sides=sides,
    )

    manifest = {
        "task": task,
        "object_name": object_name,
        "anchor_hand": anchor_hand,
        "embodiment_type": anchor_hand,
        "reference_camera": ref_cam,
        "num_frames": len(frame_ids),
        "valid_object_poses": n_obj,
        "valid_hand_frames": valid_counts,
        "retargeting_contract": contract_report,
        "files": {
            "config": str(raw_dir / "config.json"),
            "gravity": str(raw_dir / "gravity.json"),
            "hands_npz": str(npz_path),
            "object_layout": str(layout_path),
            "object_mesh": str(object_mesh_path),
            "visualization": jsonable(vis_files),
        },
    }
    with open(raw_dir / "dai_export_manifest.json", "w") as f:
        json.dump(jsonable(manifest), f, indent=2)

    logging.info("[done] Do-as-I-Do raw folder written to: %s", raw_dir)
    logging.info("[done] next command:")
    logging.info(
        "python launch.py --task %s --raw-dir %s --hand-type %s",
        task, raw_dir, anchor_hand,
    )


if __name__ == "__main__":
    main()
