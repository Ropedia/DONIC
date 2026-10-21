#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
restore_gcn_mano_flat_real_scale_v3_all_bones.py

Purpose / file responsibility
-----------------------------
This script restores an FMHR / GCN-generated MANO hand to the real-world scale
using real triangulated 3D hand keypoints. It is designed for the common FMHR
output format where the .pt file stores MANO parameters, e.g.

    [{'type': 'right', 'pose': Tensor[1,48], 'shape': Tensor[1,10], 'Rt': ...}]

instead of directly storing joints. The script therefore reconstructs the hand
mesh from MANO_RIGHT.pkl first, then obtains 21 keypoints from that generated
mesh using MANO's J_regressor plus fingertip vertices.

Pipeline
--------
1. Load GCN/FMHR .pt and extract MANO pose + betas.
2. Load MANO_RIGHT.pkl and TEMPLATE_RIGHT.obj.
3. Forward MANO with the original/unflat pose to generate posed mesh.
4. Convert generated mesh -> 21 MANO/MediaPipe-style joints:
      wrist, thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20.
5. Compute scale from the sum of all 21-keypoint skeleton bone lengths:
      scale = sum(real triangulated 3D bone lengths) / sum(GCN MANO generated bone lengths).
   Default bones are the 20 segments in the five chains:
      wrist->thumb, wrist->index, wrist->middle, wrist->ring, wrist->pinky.
6. Generate a flat/rest hand from the same betas.
7. Scale the flat hand to the real-world scale and align its wrist to real kpts[0].
8. Optionally solve a least-squares "real-scale beta" vector from:
      V_target_centered ~= V_template_centered + sum_i beta_i S_i_centered
   Note: this beta fit is approximate because MANO betas are not a true global
   scale parameter. The safest representation is still: original betas + external
   real_scale.

Example
-------
python restore_gcn_mano_flat_real_scale_v3_all_bones.py \
  --gcn-pt 0_right.pt \
  --template-obj TEMPLATE_RIGHT.obj \
  --real-kpts-json 000000.json \
  --mano-model-path MANO_RIGHT.pkl \
  --out-dir restored_flat_real_scale_out
"""

# Packaging changes: explicit triangulation/result paths; all numerical routines unchanged.
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires PyTorch because the GCN output is a .pt file.") from exc


# -----------------------------
# Basic IO utilities
# -----------------------------


def to_numpy(x: Any, dtype=np.float64) -> np.ndarray:
    """Convert torch/chumpy/scipy/list objects to numpy."""
    if x is None:
        raise ValueError("Cannot convert None to numpy array.")
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu") and hasattr(x, "numpy"):
        x = x.cpu().numpy()
    elif hasattr(x, "r"):
        # chumpy arrays in MANO pkl usually expose .r
        x = x.r
    elif hasattr(x, "toarray"):
        x = x.toarray()
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def squeeze_first(arr: np.ndarray) -> np.ndarray:
    """Remove batch dimension if arr has shape [1, ...]."""
    arr = np.asarray(arr)
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def load_torch_pt(path: Union[str, Path]) -> Any:
    """Load .pt robustly across PyTorch versions."""
    path = str(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_pickle_latin1(path: Union[str, Path]) -> Dict[str, Any]:
    path = str(path)
    with open(path, "rb") as f:
        try:
            return pickle.load(f, encoding="latin1")
        except TypeError:
            return pickle.load(f)


def read_obj(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Read vertices and faces from a simple OBJ file."""
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                idxs = []
                for p in parts[:3]:
                    # support f v, f v/vt, f v/vt/vn
                    idxs.append(int(p.split("/")[0]) - 1)
                if len(idxs) == 3:
                    faces.append(idxs)
    if not vertices:
        raise ValueError(f"No vertices found in OBJ: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def write_obj(path: Union[str, Path], vertices: np.ndarray, faces: Optional[np.ndarray] = None) -> None:
    """Write an OBJ mesh or point cloud. Faces are optional."""
    vertices = np.asarray(vertices, dtype=np.float64)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# generated by restore_gcn_mano_flat_real_scale_v3_all_bones.py\n")
        for v in vertices:
            f.write(f"v {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
        if faces is not None and len(faces) > 0:
            for tri in np.asarray(faces, dtype=np.int64):
                f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def write_keypoints_obj(path: Union[str, Path], joints: np.ndarray) -> None:
    """Write joints as OBJ vertices plus simple line segments for visualization."""
    joints = np.asarray(joints, dtype=np.float64)
    chains = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 21 hand keypoints\n")
        for v in joints:
            f.write(f"v {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
        for chain in chains:
            for a, b in zip(chain[:-1], chain[1:]):
                f.write(f"l {a + 1} {b + 1}\n")


# -----------------------------
# MANO parameter extraction
# -----------------------------


def flatten_tree(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    """Yield (path, value) pairs from nested dict/list structures."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            yield from flatten_tree(v, key)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            yield from flatten_tree(v, key)
    else:
        yield prefix, obj


def looks_like_numeric_array(x: Any) -> bool:
    try:
        if hasattr(x, "detach"):
            return True
        arr = np.asarray(x)
        return np.issubdtype(arr.dtype, np.number)
    except Exception:
        return False


def find_named_array(obj: Any, names: Sequence[str], min_size: Optional[int] = None) -> Tuple[np.ndarray, str]:
    """Find a numeric array by path/name suffix."""
    names_l = [n.lower() for n in names]
    candidates: List[Tuple[int, str, np.ndarray]] = []
    for path, value in flatten_tree(obj):
        low = path.lower().replace("-", "_")
        if not any(low.endswith(n) or f".{n}" in low or f"[{n}]" in low for n in names_l):
            continue
        if not looks_like_numeric_array(value):
            continue
        try:
            arr = to_numpy(value, dtype=np.float64)
        except Exception:
            continue
        if min_size is not None and arr.size < min_size:
            continue
        score = 0
        # exact final key is best
        final = low.split(".")[-1]
        if final in names_l:
            score += 10
        if "pose" in final or "shape" in final or "beta" in final:
            score += 2
        candidates.append((score, path, arr))
    if not candidates:
        raise KeyError(f"Could not find any of these fields: {names}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2], candidates[0][1]


def select_hand_entry(raw: Any, hand: str = "right") -> Any:
    """Select the right/left dict when FMHR saves [params_left, params_right]."""
    hand = hand.lower()
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (list, tuple)):
        # Prefer dict with type == right/left.
        for item in raw:
            if isinstance(item, dict) and str(item.get("type", "")).lower().startswith(hand[0]):
                return item
        # Otherwise if only one item exists, use it.
        if len(raw) == 1:
            return raw[0]
        # Otherwise default right -> last, left -> first, matching common [left,right].
        return raw[-1] if hand.startswith("r") else raw[0]
    return raw


def extract_mano_params_from_gcn_pt(
    pt_path: Union[str, Path],
    hand: str = "right",
    pose_key: Optional[str] = None,
    beta_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract pose and betas from FMHR/GCN .pt.

    Supported common formats:
      1) [{'type':'right', 'pose': [1,48], 'shape': [1,10], 'Rt': ...}]
      2) {'pose': ..., 'shape': ...}
      3) {'mano': [58 or 55], ...} where last 10 values are betas.
      4) direct Tensor/list with length 58/55/61, last 10 values are betas.
    """
    raw = load_torch_pt(pt_path)
    entry = select_hand_entry(raw, hand=hand)

    # Pose
    if pose_key:
        pose_arr, pose_path = find_named_array(entry, [pose_key], min_size=3)
    else:
        try:
            pose_arr, pose_path = find_named_array(
                entry,
                ["pose", "poses", "hand_pose", "mano_pose", "theta", "thetas", "rot"],
                min_size=3,
            )
        except KeyError:
            pose_arr, pose_path = None, ""

    # Betas / shape
    if beta_key:
        beta_arr, beta_path = find_named_array(entry, [beta_key], min_size=10)
    else:
        try:
            beta_arr, beta_path = find_named_array(
                entry,
                ["shape", "shapes", "beta", "betas", "mano_beta", "mano_betas"],
                min_size=10,
            )
        except KeyError:
            beta_arr, beta_path = None, ""

    # Fallback: a single MANO parameter vector with last 10 values as betas.
    if beta_arr is None or pose_arr is None:
        try:
            mano_arr, mano_path = find_named_array(entry, ["mano", "mano_param", "mano_params", "param", "params"], min_size=55)
            mano_arr = squeeze_first(np.asarray(mano_arr, dtype=np.float64)).reshape(-1)
            beta_arr = mano_arr[-10:]
            # if vector has 58: 3 global + 45 pose + 10 beta; if 55: 45 pose + 10 beta
            pose_arr = mano_arr[:-10]
            beta_path = mano_path + "[-10:]"
            pose_path = mano_path + "[:-10]"
        except KeyError:
            pass

    # Fallback: raw itself is a long numeric vector.
    if beta_arr is None or pose_arr is None:
        if looks_like_numeric_array(entry):
            arr = squeeze_first(to_numpy(entry, dtype=np.float64)).reshape(-1)
            if arr.size >= 55:
                beta_arr = arr[-10:]
                pose_arr = arr[:-10]
                beta_path = "raw[-10:]"
                pose_path = "raw[:-10]"

    if beta_arr is None or pose_arr is None:
        available = []
        for path, value in flatten_tree(entry):
            if looks_like_numeric_array(value):
                try:
                    available.append(f"{path}: shape={to_numpy(value, dtype=None).shape}")
                except Exception:
                    pass
        raise KeyError(
            "Could not extract MANO pose/betas from the .pt file.\n"
            "Available numeric fields:\n  - " + "\n  - ".join(available[:80])
        )

    pose = squeeze_first(np.asarray(pose_arr, dtype=np.float64)).reshape(-1)
    betas = squeeze_first(np.asarray(beta_arr, dtype=np.float64)).reshape(-1)[:10]

    # If pose is 45 hand-only axis-angle, prepend zero global orient.
    # If pose is 48, use as full MANO pose. If longer, keep first 48 or infer.
    if pose.size == 45:
        pose48 = np.concatenate([np.zeros(3, dtype=np.float64), pose], axis=0)
        pose_mode = "45 hand pose + zero global orient"
    elif pose.size >= 48:
        pose48 = pose[:48]
        pose_mode = "48 full MANO axis-angle pose"
    elif pose.size > 48 + 10:
        pose48 = pose[:48]
        pose_mode = "truncated to first 48 pose values"
    else:
        raise ValueError(f"Pose field {pose_path} has unsupported size {pose.size}; expected 45 or 48.")

    # Optional external Rt from FMHR.
    Rt = None
    Rt_path = None
    try:
        Rt_arr, Rt_path = find_named_array(entry, ["Rt", "rt", "extrinsic", "transform"], min_size=12)
        Rt = squeeze_first(np.asarray(Rt_arr, dtype=np.float64))
    except Exception:
        pass

    return {
        "raw": raw,
        "entry": entry,
        "pose48": pose48,
        "betas10": betas,
        "pose_path": pose_path,
        "beta_path": beta_path,
        "pose_mode": pose_mode,
        "Rt": Rt,
        "Rt_path": Rt_path,
    }


# -----------------------------
# Real keypoint JSON parser
# -----------------------------


def is_numeric_sequence(obj: Any) -> bool:
    try:
        arr = np.asarray(obj, dtype=np.float64)
        return arr.size > 0 and arr.ndim >= 1
    except Exception:
        return False


def collect_numeric_arrays(obj: Any, path: str = "") -> List[Tuple[str, np.ndarray]]:
    """Collect possible numeric keypoint arrays recursively from JSON."""
    out: List[Tuple[str, np.ndarray]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{path}.{k}" if path else str(k)
            out.extend(collect_numeric_arrays(v, child))
    elif isinstance(obj, list):
        # Try to parse this list itself.
        try:
            arr = np.asarray(obj, dtype=np.float64)
            if arr.ndim >= 2 and arr.shape[-1] >= 3:
                out.append((path, arr))
        except Exception:
            pass
        # Continue recursive search for irregular lists.
        for i, v in enumerate(obj):
            child = f"{path}[{i}]"
            if isinstance(v, (dict, list)):
                out.extend(collect_numeric_arrays(v, child))
    return out


def normalize_kpts_array(arr: np.ndarray, start_index: int = 0) -> Optional[np.ndarray]:
    """Return [21,3] if possible."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] >= start_index + 21 and arr.shape[1] >= 3:
        return arr[start_index:start_index + 21, :3]
    if arr.ndim == 3:
        # common shape [num_people/hands, 21, 3/4]
        if arr.shape[1] >= start_index + 21 and arr.shape[2] >= 3:
            return arr[0, start_index:start_index + 21, :3]
        # common shape [21, 3/4, something] is uncommon; ignore.
    if arr.ndim == 1 and arr.size >= (start_index + 21) * 3:
        flat = arr.reshape(-1, 3)
        return flat[start_index:start_index + 21, :3]
    return None


def load_real_kpts_json(
    json_path: Union[str, Path],
    hand: str = "right",
    real_kpts_key: Optional[str] = None,
    real_start_index: int = 0,
) -> Tuple[np.ndarray, str]:
    """Load real triangulated 3D hand keypoints from JSON.

    The function is intentionally permissive because EasyMocap / custom JSON
    formats vary a lot. It searches recursively for any numeric array that can
    be interpreted as [21, >=3].
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if real_kpts_key:
        # Dot-path lookup first.
        cur: Any = data
        ok = True
        for part in real_kpts_key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            arr = normalize_kpts_array(np.asarray(cur, dtype=np.float64), start_index=real_start_index)
            if arr is not None:
                return arr, real_kpts_key
        # fallback: path suffix match
        for path, arr0 in collect_numeric_arrays(data):
            if path.lower().endswith(real_kpts_key.lower()):
                arr = normalize_kpts_array(arr0, start_index=real_start_index)
                if arr is not None:
                    return arr, path
        raise KeyError(f"Could not load 21x3 keypoints using key: {real_kpts_key}")

    arrays = collect_numeric_arrays(data)
    candidates: List[Tuple[int, str, np.ndarray]] = []
    hand_l = hand.lower()
    for path, arr0 in arrays:
        arr = normalize_kpts_array(arr0, start_index=real_start_index)
        if arr is None:
            continue
        low = path.lower()
        score = 0
        if "keypoints3d" in low or "kpts3d" in low or "joints3d" in low:
            score += 10
        if "hand" in low:
            score += 5
        if hand_l.startswith("r") and ("right" in low or "rhand" in low or "handr" in low):
            score += 6
        if hand_l.startswith("l") and ("left" in low or "lhand" in low or "handl" in low):
            score += 6
        if arr0.ndim == 2 and arr0.shape[0] == 21:
            score += 5
        candidates.append((score, path, arr))

    if not candidates:
        found = [f"{p}: shape={a.shape}" for p, a in arrays[:50]]
        raise KeyError(
            "Could not find a 21x3 keypoint array in the JSON.\n"
            "Found numeric arrays:\n  - " + "\n  - ".join(found)
        )
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2], candidates[0][1]


# -----------------------------
# MANO model implementation
# -----------------------------


def batch_rodrigues_np(axisang: np.ndarray) -> np.ndarray:
    """Axis-angle to rotation matrices. Input [...,3], output [...,3,3]."""
    aa = np.asarray(axisang, dtype=np.float64).reshape(-1, 3)
    out = np.zeros((aa.shape[0], 3, 3), dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for i, r in enumerate(aa):
        theta = np.linalg.norm(r)
        if theta < 1e-12:
            # First-order approximation for tiny rotations.
            K = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]], dtype=np.float64)
            out[i] = eye + K
        else:
            k = r / theta
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
            out[i] = eye + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
    return out.reshape(axisang.shape[:-1] + (3, 3))


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


class MinimalMANO:
    """Minimal MANO forward implementation for right/left MANO pkl.

    It follows the common MANO/manopth linear blend skinning structure:
      v_shaped = v_template + shapedirs @ betas
      v_posed  = v_shaped + posedirs @ pose_feature
      vertices = LBS(v_posed, joints, rotations, weights)

    Joints are returned using mesh regression from final vertices, plus MANO
    fingertip vertices, then reordered to a common 21-point hand order.
    """

    def __init__(self, mano_pkl_path: Union[str, Path], side: str = "right", template_obj: Optional[Union[str, Path]] = None):
        self.side = side.lower()
        data = load_pickle_latin1(mano_pkl_path)
        self.data = data

        self.v_template_model = to_numpy(data["v_template"], dtype=np.float64)
        self.shapedirs = to_numpy(data["shapedirs"], dtype=np.float64)
        self.posedirs = to_numpy(data["posedirs"], dtype=np.float64)
        self.J_regressor = to_numpy(data["J_regressor"], dtype=np.float64)
        self.weights = to_numpy(data["weights"], dtype=np.float64)
        self.faces = to_numpy(data["f"], dtype=np.int64)

        if self.shapedirs.shape[0] != self.v_template_model.shape[0]:
            raise ValueError(f"Unexpected shapedirs shape: {self.shapedirs.shape}")
        if self.J_regressor.shape[1] != self.v_template_model.shape[0]:
            raise ValueError(f"Unexpected J_regressor shape: {self.J_regressor.shape}")

        self.n_verts = self.v_template_model.shape[0]
        self.num_betas = min(10, self.shapedirs.shape[-1])
        self.v_template_obj = None
        if template_obj is not None:
            v_obj, f_obj = read_obj(template_obj)
            if v_obj.shape == self.v_template_model.shape:
                self.v_template_obj = v_obj
            else:
                print(
                    f"[warn] TEMPLATE_RIGHT.obj vertex shape {v_obj.shape} does not match MANO v_template "
                    f"{self.v_template_model.shape}; using MANO pkl v_template for beta solving."
                )

        # MANO fingertip vertex ids used by manopth.
        if self.side.startswith("r"):
            self.tip_vertex_ids = np.asarray([745, 317, 444, 556, 673], dtype=np.int64)
        else:
            self.tip_vertex_ids = np.asarray([745, 317, 445, 556, 673], dtype=np.int64)

        # Reorder after appending tips to 16 MANO joints.
        # Output order: 0 wrist; 1-4 thumb; 5-8 index; 9-12 middle; 13-16 ring; 17-20 pinky.
        self.reorder_21 = np.asarray(
            [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
            dtype=np.int64,
        )

    def shape_vertices(self, betas: np.ndarray, template: Optional[np.ndarray] = None) -> np.ndarray:
        betas = np.asarray(betas, dtype=np.float64).reshape(-1)[: self.num_betas]
        base = self.v_template_model if template is None else np.asarray(template, dtype=np.float64)
        shapedirs = self.shapedirs[:, :, : betas.size]
        return base + np.einsum("vck,k->vc", shapedirs, betas)

    def pose_blend(self, pose48: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pose48 = np.asarray(pose48, dtype=np.float64).reshape(16, 3)
        rot_mats = batch_rodrigues_np(pose48)  # [16,3,3]
        ident = np.eye(3, dtype=np.float64)
        pose_feature = (rot_mats[1:] - ident).reshape(-1)  # 15*9 = 135
        return rot_mats, pose_feature

    def apply_posedirs(self, v_shaped: np.ndarray, pose_feature: np.ndarray) -> np.ndarray:
        P = np.asarray(self.posedirs, dtype=np.float64)
        pf = np.asarray(pose_feature, dtype=np.float64).reshape(-1)
        if P.shape == (self.n_verts, 3, pf.size):
            correction = np.einsum("vcp,p->vc", P, pf)
        elif P.shape == (pf.size, self.n_verts * 3):
            correction = (pf @ P).reshape(self.n_verts, 3)
        elif P.shape == (self.n_verts * 3, pf.size):
            correction = (P @ pf).reshape(self.n_verts, 3)
        else:
            raise ValueError(f"Unsupported posedirs shape {P.shape}; pose_feature size={pf.size}")
        return v_shaped + correction

    def lbs_manual(self, v_posed: np.ndarray, J: np.ndarray, rot_mats: np.ndarray) -> np.ndarray:
        """MANO-specific LBS following manopth's hand kinematic layout."""
        # MANO kinematic chains in original MANO joint index order.
        lev1 = [1, 4, 7, 10, 13]
        lev2 = [2, 5, 8, 11, 14]
        lev3 = [3, 6, 9, 12, 15]

        A_by_joint = [None for _ in range(16)]
        A0 = make_transform(rot_mats[0], J[0])
        A_by_joint[0] = A0
        for i1, i2, i3 in zip(lev1, lev2, lev3):
            A1 = A0 @ make_transform(rot_mats[i1], J[i1] - J[0])
            A2 = A1 @ make_transform(rot_mats[i2], J[i2] - J[i1])
            A3 = A2 @ make_transform(rot_mats[i3], J[i3] - J[i2])
            A_by_joint[i1] = A1
            A_by_joint[i2] = A2
            A_by_joint[i3] = A3

        A = np.stack(A_by_joint, axis=0)  # [16,4,4]
        # Remove rest joint location: A_i[:,3] -= A_i[:3,:3] @ J_i
        A_corr = A.copy()
        for j in range(16):
            A_corr[j, :3, 3] -= A_corr[j, :3, :3] @ J[j]

        W = self.weights[:, :16]
        T = np.einsum("vj,jab->vab", W, A_corr)
        v_h = np.concatenate([v_posed, np.ones((v_posed.shape[0], 1), dtype=np.float64)], axis=1)
        verts = np.einsum("vab,vb->va", T, v_h)[:, :3]
        return verts

    def forward(
        self,
        betas: np.ndarray,
        pose48: Optional[np.ndarray] = None,
        flat: bool = False,
        use_pose_blend_for_flat: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Generate MANO vertices and 21 joints.

        Args:
            betas: [10].
            pose48: [48] full axis-angle pose. If None, zeros are used.
            flat: if True, force all 48 pose values to zero.
            use_pose_blend_for_flat: if False, flat hand is pure v_shaped without
                pose blend correction. This is usually what you want for canonical
                shape/beta inspection.
        """
        betas = np.asarray(betas, dtype=np.float64).reshape(-1)[: self.num_betas]
        if pose48 is None or flat:
            pose48 = np.zeros(48, dtype=np.float64)
        else:
            pose48 = np.asarray(pose48, dtype=np.float64).reshape(-1)[:48]
        rot_mats, pose_feature = self.pose_blend(pose48)
        v_shaped = self.shape_vertices(betas)
        J_shaped = self.J_regressor @ v_shaped
        if flat and not use_pose_blend_for_flat:
            v_posed = v_shaped.copy()
        else:
            v_posed = self.apply_posedirs(v_shaped, pose_feature)
        verts = self.lbs_manual(v_posed, J_shaped, rot_mats)
        joints21 = self.mesh_to_21_joints(verts)
        return {
            "vertices": verts,
            "joints21": joints21,
            "v_shaped": v_shaped,
            "J_shaped": J_shaped,
            "pose48": pose48,
            "betas": betas,
        }

    def mesh_to_21_joints(self, vertices: np.ndarray) -> np.ndarray:
        """Use generated mesh to get MANO 21 keypoints."""
        vertices = np.asarray(vertices, dtype=np.float64)
        j16 = self.J_regressor @ vertices
        tips = vertices[self.tip_vertex_ids]
        j21_raw = np.concatenate([j16, tips], axis=0)
        return j21_raw[self.reorder_21]

    def centered_shape_basis_for_beta_fit(self, template_vertices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return centered template and centered shapedirs for beta least squares.

        Centering is done at the MANO wrist regressor joint, so translation does
        not contaminate the beta solve.
        """
        if template_vertices is None:
            template_vertices = self.v_template_obj if self.v_template_obj is not None else self.v_template_model
        V0 = np.asarray(template_vertices, dtype=np.float64)
        wrist0 = self.J_regressor[0] @ V0
        V0_centered = V0 - wrist0[None, :]

        S = self.shapedirs[:, :, : self.num_betas].copy()
        # Wrist displacement per basis: [10,3]
        wrist_S = np.einsum("v,vck->ck", self.J_regressor[0], S)  # [3,K]
        S_centered = S - np.transpose(wrist_S, (1, 0))[None, :, :] if False else None
        # clearer version: for each beta k subtract wrist displacement [3]
        S_centered = S.copy()
        for k in range(self.num_betas):
            d_wrist = self.J_regressor[0] @ S[:, :, k]
            S_centered[:, :, k] -= d_wrist[None, :]
        return V0_centered, S_centered

    def solve_betas_from_centered_vertices(
        self,
        target_vertices_centered: np.ndarray,
        template_vertices: Optional[np.ndarray] = None,
        ridge: float = 1e-8,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Least-squares solve beta from centered target mesh."""
        V0c, Sc = self.centered_shape_basis_for_beta_fit(template_vertices=template_vertices)
        Y = np.asarray(target_vertices_centered, dtype=np.float64) - V0c
        A = Sc.reshape(-1, self.num_betas)
        b = Y.reshape(-1)
        if ridge > 0:
            AtA = A.T @ A + ridge * np.eye(self.num_betas)
            Atb = A.T @ b
            beta = np.linalg.solve(AtA, Atb)
        else:
            beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        recon = V0c + np.einsum("vck,k->vc", Sc, beta)
        rmse = float(np.sqrt(np.mean(np.sum((recon - target_vertices_centered) ** 2, axis=1))))
        return beta, rmse, recon


# -----------------------------
# Scale computation
# -----------------------------


def chain_length(kpts: np.ndarray, chain: Sequence[int]) -> float:
    pts = np.asarray(kpts, dtype=np.float64)
    total = 0.0
    for a, b in zip(chain[:-1], chain[1:]):
        total += float(np.linalg.norm(pts[b] - pts[a]))
    return total


def parse_finger_chains(spec: str) -> List[List[int]]:
    """Parse '5,6,7,8;9,10,11,12;...' into list of chains."""
    chains: List[List[int]] = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        chain = [int(x.strip()) for x in part.split(",") if x.strip()]
        if len(chain) < 2:
            raise ValueError(f"Bad chain '{part}', expected at least two joint ids.")
        chains.append(chain)
    if not chains:
        raise ValueError("No valid finger chains parsed.")
    return chains


def compute_scale_from_chains(real_kpts: np.ndarray, gcn_kpts: np.ndarray, chains: List[List[int]]) -> Dict[str, Any]:
    real_lengths = []
    gcn_lengths = []
    ratios = []
    for chain in chains:
        lr = chain_length(real_kpts, chain)
        lg = chain_length(gcn_kpts, chain)
        if lg <= 1e-12:
            raise ValueError(f"GCN/MANO chain length is near zero for chain {chain}: {lg}")
        real_lengths.append(lr)
        gcn_lengths.append(lg)
        ratios.append(lr / lg)
    scale_sum = float(np.sum(real_lengths) / np.sum(gcn_lengths))
    scale_mean_ratio = float(np.mean(ratios))
    scale_median_ratio = float(np.median(ratios))
    return {
        "scale": scale_sum,
        "scale_sum_ratio": scale_sum,
        "scale_mean_ratio": scale_mean_ratio,
        "scale_median_ratio": scale_median_ratio,
        "real_lengths": real_lengths,
        "gcn_lengths": gcn_lengths,
        "ratios": ratios,
        "chains": chains,
    }


def center_at_wrist_and_scale_to_real(
    vertices: np.ndarray,
    joints21: np.ndarray,
    scale: float,
    real_wrist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Center by generated wrist, scale, then translate to real wrist."""
    wrist = np.asarray(joints21[0], dtype=np.float64)
    real_wrist = np.asarray(real_wrist, dtype=np.float64)
    V = (np.asarray(vertices, dtype=np.float64) - wrist[None, :]) * scale + real_wrist[None, :]
    J = (np.asarray(joints21, dtype=np.float64) - wrist[None, :]) * scale + real_wrist[None, :]
    return V, J


def apply_Rt_if_possible(vertices: np.ndarray, Rt: Optional[np.ndarray]) -> np.ndarray:
    """Apply a 3x4/4x4 Rt to vertices if supplied. Mostly for optional debug output."""
    if Rt is None:
        return vertices
    Rt = np.asarray(Rt, dtype=np.float64)
    if Rt.shape == (4, 4):
        R, t = Rt[:3, :3], Rt[:3, 3]
    elif Rt.shape == (3, 4):
        R, t = Rt[:3, :3], Rt[:3, 3]
    elif Rt.size >= 12:
        tmp = Rt.reshape(-1)[:12].reshape(3, 4)
        R, t = tmp[:3, :3], tmp[:3, 3]
    else:
        return vertices
    return np.asarray(vertices) @ R.T + t[None, :]


# -----------------------------
# Main
# -----------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Restore FMHR/GCN MANO hand to real scale and generate flat hand.")
    p.add_argument("--gcn-pt", default=None, help="GCN/FMHR output .pt. Default is chosen from --hand_side: gcn_out/0_<side>.pt")
    p.add_argument("--template-obj", default=None, help="MANO template OBJ. Default: TEMPLATE_LEFT.obj or TEMPLATE_RIGHT.obj based on --hand_side")
    p.add_argument("--real-kpts-json", default=None, help="Explicit triangulated frame-zero keypoints")
    p.add_argument("--out-dir", type=Path, default=None, help="Explicit result directory")
    p.add_argument("--name", required=True, help="which person's hand")
    p.add_argument("--mano-model-path", default=None, help="MANO pkl. Default: mano/MANO_LEFT.pkl or mano/MANO_RIGHT.pkl based on --hand_side")
    p.add_argument(
        "--hand_side", "--hand-side", "--hand",
        dest="hand_side",
        default="left",
        choices=["right", "left"],
        help="Hand side (default: left). --hand is kept as a backward-compatible alias.",
    )
    p.add_argument("--pose-key", default=None, help="Optional explicit pose key in .pt")
    p.add_argument("--beta-key", default=None, help="Optional explicit beta/shape key in .pt")
    p.add_argument("--real-kpts-key", default=None, help="Optional explicit key/dot-path in JSON for real 3D keypoints")
    p.add_argument("--real-start-index", type=int, default=0, help="Start index if JSON stores more than 21 keypoints")
    p.add_argument(
        "--finger-chains",
        default="0,1,2,3,4;0,5,6,7,8;0,9,10,11,12;0,13,14,15,16;0,17,18,19,20",
        help="Bone chains used to compute scale. Default = all 20 bones of the 21-keypoint hand skeleton.",
    )
    p.add_argument(
        "--scale-stat",
        default="sum",
        choices=["sum", "mean", "median"],
        help="How to combine per-chain scale ratios. Default sum means sum(all real bone lengths) / sum(all generated bone lengths).",
    )
    p.add_argument("--ridge", type=float, default=1e-8, help="Ridge term for real-scale beta least-squares solve")
    p.add_argument(
        "--flat-use-pose-blend",
        action="store_true",
        help="For flat hand, still apply pose blend shape from zero pose. Usually leave this off.",
    )
    p.add_argument(
        "--also-write-rt-debug",
        action="store_true",
        help="Also write original posed mesh transformed by FMHR Rt if Rt exists. This is only for debugging.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    from pathlib import Path

    side = args.hand_side.lower()
    side_upper = side.upper()
    mano_output = "manol" if side == "left" else "manor"

    # Side-dependent defaults. Explicit CLI paths still override these.
    if args.gcn_pt is None:
        args.gcn_pt = f"gcn_out/0_{side}.pt"
    if args.template_obj is None:
        args.template_obj = f"TEMPLATE_{side_upper}.obj"
    if args.mano_model_path is None:
        args.mano_model_path = f"mano/MANO_{side_upper}.pkl"

    out_dir = args.out_dir if args.out_dir is not None else Path(f"../{args.name}/calib")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] loading GCN/FMHR .pt: {args.gcn_pt}")
    gcn = extract_mano_params_from_gcn_pt(args.gcn_pt, hand=args.hand_side, pose_key=args.pose_key, beta_key=args.beta_key)
    pose48 = gcn["pose48"]
    betas10 = gcn["betas10"]
    print(f"[info] extracted pose from: {gcn['pose_path']} ({gcn['pose_mode']})")
    print(f"[info] extracted betas from: {gcn['beta_path']}")
    print(f"[info] betas = {np.array2string(betas10, precision=6, separator=', ')}")
    if gcn.get("Rt") is not None:
        print(f"[info] found optional Rt from: {gcn.get('Rt_path')}, shape={np.asarray(gcn['Rt']).shape}")

    print(f"[info] loading MANO model: {args.mano_model_path}")
    mano = MinimalMANO(args.mano_model_path, side=args.hand_side, template_obj=args.template_obj)
    faces = mano.faces

    print(f"[info] loading real triangulated 3D keypoints")
    real_kpts_json = args.real_kpts_json or f"../{args.name}/output/{mano_output}/keypoints3d/000000.json"
    real_kpts, real_path = load_real_kpts_json(
        real_kpts_json,
        hand=args.hand_side,
        real_kpts_key=args.real_kpts_key,
        real_start_index=args.real_start_index,
    )
    print(f"[info] real kpts path: {real_path}; shape={real_kpts.shape}")

    # 1) Generate original/unflat posed hand from MANO parameters.
    print("[info] forwarding MANO with original/unflat pose to generate mesh and mesh-regressed joints...")
    posed = mano.forward(betas10, pose48=pose48, flat=False)
    posed_vertices = posed["vertices"]
    posed_joints = mano.mesh_to_21_joints(posed_vertices)  # explicitly from generated mesh

    # 2) Compute real scale from real triangulated kpts vs generated posed mesh kpts.
    chains = parse_finger_chains(args.finger_chains)
    scale_info = compute_scale_from_chains(real_kpts, posed_joints, chains)
    if args.scale_stat == "sum":
        scale = scale_info["scale_sum_ratio"]
    elif args.scale_stat == "mean":
        scale = scale_info["scale_mean_ratio"]
    else:
        scale = scale_info["scale_median_ratio"]
    print(f"[info] scale ({args.scale_stat}) = {scale:.10f}")
    print(f"[info] total real bone length = {float(np.sum(scale_info['real_lengths'])):.8f}")
    print(f"[info] total gcn/mano bone length = {float(np.sum(scale_info['gcn_lengths'])):.8f}")
    for chain, lr, lg, ratio in zip(chains, scale_info["real_lengths"], scale_info["gcn_lengths"], scale_info["ratios"]):
        print(f"       chain {chain}: real={lr:.8f}, gcn={lg:.8f}, ratio={ratio:.8f}")

    # 3) Generate flat/rest hand with same betas.
    print("[info] generating flat/rest MANO hand from the same betas...")
    flat = mano.forward(betas10, pose48=np.zeros(48), flat=True, use_pose_blend_for_flat=args.flat_use_pose_blend)
    flat_vertices = flat["vertices"]
    flat_joints = mano.mesh_to_21_joints(flat_vertices)

    # 4) Restore posed and flat meshes to real scale and align wrist to real wrist.
    posed_real_vertices, posed_real_joints = center_at_wrist_and_scale_to_real(
        posed_vertices, posed_joints, scale=scale, real_wrist=real_kpts[0]
    )
    flat_real_vertices, flat_real_joints = center_at_wrist_and_scale_to_real(
        flat_vertices, flat_joints, scale=scale, real_wrist=real_kpts[0]
    )

    # 5) Solve approximate real-scale beta vector in canonical wrist-centered coordinates.
    print("[info] solving approximate real-scale betas by least squares...")
    # Target centered by flat generated wrist, scaled to real physical size but still canonical/wrist-centered.
    target_flat_centered = (flat_vertices - flat_joints[0][None, :]) * scale
    template_for_solve = mano.v_template_obj if mano.v_template_obj is not None else mano.v_template_model
    real_scale_betas, beta_fit_rmse, beta_recon_centered = mano.solve_betas_from_centered_vertices(
        target_flat_centered,
        template_vertices=template_for_solve,
        ridge=args.ridge,
    )
    print(f"[info] approximate real-scale betas = {np.array2string(real_scale_betas, precision=6, separator=', ')}")
    print(f"[info] beta fit RMSE per vertex = {beta_fit_rmse:.10f} in JSON/real-kpt units")

    # Reconstruct canonical flat mesh from solved real-scale betas, then put wrist at real wrist for visualization.
    V0c, Sc = mano.centered_shape_basis_for_beta_fit(template_vertices=template_for_solve)
    beta_recon_world = beta_recon_centered + real_kpts[0][None, :]
    beta_recon_joints = mano.mesh_to_21_joints(beta_recon_world)  # approximate because mesh already centered/translated

    # 6) Save files.
    print(f"[info] writing outputs to: {out_dir}")





    write_obj(out_dir / "05_flat_real_scale_beta_lstsq_world.obj", beta_recon_world, faces)




    if args.also_write_rt_debug and gcn.get("Rt") is not None:
        rt_vertices = apply_Rt_if_possible(posed_vertices, gcn["Rt"])
        rt_joints = mano.mesh_to_21_joints(rt_vertices)
        write_obj(out_dir / "debug_posed_original_scale_after_Rt.obj", rt_vertices, faces)
        write_keypoints_obj(out_dir / "debug_posed_original_scale_after_Rt_joints.obj", rt_joints)

    save_dict = {
        "pose48": torch.tensor(pose48, dtype=torch.float32),
        "original_betas10": torch.tensor(betas10, dtype=torch.float32),
        "real_scale": float(scale),
        "real_scale_betas10_lstsq": torch.tensor(real_scale_betas, dtype=torch.float32),
        "posed_vertices_original_scale": torch.tensor(posed_vertices, dtype=torch.float32),
        "posed_joints21_original_scale_from_mesh": torch.tensor(posed_joints, dtype=torch.float32),
        "posed_vertices_real_scale_world": torch.tensor(posed_real_vertices, dtype=torch.float32),
        "posed_joints21_real_scale_world": torch.tensor(posed_real_joints, dtype=torch.float32),
        "flat_vertices_original_scale": torch.tensor(flat_vertices, dtype=torch.float32),
        "flat_joints21_original_scale_from_mesh": torch.tensor(flat_joints, dtype=torch.float32),
        "flat_vertices_real_scale_world": torch.tensor(flat_real_vertices, dtype=torch.float32),
        "flat_joints21_real_scale_world": torch.tensor(flat_real_joints, dtype=torch.float32),
        "real_triangulated_kpts21": torch.tensor(real_kpts, dtype=torch.float32),
    }
    torch.save(save_dict, out_dir / "restored_flat_real_scale.pt")

    summary = {
        "gcn_pt": str(args.gcn_pt),
        "mano_model_path": str(args.mano_model_path),
        "template_obj": str(args.template_obj),
        "hand": args.hand_side,
        "pose_path": gcn["pose_path"],
        "beta_path": gcn["beta_path"],
        "pose_mode": gcn["pose_mode"],
        "finger_chains": chains,
        "scale_stat": args.scale_stat,
        "scale": float(scale),
        "scale_details": scale_info,
        "original_betas10": betas10.tolist(),
        "real_scale_betas10_lstsq": real_scale_betas.tolist(),
        "beta_fit_rmse_per_vertex": float(beta_fit_rmse),
        "notes": [
            "Scale is computed as sum(all selected real bone lengths) / sum(all selected MANO-generated posed bone lengths). Default selected bones are all 20 bones of the 21-keypoint skeleton.",
            "Flat hand is generated after scale estimation using the same original betas with pose set to zero.",
            "real_scale_betas10_lstsq is only an approximation because MANO betas are shape coefficients, not a pure global scale parameter.",
            "For physically correct rendering, prefer original_betas10 + external real_scale, or use 04_gcn_mano_flat_real_scale_world.obj directly.",
        ],
        "output_files": [

            "05_flat_real_scale_beta_lstsq_world.obj",
            "restored_flat_real_scale.pt",
            "summary.json",
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[done] finished.")
    print(f"[done] main flat real-scale mesh: {out_dir / '04_gcn_mano_flat_real_scale_world.obj'}")
    print(f"[done] saved pt: {out_dir / 'restored_flat_real_scale.pt'}")


if __name__ == "__main__":
    main()
