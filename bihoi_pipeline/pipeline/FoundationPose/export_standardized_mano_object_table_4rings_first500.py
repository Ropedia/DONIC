#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export contact-refined EasyMocap / FoundationPose outputs into a compact,
explicit sequence format:

  <out_dir>/
    frames.npy                    (T,) int64, always 0..T-1
    source_frame_ids.npy         (T,) float64, original/resampled frame coordinate
    timestamps_sec.npy           (T,) float64
    mano_right.npz               global_orient(T,3), hand_pose(T,45), transl(T,3), betas(10,)
    mano_left.npz                same, standard MANO_LEFT (never mirrored)
    obj_pose.npz                 trans(T,3), quat(T,4)
    object_mesh.ply/.obj         unchanged canonical mesh (unless explicit mesh scale != 1)
    table.json                   table plane/height/range metadata
    metadata.json                full coordinate/convention declaration

Input expected from:
  bash run_contact_mano_refine_and_export_bimanual_args_30fps.sh \
      --data_name <name> --hand bimanual

Default source locations:
  <data_dir>/outputmano/contact_mano/right/smpl/*.json
  <data_dir>/outputmano/contact_mano/left/smpl/*.json
  <data_dir>/outputmano/smoothed_ob_in_world/*.txt
  <data_dir>/meshes/compressed.ply  (or <data_dir>/compressed.ply)
  <data_dir>/extri.yml

Coordinate contract written by this exporter
--------------------------------------------
* right-handed world frame
* +Z is physical up estimated from the three camera rings in extri.yml
* by default Z=0 is an automatically estimated tabletop plane
* metres for translation/mesh
* radians for MANO axis-angle
* MANO hand_pose is FULL 45D axis-angle, non-PCA, intended for
  flat_hand_mean=True. A downstream loader MUST NOT add MANO hand_mean again.
* left hand is standard MANO_LEFT.pkl geometry, with no mirroring.
* object pose maps the exported mesh's unchanged local/canonical coordinates
  into the exported world frame.
* object quaternion order is explicitly stored in metadata (default wxyz).

The exporter supports resampling to 30 fps. Translation is linearly
interpolated; rotations (object/global hand/15 MANO joints) use quaternion
SLERP.
"""

import argparse
import json
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


# -----------------------------------------------------------------------------
# basic utilities
# -----------------------------------------------------------------------------


def natural_key(path):
    s = Path(path).stem
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def numeric_stem(path):
    s = Path(path).stem
    m = re.search(r"(-?\d+)$", s)
    if m is None:
        raise ValueError(f"Cannot parse numeric frame id from: {path}")
    return int(m.group(1))


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_floats(text, n, name):
    arr = np.fromstring(str(text).replace(",", " "), sep=" ", dtype=np.float64)
    if arr.size != n:
        raise ValueError(f"{name} needs {n} numbers, got {arr.size}: {text!r}")
    return arr


def normalize(v, name="vector"):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError(f"Invalid {name}: {v}")
    return v / n


def aa_to_R(aa):
    return cv2.Rodrigues(np.asarray(aa, dtype=np.float64).reshape(3))[0]


def R_to_aa(R):
    return cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))[0].reshape(3)


def project_to_so3(R):
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64).reshape(3, 3))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def R_to_quat_xyzw(R):
    """Rotation matrix -> normalized quaternion [x,y,z,w]."""
    R = project_to_so3(R)
    tr = float(np.trace(R))
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-15)) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-15)) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-15)) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-15)
    return q


def quat_xyzw_to_R(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q /= max(float(np.linalg.norm(q)), 1e-15)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def quat_slerp(q0, q1, a):
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q0 /= max(np.linalg.norm(q0), 1e-15)
    q1 /= max(np.linalg.norm(q1), 1e-15)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = (1.0 - a) * q0 + a * q1
        q /= max(np.linalg.norm(q), 1e-15)
        return q
    theta = math.acos(dot)
    s = math.sin(theta)
    return (math.sin((1-a)*theta)/s) * q0 + (math.sin(a*theta)/s) * q1


def interp_linear(t_src, values, t_dst):
    values = np.asarray(values, dtype=np.float64)
    out = np.empty((len(t_dst),) + values.shape[1:], dtype=np.float64)
    flat = values.reshape(len(values), -1)
    out_flat = out.reshape(len(t_dst), -1)
    for j in range(flat.shape[1]):
        out_flat[:, j] = np.interp(t_dst, t_src, flat[:, j])
    return out


def interp_rotmat_sequence(t_src, R_src, t_dst):
    R_src = np.asarray(R_src, dtype=np.float64).reshape(-1, 3, 3)
    q_src = np.stack([R_to_quat_xyzw(R) for R in R_src], axis=0)
    out = []
    for t in t_dst:
        if t <= t_src[0]:
            q = q_src[0]
        elif t >= t_src[-1]:
            q = q_src[-1]
        else:
            hi = int(np.searchsorted(t_src, t, side="right"))
            lo = hi - 1
            den = float(t_src[hi] - t_src[lo])
            a = 0.0 if den <= 0 else float((t - t_src[lo]) / den)
            q = quat_slerp(q_src[lo], q_src[hi], a)
        out.append(quat_xyzw_to_R(q))
    return np.stack(out, axis=0)


def interp_axis_angle_sequence(t_src, aa_src, t_dst):
    aa_src = np.asarray(aa_src, dtype=np.float64)
    if aa_src.ndim != 2 or aa_src.shape[1] % 3 != 0:
        raise ValueError(f"Expected (T, 3*K) axis-angle, got {aa_src.shape}")
    K = aa_src.shape[1] // 3
    out = np.empty((len(t_dst), 3*K), dtype=np.float64)
    for k in range(K):
        Rs = np.stack([aa_to_R(v) for v in aa_src[:, 3*k:3*k+3]], axis=0)
        Ri = interp_rotmat_sequence(t_src, Rs, t_dst)
        out[:, 3*k:3*k+3] = np.stack([R_to_aa(R) for R in Ri], axis=0)
    return out


# -----------------------------------------------------------------------------
# camera-rig up estimation
# -----------------------------------------------------------------------------


def read_extri_yml(data_dir):
    path = Path(data_dir) / "extri.yml"
    if not path.exists():
        raise FileNotFoundError(path)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    names = fs.getNode("names")
    if names.empty():
        fs.release()
        raise ValueError(f"No names in {path}")
    cams = {}
    for i in range(names.size()):
        name = str(names.at(i).string())
        rn, tn = fs.getNode(f"Rot_{name}"), fs.getNode(f"T_{name}")
        if rn.empty() or tn.empty():
            continue
        R = np.asarray(rn.mat(), dtype=np.float64).reshape(3, 3)
        t = np.asarray(tn.mat(), dtype=np.float64).reshape(3)
        cams[name] = (R, t)
    fs.release()
    if not cams:
        raise RuntimeError("No valid cameras in extri.yml")
    return cams


def kmeans_1d(values, n_clusters, max_iter=100):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    centers = np.quantile(values, (np.arange(n_clusters) + 0.5) / n_clusters)
    labels = None
    for _ in range(max_iter):
        new_labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        new_centers = centers.copy()
        for k in range(n_clusters):
            mem = values[new_labels == k]
            if len(mem) == 0:
                raise RuntimeError("empty camera ring cluster")
            new_centers[k] = float(np.mean(mem))
        if labels is not None and np.array_equal(labels, new_labels):
            labels, centers = new_labels, new_centers
            break
        labels, centers = new_labels, new_centers
    order = np.argsort(centers)
    remap = np.empty(n_clusters, dtype=np.int64)
    remap[order] = np.arange(n_clusters)
    return remap[labels]


def fit_plane(points, orient_like=None):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c[None], full_matrices=False)
    n = normalize(vt[-1], "plane normal")
    if orient_like is not None and float(np.dot(n, orient_like)) < 0:
        n = -n
    return c, n


def estimate_up_world_camera_rings(cams, n_rings=4):
    """Estimate physical world-up from a multi-ring camera rig.

    Camera centers are projected onto the rig's dominant vertical axis, clustered
    into ``n_rings`` 1D height groups, and one plane is fit to every ring.  The
    averaged ring normal gives the up axis.  The sign is resolved from the common
    rig geometry used here: lower rings have a larger radius and upper rings a
    smaller radius.

    This works for the current 4-ring / 31-camera calibration and also keeps the
    ring count configurable for other captures.
    """
    n_rings = int(n_rings)
    if n_rings < 2:
        raise ValueError(f"num_camera_rings must be >= 2, got {n_rings}")
    names = sorted(cams.keys(), key=lambda x: natural_key(Path(str(x))))
    centers = []
    for name in names:
        R, t = cams[name]
        centers.append(-R.T @ t)
    centers = np.asarray(centers, dtype=np.float64)
    if len(centers) < n_rings * 3:
        raise ValueError(f"Need >= {n_rings*3} cameras, got {len(centers)}")

    rig_center = centers.mean(axis=0)
    _, _, vt = np.linalg.svd(centers - rig_center[None], full_matrices=False)
    axis0 = normalize(vt[-1], "camera rig axis")
    heights = (centers - rig_center[None]) @ axis0
    labels = kmeans_1d(heights, n_rings)

    ring = []
    normals = []
    for rid in range(n_rings):
        mask = labels == rid
        pts = centers[mask]
        c, n = fit_plane(pts, axis0)
        radial = pts - c[None]
        radial -= (radial @ n)[:, None] * n[None]
        radius = np.linalg.norm(radial, axis=1)
        mean_radius = float(np.mean(radius))
        mean_height = float(np.mean(heights[mask]))
        ring_names = [names[i] for i in np.where(mask)[0]]
        ring.append((c, n, mean_radius))
        normals.append(n)
        print(
            f"[Gravity] ring {rid}: cameras={ring_names}, "
            f"height={mean_height:.6f} m, radius={mean_radius:.6f} m"
        )

    up = normalize(np.sum(normals, axis=0), "averaged ring normal")
    low = int(np.argmax([r[2] for r in ring]))
    high = int(np.argmin([r[2] for r in ring]))
    up_hint = normalize(ring[high][0] - ring[low][0], "low-to-high ring direction")
    if abs(float(np.dot(up, up_hint))) < 0.75:
        raise RuntimeError("Camera-ring sign is ambiguous; use --up_world manually")
    if float(np.dot(up, up_hint)) < 0:
        up = -up
    return up, centers


def make_up_aligned_rotation(up_world):
    """Return R_export_from_world with rows = export x/y/z axes in old world."""
    z = normalize(up_world, "up_world")
    x_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x = x_hint - float(np.dot(x_hint, z)) * z
    if np.linalg.norm(x) < 1e-6:
        x_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x = x_hint - float(np.dot(x_hint, z)) * z
    x = normalize(x, "export x")
    y = normalize(np.cross(z, x), "export y")
    R = np.stack([x, y, z], axis=0)
    return project_to_so3(R)


# -----------------------------------------------------------------------------
# source parsing
# -----------------------------------------------------------------------------


def file_index(folder, suffix):
    folder = Path(folder)
    out = {}
    for p in sorted(folder.glob(f"*{suffix}"), key=natural_key):
        try:
            out[numeric_stem(p)] = p
        except ValueError:
            pass
    return out


def read_pose_txt(path):
    M = np.loadtxt(path)
    M = np.asarray(M, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(M)):
        raise ValueError(f"Non-finite object pose: {path}")
    return M


def read_hand_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            raise ValueError(f"Empty json: {path}")
        rec = data[0]
    elif isinstance(data, dict):
        rec = data
    else:
        raise ValueError(f"Unsupported JSON root: {type(data)}")

    for key in ("poses", "shapes", "Rh", "Th"):
        if key not in rec:
            raise KeyError(f"{path}: missing {key}")

    poses = np.asarray(rec["poses"], dtype=np.float64).reshape(-1)
    if poses.size != 48:
        raise ValueError(
            f"{path}: expected full48 from --contact_mano_save_pose_format full48, got {poses.size}"
        )
    betas = np.asarray(rec["shapes"], dtype=np.float64).reshape(-1)
    if betas.size < 10:
        betas = np.pad(betas, (0, 10 - betas.size))
    return {
        "poses": poses[:48],
        "betas": betas[:10],
        "Rh": np.asarray(rec["Rh"], dtype=np.float64).reshape(-1)[:3],
        "Th": np.asarray(rec["Th"], dtype=np.float64).reshape(-1)[:3],
        "pose_format": rec.get("contact_mano_pose_format", "unknown"),
    }


def resolve_mano_model_path(mano_dir, side):
    mano_dir = Path(mano_dir).expanduser().resolve()
    fname = "MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl"
    candidates = [
        mano_dir / fname,
        mano_dir / "mano" / fname,
        mano_dir / "MANO" / fname,
        mano_dir / "models" / "mano" / fname,
        mano_dir / "models" / "MANO" / fname,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Cannot find {fname}; tried:\n" + "\n".join(map(str, candidates)))


def create_mano_model(mano_dir, side, device):
    model_path = resolve_mano_model_path(mano_dir, side)
    kwargs = dict(
        model_path=str(model_path),
        is_rhand=(side == "right"),
        use_pca=False,
        flat_hand_mean=True,
    )
    try:
        model = smplx.MANO(**kwargs, batch_size=1)
    except TypeError:
        model = smplx.MANO(**kwargs)
    return model.to(device), model_path


def mano_root_from_betas(model, betas):
    """MANO root joint in model-local canonical coordinates for each beta row."""
    betas = np.asarray(betas, dtype=np.float32).reshape(-1, 10)
    device = next(model.parameters()).device
    with torch.inference_mode():
        b = torch.as_tensor(betas, dtype=torch.float32, device=device)
        v_template = model.v_template
        if v_template.ndim == 2:
            v_template = v_template.unsqueeze(0)
        shapedirs = model.shapedirs[..., :10]
        # (B,V,3) = (1,V,3) + (B,10) x (V,3,10)
        v_shaped = v_template + torch.einsum("bl,vcl->bvc", b, shapedirs)
        Jreg = model.J_regressor
        if getattr(Jreg, "is_sparse", False):
            Jreg = Jreg.to_dense()
        root_w = Jreg[0].reshape(-1)
        root = torch.einsum("v,bvc->bc", root_w, v_shaped)
    return root.detach().cpu().numpy().astype(np.float64)


def load_aligned_source(data_dir, right_dir, left_dir, object_dir):
    ri = file_index(right_dir, ".json")
    li = file_index(left_dir, ".json")
    oi = file_index(object_dir, ".txt")
    common = sorted(set(ri) & set(li) & set(oi))
    if len(common) < 2:
        raise RuntimeError(
            f"Need >=2 common frames; right={len(ri)}, left={len(li)}, object={len(oi)}, common={len(common)}"
        )
    dropped = {
        "right_only_or_partial": len(set(ri) - set(common)),
        "left_only_or_partial": len(set(li) - set(common)),
        "object_only_or_partial": len(set(oi) - set(common)),
    }
    hands = {"right": [], "left": []}
    objects = []
    for fid in common:
        hands["right"].append(read_hand_json(ri[fid]))
        hands["left"].append(read_hand_json(li[fid]))
        objects.append(read_pose_txt(oi[fid]))
    return np.asarray(common, dtype=np.int64), hands, np.stack(objects), dropped


# -----------------------------------------------------------------------------
# canonical/world conversion
# -----------------------------------------------------------------------------


def mesh_vertices(mesh):
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError("Empty mesh scene")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh, np.asarray(mesh.vertices, dtype=np.float64)


def estimate_table_height_from_object(mesh, object_poses_up, percentile=5.0, max_vertices=20000):
    mesh, verts = mesh_vertices(mesh)
    if len(verts) == 0:
        raise ValueError("Object mesh has no vertices")
    if len(verts) > max_vertices:
        idx = np.linspace(0, len(verts)-1, max_vertices).astype(np.int64)
        verts = verts[idx]
    bottoms = []
    for T in object_poses_up:
        # Need only z coordinate after object pose.
        z = verts @ T[2, :3].reshape(3, 1)
        z = z.reshape(-1) + float(T[2, 3])
        bottoms.append(float(np.min(z)))
    bottoms = np.asarray(bottoms, dtype=np.float64)
    h = float(np.percentile(bottoms, float(percentile)))
    return h, bottoms


def build_hand_standard(side, records, R_export_from_world, t_export_from_world, mano_dir, device):
    model, model_path = create_mano_model(mano_dir, side, device)
    poses = np.stack([r["poses"] for r in records], axis=0)
    betas_all = np.stack([r["betas"] for r in records], axis=0)
    Rh = np.stack([r["Rh"] for r in records], axis=0)
    Th = np.stack([r["Th"] for r in records], axis=0)

    # Contract requires one beta vector. Use robust median and report deviation.
    betas = np.median(betas_all, axis=0)
    beta_max_abs_dev = float(np.max(np.abs(betas_all - betas[None])))

    root_src = mano_root_from_betas(model, betas_all)
    root_out = mano_root_from_betas(model, betas[None])[0]

    Tn = len(records)
    global_orient = np.zeros((Tn, 3), dtype=np.float64)
    hand_pose = poses[:, 3:48].copy()
    transl = np.zeros((Tn, 3), dtype=np.float64)

    for i in range(Tn):
        R_rigid = aa_to_R(Rh[i])
        R_pose = aa_to_R(poses[i, :3])
        R_global = R_export_from_world @ R_rigid @ R_pose
        global_orient[i] = R_to_aa(R_global)

        # Source world root after external EasyMocap Rh/Th.
        root_world = R_rigid @ root_src[i] + Th[i]
        root_export = R_export_from_world @ root_world + t_export_from_world

        # In standard MANO, global_orient rotates about the root; transl simply
        # offsets the root from its model-local location root_out.
        transl[i] = root_export - root_out

    # Numerical convention sanity check on a few frames: reconstruct the source
    # representation (full48 then external Rh/Th), transform it into export world,
    # and compare against standard MANO(global_orient, hand_pose, transl).
    check_ids = sorted(set([0, Tn // 2, Tn - 1]))
    check_errors = []
    model.eval()
    with torch.inference_mode():
        for i in check_ids:
            src_out = model(
                global_orient=torch.as_tensor(poses[i:i+1, :3], dtype=torch.float32, device=device),
                hand_pose=torch.as_tensor(poses[i:i+1, 3:48], dtype=torch.float32, device=device),
                betas=torch.as_tensor(betas_all[i:i+1], dtype=torch.float32, device=device),
                return_verts=True,
            )
            v_src = src_out.vertices[0].detach().cpu().numpy().astype(np.float64)
            R_rigid = aa_to_R(Rh[i])
            v_world = v_src @ R_rigid.T + Th[i][None]
            v_export_target = v_world @ R_export_from_world.T + t_export_from_world[None]

            out_std = model(
                global_orient=torch.as_tensor(global_orient[i:i+1], dtype=torch.float32, device=device),
                hand_pose=torch.as_tensor(hand_pose[i:i+1], dtype=torch.float32, device=device),
                betas=torch.as_tensor(betas[None], dtype=torch.float32, device=device),
                transl=torch.as_tensor(transl[i:i+1], dtype=torch.float32, device=device),
                return_verts=True,
            )
            v_std = out_std.vertices[0].detach().cpu().numpy().astype(np.float64)
            e = np.linalg.norm(v_std - v_export_target, axis=1)
            check_errors.append({
                "frame_local_index": int(i),
                "mean_vertex_error_m": float(np.mean(e)),
                "max_vertex_error_m": float(np.max(e)),
            })

    max_check_error = max(x["max_vertex_error_m"] for x in check_errors) if check_errors else 0.0
    if max_check_error > 1e-3:
        print(
            f"[MANO][WARN] {side} standard-parameter reconstruction max vertex error "
            f"is {max_check_error*1000:.3f} mm. If source betas vary, this is expected because "
            "the requested contract stores one betas(10,) vector."
        )
    else:
        print(f"[MANO][check] {side}: max reconstruction error {max_check_error*1000:.4f} mm")

    del model
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "global_orient": global_orient,
        "hand_pose": hand_pose,
        "transl": transl,
        "betas": betas,
        "betas_per_frame": betas_all,
        "beta_max_abs_dev": beta_max_abs_dev,
        "model_path": str(model_path),
        "pose_formats": sorted(set(str(r["pose_format"]) for r in records)),
        "conversion_check": check_errors,
    }


def transform_object_poses(object_poses_world, R_export_from_world, t_export_from_world):
    Tew = np.eye(4, dtype=np.float64)
    Tew[:3, :3] = R_export_from_world
    Tew[:3, 3] = t_export_from_world
    return np.stack([Tew @ T for T in object_poses_world], axis=0)


def object_pose_to_trans_quat(object_poses, quat_order):
    trans = object_poses[:, :3, 3].astype(np.float64)
    qxyzw = np.stack([R_to_quat_xyzw(T[:3, :3]) for T in object_poses], axis=0)
    if quat_order == "xyzw":
        quat = qxyzw
    elif quat_order == "wxyz":
        quat = qxyzw[:, [3, 0, 1, 2]]
    else:
        raise ValueError(quat_order)
    return trans, quat


def observed_xy_range(obj_trans, hands, margin):
    pts = [np.asarray(obj_trans, dtype=np.float64)]
    for side in ("right", "left"):
        if side in hands:
            pts.append(np.asarray(hands[side]["transl"], dtype=np.float64))
    P = np.concatenate(pts, axis=0)
    good = np.isfinite(P).all(axis=1)
    P = P[good]
    if len(P) == 0:
        return [-1.0, 1.0, -1.0, 1.0]
    xmin, ymin = np.min(P[:, :2], axis=0)
    xmax, ymax = np.max(P[:, :2], axis=0)
    m = float(margin)
    return [float(xmin-m), float(xmax+m), float(ymin-m), float(ymax+m)]


# -----------------------------------------------------------------------------
# resampling
# -----------------------------------------------------------------------------


def resample_all(frame_ids, input_fps, output_fps, hands, object_poses):
    frame_ids = np.asarray(frame_ids, dtype=np.float64)
    t_src = (frame_ids - frame_ids[0]) / float(input_fps)
    if len(t_src) < 2 or t_src[-1] <= 0:
        raise ValueError("Invalid source timeline")

    contiguous = bool(np.all(np.diff(frame_ids) == 1))
    if abs(float(input_fps) - float(output_fps)) < 1e-9 and contiguous:
        return (
            np.arange(len(frame_ids), dtype=np.int64),
            frame_ids.copy(),
            t_src.copy(),
            hands,
            object_poses,
        )

    dt = 1.0 / float(output_fps)
    n = int(math.floor(t_src[-1] / dt + 1e-9)) + 1
    t_dst = np.arange(n, dtype=np.float64) * dt
    source_coord = frame_ids[0] + t_dst * float(input_fps)

    hands_out = {}
    for side, h in hands.items():
        hands_out[side] = {
            "global_orient": interp_axis_angle_sequence(t_src, h["global_orient"], t_dst),
            "hand_pose": interp_axis_angle_sequence(t_src, h["hand_pose"], t_dst),
            "transl": interp_linear(t_src, h["transl"], t_dst),
            "betas": h["betas"].copy(),
            "beta_max_abs_dev": h["beta_max_abs_dev"],
            "model_path": h["model_path"],
            "pose_formats": h["pose_formats"],
            "conversion_check": h.get("conversion_check", []),
        }

    R_obj = object_poses[:, :3, :3]
    t_obj = object_poses[:, :3, 3]
    Ri = interp_rotmat_sequence(t_src, R_obj, t_dst)
    ti = interp_linear(t_src, t_obj, t_dst)
    obj_out = np.tile(np.eye(4, dtype=np.float64), (len(t_dst), 1, 1))
    obj_out[:, :3, :3] = Ri
    obj_out[:, :3, 3] = ti

    return np.arange(len(t_dst), dtype=np.int64), source_coord, t_dst, hands_out, obj_out


# -----------------------------------------------------------------------------
# mesh export
# -----------------------------------------------------------------------------


def export_mesh(mesh_path, out_dir, mesh_scale_to_m):
    mesh_path = Path(mesh_path).expanduser().resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    suffix = mesh_path.suffix.lower()
    if suffix not in (".ply", ".obj"):
        suffix = ".ply"
    out_path = Path(out_dir) / f"object_mesh{suffix}"

    mesh = trimesh.load(str(mesh_path), process=False)
    mesh, verts = mesh_vertices(mesh)
    scale = float(mesh_scale_to_m)
    if abs(scale - 1.0) < 1e-12 and mesh_path.suffix.lower() == suffix:
        shutil.copy2(mesh_path, out_path)
    else:
        mesh = mesh.copy()
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
        mesh.export(str(out_path))

    mesh_check = trimesh.load(str(out_path), process=False)
    mesh_check, vcheck = mesh_vertices(mesh_check)
    extents = np.asarray(mesh_check.bounds[1] - mesh_check.bounds[0], dtype=np.float64)
    return out_path, mesh_check, {
        "source": str(mesh_path),
        "output": str(out_path),
        "scale_to_meter": scale,
        "num_vertices": int(len(vcheck)),
        "num_faces": int(len(mesh_check.faces)) if hasattr(mesh_check, "faces") else None,
        "extents_m": extents,
        "is_watertight": bool(getattr(mesh_check, "is_watertight", False)),
        "canonical_policy": "vertices preserved; no recentering, no axis flip, no mirroring",
    }


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Export standardized 30fps MANO + object + table sequence")
    ap.add_argument("--data_name", required=True)
    ap.add_argument("--easymocap_root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--data_dir", default="", help="Override <easymocap_root>/<data_name>")
    ap.add_argument("--output_dir", default="", help="Default: <data_dir>/standardized_hoi_30fps")
    ap.add_argument("--mano_dir", default="", help="Default: <easymocap_root>/FoundationPose/mycpp")
    ap.add_argument("--right_hand_dir", default="")
    ap.add_argument("--left_hand_dir", default="")
    ap.add_argument("--object_pose_dir", default="")
    ap.add_argument("--mesh_file", default="")

    ap.add_argument("--input_fps", type=float, default=30.0)
    ap.add_argument("--output_fps", type=float, default=30.0)
    ap.add_argument("--quat_order", choices=["wxyz", "xyzw"], default="wxyz")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")

    ap.add_argument(
        "--up_mode",
        choices=["auto_camera_rings", "auto_three_ring", "manual", "keep_z"],
        default="auto_camera_rings",
        help=(
            "How to define physical +Z. auto_camera_rings clusters camera centers by "
            "height and fits one plane per ring; auto_three_ring is kept only as a "
            "backward-compatible alias and still uses --num_camera_rings."
        ),
    )
    ap.add_argument("--up_world", default="", help='Manual old-world up vector, e.g. "0,0,1"')
    ap.add_argument(
        "--num_camera_rings", type=int, default=4,
        help="Number of physical camera rings in extri.yml. Default: 4."
    )

    ap.add_argument("--z_zero_mode", choices=["auto_table", "manual_table", "old_origin"], default="auto_table")
    ap.add_argument("--table_z", type=float, default=0.0,
                    help="For manual_table: table z after up-alignment but before z-zero translation")
    ap.add_argument("--table_bottom_percentile", type=float, default=5.0,
                    help="auto_table: percentile of transformed object-mesh bottom z values")
    ap.add_argument("--table_xy_range", default="",
                    help='Optional exported-world table range "xmin,xmax,ymin,ymax" in meters')
    ap.add_argument("--table_auto_margin", type=float, default=0.20,
                    help="If table_xy_range omitted, expand observed object/hand XY bounds by this margin")

    ap.add_argument("--mesh_scale_to_m", type=float, default=1.0,
                    help="Scale local mesh vertices to meters. Default assumes mesh already in meters.")
    ap.add_argument(
        "--max_frames",
        type=int,
        default=500,
        help="Only export the first N output frames after resampling. Default: 500. Use 0 to export all frames.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.easymocap_root).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else root / args.data_name
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "standardized_hoi_30fps"
    mano_dir = Path(args.mano_dir).expanduser().resolve() if args.mano_dir else root / "FoundationPose" / "mycpp"

    right_dir = Path(args.right_hand_dir).expanduser().resolve() if args.right_hand_dir else data_dir / "outputmano" / "contact_mano" / "right" / "smpl"
    left_dir = Path(args.left_hand_dir).expanduser().resolve() if args.left_hand_dir else data_dir / "outputmano" / "contact_mano" / "left" / "smpl"
    object_dir = Path(args.object_pose_dir).expanduser().resolve() if args.object_pose_dir else data_dir / "outputmano" / "smoothed_ob_in_world"
    if not object_dir.is_dir():
        alt = data_dir / "outputmano" / "ob_in_world"
        if alt.is_dir():
            object_dir = alt

    if args.mesh_file:
        mesh_file = Path(args.mesh_file).expanduser().resolve()
    elif (data_dir / "compressed.ply").is_file():
        mesh_file = data_dir / "compressed.ply"
    else:
        mesh_file = data_dir / "meshes" / "compressed.ply"

    for p, label in [(right_dir, "right hand"), (left_dir, "left hand"), (object_dir, "object pose"), (mano_dir, "MANO")]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {label}: {p}")
    if not mesh_file.exists():
        raise FileNotFoundError(f"Missing mesh: {mesh_file}")

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    elif out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output exists and is non-empty: {out_dir}; pass --overwrite")
    ensure_dir(out_dir)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU")
        device = "cpu"

    print(f"[Input] data_dir       = {data_dir}")
    print(f"[Input] right_hand_dir = {right_dir}")
    print(f"[Input] left_hand_dir  = {left_dir}")
    print(f"[Input] object_pose_dir= {object_dir}")
    print(f"[Input] mesh           = {mesh_file}")
    print(f"[Config] fps           = {args.input_fps} -> {args.output_fps}")
    print(f"[Config] device        = {device}")

    source_ids, source_hands, object_world, dropped = load_aligned_source(
        data_dir, right_dir, left_dir, object_dir
    )
    print(f"[Align] common source frames = {len(source_ids)}, {source_ids[0]}..{source_ids[-1]}")
    if any(v > 0 for v in dropped.values()):
        print(f"[Align][WARN] non-common files dropped: {dropped}")

    # 1) Determine physical +Z.
    camera_centers_old = None
    if args.up_mode in ("auto_camera_rings", "auto_three_ring"):
        cams = read_extri_yml(data_dir)
        up_world, camera_centers_old = estimate_up_world_camera_rings(
            cams, args.num_camera_rings
        )
        up_source = f"auto_camera_rings({args.num_camera_rings}) from extri.yml"
    elif args.up_mode == "manual":
        if not args.up_world:
            raise ValueError("--up_world required for --up_mode manual")
        up_world = normalize(parse_floats(args.up_world, 3, "--up_world"), "manual up")
        up_source = "manual --up_world"
    else:
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        up_source = "keep_z: original EasyMocap +Z assumed up"

    R_up = make_up_aligned_rotation(up_world)
    object_up = transform_object_poses(object_world, R_up, np.zeros(3))

    # Load source mesh in its original canonical coordinates for table estimate.
    mesh_src = trimesh.load(str(mesh_file), process=False)
    if abs(float(args.mesh_scale_to_m) - 1.0) > 1e-12:
        # Object poses are translations in metres. If source mesh needs a scale,
        # use the scaled canonical geometry for tabletop estimation and export.
        mesh_src, _ = mesh_vertices(mesh_src)
        mesh_src = mesh_src.copy()
        mesh_src.vertices = np.asarray(mesh_src.vertices, dtype=np.float64) * float(args.mesh_scale_to_m)

    # 2) Choose where Z=0 lives.
    table_bottoms = None
    if args.z_zero_mode == "auto_table":
        table_z_before_zero, table_bottoms = estimate_table_height_from_object(
            mesh_src,
            object_up,
            percentile=args.table_bottom_percentile,
        )
        z_source = (
            f"auto object-bottom percentile={args.table_bottom_percentile:g}; "
            "assumes object reaches/rests on tabletop during sequence"
        )
    elif args.z_zero_mode == "manual_table":
        table_z_before_zero = float(args.table_z)
        z_source = "manual --table_z after up-alignment"
    else:
        table_z_before_zero = 0.0
        z_source = "old_origin: rotated EasyMocap origin retained at Z=0"

    t_export = np.array([0.0, 0.0, -table_z_before_zero], dtype=np.float64)
    T_export_from_old = np.eye(4, dtype=np.float64)
    T_export_from_old[:3, :3] = R_up
    T_export_from_old[:3, 3] = t_export

    object_export = transform_object_poses(object_world, R_up, t_export)

    # 3) Convert EasyMocap Rh/Th + saved full48 to standard MANO params.
    hands = {}
    for side in ("right", "left"):
        print(f"[MANO] converting {side} hand to standard full-axis-angle params...")
        hands[side] = build_hand_standard(
            side,
            source_hands[side],
            R_up,
            t_export,
            mano_dir,
            device,
        )
        if hands[side]["beta_max_abs_dev"] > 1e-4:
            print(
                f"[MANO][WARN] {side} per-frame betas vary; max abs deviation from median = "
                f"{hands[side]['beta_max_abs_dev']:.6g}. Export contract uses median betas(10,)."
            )

    # 4) Optional fps normalization.
    frames, source_coord, timestamps, hands, object_export = resample_all(
        source_ids,
        args.input_fps,
        args.output_fps,
        hands,
        object_export,
    )

    # Keep only the first N OUTPUT frames after optional FPS resampling.
    # Static MANO betas stay as shape (10,) and therefore are not sliced.
    if args.max_frames > 0:
        N = min(int(args.max_frames), len(frames))
        frames = frames[:N]
        source_coord = source_coord[:N]
        timestamps = timestamps[:N]
        object_export = object_export[:N]
        for side in ("right", "left"):
            hands[side]["global_orient"] = hands[side]["global_orient"][:N]
            hands[side]["hand_pose"] = hands[side]["hand_pose"][:N]
            hands[side]["transl"] = hands[side]["transl"][:N]
        print(f"[Timeline] max_frames={args.max_frames}: keeping first {N} output frames")

    T = len(frames)
    if T < 1:
        raise RuntimeError("No output frames remain after applying --max_frames")
    print(f"[Timeline] output T={T}, fps={args.output_fps:g}, duration={timestamps[-1]:.3f}s")

    # 5) Object pose compact representation.
    obj_trans, obj_quat = object_pose_to_trans_quat(object_export, args.quat_order)

    # 6) Table XY range.
    if args.table_xy_range:
        table_range = parse_floats(args.table_xy_range, 4, "--table_xy_range").tolist()
        range_source = "manual --table_xy_range"
    else:
        table_range = observed_xy_range(obj_trans, hands, args.table_auto_margin)
        range_source = (
            f"AUTO observed interaction XY bounds + {args.table_auto_margin:g} m margin; "
            "this is NOT a measured physical table edge"
        )

    # 7) Mesh export. Pose maps this exact canonical mesh to world.
    mesh_out, mesh_check, mesh_meta = export_mesh(mesh_file, out_dir, args.mesh_scale_to_m)
    if not mesh_meta["is_watertight"]:
        print("[Mesh][WARN] source/exported object mesh is not watertight; geometry was kept unchanged.")

    # 8) Write arrays.
    np.save(out_dir / "frames.npy", frames.astype(np.int64))
    np.save(out_dir / "source_frame_ids.npy", source_coord.astype(np.float64))
    np.save(out_dir / "timestamps_sec.npy", timestamps.astype(np.float64))

    for side in ("right", "left"):
        h = hands[side]
        np.savez(
            out_dir / f"mano_{side}.npz",
            global_orient=np.asarray(h["global_orient"], dtype=np.float32),
            hand_pose=np.asarray(h["hand_pose"], dtype=np.float32),
            transl=np.asarray(h["transl"], dtype=np.float32),
            betas=np.asarray(h["betas"], dtype=np.float32),
        )

    np.savez(
        out_dir / "obj_pose.npz",
        trans=obj_trans.astype(np.float32),
        quat=obj_quat.astype(np.float32),
    )

    table = {
        "frame": "export_world",
        "units": "meters",
        "height_z_m": 0.0 if args.z_zero_mode != "old_origin" else None,
        "plane_abcd": [0.0, 0.0, 1.0, 0.0] if args.z_zero_mode != "old_origin" else None,
        "range_xy_m": {
            "xmin": float(table_range[0]),
            "xmax": float(table_range[1]),
            "ymin": float(table_range[2]),
            "ymax": float(table_range[3]),
        },
        "range_source": range_source,
        "z_zero_definition": (
            "Z=0 is the tabletop plane" if args.z_zero_mode != "old_origin"
            else "Z=0 is the rotated original EasyMocap world origin; table was not estimated"
        ),
        "table_z_before_zero_shift_m": float(table_z_before_zero),
        "table_height_source": z_source,
    }
    if table_bottoms is not None:
        table["auto_diagnostics"] = {
            "object_bottom_z_min_m": float(np.min(table_bottoms)),
            "object_bottom_z_median_m": float(np.median(table_bottoms)),
            "object_bottom_z_max_m": float(np.max(table_bottoms)),
            "percentile_used": float(args.table_bottom_percentile),
        }
    with open(out_dir / "table.json", "w") as f:
        json.dump(jsonable(table), f, indent=2)

    metadata = {
        "format": "standardized_bimanual_mano_object_table_v1",
        "num_frames": int(T),
        "fps": float(args.output_fps),
        "source_fps": float(args.input_fps),
        "resampled": bool(abs(args.input_fps - args.output_fps) > 1e-9),
        "units": {"length": "meter", "angle": "radian"},
        "timeline": {
            "frames_file": "frames.npy",
            "frames_definition": "contiguous output indices 0..T-1",
            "source_frame_ids_file": "source_frame_ids.npy",
            "timestamps_file": "timestamps_sec.npy",
            "source_common_frame_first": int(source_ids[0]),
            "source_common_frame_last": int(source_ids[-1]),
            "source_common_frame_count": int(len(source_ids)),
            "dropped_non_common_counts": dropped,
        },
        "world_frame": {
            "handedness": "right-handed",
            "up_axis": "+Z",
            "up_source": up_source,
            "up_vector_in_original_easymocap_world": up_world,
            "z_zero": table["z_zero_definition"],
            "T_export_world_from_original_easymocap_world": T_export_from_old,
            "xy_orientation_rule": (
                "export +X = original EasyMocap +X projected onto plane orthogonal to up; "
                "export +Y completes right-handed frame"
            ),
        },
        "mano_right": {
            "file": "mano_right.npz",
            "model": "standard MANO_RIGHT.pkl",
            "mirrored": False,
            "global_orient": {"shape": [T, 3], "representation": "axis-angle in export world"},
            "hand_pose": {
                "shape": [T, 45],
                "representation": "15x axis-angle flattened",
                "pca": False,
                "flat_hand_mean": True,
                "mean_policy": "DO NOT add hands_mean/hand_mean downstream",
            },
            "transl": {"shape": [T, 3], "units": "meter", "semantics": "standard MANO transl in export world"},
            "betas": {"shape": [10], "policy": "median of source per-frame shapes"},
            "source_pose_formats": hands["right"]["pose_formats"],
            "source_beta_max_abs_deviation": hands["right"]["beta_max_abs_dev"],
            "model_path_used_for_conversion": hands["right"]["model_path"],
            "conversion_check": hands["right"].get("conversion_check", []),
        },
        "mano_left": {
            "file": "mano_left.npz",
            "model": "standard MANO_LEFT.pkl",
            "mirrored": False,
            "global_orient": {"shape": [T, 3], "representation": "axis-angle in export world"},
            "hand_pose": {
                "shape": [T, 45],
                "representation": "15x axis-angle flattened",
                "pca": False,
                "flat_hand_mean": True,
                "mean_policy": "DO NOT add hands_mean/hand_mean downstream",
            },
            "transl": {"shape": [T, 3], "units": "meter", "semantics": "standard MANO transl in export world"},
            "betas": {"shape": [10], "policy": "median of source per-frame shapes"},
            "source_pose_formats": hands["left"]["pose_formats"],
            "source_beta_max_abs_deviation": hands["left"]["beta_max_abs_dev"],
            "model_path_used_for_conversion": hands["left"]["model_path"],
            "conversion_check": hands["left"].get("conversion_check", []),
        },
        "object": {
            "pose_file": "obj_pose.npz",
            "trans": {"shape": [T, 3], "units": "meter"},
            "quat": {"shape": [T, 4], "order": args.quat_order, "rotation": "object canonical -> export world"},
            "mesh_file": mesh_out.name,
            "mesh": mesh_meta,
            "canonical_consistency": (
                "obj_pose maps this exported mesh canonical/local coordinates directly into export world; "
                "the mesh is not recentered, mirrored, or axis-flipped"
            ),
        },
        "table": table,
        "source": {
            "data_dir": str(data_dir),
            "right_contact_mano": str(right_dir),
            "left_contact_mano": str(left_dir),
            "smoothed_object_pose": str(object_dir),
            "source_mesh": str(mesh_file),
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(jsonable(metadata), f, indent=2)

    # Human-readable contract summary.
    readme = "# Standardized bimanual HOI sequence\n\n"
    readme += f"- frames: T={T}, fps={args.output_fps:g}\n"
    readme += "- length unit: meter\n- angle unit: radian\n"
    readme += "- world: right-handed, +Z up\n"
    readme += f"- Z=0: {table['z_zero_definition']}\n"
    readme += "- mano_right/mano_left: standard MANO, non-PCA, hand_pose=(T,45) axis-angle, flat_hand_mean=True; DO NOT add hand mean\n"
    readme += f"- obj_pose quaternion order: {args.quat_order}\n"
    readme += f"- object canonical mesh: {mesh_out.name}; obj_pose maps exactly this canonical mesh into world\n"
    readme += f"- object mesh watertight: {mesh_meta['is_watertight']}\n"
    readme += f"- table XY range source: {range_source}\n"
    (out_dir / "README.md").write_text(readme)

    print("\n[Done] standardized export:", out_dir)
    print(f"  frames      : ({T},) @ {args.output_fps:g} fps")
    print(f"  mano_right  : global_orient({T},3), hand_pose({T},45), transl({T},3), betas(10,)")
    print(f"  mano_left   : global_orient({T},3), hand_pose({T},45), transl({T},3), betas(10,)")
    print(f"  obj_pose    : trans({T},3), quat({T},4), order={args.quat_order}")
    print(f"  object_mesh : {mesh_out.name}, watertight={mesh_meta['is_watertight']}")
    print(f"  world       : +Z up; {table['z_zero_definition']}")
    if table["height_z_m"] is None:
        print(f"  table       : height not defined; range={table['range_xy_m']}")
    else:
        print(f"  table       : z={table['height_z_m']:.6f} m, range={table['range_xy_m']}")


if __name__ == "__main__":
    main()
