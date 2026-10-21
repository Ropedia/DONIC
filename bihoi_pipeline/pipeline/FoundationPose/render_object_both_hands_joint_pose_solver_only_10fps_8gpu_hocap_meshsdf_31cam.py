import os
import glob
import math
import json
import csv
import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import cv2
import imageio.v2 as imageio
import numpy as np
import trimesh
import pyrender
import torch
import smplx

try:
    # Exact HO-Cap MeshSDF CUDA loss. This pipeline intentionally has NO proxy fallback.
    from hocap_annotation.loss import MeshSDFLoss as HOCapMeshSDFLoss
    _HOCAP_MESHSDF_IMPORT_ERROR = None
except Exception as _e:
    HOCapMeshSDFLoss = None
    _HOCAP_MESHSDF_IMPORT_ERROR = _e

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
DEPTH_EXTS = ('.npy', '.png', '.tif', '.tiff', '.exr')
JSON_EXTS = ('.json',)


def natural_sort_key(path):
    import re
    name = Path(path).stem
    parts = re.split(r'(\d+)', str(name))
    return [int(s) if s.isdigit() else s.lower() for s in parts]


def set_logging_format():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_existing_file(folder, stem, exts):
    folder = str(folder)
    for ext in exts:
        p = os.path.join(folder, f'{stem}{ext}')
        if os.path.exists(p):
            return p
    for p in sorted(glob.glob(os.path.join(folder, f'{stem}.*'))):
        if os.path.splitext(p)[1].lower() in exts:
            return p
    return None


def list_camera_ids(images_root):
    if not os.path.isdir(images_root):
        return []
    return [p.name for p in sorted(Path(images_root).iterdir(), key=lambda x: natural_sort_key(str(x))) if p.is_dir()]


def readextfromyml(scene_dir):
    path = os.path.join(scene_dir, 'extri.yml')
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    names = fs.getNode('names')
    out = {}
    for i in range(names.size()):
        name = names.at(i).string()
        R = np.asarray(fs.getNode(f'Rot_{name}').mat(), dtype=np.float64).reshape(3, 3)
        t = np.asarray(fs.getNode(f'T_{name}').mat(), dtype=np.float64).reshape(3)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        out[str(name)] = {'Rot': R, 'T': t.reshape(3, 1), 'ext': T}
    fs.release()
    return out


def read_intrinsics_from_txt(scene_dir, downscale=1.0, txt_name='cam_K.txt'):
    path = os.path.join(scene_dir, txt_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    vals = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals.extend(float(x) for x in line.replace(',', ' ').split())
    if len(vals) != 9:
        raise ValueError(f'Expected 9 numbers in {path}, got {len(vals)}')
    K = np.asarray(vals, dtype=np.float64).reshape(3, 3)
    K[:2] *= float(downscale)
    return K


def read_intrinsics_per_camera(scene_dir, cam_ids, downscale=1.0):
    """Read EasyMocap intri.yml per-camera K when available; otherwise use cam_K.txt.

    Depth backprojection should use the calibration belonging to each camera.  Many of
    the user's captures share one K, but supporting K_<cam> keeps the 31-camera fusion
    correct when calibration differs slightly between cameras.
    """
    cam_ids = [str(c) for c in cam_ids]
    shared = read_intrinsics_from_txt(scene_dir, downscale=downscale)
    out = {c: shared.copy() for c in cam_ids}
    path = os.path.join(scene_dir, 'intri.yml')
    if not os.path.isfile(path):
        logging.info('[SDF] intri.yml not found; using shared cam_K.txt for all depth cameras.')
        return out

    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        logging.warning(f'[SDF] failed to open {path}; using shared cam_K.txt.')
        return out
    try:
        for cam in cam_ids:
            keys = [f'K_{cam}']
            if str(cam).isdigit():
                keys += [f'K_{int(cam):02d}', f'K_{int(cam):03d}']
            K = None
            for key in keys:
                node = fs.getNode(key)
                if not node.empty():
                    mat = node.mat()
                    if mat is not None and np.asarray(mat).size == 9:
                        K = np.asarray(mat, dtype=np.float64).reshape(3, 3)
                        break
            if K is not None:
                K = K.copy()
                K[:2] *= float(downscale)
                out[cam] = K
    finally:
        fs.release()
    return out


def _require_hocap_mesh_sdf(device):
    """Return the exact HO-Cap MeshSDFLoss or fail loudly; never substitute a proxy."""
    if HOCapMeshSDFLoss is None:
        raise RuntimeError(
            'HO-Cap MeshSDFLoss is required but could not be imported. '
            f'Original import error: {_HOCAP_MESHSDF_IMPORT_ERROR!r}.\\n'
            'Install HO-Cap-Annotation in THIS conda environment and build its CUDA extension:\\n'
            '  cd <HO-Cap-Annotation>\\n'
            '  python -m pip install -e . --no-cache-dir\\n'
            '  cd hocap_annotation/loss/meshsdf_loss\\n'
            '  python -m pip install . --no-cache-dir\\n'
            'Then verify: python -c "from hocap_annotation.loss import MeshSDFLoss; import meshsdf_loss_cuda; print(\\\"OK\\\")"'
        )
    try:
        loss = HOCapMeshSDFLoss().to(device)
        logging.info('[JointPoseSolver] exact HO-Cap hocap_annotation.loss.MeshSDFLoss loaded; proxy fallback is disabled.')
        return loss
    except Exception as e:
        raise RuntimeError(f'Failed to initialize exact HO-Cap MeshSDFLoss on {device}: {e}') from e


def read_pose_txt(path):
    return np.asarray(np.loadtxt(path), dtype=np.float64).reshape(4, 4)


def load_hand_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            raise ValueError(f'Empty json: {path}')
        return data[0]
    if isinstance(data, dict):
        return data
    raise ValueError(f'Unsupported hand json format: {path}')


def rodrigues_to_matrix(rvec):
    return cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))[0]


def matrix_to_rodrigues(R):
    return cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))[0].reshape(3)


def build_world_pose_from_Rh_Th(Rh, Th):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rodrigues_to_matrix(Rh)
    T[:3, 3] = np.asarray(Th, dtype=np.float64).reshape(3)
    return T


def transform_points(T, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    return pts @ T[:3, :3].T + T[:3, 3][None]


def resolve_mano_model_path(mano_dir, is_rhand=False):
    mano_dir = os.path.abspath(os.path.expanduser(str(mano_dir)))
    name = 'MANO_RIGHT.pkl' if is_rhand else 'MANO_LEFT.pkl'
    cands = [
        mano_dir,
        os.path.join(mano_dir, name),
        os.path.join(mano_dir, 'mano', name),
        os.path.join(mano_dir, 'MANO', name),
        os.path.join(mano_dir, 'models', 'mano', name),
        os.path.join(mano_dir, 'models', 'MANO', name),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, name)):
            return c
    raise FileNotFoundError(f'Cannot find {name} under {mano_dir}')


def create_mano_layer(mano_dir, is_rhand, num_pca_comps, flat_hand_mean, device):
    model_path = resolve_mano_model_path(mano_dir, is_rhand)
    kwargs = dict(
        model_path=model_path,
        is_rhand=bool(is_rhand),
        use_pca=True,
        num_pca_comps=int(num_pca_comps),
        flat_hand_mean=bool(flat_hand_mean),
    )
    try:
        layer = smplx.MANO(batch_size=1, **kwargs)
    except TypeError:
        layer = smplx.MANO(**kwargs)
    return layer.to(device)


class ManoHandBuilder:
    def __init__(self, mano_dir, is_rhand, device, num_pca_comps=20, flat_hand_mean=True):
        self.device = torch.device(device)
        self.num_pca_comps = int(num_pca_comps)
        self.model = create_mano_layer(
            mano_dir, is_rhand, self.num_pca_comps, flat_hand_mean, self.device
        )
        self.faces = np.asarray(self.model.faces, dtype=np.int64)

    def forward_local(self, pose23, betas10):
        pose23 = torch.as_tensor(pose23, dtype=torch.float32, device=self.device)
        betas10 = torch.as_tensor(betas10, dtype=torch.float32, device=self.device)
        if pose23.ndim == 1:
            pose23 = pose23[None]
        if betas10.ndim == 1:
            betas10 = betas10[None]
        out = self.model(
            global_orient=pose23[:, :3],
            hand_pose=pose23[:, 3:3 + self.num_pca_comps],
            betas=betas10[:, :10],
            return_verts=True,
        )
        return out.vertices

    def restore_full_pose_48(self, pose23_np):
        p = torch.as_tensor(pose23_np, dtype=torch.float32, device=self.device).reshape(1, -1)
        comps = self.model.hand_components[:self.num_pca_comps]
        if not torch.is_tensor(comps):
            comps = torch.tensor(comps, dtype=torch.float32, device=self.device)
        else:
            comps = comps.to(self.device).float()
        hand45 = p[:, 3:] @ comps
        full = torch.cat([p[:, :3], hand45], dim=1)
        pose_mean = getattr(self.model, 'pose_mean', None)
        if pose_mean is not None:
            if not torch.is_tensor(pose_mean):
                pose_mean = torch.tensor(pose_mean, dtype=torch.float32, device=self.device)
            full = full + pose_mean.reshape(1, 48).to(self.device).float()
        return full.detach().cpu().numpy().reshape(-1).astype(np.float32)


def torch_batch_rodrigues(rvec):
    rvec = rvec.reshape(-1, 3)
    theta = torch.linalg.norm(rvec, dim=1, keepdim=True)
    axis = rvec / torch.clamp(theta, min=1e-8)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = torch.zeros_like(x)
    K = torch.stack([
        zero, -z, y,
        z, zero, -x,
        -y, x, zero,
    ], dim=1).reshape(-1, 3, 3)
    I = torch.eye(3, dtype=rvec.dtype, device=rvec.device)[None].expand(len(rvec), -1, -1)
    s = torch.sin(theta)[:, None]
    c = torch.cos(theta)[:, None]
    R = I + s * K + (1.0 - c) * (K @ K)
    tiny = (theta.reshape(-1) < 1e-7)
    if torch.any(tiny):
        # First-order approximation keeps gradients stable near zero.
        rv = rvec[tiny]
        Ksmall = torch.stack([
            torch.zeros_like(rv[:, 0]), -rv[:, 2], rv[:, 1],
            rv[:, 2], torch.zeros_like(rv[:, 0]), -rv[:, 0],
            -rv[:, 1], rv[:, 0], torch.zeros_like(rv[:, 0]),
        ], dim=1).reshape(-1, 3, 3)
        R = R.clone()
        R[tiny] = I[tiny] + Ksmall
    return R


def prepare_pose23_and_betas10(rec, num_pca_comps=20):
    # Prefer our own saved PCA representation when present.
    for key in ('contact_mano_pose23', 'joint_mano_pose23'):
        if key in rec:
            p = np.asarray(rec[key], dtype=np.float32).reshape(-1)
            if p.size >= 3 + num_pca_comps:
                pose23 = p[:3 + num_pca_comps]
                break
    else:
        p = np.asarray(rec.get('poses', []), dtype=np.float32).reshape(-1)
        if p.size == 3 + num_pca_comps:
            pose23 = p.copy()
        elif p.size >= 48:
            # A full48 json cannot be uniquely converted back to PCA without the
            # MANO basis. If our PCA side-field is absent, use the first local root
            # and zeros for PCA rather than silently misreading full48 as PCA23.
            pose23 = np.zeros(3 + num_pca_comps, dtype=np.float32)
            pose23[:3] = p[:3]
            logging.warning('full48 hand json without contact_mano_pose23/joint_mano_pose23; PCA coefficients set to zero.')
        else:
            pose23 = np.zeros(3 + num_pca_comps, dtype=np.float32)
            pose23[:min(p.size, pose23.size)] = p[:min(p.size, pose23.size)]

    shapes = np.asarray(rec.get('shapes', rec.get('betas', np.zeros(10))), dtype=np.float32).reshape(-1)
    betas10 = np.zeros(10, dtype=np.float32)
    betas10[:min(10, shapes.size)] = shapes[:min(10, shapes.size)]
    Rh = np.asarray(rec.get('Rh', np.zeros(3)), dtype=np.float32).reshape(-1)[:3]
    Th = np.asarray(rec.get('Th', np.zeros(3)), dtype=np.float32).reshape(-1)[:3]
    return pose23, betas10, Rh, Th


def save_joint_mano_json(src_json, dst_json, pose23, pose48, betas10, Rh, Th, save_format='full48'):
    with open(src_json, 'r') as f:
        src = json.load(f)
    if isinstance(src, list):
        rec = dict(src[0]) if src else {}
        wrapper = 'list'
    else:
        rec = dict(src)
        wrapper = 'dict'
    rec['poses'] = [pose48.astype(float).tolist() if save_format == 'full48' else pose23.astype(float).tolist()]
    rec['shapes'] = [np.asarray(betas10, dtype=float).reshape(-1)[:10].tolist()]
    rec['Rh'] = [np.asarray(Rh, dtype=float).reshape(3).tolist()]
    rec['Th'] = [np.asarray(Th, dtype=float).reshape(3).tolist()]
    rec['contact_mano_pose23'] = [np.asarray(pose23, dtype=float).reshape(-1).tolist()]
    rec['joint_mano_method'] = 'joint_pose_solver_only'
    ensure_dir(os.path.dirname(dst_json))
    with open(dst_json, 'w') as f:
        json.dump([rec] if wrapper == 'list' else rec, f, indent=2)


def infer_hand_json_dirs(args):
    out = {'left': args.left_hand_json_dir, 'right': args.right_hand_json_dir}
    if args.hand_json_dir:
        if out['left'] is None:
            for name in ('left', 'manol', 'handl'):
                p = os.path.join(args.hand_json_dir, name, 'smpl')
                if os.path.isdir(p):
                    out['left'] = p
                    break
        if out['right'] is None:
            for name in ('right', 'manor', 'handr'):
                p = os.path.join(args.hand_json_dir, name, 'smpl')
                if os.path.isdir(p):
                    out['right'] = p
                    break
    return out


def _seq_vel_acc(x, temporal_scale):
    if x.shape[0] < 2:
        z = x.new_zeros(())
        return z, z
    v = (x[1:] - x[:-1]) * float(temporal_scale)
    vel = torch.mean(v * v)
    if x.shape[0] < 3:
        return vel, x.new_zeros(())
    a = (v[1:] - v[:-1]) * float(temporal_scale)
    return vel, torch.mean(a * a)


def _rot_vel_acc(rvec, temporal_scale):
    if rvec.shape[0] < 2:
        z = rvec.new_zeros(())
        return z, z
    R = torch_batch_rodrigues(rvec)
    dR = R[1:] - R[:-1]
    vel = torch.mean(dR * dR) * float(temporal_scale) ** 2
    if rvec.shape[0] < 3:
        return vel, rvec.new_zeros(())
    ddR = dR[1:] - dR[:-1]
    acc = torch.mean(ddR * ddR) * float(temporal_scale) ** 4
    return vel, acc


def _contiguous_segments(indices):
    if not indices:
        return []
    segs, cur = [], [indices[0]]
    for x in indices[1:]:
        if x == cur[-1] + 1:
            cur.append(x)
        else:
            segs.append(cur)
            cur = [x]
    segs.append(cur)
    return segs


def _load_depth_file(path, depth_scale):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        d = np.load(path)
    else:
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise RuntimeError(f'Failed to read depth: {path}')
    d = np.asarray(d)
    if d.ndim == 3:
        d = d[..., 0]
    d = d.astype(np.float32)
    valid = d[np.isfinite(d) & (d > 0)]
    if valid.size == 0:
        return d
    if depth_scale > 0:
        d *= float(depth_scale)
    else:
        med = float(np.median(valid))
        if np.issubdtype(np.asarray(d).dtype, np.integer) or med > 20.0:
            d *= 0.001
    return d


# SDF_FAST_PATCH_V1
def _make_backproject_cache(K, cam_to_world, H, W, stride=8):
    """Precompute sampled pixels, camera rays, and camera-to-world transform."""
    step = max(1, int(stride))
    ys = np.arange(0, H, step, dtype=np.int32)
    xs = np.arange(0, W, step, dtype=np.int32)
    xx, yy = np.meshgrid(xs, ys)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    ray_x = (xx.astype(np.float64) - cx) / fx
    ray_y = (yy.astype(np.float64) - cy) / fy

    T = np.asarray(cam_to_world, dtype=np.float64)
    return {
        'shape': (int(H), int(W)),
        'xx': xx,
        'yy': yy,
        'ray_x': ray_x,
        'ray_y': ray_y,
        'R': T[:3, :3].copy(),
        't': T[:3, 3].copy(),
    }


def _backproject_depth_world_fast(depth, cache, max_depth=5.0, mask=None):
    """Fast depth-to-world backprojection using precomputed rays/transforms."""
    xx = cache['xx']
    yy = cache['yy']
    z = depth[yy, xx]
    good = np.isfinite(z) & (z > 1e-6) & (z < float(max_depth))
    if mask is not None:
        good &= mask[yy, xx] > 0
    if not np.any(good):
        return np.zeros((0, 3), dtype=np.float32)

    z = z[good].astype(np.float64, copy=False)
    x = cache['ray_x'][good] * z
    y = cache['ray_y'][good] * z
    pc = np.stack([x, y, z], axis=1)
    pw = pc @ cache['R'].T + cache['t'][None]
    return pw.astype(np.float32, copy=False)


def _voxel_downsample_np(points, voxel):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0 or voxel <= 0:
        return points
    q = np.floor(points / float(voxel)).astype(np.int64)
    _, inds = np.unique(q, axis=0, return_index=True)
    return points[np.sort(inds)]


def _sample_rows(points, n, seed):
    points = np.asarray(points)
    if n <= 0 or len(points) <= n:
        return points
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(points), size=int(n), replace=False)
    return points[idx]


def _current_hand_world_numpy(builder, pose23, betas, Rh, Th):
    with torch.no_grad():
        p = torch.as_tensor(pose23, dtype=torch.float32, device=builder.device).reshape(1, -1)
        b = torch.as_tensor(betas, dtype=torch.float32, device=builder.device).reshape(1, -1)
        v = builder.forward_local(p, b)[0].detach().cpu().numpy()
    return transform_points(build_world_pose_from_Rh_Th(Rh, Th), v)


def _combined_numpy_geometry_for_frame(
    fi,
    object_mesh_vertices,
    object_faces,
    object_rvec_np,
    object_t_np,
    hand_states,
    hand_builders,
):
    """Build the exact per-frame combined hand+object mesh and valid face offsets."""
    R = rodrigues_to_matrix(object_rvec_np[fi])
    objw = np.asarray(object_mesh_vertices, dtype=np.float64) @ R.T + object_t_np[fi][None]
    verts_parts = [objw.astype(np.float32)]
    faces_parts = [np.asarray(object_faces, dtype=np.int64)]
    offset = len(objw)
    present_sides = []
    for side in ('left', 'right'):
        st = hand_states.get(side)
        if st is None:
            continue
        hi = st['frame_to_local'].get(fi)
        if hi is None:
            continue
        hv = _current_hand_world_numpy(
            hand_builders[side], st['pose23'][hi], st['betas'][hi], st['Rh'][hi], st['Th'][hi]
        ).astype(np.float32)
        verts_parts.append(hv)
        faces_parts.append(np.asarray(hand_builders[side].faces, dtype=np.int64) + offset)
        offset += len(hv)
        present_sides.append(side)
    return (
        np.concatenate(verts_parts, axis=0).astype(np.float32),
        np.concatenate(faces_parts, axis=0).astype(np.int64),
        present_sides,
    )


def _exact_meshsdf_filter_points(verts_np, faces_np, points_np, exact_sdf, device, dist_thresh):
    """HO-Cap-style observation filtering using MeshSDFLoss' returned distance."""
    if len(points_np) == 0:
        return points_np.astype(np.float32)
    v = torch.as_tensor(verts_np, dtype=torch.float32, device=device).contiguous()
    f = torch.as_tensor(faces_np, dtype=torch.long, device=device).contiguous()
    p = torch.as_tensor(points_np, dtype=torch.float32, device=device).contiguous()
    with torch.no_grad():
        _, dist, _ = exact_sdf(v, f, p)
    keep = (dist < float(dist_thresh)).detach().cpu().numpy().reshape(-1)
    return np.asarray(points_np, dtype=np.float32)[keep]


def _prepare_depth_points_for_sdf(
    frame_ids,
    object_mesh_vertices,
    object_faces,
    object_rvec_np,
    object_t_np,
    hand_states,
    hand_builders,
    scene_dir,
    args,
    exact_sdf,
    device,
):
    """Fuse all requested cameras and prepare exact HO-Cap MeshSDF observations.

    Same observation algorithm as before:
      depth -> world -> geometry crop -> exact MeshSDF distance filter
      -> voxel downsample -> point cap.

    Speedups only:
      * camera-to-world inverse computed once per camera
      * sampled backprojection rays computed once per camera
      * depth I/O + CPU backprojection parallelized across cameras
      * final exact-MeshSDF-filtered points cached for reruns
    """
    depth_root = args.depth_root or os.path.join(scene_dir, 'depth')
    if not os.path.isdir(depth_root):
        raise FileNotFoundError(f'[SDF] depth root missing: {depth_root}')

    cam_ext = readextfromyml(scene_dir)
    if args.sdf_cams:
        cams = [str(c) for c in args.sdf_cams]
    else:
        cams = [str(i) for i in range(1, int(args.sdf_expected_cam_count) + 1)]

    cam_stride = max(1, int(args.sdf_cam_stride))
    if cam_stride != 1:
        cams = cams[::cam_stride]

    if bool(args.sdf_require_all_cams):
        expected = int(args.sdf_expected_cam_count)
        if len(cams) != expected:
            raise RuntimeError(
                f'[SDF] expected exactly {expected} cameras after selection, got {len(cams)}: {cams}'
            )
        missing_ext = [c for c in cams if c not in cam_ext]
        missing_dirs = [c for c in cams if not os.path.isdir(os.path.join(depth_root, c))]
        if missing_ext or missing_dirs:
            raise RuntimeError(
                f'[SDF] all {expected} cameras are required. '
                f'missing extrinsics={missing_ext}, missing depth dirs={missing_dirs}'
            )

    Ks = read_intrinsics_per_camera(scene_dir, cams, downscale=args.downscale)

    # Expensive but constant values: compute once rather than cam x frame.
    cam_to_world = {
        cam: np.linalg.inv(np.asarray(cam_ext[cam]['ext'], dtype=np.float64))
        for cam in cams
    }

    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    io_workers = max(1, min(int(getattr(args, 'sdf_io_workers', 8)), len(cams)))

    logging.info(f'[SDF] using {len(cams)} depth cameras: {cams}')
    if len(cams) == 31:
        logging.info('[SDF] confirmed full 31-camera depth fusion.')
    logging.info(f'[SDF-fast] camera I/O/backprojection workers={io_workers}')

    # Precompute a backprojection lookup per camera using first available frame.
    bp_cache = {}
    if frame_ids:
        first_fid = frame_ids[0]
        for cam in cams:
            folder = os.path.join(depth_root, cam)
            dp = find_existing_file(folder, first_fid, DEPTH_EXTS)
            if dp is None:
                if bool(args.sdf_require_all_cams):
                    raise FileNotFoundError(
                        f'[SDF] first frame {first_fid}: missing depth for camera {cam}'
                    )
                continue
            d0 = _load_depth_file(dp, args.depth_scale)
            H, W = d0.shape[:2]
            bp_cache[cam] = _make_backproject_cache(
                Ks[cam], cam_to_world[cam], H, W, stride=args.sdf_pixel_stride
            )

    logging.info(
        f'[SDF-fast] precomputed backprojection rays for {len(bp_cache)}/{len(cams)} cameras'
    )

    reuse_cache = bool(int(getattr(args, 'sdf_reuse_cache', 1)))
    cache_dir = getattr(args, 'sdf_cache_dir', None)
    if not cache_dir:
        cache_dir = os.path.join(args.output_dir, 'sdf_dpts_cache_exact31')
    if reuse_cache:
        ensure_dir(cache_dir)
        logging.info(f'[SDF-cache] enabled: {cache_dir}')

    def process_one_camera(cam, fid, bbox_min, bbox_max):
        folder = os.path.join(depth_root, cam)
        dp = find_existing_file(folder, fid, DEPTH_EXTS)
        if dp is None:
            return cam, None, True

        depth = _load_depth_file(dp, args.depth_scale)
        cache = bp_cache.get(cam)
        if cache is None or tuple(cache['shape']) != tuple(depth.shape[:2]):
            H, W = depth.shape[:2]
            cache = _make_backproject_cache(
                Ks[cam], cam_to_world[cam], H, W, stride=args.sdf_pixel_stride
            )

        pts = _backproject_depth_world_fast(
            depth, cache, max_depth=args.sdf_max_depth, mask=None
        )
        if len(pts):
            in_box = np.all(
                (pts >= bbox_min[None]) & (pts <= bbox_max[None]), axis=1
            )
            pts = pts[in_box]
        return cam, pts, False

    def cache_is_fresh(cache_path, fid):
        """A cached point set is only valid if it is newer than the depth images and the
        object pose it was cropped around; frame ids get reused after a re-extraction."""
        try:
            cache_mtime = os.path.getmtime(cache_path)
        except OSError:
            return False
        inputs = [find_existing_file(os.path.join(depth_root, cam), fid, DEPTH_EXTS) for cam in cams]
        pose_dir = getattr(args, 'ob_in_world_dir', None)
        if pose_dir:
            inputs.append(find_existing_file(pose_dir, fid, ('.txt',)))
        for path in inputs:
            if path is not None and os.path.getmtime(path) > cache_mtime:
                return False
        return True

    stale_cache = 0
    out = {}
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        for fi, fid in enumerate(frame_ids):
            cache_path = os.path.join(cache_dir, f'{fid}.npy')

            use_cached = reuse_cache and os.path.isfile(cache_path)
            if use_cached and not cache_is_fresh(cache_path, fid):
                use_cached = False
                stale_cache += 1
                if stale_cache == 1:
                    logging.warning(f'[SDF-cache] {cache_path} is older than its inputs; recomputing stale entries')

            if use_cached:
                pts = np.asarray(np.load(cache_path), dtype=np.float32).reshape(-1, 3)
                out[fid] = pts
                if fi % 20 == 0 or fi + 1 == len(frame_ids):
                    logging.info(
                        f'[SDF-cache] loaded frame {fi+1}/{len(frame_ids)} ({fid}): '
                        f'{len(pts)} points'
                    )
                continue

            combined, combined_faces_np, _ = _combined_numpy_geometry_for_frame(
                fi, object_mesh_vertices, object_faces, object_rvec_np, object_t_np,
                hand_states, hand_builders,
            )
            bbox_min = combined.min(axis=0) - float(args.sdf_bbox_margin)
            bbox_max = combined.max(axis=0) + float(args.sdf_bbox_margin)

            futures = [
                pool.submit(process_one_camera, cam, fid, bbox_min, bbox_max)
                for cam in cams
            ]

            pts_all = []
            missing_depth = []
            per_cam_counts = {}
            for fut in futures:
                cam, pts_cam, missing = fut.result()
                if missing:
                    missing_depth.append(cam)
                    per_cam_counts[cam] = 0
                    continue
                ncam = 0 if pts_cam is None else int(len(pts_cam))
                per_cam_counts[cam] = ncam
                if ncam:
                    pts_all.append(pts_cam)

            if missing_depth and bool(args.sdf_require_all_cams):
                raise FileNotFoundError(
                    f'[SDF] frame {fid}: missing depth image for '
                    f'{len(missing_depth)} camera(s): {missing_depth}'
                )

            pts = (
                np.concatenate(pts_all, axis=0).astype(np.float32, copy=False)
                if pts_all else np.zeros((0, 3), dtype=np.float32)
            )

            if len(pts) == 0:
                out[fid] = pts
                if reuse_cache:
                    np.save(cache_path, pts)
                continue

            # Keep exact HO-Cap MeshSDF observation selection unchanged.
            pts = _exact_meshsdf_filter_points(
                combined, combined_faces_np, pts, exact_sdf, device, args.sdf_dist_thresh
            )
            pts = _voxel_downsample_np(pts, args.sdf_voxel_size)
            pts = _sample_rows(
                pts, int(args.sdf_max_points_per_frame), seed=fi + 1234
            ).astype(np.float32)
            out[fid] = pts

            if reuse_cache:
                tmp_path = cache_path + '.tmp.npy'
                np.save(tmp_path, pts)
                os.replace(tmp_path, cache_path)

            if fi % 20 == 0 or fi + 1 == len(frame_ids):
                nonzero_cams = sum(v > 0 for v in per_cam_counts.values())
                logging.info(
                    f'[SDF-fast] prepared frame {fi+1}/{len(frame_ids)} ({fid}): '
                    f'{len(pts)} MeshSDF-filtered points from '
                    f'{nonzero_cams}/{len(cams)} depth cameras'
                )

    return out


def _mesh_sdf_exact(verts, faces, dpts, exact_loss):
    """Exact HO-Cap MeshSDF loss. No proxy/fallback path exists."""
    if dpts.shape[0] == 0:
        return verts.new_zeros(())
    loss, _, _ = exact_loss(
        verts.contiguous(), faces.contiguous(), dpts.contiguous()
    )
    # Match tools/07_joint_pose_solver.py: loss *= 1e3 after MeshSDFLoss.
    return loss * 1e3


def run_joint_pose_solver_only(pose_files, hand_json_dirs, object_mesh, args):
    """Adapt the uploaded 07_joint_pose_solver idea to the user's current file layout.

    The objective contains ONLY:
        L = w_sdf * L_sdf
          + w_reg_m * L_reg_hand
          + w_reg_o * L_reg_object
          + w_smooth_m * L_smooth_hand
          + w_smooth_o * L_smooth_object

    There is deliberately no contact, collision, penetration, snap, push/pull, or polish.
    """
    device = torch.device(args.device)
    frame_ids = [Path(p).stem for p in pose_files]
    n = len(frame_ids)

    raw_obj = np.stack([read_pose_txt(p) for p in pose_files], axis=0)
    obj_r0_np = np.stack([matrix_to_rodrigues(T[:3, :3]) for T in raw_obj], axis=0).astype(np.float32)
    obj_t0_np = raw_obj[:, :3, 3].astype(np.float32)
    obj_r0 = torch.tensor(obj_r0_np, device=device)
    obj_t0 = torch.tensor(obj_t0_np, device=device)
    obj_r = torch.nn.Parameter(obj_r0.clone())
    obj_t = torch.nn.Parameter(obj_t0.clone())

    hand_builders = {}
    hand_states = {}
    for side in ('left', 'right'):
        folder = hand_json_dirs.get(side)
        if not folder:
            continue
        builder = ManoHandBuilder(
            args.mano_dir, is_rhand=(side == 'right'), device=device,
            num_pca_comps=args.num_pca_comps, flat_hand_mean=bool(args.flat_hand_mean),
        )
        records = []
        for fi, fid in enumerate(frame_ids):
            jp = find_existing_file(folder, fid, JSON_EXTS)
            if jp is None:
                continue
            rec = load_hand_json(jp)
            pose23, betas, Rh, Th = prepare_pose23_and_betas10(rec, args.num_pca_comps)
            records.append((fi, fid, jp, pose23, betas, Rh, Th))
        if not records:
            continue
        pose_np = np.stack([x[3] for x in records]).astype(np.float32)
        betas_np = np.stack([x[4] for x in records]).astype(np.float32)
        Rh_np = np.stack([x[5] for x in records]).astype(np.float32)
        Th_np = np.stack([x[6] for x in records]).astype(np.float32)
        state = {
            'records': records,
            'indices': [x[0] for x in records],
            'frame_to_local': {x[0]: i for i, x in enumerate(records)},
            'pose0': torch.tensor(pose_np, device=device),
            'Rh0': torch.tensor(Rh_np, device=device),
            'Th0': torch.tensor(Th_np, device=device),
            'pose': torch.nn.Parameter(torch.tensor(pose_np, device=device)),
            'Rh': torch.nn.Parameter(torch.tensor(Rh_np, device=device)),
            'Th': torch.nn.Parameter(torch.tensor(Th_np, device=device)),
            'betas': betas_np,
        }
        hand_builders[side] = builder
        hand_states[side] = state

    obj_vertices_np = np.asarray(object_mesh.vertices, dtype=np.float32)
    obj_faces_np = np.asarray(object_mesh.faces, dtype=np.int64)
    obj_vertices = torch.tensor(obj_vertices_np, dtype=torch.float32, device=device)

    # One GLOBAL object scale shared by the whole sequence. Scaling is around the
    # local mesh centroid so the object's world-space center is not shifted merely
    # because its size changes. log-scale keeps the value strictly positive.
    object_scale_enable = bool(int(args.object_scale_enable))
    obj_scale_center = obj_vertices.mean(dim=0, keepdim=True)
    if object_scale_enable:
        obj_log_scale = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))
    else:
        obj_log_scale = torch.zeros((), dtype=torch.float32, device=device)

    params = [obj_r, obj_t]
    for st in hand_states.values():
        params.extend([st['pose'], st['Rh'], st['Th']])

    if object_scale_enable:
        optimizer = torch.optim.Adam([
            {'params': params, 'lr': float(args.joint_lr)},
            {'params': [obj_log_scale], 'lr': float(args.object_scale_lr)},
        ])
        params_for_clip = params + [obj_log_scale]
    else:
        optimizer = torch.optim.Adam(params, lr=float(args.joint_lr))
        params_for_clip = params

    # Build a single combined face array: object first, then every present hand.
    faces_list = [obj_faces_np]
    offset = len(obj_vertices_np)
    for side in ('left', 'right'):
        if side in hand_builders:
            faces_list.append(hand_builders[side].faces + offset)
            offset += len(hand_builders[side].model.v_template)
    combined_faces_np = np.concatenate(faces_list, axis=0).astype(np.int64)
    combined_faces = torch.tensor(combined_faces_np, dtype=torch.long, device=device)

    # Strict requirement: the exact HO-Cap CUDA MeshSDF loss must be available.
    exact_sdf = _require_hocap_mesh_sdf(device)

    dpts_by_frame = None
    log_rows = []
    temporal_scale = float(args.sequence_fps) / max(float(args.temporal_reference_fps), 1e-6)

    def current_object_scale():
        if object_scale_enable:
            return torch.exp(obj_log_scale)
        return obj_vertices.new_tensor(1.0)

    def scaled_object_vertices_local():
        scale = current_object_scale()
        return obj_scale_center + scale * (obj_vertices - obj_scale_center)

    def current_object_vertices(frame_indices):
        rr = obj_r[frame_indices]
        tt = obj_t[frame_indices]
        R = torch_batch_rodrigues(rr)
        local = scaled_object_vertices_local()
        return torch.einsum('vj,fij->fvi', local, R) + tt[:, None, :]

    def current_hand_vertices_for_frame(side, fi):
        st = hand_states[side]
        li = st['frame_to_local'].get(fi)
        if li is None:
            return None
        p = st['pose'][li:li+1]
        b = torch.tensor(st['betas'][li:li+1], dtype=torch.float32, device=device)
        Rh = st['Rh'][li:li+1]
        Th = st['Th'][li:li+1]
        vl = hand_builders[side].forward_local(p, b)
        R = torch_batch_rodrigues(Rh)
        vw = torch.einsum('fvi,fji->fvj', vl, R) + Th[:, None, :]
        return vw[0]

    total_steps = int(args.joint_total_steps)
    sdf_start = max(0, total_steps - int(args.joint_sdf_steps))
    for step in range(total_steps):
        optimizer.zero_grad(set_to_none=True)

        loss_reg_m = obj_r.new_zeros(())
        loss_smooth_m = obj_r.new_zeros(())
        for side, st in hand_states.items():
            reg = (
                float(args.hand_pose_reg_scale) * torch.mean((st['pose'] - st['pose0']) ** 2)
                + float(args.hand_rh_reg_scale) * torch.mean((st['Rh'] - st['Rh0']) ** 2)
                + float(args.hand_th_reg_scale) * torch.mean((st['Th'] - st['Th0']) ** 2)
            )
            loss_reg_m = loss_reg_m + reg

            # Do not smooth across missing-frame gaps.
            for seg_global in _contiguous_segments(st['indices']):
                loc = [st['frame_to_local'][i] for i in seg_global]
                loc_t = torch.tensor(loc, dtype=torch.long, device=device)
                pv, pa = _seq_vel_acc(st['pose'][loc_t], temporal_scale)
                rv, ra = _rot_vel_acc(st['Rh'][loc_t], temporal_scale)
                tv, ta = _seq_vel_acc(st['Th'][loc_t], temporal_scale)
                loss_smooth_m = loss_smooth_m + (
                    float(args.hand_pose_vel_scale) * pv
                    + float(args.hand_pose_acc_scale) * pa
                    + float(args.hand_rh_vel_scale) * rv
                    + float(args.hand_rh_acc_scale) * ra
                    + float(args.hand_th_vel_scale) * tv
                    + float(args.hand_th_acc_scale) * ta
                )

        # Object alignment and temporal smoothness.
        R_cur = torch_batch_rodrigues(obj_r)
        R0 = torch_batch_rodrigues(obj_r0)
        loss_reg_o = (
            float(args.object_rot_reg_scale) * torch.mean((R_cur - R0) ** 2)
            + float(args.object_th_reg_scale) * torch.mean((obj_t - obj_t0) ** 2)
        )
        if object_scale_enable:
            # Keep this inside the existing object regularization group, preserving
            # the five-group JointPoseSolver objective while allowing depth-driven scale.
            loss_reg_o = loss_reg_o + float(args.object_scale_reg_scale) * (current_object_scale() - 1.0) ** 2
        orv, ora = _rot_vel_acc(obj_r, temporal_scale)
        otv, ota = _seq_vel_acc(obj_t, temporal_scale)
        loss_smooth_o = (
            float(args.object_rot_vel_scale) * orv
            + float(args.object_rot_acc_scale) * ora
            + float(args.object_th_vel_scale) * otv
            + float(args.object_th_acc_scale) * ota
        )

        loss_sdf = obj_r.new_zeros(())
        if float(args.w_sdf) > 0 and step >= sdf_start:
            if dpts_by_frame is None:
                # Match the original solver: prepare observation points only when SDF phase begins,
                # using the then-current geometry rather than the raw input geometry.
                cur_r_np = obj_r.detach().cpu().numpy()
                cur_t_np = obj_t.detach().cpu().numpy()
                hand_states_np = {}
                for side, st in hand_states.items():
                    hand_states_np[side] = {
                        'frame_to_local': st['frame_to_local'],
                        'pose23': st['pose'].detach().cpu().numpy(),
                        'betas': st['betas'],
                        'Rh': st['Rh'].detach().cpu().numpy(),
                        'Th': st['Th'].detach().cpu().numpy(),
                    }
                dpts_by_frame = _prepare_depth_points_for_sdf(
                    frame_ids, obj_vertices_np, obj_faces_np, cur_r_np, cur_t_np,
                    hand_states_np, hand_builders, args.test_scene_dir, args,
                    exact_sdf=exact_sdf, device=device,
                )

            # Cycle through a small frame batch each step to keep exact MeshSDF practical.
            batch_n = max(1, min(int(args.sdf_frame_batch), n))
            start = ((step - sdf_start) * batch_n) % n
            frame_batch = [(start + j) % n for j in range(batch_n)]
            sdf_losses = []
            obj_batch = current_object_vertices(frame_batch)
            for bj, fi in enumerate(frame_batch):
                dnp = dpts_by_frame.get(frame_ids[fi], np.zeros((0, 3), np.float32))
                if len(dnp) < int(args.sdf_min_points):
                    continue
                verts_parts = [obj_batch[bj]]
                for side in ('left', 'right'):
                    if side in hand_states:
                        hv = current_hand_vertices_for_frame(side, fi)
                        if hv is not None:
                            verts_parts.append(hv)
                # Build valid face offsets for exactly the hands present in this frame.
                face_parts = [obj_faces_np]
                face_offset = len(obj_vertices_np)
                for side in ('left', 'right'):
                    if side not in hand_states:
                        continue
                    hv = current_hand_vertices_for_frame(side, fi)
                    if hv is None:
                        continue
                    face_parts.append(np.asarray(hand_builders[side].faces, dtype=np.int64) + face_offset)
                    face_offset += int(hv.shape[0])
                frame_faces = torch.as_tensor(
                    np.concatenate(face_parts, axis=0), dtype=torch.long, device=device
                )
                verts = torch.cat(verts_parts, dim=0)
                dpts = torch.as_tensor(dnp, dtype=torch.float32, device=device)
                sdf_losses.append(_mesh_sdf_exact(verts, frame_faces, dpts, exact_sdf))
            if sdf_losses:
                loss_sdf = torch.stack(sdf_losses).mean()

        loss = (
            float(args.w_sdf) * loss_sdf
            + float(args.w_reg_m) * loss_reg_m
            + float(args.w_reg_o) * loss_reg_o
            + float(args.w_smooth_m) * loss_smooth_m
            + float(args.w_smooth_o) * loss_smooth_o
        )
        loss.backward()
        if float(args.joint_grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(params_for_clip, float(args.joint_grad_clip_norm))
        optimizer.step()
        if object_scale_enable:
            with torch.no_grad():
                lo = math.log(max(float(args.object_scale_min), 1e-6))
                hi = math.log(max(float(args.object_scale_max), float(args.object_scale_min) + 1e-6))
                obj_log_scale.clamp_(lo, hi)
        if float(args.mano_pca_abs_limit) > 0:
            with torch.no_grad():
                for st in hand_states.values():
                    st['pose'][:, 3:] = torch.clamp(
                        st['pose'][:, 3:], -float(args.mano_pca_abs_limit), float(args.mano_pca_abs_limit)
                    )

        row = dict(
            step=step + 1,
            total=float(loss.detach().cpu()),
            sdf=float(loss_sdf.detach().cpu()),
            reg_m=float(loss_reg_m.detach().cpu()),
            smooth_m=float(loss_smooth_m.detach().cpu()),
            reg_o=float(loss_reg_o.detach().cpu()),
            smooth_o=float(loss_smooth_o.detach().cpu()),
            object_scale=float(current_object_scale().detach().cpu()),
        )
        log_rows.append(row)
        if step == 0 or (step + 1) % int(args.joint_log_every) == 0 or step + 1 == total_steps:
            logging.info(
                f'[JointPoseSolver] step={step+1:04d}/{total_steps:04d} '
                f'loss={row["total"]:.6e} sdf={row["sdf"]:.6e} '
                f'reg_m={row["reg_m"]:.6e} smooth_m={row["smooth_m"]:.6e} '
                f'reg_o={row["reg_o"]:.6e} smooth_o={row["smooth_o"]:.6e} '
                f'obj_scale={row["object_scale"]:.6f}'
            )

    # -------------------------------------------------------------------------
    # Global object-scale fine-tune using ALL frames' fused 31-camera depth data.
    # dpts_by_frame already comes from every frame's 31 depth maps in world coords.
    # We first keep only object-near observations (to avoid hand points driving scale),
    # then run complete all-frame sweeps with object pose and hands frozen.
    # -------------------------------------------------------------------------
    scale_rows = []
    if (
        object_scale_enable
        and dpts_by_frame is not None
        and int(args.object_scale_epochs) > 0
    ):
        logging.info(
            f'[ObjectScale] begin all-frame fine-tune: frames={n}, '
            f'epochs={int(args.object_scale_epochs)}, '
            f'frame_batch={int(args.object_scale_frame_batch)}'
        )

        # Fixed optimized object poses for the scale-only pass.
        with torch.no_grad():
            fixed_R = torch_batch_rodrigues(obj_r.detach()).detach()
            fixed_t = obj_t.detach().clone()
            scale_at_selection = float(current_object_scale().detach().cpu())
            local_at_selection = (
                obj_scale_center
                + current_object_scale().detach() * (obj_vertices - obj_scale_center)
            )

        # Build object-only observation subsets from the already fused all-camera
        # observations. This removes points that belonged to either MANO hand.
        scale_dpts_by_frame = {}
        object_faces_t = torch.as_tensor(obj_faces_np, dtype=torch.long, device=device)
        total_selected = 0
        frames_selected = 0
        select_thresh = float(args.object_scale_select_thresh)

        for fi, fid in enumerate(frame_ids):
            dnp = dpts_by_frame.get(fid, np.zeros((0, 3), np.float32))
            if len(dnp) < int(args.object_scale_min_points):
                continue
            with torch.no_grad():
                vw = torch.einsum('vj,ij->vi', local_at_selection, fixed_R[fi]) + fixed_t[fi][None, :]
                _, dist, _ = exact_sdf(
                    vw.contiguous(),
                    object_faces_t.contiguous(),
                    torch.as_tensor(dnp, dtype=torch.float32, device=device).contiguous(),
                )
                keep = (dist < select_thresh).detach().cpu().numpy().reshape(-1)
            selected = np.asarray(dnp, dtype=np.float32)[keep]
            if len(selected) >= int(args.object_scale_min_points):
                scale_dpts_by_frame[fid] = selected
                total_selected += int(len(selected))
                frames_selected += 1

        logging.info(
            f'[ObjectScale] object-only depth observations: '
            f'{frames_selected}/{n} frames, {total_selected} points, '
            f'initial_scale={scale_at_selection:.6f}, '
            f'select_thresh={select_thresh:.4f} m'
        )

        if frames_selected > 0:
            scale_optimizer = torch.optim.Adam(
                [obj_log_scale], lr=float(args.object_scale_lr)
            )
            frame_batch_n = max(1, int(args.object_scale_frame_batch))
            min_pts = int(args.object_scale_min_points)
            scale_step = 0

            # Every epoch iterates the complete frame list: all available frames
            # contribute to the global sequence-level scale estimate.
            for epoch in range(int(args.object_scale_epochs)):
                epoch_losses = []
                used_frames_epoch = 0
                for start_i in range(0, n, frame_batch_n):
                    inds = list(range(start_i, min(start_i + frame_batch_n, n)))
                    scale_optimizer.zero_grad(set_to_none=True)
                    local_scaled = scaled_object_vertices_local()
                    batch_losses = []

                    for fi in inds:
                        fid = frame_ids[fi]
                        dnp = scale_dpts_by_frame.get(fid)
                        if dnp is None or len(dnp) < min_pts:
                            continue
                        # Object pose is fixed; only the single global scale receives gradients.
                        vw = torch.einsum('vj,ij->vi', local_scaled, fixed_R[fi]) + fixed_t[fi][None, :]
                        dpts = torch.as_tensor(dnp, dtype=torch.float32, device=device)
                        batch_losses.append(
                            _mesh_sdf_exact(vw, object_faces_t, dpts, exact_sdf)
                        )
                        used_frames_epoch += 1

                    if not batch_losses:
                        continue

                    scale_sdf = torch.stack(batch_losses).mean()
                    scale_reg = float(args.object_scale_reg_scale) * (current_object_scale() - 1.0) ** 2
                    scale_loss = float(args.w_sdf) * scale_sdf + float(args.w_reg_o) * scale_reg
                    scale_loss.backward()
                    if float(args.joint_grad_clip_norm) > 0:
                        torch.nn.utils.clip_grad_norm_([obj_log_scale], float(args.joint_grad_clip_norm))
                    scale_optimizer.step()
                    with torch.no_grad():
                        lo = math.log(max(float(args.object_scale_min), 1e-6))
                        hi = math.log(max(float(args.object_scale_max), float(args.object_scale_min) + 1e-6))
                        obj_log_scale.clamp_(lo, hi)

                    scale_step += 1
                    row_scale = {
                        'step': scale_step,
                        'epoch': epoch + 1,
                        'start_frame_index': start_i,
                        'scale_sdf': float(scale_sdf.detach().cpu()),
                        'scale_reg': float(scale_reg.detach().cpu()),
                        'loss': float(scale_loss.detach().cpu()),
                        'object_scale': float(current_object_scale().detach().cpu()),
                    }
                    scale_rows.append(row_scale)
                    epoch_losses.append(row_scale['loss'])

                logging.info(
                    f'[ObjectScale] epoch={epoch+1}/{int(args.object_scale_epochs)} '
                    f'used_frames={used_frames_epoch}/{n} '
                    f'scale={float(current_object_scale().detach().cpu()):.6f} '
                    f'mean_loss={(float(np.mean(epoch_losses)) if epoch_losses else float("nan")):.6e}'
                )

            logging.info(
                f'[ObjectScale] final global scale='
                f'{float(current_object_scale().detach().cpu()):.6f}'
            )
        else:
            logging.warning('[ObjectScale] no object-only depth observations passed the selection threshold; keeping scale=1/current.')

    # Save optimized object poses to the LEGACY path expected by downstream code.
    # The optimization method is JointPoseSolver-only; the directory name is kept
    # solely for backward compatibility with the existing exporter / WebUI.
    compat_obj_dir = os.path.join(args.output_dir, 'smoothed_ob_in_world')
    # Rewritten in full: leftovers from a run with a different frame set must not survive.
    if os.path.isdir(compat_obj_dir):
        shutil.rmtree(compat_obj_dir)
    ensure_dir(compat_obj_dir)
    obj_r_np = obj_r.detach().cpu().numpy()
    obj_t_np = obj_t.detach().cpu().numpy()
    object_pose_by_frame = {}
    for i, fid in enumerate(frame_ids):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rodrigues_to_matrix(obj_r_np[i])
        T[:3, 3] = obj_t_np[i]
        object_pose_by_frame[fid] = T
        np.savetxt(os.path.join(compat_obj_dir, f'{fid}.txt'), T, fmt='%.9f')

    # Save legal MANO parameters, with shapes fixed exactly.
    final_hand_dirs = dict(hand_json_dirs)
    for side, st in hand_states.items():
        out_dir = os.path.join(args.output_dir, 'contact_mano', side, 'smpl')
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        ensure_dir(out_dir)
        pnp = st['pose'].detach().cpu().numpy()
        rnp = st['Rh'].detach().cpu().numpy()
        tnp = st['Th'].detach().cpu().numpy()
        for li, rec_tuple in enumerate(st['records']):
            fi, fid, src_json, _, betas, _, _ = rec_tuple
            p48 = hand_builders[side].restore_full_pose_48(pnp[li])
            save_joint_mano_json(
                src_json,
                os.path.join(out_dir, f'{fid}.json'),
                pnp[li], p48, betas, rnp[li], tnp[li],
                save_format=args.mano_save_pose_format,
            )
        final_hand_dirs[side] = out_dir

    # Save the globally scaled object mesh for downstream export.
    final_scale = float(current_object_scale().detach().cpu())
    center_np = obj_vertices_np.mean(axis=0, keepdims=True)
    scaled_vertices_np = center_np + final_scale * (obj_vertices_np - center_np)
    object_mesh.vertices = scaled_vertices_np
    scaled_mesh_path = os.path.join(args.output_dir, 'object_mesh_scaled.ply')
    object_mesh.export(scaled_mesh_path)
    with open(os.path.join(args.output_dir, 'object_scale.txt'), 'w') as f:
        f.write(f'{final_scale:.9f}\n')
    logging.info(f'[ObjectScale] saved scaled mesh: {scaled_mesh_path} (scale={final_scale:.6f})')

    log_path = os.path.join(args.output_dir, 'joint_pose_solver_loss.csv')
    with open(log_path, 'w', newline='') as f:
        w = csv.DictWriter(
            f,
            fieldnames=['step', 'total', 'sdf', 'reg_m', 'smooth_m', 'reg_o', 'smooth_o', 'object_scale']
        )
        w.writeheader()
        w.writerows(log_rows)
    logging.info(f'[JointPoseSolver] saved loss log: {log_path}')

    if scale_rows:
        scale_log_path = os.path.join(args.output_dir, 'object_scale_finetune.csv')
        with open(scale_log_path, 'w', newline='') as f:
            w = csv.DictWriter(
                f,
                fieldnames=['step', 'epoch', 'start_frame_index', 'scale_sdf', 'scale_reg', 'loss', 'object_scale']
            )
            w.writeheader()
            w.writerows(scale_rows)
        logging.info(f'[ObjectScale] saved fine-tune log: {scale_log_path}')

    return object_pose_by_frame, final_hand_dirs


def colorize_mesh(mesh, rgba):
    m = mesh.copy()
    m.visual.vertex_colors = np.tile(np.asarray(rgba, dtype=np.uint8).reshape(1, 4), (len(m.vertices), 1))
    return m


def cv_pose_to_gl_pose(T):
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    return cv_to_gl @ np.asarray(T, dtype=np.float64)


class CombinedRenderer:
    def __init__(self, object_mesh):
        self.object_mesh = colorize_mesh(object_mesh, [20, 20, 20, 255])

    def render(self, object_pose_cv, hand_meshes_local, hand_poses_cv, K, image_size, bg_rgb, alpha=0.55, white_background=False):
        W, H = image_size
        scene = pyrender.Scene(bg_color=[255, 255, 255, 0] if white_background else [0, 0, 0, 0])
        scene.add(pyrender.Mesh.from_trimesh(self.object_mesh, smooth=False), pose=cv_pose_to_gl_pose(object_pose_cv))
        for mesh, pose in zip(hand_meshes_local, hand_poses_cv):
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=cv_pose_to_gl_pose(pose))
        cam = pyrender.IntrinsicsCamera(K[0,0], K[1,1], K[0,2], K[1,2], znear=0.01, zfar=20.0)
        scene.add(cam, pose=np.eye(4))
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
        scene.add(light, pose=np.eye(4))
        renderer = pyrender.OffscreenRenderer(W, H)
        try:
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        finally:
            renderer.delete()
        mask = depth > 0
        render_rgb = color[..., :3]
        if white_background:
            overlay = np.full_like(render_rgb, 255)
        else:
            overlay = np.asarray(bg_rgb, dtype=np.uint8).copy()
        a = float(alpha)
        overlay[mask] = np.clip((1-a) * overlay[mask].astype(np.float32) + a * render_rgb[mask].astype(np.float32), 0, 255).astype(np.uint8)
        return overlay, render_rgb, mask.astype(np.uint8) * 255, depth


def make_grid(images, ncols=0, cell_width=360, pad=4):
    if not images:
        return None
    resized = []
    for img in images:
        h, w = img.shape[:2]
        nw = int(cell_width)
        nh = max(1, int(round(h * nw / max(w, 1))))
        resized.append(cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA))
    n = len(resized)
    cols = int(ncols) if int(ncols) > 0 else int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    ch = max(x.shape[0] for x in resized)
    canvas = np.zeros((rows * ch + (rows-1)*pad, cols*cell_width + (cols-1)*pad, 3), np.uint8)
    for i, img in enumerate(resized):
        r, c = divmod(i, cols)
        y, x = r*(ch+pad), c*(cell_width+pad)
        canvas[y:y+img.shape[0], x:x+img.shape[1]] = img
    return canvas


def render_frames(pose_files, object_pose_by_frame, hand_json_dirs, object_mesh, args):
    images_root = os.path.join(args.test_scene_dir, 'images')
    cam_ext = readextfromyml(args.test_scene_dir)
    cams = args.cams if args.cams else list_camera_ids(images_root)
    # Same per-camera intrinsics the depth fusion uses (intri.yml, cam_K.txt fallback).
    Ks = read_intrinsics_per_camera(args.test_scene_dir, cams, downscale=args.downscale)
    renderer = CombinedRenderer(object_mesh)
    builders = {
        'left': ManoHandBuilder(args.mano_dir, False, args.device, args.num_pca_comps, bool(args.flat_hand_mean)),
        'right': ManoHandBuilder(args.mano_dir, True, args.device, args.num_pca_comps, bool(args.flat_hand_mean)),
    }
    colors = {'left': [220, 0, 0, 255], 'right': [0, 90, 220, 255]}
    by_cam = os.path.join(args.output_dir, 'by_cam')
    grid_dir = os.path.join(args.output_dir, 'grid')
    ensure_dir(by_cam)
    if not args.no_grid:
        ensure_dir(grid_dir)
    for cam in cams:
        ensure_dir(os.path.join(by_cam, str(cam), 'overlay'))
        if args.save_render:
            ensure_dir(os.path.join(by_cam, str(cam), 'render'))
        if args.save_mask:
            ensure_dir(os.path.join(by_cam, str(cam), 'mask'))
        if args.save_depth:
            ensure_dir(os.path.join(by_cam, str(cam), 'depth'))

    for pp in pose_files:
        fid = Path(pp).stem
        ob_world = object_pose_by_frame.get(fid, read_pose_txt(pp))
        hand_meshes, hand_worlds = [], []
        for side in ('left', 'right'):
            folder = hand_json_dirs.get(side)
            if not folder:
                continue
            jp = find_existing_file(folder, fid, JSON_EXTS)
            if not jp:
                continue
            rec = load_hand_json(jp)
            pose23, betas, Rh, Th = prepare_pose23_and_betas10(rec, args.num_pca_comps)
            with torch.no_grad():
                vl = builders[side].forward_local(pose23, betas)[0].detach().cpu().numpy()
            mesh = trimesh.Trimesh(vertices=vl, faces=builders[side].faces, process=False)
            hand_meshes.append(colorize_mesh(mesh, colors[side]))
            hand_worlds.append(build_world_pose_from_Rh_Th(Rh, Th))

        cells = []
        for cam in cams:
            cam = str(cam)
            if cam not in cam_ext:
                continue
            ip = find_existing_file(os.path.join(images_root, cam), fid, IMAGE_EXTS)
            if not ip:
                continue
            rgb = imageio.imread(ip)[..., :3]
            H, W = rgb.shape[:2]
            ext = cam_ext[cam]['ext']
            ob_cam = ext @ ob_world
            hand_cam = [ext @ T for T in hand_worlds]
            overlay, raw_render, mask, depth = renderer.render(
                ob_cam, hand_meshes, hand_cam, Ks[cam], (W, H), rgb,
                alpha=args.alpha, white_background=bool(args.white_background),
            )
            imageio.imwrite(os.path.join(by_cam, cam, 'overlay', f'{fid}.png'), overlay)
            if args.save_render:
                imageio.imwrite(os.path.join(by_cam, cam, 'render', f'{fid}.png'), raw_render)
            if args.save_mask:
                imageio.imwrite(os.path.join(by_cam, cam, 'mask', f'{fid}.png'), mask)
            if args.save_depth:
                np.save(os.path.join(by_cam, cam, 'depth', f'{fid}.npy'), depth.astype(np.float32))
            if not args.no_grid:
                cv2.putText(overlay, f'cam {cam}', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
                cells.append(overlay)
        if not args.no_grid and cells:
            grid = make_grid(cells, args.grid_cols, args.grid_cell_width, args.grid_pad)
            imageio.imwrite(os.path.join(grid_dir, f'{fid}.jpg'), grid)


def assemble_grid_video(output_dir, fps):
    files = sorted(glob.glob(os.path.join(output_dir, 'grid', '*.jpg')), key=natural_sort_key)
    if not files:
        return
    # The grid frames are the product; the mp4 is a convenience and must not fail the
    # run when imageio's video backend is missing or incompatible (the WebUI encodes
    # its own video from grid/*.jpg with ffmpeg).
    try:
        writer = imageio.get_writer(os.path.join(output_dir, 'grid.mp4'), fps=int(fps), codec='libx264', quality=8)
        try:
            for p in files:
                writer.append_data(imageio.imread(p))
        finally:
            writer.close()
    except Exception as exc:
        logging.warning(f'[grid] skipping grid.mp4 (imageio video writer failed): {exc}')


def parse_gpu_ids(s):
    out = []
    for x in str(s).split(','):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def run_multi_gpu_render(args, joint_obj_dir, joint_hand_dirs):
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        return False
    pose_files = sorted(glob.glob(os.path.join(joint_obj_dir, '*.txt')), key=natural_sort_key)
    n = len(pose_files)
    workers = min(len(gpu_ids), int(args.num_gpu_workers), n)
    if workers <= 1:
        return False
    chunks = []
    base = n // workers
    rem = n % workers
    s = 0
    for i in range(workers):
        e = s + base + (1 if i < rem else 0)
        chunks.append((s, e))
        s = e
    procs = []
    for rank, ((st, ed), gid) in enumerate(zip(chunks, gpu_ids[:workers])):
        cmd = [
            sys.executable, os.path.abspath(__file__),
            '--mesh_file', args.mesh_file,
            '--test_scene_dir', args.test_scene_dir,
            '--ob_in_world_dir', joint_obj_dir,
            '--mano_dir', args.mano_dir,
            '--output_dir', args.output_dir,
            '--start_frame', str(st), '--end_frame', str(ed),
            '--device', 'cuda', '--multi_gpu', '0', '--worker_mode',
            '--joint_solver_enable', '0',
            '--grid_video_fps', str(args.grid_video_fps),
            '--downscale', str(args.downscale),
            '--alpha', str(args.alpha),
            '--white_background', str(args.white_background),
            '--num_pca_comps', str(args.num_pca_comps),
            '--flat_hand_mean', str(args.flat_hand_mean),
        ]
        if joint_hand_dirs.get('left'):
            cmd += ['--left_hand_json_dir', joint_hand_dirs['left']]
        if joint_hand_dirs.get('right'):
            cmd += ['--right_hand_json_dir', joint_hand_dirs['right']]
        if args.no_grid:
            cmd += ['--no_grid']
        if args.save_render:
            cmd += ['--save_render']
        if args.save_mask:
            cmd += ['--save_mask']
        if args.save_depth:
            cmd += ['--save_depth']
        if args.cams:
            cmd += ['--cams'] + list(args.cams)
        env = os.environ.copy()
        # gid is logical (an index into the parent's CUDA_VISIBLE_DEVICES); the child's
        # CUDA_VISIBLE_DEVICES is interpreted by the driver as physical ids, so map it back.
        parent_visible = [x.strip() for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x.strip()]
        physical = parent_visible[gid] if gid < len(parent_visible) else str(gid)
        env['CUDA_VISIBLE_DEVICES'] = physical
        env['EGL_DEVICE_ID'] = '0'
        logging.info(f'[render worker {rank}] logical GPU {gid} (physical {physical}), frames [{st},{ed})')
        procs.append(subprocess.Popen(cmd, env=env))
    rc = [p.wait() for p in procs]
    if any(x != 0 for x in rc):
        raise RuntimeError(f'Multi-GPU render worker failure: {rc}')
    if not args.no_grid:
        assemble_grid_video(args.output_dir, args.grid_video_fps)
    return True


def build_arg_parser():
    p = argparse.ArgumentParser(
        description='10fps JointPoseSolver-only hand+object optimization. No contact/collision/penetration optimization.'
    )
    p.add_argument('--mesh_file', required=True)
    p.add_argument('--test_scene_dir', required=True)
    p.add_argument('--ob_in_world_dir', required=True)
    p.add_argument('--left_hand_json_dir', default=None)
    p.add_argument('--right_hand_json_dir', default=None)
    p.add_argument('--hand_json_dir', default=None)
    p.add_argument('--mano_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--num_pca_comps', type=int, default=20)
    p.add_argument('--flat_hand_mean', type=int, default=1)
    p.add_argument('--mano_save_pose_format', choices=['full48', 'pca23'], default='full48')
    p.add_argument('--mano_pca_abs_limit', type=float, default=8.0)
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=-1)
    p.add_argument('--sequence_fps', type=float, default=10.0)
    p.add_argument('--temporal_reference_fps', type=float, default=30.0)

    # Exactly the five JointPoseSolver loss groups.
    p.add_argument('--joint_solver_enable', type=int, default=1)
    p.add_argument('--joint_lr', type=float, default=0.003)
    p.add_argument('--joint_total_steps', type=int, default=160)
    p.add_argument('--joint_sdf_steps', type=int, default=60)
    p.add_argument('--joint_log_every', type=int, default=10)
    p.add_argument('--joint_grad_clip_norm', type=float, default=1.0)
    p.add_argument('--w_sdf', type=float, default=1.0)
    p.add_argument('--w_reg_m', type=float, default=1.0)
    p.add_argument('--w_reg_o', type=float, default=1.0)
    p.add_argument('--w_smooth_m', type=float, default=1.0)
    p.add_argument('--w_smooth_o', type=float, default=1.0)

    # Internal component scales inside hand/object regularization/smoothness terms.
    p.add_argument('--hand_pose_reg_scale', type=float, default=1e-4)
    p.add_argument('--hand_rh_reg_scale', type=float, default=1e-4)
    p.add_argument('--hand_th_reg_scale', type=float, default=1e-2)
    p.add_argument('--hand_pose_vel_scale', type=float, default=5e-5)
    p.add_argument('--hand_pose_acc_scale', type=float, default=2e-4)
    p.add_argument('--hand_rh_vel_scale', type=float, default=1e-4)
    p.add_argument('--hand_rh_acc_scale', type=float, default=5e-4)
    p.add_argument('--hand_th_vel_scale', type=float, default=1.0)
    p.add_argument('--hand_th_acc_scale', type=float, default=4.0)
    p.add_argument('--object_rot_reg_scale', type=float, default=1e-3)
    p.add_argument('--object_th_reg_scale', type=float, default=1e-2)
    p.add_argument('--object_rot_vel_scale', type=float, default=1e-3)
    p.add_argument('--object_rot_acc_scale', type=float, default=5e-3)
    p.add_argument('--object_th_vel_scale', type=float, default=1.0)
    p.add_argument('--object_th_acc_scale', type=float, default=4.0)

    # Global sequence-level object scale, optimized from all-frame fused depth observations.
    p.add_argument('--object_scale_enable', type=int, default=1)
    p.add_argument('--object_scale_lr', type=float, default=5e-4)
    p.add_argument('--object_scale_min', type=float, default=0.5)
    p.add_argument('--object_scale_max', type=float, default=2.0)
    p.add_argument('--object_scale_reg_scale', type=float, default=1e-3)
    p.add_argument('--object_scale_epochs', type=int, default=2,
                   help='Complete all-frame scale-only sweeps after the main joint optimization.')
    p.add_argument('--object_scale_frame_batch', type=int, default=8)
    p.add_argument('--object_scale_select_thresh', type=float, default=0.015,
                   help='Meters: keep fused depth observations close to the object mesh, excluding hand points.')
    p.add_argument('--object_scale_min_points', type=int, default=50)

    # SDF observation preparation; these are observation controls, not contact controls.
    p.add_argument('--depth_root', default=None)
    p.add_argument('--depth_mask_root', default=None,
                   help='Optional foreground mask root. Leave unset to avoid assuming object-only masks cover hands.')
    p.add_argument('--depth_scale', type=float, default=-1.0,
                   help='>0 multiplies raw depth by this scale; <=0 auto-detects mm-like values.')
    p.add_argument('--sdf_cams', nargs='*', default=None,
                   help='Depth camera ids. Default is 1..sdf_expected_cam_count (31).')
    p.add_argument('--sdf_expected_cam_count', type=int, default=31)
    p.add_argument('--sdf_require_all_cams', type=int, default=1,
                   help='1: require all 31 camera extrinsics/directories and every frame depth; fail if any is missing.')
    p.add_argument('--sdf_cam_stride', type=int, default=1,
                   help='Camera subsampling stride. Keep 1 to use all 31 cameras.')
    p.add_argument('--sdf_pixel_stride', type=int, default=8)
    p.add_argument('--sdf_max_depth', type=float, default=5.0)
    p.add_argument('--sdf_bbox_margin', type=float, default=0.05)
    p.add_argument('--sdf_dist_thresh', type=float, default=0.01,
                   help='Threshold applied directly to the distance returned by HO-Cap MeshSDFLoss; HO-Cap config uses 0.01.')
    p.add_argument('--sdf_voxel_size', type=float, default=0.003)
    p.add_argument('--sdf_max_points_per_frame', type=int, default=2000)
    p.add_argument('--sdf_min_points', type=int, default=100)
    p.add_argument('--sdf_frame_batch', type=int, default=4)
    p.add_argument('--sdf_io_workers', type=int, default=8,
                   help='Parallel workers for 31-camera depth I/O/backprojection.')
    p.add_argument('--sdf_reuse_cache', type=int, default=1,
                   help='1: save/load final exact-MeshSDF observation cache.')
    p.add_argument('--sdf_cache_dir', default=None,
                   help='Optional exact-MeshSDF dpts cache directory.')

    # Rendering only.
    p.add_argument('--downscale', type=float, default=1.0)
    p.add_argument('--cams', nargs='*', default=None)
    p.add_argument('--alpha', type=float, default=0.55)
    p.add_argument('--white_background', type=int, default=1)
    p.add_argument('--save_render', action='store_true')
    p.add_argument('--save_mask', action='store_true')
    p.add_argument('--save_depth', action='store_true')
    p.add_argument('--no_grid', action='store_true')
    p.add_argument('--grid_cols', type=int, default=0)
    p.add_argument('--grid_cell_width', type=int, default=360)
    p.add_argument('--grid_pad', type=int, default=4)
    p.add_argument('--grid_video_fps', type=int, default=10)
    p.add_argument('--multi_gpu', type=int, default=1)
    p.add_argument('--gpu_ids', type=str, default='0,1,2,3,4,5,6,7')
    p.add_argument('--num_gpu_workers', type=int, default=8)
    p.add_argument('--worker_mode', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--skip_render', action='store_true',
                   help='Skip pyrender visualization after JointPoseSolver optimization.')
    return p


def main():
    args = build_arg_parser().parse_args()
    set_logging_format()
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    ensure_dir(args.output_dir)

    object_mesh = trimesh.load(args.mesh_file, process=False)
    if isinstance(object_mesh, trimesh.Scene):
        object_mesh = trimesh.util.concatenate(tuple(object_mesh.geometry.values()))
    pose_files = sorted(glob.glob(os.path.join(args.ob_in_world_dir, '*.txt')), key=natural_sort_key)
    if not pose_files:
        raise FileNotFoundError(f'No object poses in {args.ob_in_world_dir}')
    start = max(0, int(args.start_frame))
    end = len(pose_files) if int(args.end_frame) < 0 else min(len(pose_files), int(args.end_frame))
    pose_files = pose_files[start:end]
    hand_dirs = infer_hand_json_dirs(args)

    if bool(args.joint_solver_enable) and not args.worker_mode:
        if args.device != 'cuda':
            logging.warning('Joint solver is running on CPU; this will be slow.')
        object_pose_by_frame, hand_dirs = run_joint_pose_solver_only(pose_files, hand_dirs, object_mesh, args)
        compat_obj_dir = os.path.join(args.output_dir, 'smoothed_ob_in_world')

        # Solver outputs are already saved. Rendering is optional and must not
        # make a successful JointPoseSolver run fail because of EGL/OpenGL.
        if args.skip_render:
            logging.info(
                '[JointPoseSolver] optimization finished; skip_render=True, '
                'skipping pyrender visualization.'
            )
            logging.info(f'Done. JointPoseSolver-only output: {args.output_dir}')
            return

        if bool(args.multi_gpu) and args.device == 'cuda':
            if run_multi_gpu_render(args, compat_obj_dir, hand_dirs):
                logging.info(f'Done. JointPoseSolver-only output: {args.output_dir}')
                return
    else:
        object_pose_by_frame = {Path(p).stem: read_pose_txt(p) for p in pose_files}
        # Render-only mode (e.g. visualizing an already optimized outputmano/): use the
        # same multi-GPU worker split as the solver path when several GPUs are given.
        if (not args.worker_mode and bool(args.multi_gpu) and args.device == 'cuda'
                and run_multi_gpu_render(args, args.ob_in_world_dir, hand_dirs)):
            logging.info(f'Done. Render-only output: {args.output_dir}')
            return

    render_frames(pose_files, object_pose_by_frame, hand_dirs, object_mesh, args)
    if not args.no_grid and not args.worker_mode:
        assemble_grid_video(args.output_dir, args.grid_video_fps)
    logging.info(f'Done. Results: {args.output_dir}')


if __name__ == '__main__':
    main()
