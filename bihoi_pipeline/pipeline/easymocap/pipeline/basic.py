'''
  @ Date: 2021-04-13 20:43:16
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-14 15:32:24
  @ FilePath: /EasyMocap/easymocap/pipeline/basic.py

This version adds multi-stage-optimization-specific 6D Kalman filters on
optimized global RT parameters and internal pose parameters. It uses a layered
design for hand/body sequences:
    1) pose-joint Kalman: each internal joint uses [rx, ry, rz, vrx, vry, vrz]
    2) global RT Kalman: root/global motion uses [Tx, Ty, Tz, Rx, Ry, Rz, velocity]
The filters run after 3D/2D optimization, so optimized parameters are treated as
measurements. Defaults are tuned to reduce jitter without making hand motion too
laggy.
'''
import numpy as np

from ..pyfitting import optimizeShape, optimizePose2D, optimizePose3D
from ..mytools import Timer
from ..dataset import CONFIG
from .weight import load_weight_pose, load_weight_shape
from .config import Config


class KalmanFilterRT6D(object):
    """
    6D Kalman filter for optimized body/object global RT parameters.

    Measurement:
        z = [Tx, Ty, Tz, Rx, Ry, Rz]

    State:
        x = [Tx, Ty, Tz, Rx, Ry, Rz, vTx, vTy, vTz, vRx, vRy, vRz]

    This follows the same idea as FoundationPose-style KalmanFilter6D, but it is
    self-contained here so EasyMocap does not need an extra import path.
    """

    def __init__(
        self,
        measurement_noise_scale=5.0,
        process_noise_scale=1.0,
        init_cov_scale=1.0,
        dt=1.0,
    ):
        ndim = 6
        self.ndim = ndim
        self.dt = dt
        self.measurement_noise_scale = float(measurement_noise_scale)
        self.process_noise_scale = float(process_noise_scale)
        self.init_cov_scale = float(init_cov_scale)

        self._motion_mat = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float64)

        # Similar relative scale idea to kalman_filter_6d.py, but with safer
        # absolute floors because MANO/SMPL translations and rotations can be
        # small or near zero.
        self._std_weight_trans = 1.0 / 10.0
        self._std_weight_rot = 1.0 / 20.0
        self._std_weight_vel_trans = 1.0 / 20.0
        self._std_weight_vel_rot = 1.0 / 40.0
        self._min_trans_scale = 1e-3
        self._min_rot_scale = 1e-3

    def _scales(self, measurement_or_mean):
        arr = np.asarray(measurement_or_mean, dtype=np.float64)
        trans_scale = max(float(np.linalg.norm(arr[:3])), self._min_trans_scale)
        rot_scale = max(float(np.linalg.norm(arr[3:6])), self._min_rot_scale)
        return trans_scale, rot_scale

    def initiate(self, measurement):
        measurement = np.asarray(measurement, dtype=np.float64).reshape(6)
        mean_pos = measurement.copy()
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        trans_scale, rot_scale = self._scales(measurement)
        std = [
            0.2 * self._std_weight_trans * trans_scale,
            0.2 * self._std_weight_trans * trans_scale,
            0.2 * self._std_weight_trans * trans_scale,
            0.2 * self._std_weight_rot * rot_scale,
            0.2 * self._std_weight_rot * rot_scale,
            0.2 * self._std_weight_rot * rot_scale,
            self._std_weight_vel_trans * trans_scale,
            self._std_weight_vel_trans * trans_scale,
            self._std_weight_vel_trans * trans_scale,
            self._std_weight_vel_rot * rot_scale,
            self._std_weight_vel_rot * rot_scale,
            self._std_weight_vel_rot * rot_scale,
        ]
        covariance = np.diag(np.square(std)) * self.init_cov_scale
        return mean, covariance

    def predict(self, mean, covariance):
        mean = np.asarray(mean, dtype=np.float64).reshape(12)
        covariance = np.asarray(covariance, dtype=np.float64).reshape(12, 12)

        trans_scale, rot_scale = self._scales(mean[:6])
        std_pos = [
            self._std_weight_trans * trans_scale,
            self._std_weight_trans * trans_scale,
            self._std_weight_trans * trans_scale,
            self._std_weight_rot * rot_scale,
            self._std_weight_rot * rot_scale,
            self._std_weight_rot * rot_scale,
        ]
        std_vel = [
            self._std_weight_vel_trans * trans_scale,
            self._std_weight_vel_trans * trans_scale,
            self._std_weight_vel_trans * trans_scale,
            self._std_weight_vel_rot * rot_scale,
            self._std_weight_vel_rot * rot_scale,
            self._std_weight_vel_rot * rot_scale,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])) * self.process_noise_scale

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        trans_scale, rot_scale = self._scales(mean[:6])
        std = [
            self.measurement_noise_scale * self._std_weight_trans * trans_scale,
            self.measurement_noise_scale * self._std_weight_trans * trans_scale,
            self.measurement_noise_scale * self._std_weight_trans * trans_scale,
            self.measurement_noise_scale * self._std_weight_rot * rot_scale,
            self.measurement_noise_scale * self._std_weight_rot * rot_scale,
            self.measurement_noise_scale * self._std_weight_rot * rot_scale,
        ]
        innovation_cov = np.diag(np.square(std))
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return projected_mean, projected_cov

    def update(self, mean, covariance, measurement):
        measurement = np.asarray(measurement, dtype=np.float64).reshape(6)
        projected_mean, projected_cov = self.project(mean, covariance)

        # K = P H^T (H P H^T + R)^-1, solved without explicit inverse.
        kalman_gain = _stable_kalman_gain(covariance, self._update_mat, projected_cov)
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance


class KalmanFilterPoseJoint6D(object):
    """
    6D Kalman filter for one SMPL/SMPLH/SMPLX pose joint.

    Measurement:
        z = [rx, ry, rz]  axis-angle rotation vector of one joint

    State:
        x = [rx, ry, rz, vrx, vry, vrz]

    This is a constant-angular-velocity model. It is intentionally lightweight
    and self-contained because EasyMocap stores poses as axis-angle vectors.
    """

    def __init__(
        self,
        measurement_noise_scale=5.0,
        process_noise_scale=1.0,
        init_cov_scale=1.0,
        dt=1.0,
    ):
        ndim = 3
        self.ndim = ndim
        self.dt = dt
        self.measurement_noise_scale = float(measurement_noise_scale)
        self.process_noise_scale = float(process_noise_scale)
        self.init_cov_scale = float(init_cov_scale)

        self._motion_mat = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float64)

        # Axis-angle values are usually small. Use absolute noise floors so a
        # near-zero rotation does not create an over-confident Kalman state.
        self._std_weight_rot = 1.0 / 20.0
        self._std_weight_vel_rot = 1.0 / 40.0
        self._min_rot_scale = 1e-2

    def _scale(self, measurement_or_mean):
        arr = np.asarray(measurement_or_mean, dtype=np.float64)
        return max(float(np.linalg.norm(arr[:3])), self._min_rot_scale)

    def initiate(self, measurement):
        measurement = np.asarray(measurement, dtype=np.float64).reshape(3)
        mean_pos = measurement.copy()
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        rot_scale = self._scale(measurement)
        std = [
            0.2 * self._std_weight_rot * rot_scale,
            0.2 * self._std_weight_rot * rot_scale,
            0.2 * self._std_weight_rot * rot_scale,
            self._std_weight_vel_rot * rot_scale,
            self._std_weight_vel_rot * rot_scale,
            self._std_weight_vel_rot * rot_scale,
        ]
        covariance = np.diag(np.square(std)) * self.init_cov_scale
        return mean, covariance

    def predict(self, mean, covariance):
        mean = np.asarray(mean, dtype=np.float64).reshape(6)
        covariance = np.asarray(covariance, dtype=np.float64).reshape(6, 6)

        rot_scale = self._scale(mean[:3])
        std_pos = [self._std_weight_rot * rot_scale] * 3
        std_vel = [self._std_weight_vel_rot * rot_scale] * 3
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])) * self.process_noise_scale

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        rot_scale = self._scale(mean[:3])
        std = [self.measurement_noise_scale * self._std_weight_rot * rot_scale] * 3
        innovation_cov = np.diag(np.square(std))
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return projected_mean, projected_cov

    def update(self, mean, covariance, measurement):
        measurement = np.asarray(measurement, dtype=np.float64).reshape(3)
        projected_mean, projected_cov = self.project(mean, covariance)

        # K = P H^T (H P H^T + R)^-1, solved without explicit inverse.
        kalman_gain = _stable_kalman_gain(covariance, self._update_mat, projected_cov)
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance


def _get_cfg_value(cfg, name, default):
    """Read attributes from EasyMocap Config/OmegaConf-like objects safely."""
    if cfg is None:
        return default
    if hasattr(cfg, name):
        return getattr(cfg, name)
    try:
        return cfg[name]
    except Exception:
        return default





def _stable_kalman_gain(covariance, update_mat, projected_cov):
    """
    Compute Kalman gain without explicitly inverting the innovation covariance.

    K = P H^T (H P H^T + R)^-1

    Using solve is more stable than np.linalg.inv. A small diagonal jitter is
    added as fallback for near-singular covariance matrices, which can happen
    after long optimization sequences or very small rotations.
    """
    PHt = covariance @ update_mat.T
    try:
        return np.linalg.solve(projected_cov.T, PHt.T).T
    except np.linalg.LinAlgError:
        jitter = 1e-9 * np.eye(projected_cov.shape[0], dtype=projected_cov.dtype)
        return np.linalg.solve((projected_cov + jitter).T, PHt.T).T


def _clip_vector_norm(vec, max_norm):
    """Clip a vector by L2 norm while preserving direction."""
    if max_norm is None:
        return vec
    norm = float(np.linalg.norm(vec))
    if norm > max_norm and norm > 1e-12:
        return vec * (float(max_norm) / norm)
    return vec

def apply_kalman_filter_to_params(
    params,
    measurement_noise_scale=2.0,
    process_noise_scale=3.0,
    init_cov_scale=1.0,
    max_jump=None,
    max_velocity=None,
):
    """
    Apply one forward Kalman pass to optimized global RT parameters.

    Args:
        params: EasyMocap parameter dict. Must contain 'Rh' and 'Th'.
        measurement_noise_scale: larger value means trust optimized measurement
            less and produce stronger smoothing.
        process_noise_scale: larger value means allow faster motion/change.
        init_cov_scale: initial covariance scale.
        max_jump: optional translation jump threshold. If the measurement jumps
            farther than this value from prediction, reset the state for that
            frame instead of dragging the sequence.
        max_velocity: optional translation velocity clamp for the internal state.
            This prevents one bad frame from creating long-lasting drift.

    Returns:
        params with filtered 'Rh' and 'Th'. Other fields are unchanged.
    """
    if params is None or ('Rh' not in params) or ('Th' not in params):
        return params

    Rh = np.asarray(params['Rh']).copy()
    Th = np.asarray(params['Th']).copy()
    Rh_shape = Rh.shape
    Th_shape = Th.shape

    if Rh.shape[0] != Th.shape[0]:
        return params

    n_frames = Rh.shape[0]
    Rh_flat = Rh.reshape(n_frames, -1)
    Th_flat = Th.reshape(n_frames, -1)
    if Rh_flat.shape[1] < 3 or Th_flat.shape[1] < 3:
        return params

    kf = KalmanFilterRT6D(
        measurement_noise_scale=measurement_noise_scale,
        process_noise_scale=process_noise_scale,
        init_cov_scale=init_cov_scale,
    )

    mean, covariance = None, None
    Rh_filtered = Rh_flat.copy()
    Th_filtered = Th_flat.copy()

    for nf in range(n_frames):
        z = np.r_[Th_flat[nf, :3], Rh_flat[nf, :3]].astype(np.float64)

        if mean is None:
            mean, covariance = kf.initiate(z)
        else:
            mean, covariance = kf.predict(mean, covariance)
            if max_jump is not None:
                jump = np.linalg.norm(z[:3] - mean[:3])
                if jump > max_jump:
                    mean, covariance = kf.initiate(z)
                else:
                    mean, covariance = kf.update(mean, covariance, z)
            else:
                mean, covariance = kf.update(mean, covariance, z)

        if max_velocity is not None:
            mean[6:9] = _clip_vector_norm(mean[6:9], max_velocity)

        Th_filtered[nf, :3] = mean[:3]
        Rh_filtered[nf, :3] = mean[3:6]

    params['Rh'] = Rh_filtered.reshape(Th_shape if False else Rh_shape)
    params['Th'] = Th_filtered.reshape(Th_shape)
    return params



def apply_kalman_filter_to_poses(
    params,
    measurement_noise_scale=1.5,
    process_noise_scale=8.0,
    init_cov_scale=1.0,
    max_angle_jump=None,
    max_angular_velocity=None,
    skip_root_pose=True,
):
    """
    Apply one forward 6D Kalman pass to internal pose parameters.

    Args:
        params: EasyMocap parameter dict. Must contain 'poses'.
        measurement_noise_scale: larger value means trust optimized pose less
            and produce stronger smoothing.
        process_noise_scale: larger value means allow faster joint motion.
        init_cov_scale: initial covariance scale.
        max_angle_jump: optional per-joint axis-angle jump threshold in radians.
            If the optimized joint rotation jumps farther than this threshold
            from prediction, reset that joint state for that frame.
        max_angular_velocity: optional per-joint angular velocity clamp in
            radians/frame. This helps suppress long-lasting finger explosions.
        skip_root_pose: if True, do not filter poses[:, 0:3] because global
            orientation is already filtered through params['Rh'].

    Returns:
        params with filtered 'poses'. Other fields are unchanged.
    """
    if params is None or 'poses' not in params:
        return params

    poses = np.asarray(params['poses']).copy()
    pose_shape = poses.shape
    if poses.shape[0] <= 1:
        return params

    n_frames = poses.shape[0]
    poses_flat = poses.reshape(n_frames, -1)
    if poses_flat.shape[1] < 3:
        return params

    n_joints = poses_flat.shape[1] // 3
    if n_joints < 1:
        return params

    usable_dim = n_joints * 3
    poses_joint = poses_flat[:, :usable_dim].reshape(n_frames, n_joints, 3)
    poses_filtered = poses_joint.copy()

    kf_list = [
        KalmanFilterPoseJoint6D(
            measurement_noise_scale=measurement_noise_scale,
            process_noise_scale=process_noise_scale,
            init_cov_scale=init_cov_scale,
        )
        for _ in range(n_joints)
    ]
    states = [None] * n_joints

    start_joint = 1 if skip_root_pose and n_joints > 1 else 0

    for nf in range(n_frames):
        for nj in range(start_joint, n_joints):
            z = poses_joint[nf, nj].astype(np.float64)

            if states[nj] is None:
                mean, covariance = kf_list[nj].initiate(z)
            else:
                mean, covariance = states[nj]
                mean, covariance = kf_list[nj].predict(mean, covariance)

                if max_angle_jump is not None:
                    jump = np.linalg.norm(z - mean[:3])
                    if jump > max_angle_jump:
                        mean, covariance = kf_list[nj].initiate(z)
                    else:
                        mean, covariance = kf_list[nj].update(mean, covariance, z)
                else:
                    mean, covariance = kf_list[nj].update(mean, covariance, z)

            if max_angular_velocity is not None:
                mean[3:6] = _clip_vector_norm(mean[3:6], max_angular_velocity)

            states[nj] = (mean, covariance)
            poses_filtered[nf, nj] = mean[:3]

    poses_flat[:, :usable_dim] = poses_filtered.reshape(n_frames, usable_dim)
    params['poses'] = poses_flat.reshape(pose_shape)
    return params

def multi_stage_optimize(body_model, params, kp3ds, kp2ds=None, bboxes=None, Pall=None, weight={}, cfg=None):
    with Timer('Optimize global RT'):
        cfg.OPT_R = True
        cfg.OPT_T = True
        # Disable smooth_body during global RT optimization: when only Rh/Th
        # are optimized (body pose is zero), smooth_body penalizes natural
        # inter-frame joint movement in world coordinates, which conflicts
        # with the k3d data term and causes incorrect global rotation.
        saved_smooth_body = weight.get('smooth_body', 0.)
        weight['smooth_body'] = 0.
        params = optimizePose3D(body_model, params, kp3ds, weight=weight, cfg=cfg)
        weight['smooth_body'] = saved_smooth_body
        # params = optimizePose(body_model, params, kp3ds, weight_loss=weight, kintree=config['kintree'], cfg=cfg)
    with Timer('Optimize 3D Pose/{} frames'.format(kp3ds.shape[0])):
        cfg.OPT_POSE = True
        cfg.ROBUST_3D = False
        params = optimizePose3D(body_model, params, kp3ds, weight=weight, cfg=cfg)
        if False:
            cfg.ROBUST_3D = True
            params = optimizePose3D(body_model, params, kp3ds, weight=weight, cfg=cfg)
        if cfg.model in ['smplh', 'smplx']:
            cfg.OPT_HAND = True
            params = optimizePose3D(body_model, params, kp3ds, weight=weight, cfg=cfg)
        if cfg.model == 'smplx':
            cfg.OPT_EXPR = True
            params = optimizePose3D(body_model, params, kp3ds, weight=weight, cfg=cfg)

    return params


def multi_stage_optimize2d(body_model, params, kp2ds, bboxes, Pall, weight={}, args=None):
    cfg = Config(args)
    cfg.device = body_model.device
    cfg.device = body_model.device
    cfg.model = body_model.model_type
    with Timer('Optimize global RT'):
        cfg.OPT_R = True
        cfg.OPT_T = True
        params = optimizePose2D(body_model, params, bboxes, kp2ds, Pall, weight=weight, cfg=cfg)
    with Timer('Optimize 2D Pose/{} frames'.format(kp2ds.shape[0])):
        cfg.OPT_POSE = True
        cfg.OPT_SHAPE = True
        # bboxes => (nFrames, nViews, 5), keypoints2d => (nFrames, nViews, nJoints, 3)
        params = optimizePose2D(body_model, params, bboxes, kp2ds, Pall, weight=weight, cfg=cfg)


    with Timer('Kalman filter optimized poses/{} frames'.format(kp2ds.shape[0])):
        params = apply_kalman_filter_to_poses(
            params,
            measurement_noise_scale=getattr(args, 'pose_kalman_measurement_noise', 5),
            process_noise_scale=getattr(args, 'pose_kalman_process_noise', 1),
            init_cov_scale=getattr(args, 'pose_kalman_init_cov', 1.0),
            max_angle_jump=getattr(args, 'pose_kalman_max_angle_jump', None),
            max_angular_velocity=getattr(args, 'pose_kalman_max_angular_velocity', None),
            skip_root_pose=getattr(args, 'pose_kalman_skip_root', True),
        )

import numpy as np
import torch
import copy


from pathlib import Path
import torch
import numpy as np


def load_betas_from_restored_pt(name):
    """
    Load 10 MANO betas from:
        ../<name>/calib/restored_flat_real_scale.pt
    """
    pt_path = Path(".") / name / "calib" / "restored_flat_real_scale.pt"

    if not pt_path.exists():
        raise FileNotFoundError(f"Cannot find beta pt file: {pt_path}")

    try:
        data = torch.load(pt_path, map_location="cpu")
    except TypeError:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Common possible keys
    candidate_keys = [
        "real_scale_betas10_lstsq",
        "real_scale_betas10",
        "betas",
        "beta",
        "shapes",
        "shape",
    ]

    betas = None

    if isinstance(data, dict):
        for key in candidate_keys:
            if key in data:
                betas = data[key]
                print(f"[info] Loaded betas from key: {key}")
                break
    else:
        # If the whole pt itself is betas
        betas = data

    if betas is None:
        raise KeyError(
            f"Cannot find betas in {pt_path}. "
            f"Available keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        )

    if isinstance(betas, torch.Tensor):
        betas = betas.detach().cpu().numpy()

    betas = np.asarray(betas, dtype=np.float32).reshape(-1)

    if betas.shape[0] < 10:
        raise ValueError(f"Betas should have at least 10 values, but got shape {betas.shape}")

    betas = betas[:10]

    print("[info] betas loaded:", betas)

    return betas


def set_fixed_betas_to_params(params, name):
    """
    Keep params as a dict.
    Overwrite all MANO/SMPL shape betas beta[0:10]
    using betas loaded from:

        ../<name>/calib/restored_flat_real_scale.pt

    EasyMocap usually uses params['shapes'] with shape (nFrames, 10).
    """
    fixed_betas = load_betas_from_restored_pt(name)

    key = "shapes" if "shapes" in params else "shape"

    if key not in params:
        params["shapes"] = fixed_betas[None, :].copy()
        return params

    old_shape = params[key]
    n_frames = old_shape.shape[0]

    if isinstance(old_shape, torch.Tensor):
        device = old_shape.device
        dtype = old_shape.dtype

        fixed = torch.tensor(fixed_betas, dtype=dtype, device=device).view(1, 10)

        new_shape = old_shape.clone()

        # overwrite all beta0 ~ beta9
        new_shape[:, 0:10] = fixed.repeat(n_frames, 1)

        params[key] = new_shape.detach()

    else:
        fixed = np.asarray(fixed_betas, dtype=np.float32).reshape(1, 10)

        new_shape = np.asarray(old_shape, dtype=np.float32).copy()

        # overwrite all beta0 ~ beta9
        new_shape[:, 0:10] = np.repeat(fixed, n_frames, axis=0)

        params[key] = new_shape

    return params
def smpl_from_keypoints3d2d(body_model, kp3ds, kp2ds, bboxes, Pall, config, args,
    weight_shape=None, weight_pose=None):

    model_type = body_model.model_type
    params_init = body_model.init_params(nFrames=1)
    if weight_shape is None:
        weight_shape = load_weight_shape(model_type, args.opts)
    if args.name is not None:
        # For MANO hand model, do NOT optimize shape.
        # Keep the whole params dict, only fix betas/shapes.
        params_shape = copy.deepcopy(params_init)

        params_shape = set_fixed_betas_to_params(params_shape, args.name)
    else:
        params_shape = optimizeShape(body_model, params_init, kp3ds, 
            weight_loss=weight_shape, kintree=config['kintree'])       

    # optimize 3D pose
    cfg = Config(args)
    cfg.device = body_model.device

    # Pass Kalman options into cfg because multi_stage_optimize only receives cfg.
    cfg.use_param_kalman = getattr(args, 'use_param_kalman', True)
    cfg.param_kalman_measurement_noise = getattr(args, 'param_kalman_measurement_noise', 2.0)
    cfg.param_kalman_process_noise = getattr(args, 'param_kalman_process_noise', 3.0)
    cfg.param_kalman_init_cov = getattr(args, 'param_kalman_init_cov', 1.0)
    cfg.param_kalman_max_jump = getattr(args, 'param_kalman_max_jump', None)
    cfg.param_kalman_max_velocity = getattr(args, 'param_kalman_max_velocity', None)
    cfg.use_pose_kalman = getattr(args, 'use_pose_kalman', True)
    cfg.pose_kalman_measurement_noise = getattr(args, 'pose_kalman_measurement_noise', 1.5)
    cfg.pose_kalman_process_noise = getattr(args, 'pose_kalman_process_noise', 8.0)
    cfg.pose_kalman_init_cov = getattr(args, 'pose_kalman_init_cov', 1.0)
    cfg.pose_kalman_max_angle_jump = getattr(args, 'pose_kalman_max_angle_jump', None)
    cfg.pose_kalman_max_angular_velocity = getattr(args, 'pose_kalman_max_angular_velocity', None)
    cfg.pose_kalman_skip_root = getattr(args, 'pose_kalman_skip_root', True)

    params = body_model.init_params(nFrames=kp3ds.shape[0])
    params['shapes'] = params_shape['shapes'].copy()
    if weight_pose is None:
        weight_pose = load_weight_pose(model_type, args.opts)
    # We divide this step to two functions, because we can have different initialization method
    
    params = multi_stage_optimize(body_model, params, kp3ds, kp2ds, bboxes, Pall, weight_pose, cfg)
    return params


def smpl_from_keypoints3d(body_model, kp3ds, config, args,
    weight_shape=None, weight_pose=None):
    model_type = body_model.model_type
    params_init = body_model.init_params(nFrames=1)
    if weight_shape is None:
        weight_shape = load_weight_shape(model_type, args.opts)
    if model_type in ['smpl', 'smplh', 'smplx']:
        # when use SMPL model, optimize the shape only with first 1-14 limbs,
        # don't use (nose, neck)
        params_shape = optimizeShape(body_model, params_init, kp3ds,
            weight_loss=weight_shape, kintree=CONFIG['body15']['kintree'][1:])
    else:
        params_shape = optimizeShape(body_model, params_init, kp3ds,
            weight_loss=weight_shape, kintree=config['kintree'])
    # optimize 3D pose
    cfg = Config(args)
    cfg.device = body_model.device
    cfg.model_type = model_type

    # Pass Kalman options into cfg because multi_stage_optimize only receives cfg.
    cfg.use_param_kalman = getattr(args, 'use_param_kalman', True)
    cfg.param_kalman_measurement_noise = getattr(args, 'param_kalman_measurement_noise', 2.0)
    cfg.param_kalman_process_noise = getattr(args, 'param_kalman_process_noise', 3.0)
    cfg.param_kalman_init_cov = getattr(args, 'param_kalman_init_cov', 1.0)
    cfg.param_kalman_max_jump = getattr(args, 'param_kalman_max_jump', None)
    cfg.param_kalman_max_velocity = getattr(args, 'param_kalman_max_velocity', None)
    cfg.use_pose_kalman = getattr(args, 'use_pose_kalman', True)
    cfg.pose_kalman_measurement_noise = getattr(args, 'pose_kalman_measurement_noise', 1.5)
    cfg.pose_kalman_process_noise = getattr(args, 'pose_kalman_process_noise', 8.0)
    cfg.pose_kalman_init_cov = getattr(args, 'pose_kalman_init_cov', 1.0)
    cfg.pose_kalman_max_angle_jump = getattr(args, 'pose_kalman_max_angle_jump', None)
    cfg.pose_kalman_max_angular_velocity = getattr(args, 'pose_kalman_max_angular_velocity', None)
    cfg.pose_kalman_skip_root = getattr(args, 'pose_kalman_skip_root', True)

    params = body_model.init_params(nFrames=kp3ds.shape[0])
    params['shapes'] = params_shape['shapes'].copy()
    if weight_pose is None:
        weight_pose = load_weight_pose(model_type, args.opts)
    # We divide this step to two functions, because we can have different initialization method
    params = multi_stage_optimize(body_model, params, kp3ds, None, None, None, weight_pose, cfg)
    return params
