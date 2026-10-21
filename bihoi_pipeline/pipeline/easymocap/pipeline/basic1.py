'''
  @ Date: 2021-04-13 20:43:16
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-14 15:32:24
  @ FilePath: /EasyMocap/easymocap/pipeline/basic.py

This version adds an optional Kalman filter on optimized global RT parameters.
The filtering is applied inside multi_stage_optimize after 3D/2D optimization,
so it behaves like the pose-level Kalman filter used in predict_pose_refine.py:
measurement first, then predict/update, then write the filtered pose sequence back.
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

        # K = P H^T (H P H^T + R)^-1
        kalman_gain = covariance @ self._update_mat.T @ np.linalg.inv(projected_cov)
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


def apply_kalman_filter_to_params(
    params,
    measurement_noise_scale=5.0,
    process_noise_scale=1.0,
    init_cov_scale=1.0,
    max_jump=None,
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

        Th_filtered[nf, :3] = mean[:3]
        Rh_filtered[nf, :3] = mean[3:6]

    params['Rh'] = Rh_filtered.reshape(Th_shape if False else Rh_shape)
    params['Th'] = Th_filtered.reshape(Th_shape)
    return params


def multi_stage_optimize(body_model, params, kp3ds, kp2ds=None, bboxes=None, Pall=None, weight={}, cfg=None):
    with Timer('Optimize global RT'):
        cfg.OPT_R = True
        cfg.OPT_T = True
        params = optimizePose3D(body_model, params, kp3ds, weight=weight, cfg=cfg)
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
    if kp2ds is not None:
        with Timer('Optimize 2D Pose/{} frames'.format(kp3ds.shape[0])):
            # bboxes => (nFrames, nViews, 5), keypoints2d => (nFrames, nViews, nJoints, 3)
            params = optimizePose2D(body_model, params, bboxes, kp2ds, Pall, weight=weight, cfg=cfg)

    # ------------------------------------------------------------
    # Kalman filter on optimized global RT.
    # This mirrors predict_pose_refine.py: after the pose measurement is
    # produced by optimization, run a temporal predict/update pass and write
    # the filtered Rh/Th back into params.
    # ------------------------------------------------------------
    if _get_cfg_value(cfg, 'use_param_kalman', False):
        with Timer('Kalman filter optimized global RT/{} frames'.format(kp3ds.shape[0])):
            params = apply_kalman_filter_to_params(
                params,
                measurement_noise_scale=_get_cfg_value(cfg, 'param_kalman_measurement_noise', 5.0),
                process_noise_scale=_get_cfg_value(cfg, 'param_kalman_process_noise', 1.0),
                init_cov_scale=_get_cfg_value(cfg, 'param_kalman_init_cov', 1.0),
                max_jump=_get_cfg_value(cfg, 'param_kalman_max_jump', None),
            )
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

    if getattr(args, 'use_param_kalman', False):
        with Timer('Kalman filter optimized global RT/{} frames'.format(kp2ds.shape[0])):
            params = apply_kalman_filter_to_params(
                params,
                measurement_noise_scale=getattr(args, 'param_kalman_measurement_noise', 5.0),
                process_noise_scale=getattr(args, 'param_kalman_process_noise', 1.0),
                init_cov_scale=getattr(args, 'param_kalman_init_cov', 1.0),
                max_jump=getattr(args, 'param_kalman_max_jump', None),
            )
    return params


def smpl_from_keypoints3d2d(body_model, kp3ds, kp2ds, bboxes, Pall, config, args,
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

    # Pass Kalman options into cfg because multi_stage_optimize only receives cfg.
    cfg.use_param_kalman = getattr(args, 'use_param_kalman', True)
    cfg.param_kalman_measurement_noise = getattr(args, 'param_kalman_measurement_noise', 5.0)
    cfg.param_kalman_process_noise = getattr(args, 'param_kalman_process_noise', 1.0)
    cfg.param_kalman_init_cov = getattr(args, 'param_kalman_init_cov', 1.0)
    cfg.param_kalman_max_jump = getattr(args, 'param_kalman_max_jump', None)

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
    cfg.param_kalman_measurement_noise = getattr(args, 'param_kalman_measurement_noise', 5.0)
    cfg.param_kalman_process_noise = getattr(args, 'param_kalman_process_noise', 1.0)
    cfg.param_kalman_init_cov = getattr(args, 'param_kalman_init_cov', 1.0)
    cfg.param_kalman_max_jump = getattr(args, 'param_kalman_max_jump', None)

    params = body_model.init_params(nFrames=kp3ds.shape[0])
    params['shapes'] = params_shape['shapes'].copy()
    if weight_pose is None:
        weight_pose = load_weight_pose(model_type, args.opts)
    # We divide this step to two functions, because we can have different initialization method
    params = multi_stage_optimize(body_model, params, kp3ds, None, None, None, weight_pose, cfg)
    return params
