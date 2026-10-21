'''
 * @ Date: 2020-09-14 11:01:52
 * @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-04-13 20:31:34
  @ FilePath: /EasyMocapRelease/media/qing/Project/mirror/EasyMocap/easymocap/mytools/reconstruction.py

This version adds a joint-level Kalman filter for triangulated 3D keypoints.
It follows the same calling style as predict_pose_refine.py:
    measurement -> initiate on first frame -> predict -> update -> filtered output
For triangulation, each joint uses a 6D constant-velocity state:
    [x, y, z, vx, vy, vz]
'''

import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None


def solveZ(A):
    u, s, v = np.linalg.svd(A)
    X = v[-1, :]
    X = X / X[3]
    return X[:3]


def projectN3(kpts3d, Pall):
    # kpts3d: (N, 3) or (N, 4)
    nViews = len(Pall)
    kp3d = np.hstack((kpts3d[:, :3], np.ones((kpts3d.shape[0], 1))))
    kp2ds = []
    for nv in range(nViews):
        kp2d = Pall[nv] @ kp3d.T
        kp2d[:2, :] /= kp2d[2:, :]
        kp2ds.append(kp2d.T[None, :, :])
    kp2ds = np.vstack(kp2ds)
    if kpts3d.shape[-1] == 4:
        kp2ds[..., -1] = kp2ds[..., -1] * (kpts3d[None, :, -1] > 0.)
    return kp2ds


def simple_reprojection_error(kpts1, kpts1_proj):
    # (N, 3)
    error = np.mean((kpts1[:, :2] - kpts1_proj[:, :2]) ** 2)
    return error


def simple_triangulate(kpts, Pall):
    # kpts: (nViews, 3)
    # Pall: (nViews, 3, 4)
    # return: kpts3d(4,), [x, y, z, conf]
    nViews = len(kpts)
    A = np.zeros((nViews * 2, 4), dtype=np.float64)
    result = np.zeros(4, dtype=np.float64)
    valid = kpts[:, 2] > 0
    if valid.sum() < 2:
        return result
    result[3] = kpts[:, 2].sum() / valid.sum()
    for i in range(nViews):
        P = Pall[i]
        A[i * 2, :] = kpts[i, 2] * (kpts[i, 0] * P[2:3, :] - P[0:1, :])
        A[i * 2 + 1, :] = kpts[i, 2] * (kpts[i, 1] * P[2:3, :] - P[1:2, :])
    result[:3] = solveZ(A)
    return result


class KalmanFilter3D(object):
    """
    3D Kalman filter for triangulated joints.

    State space, 6-dimensional:
        [x, y, z, vx, vy, vz]

    It is intentionally parallel to KalmanFilter6D used by predict_pose_refine.py,
    but the measurement is only [x, y, z].
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

        # Similar spirit to FoundationPose's pose KF: position/velocity noise
        # scales with the current distance from the camera/world origin.
        self._std_weight_pos = 1.0 / 20.0
        self._std_weight_vel = 1.0 / 40.0

    def initiate(self, measurement):
        measurement = np.asarray(measurement, dtype=np.float64).reshape(3)
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        scale_xyz = max(np.linalg.norm(measurement), 1e-5)
        std = [
            0.2 * self._std_weight_pos * scale_xyz,
            0.2 * self._std_weight_pos * scale_xyz,
            0.2 * self._std_weight_pos * scale_xyz,
            self._std_weight_vel * scale_xyz,
            self._std_weight_vel * scale_xyz,
            self._std_weight_vel * scale_xyz,
        ]
        covariance = np.diag(np.square(std)) * self.init_cov_scale
        return mean, covariance

    def predict(self, mean, covariance):
        mean = np.asarray(mean, dtype=np.float64)
        covariance = np.asarray(covariance, dtype=np.float64)

        scale_xyz = max(np.linalg.norm(mean[:3]), 1e-5)
        std_pos = [self._std_weight_pos * scale_xyz] * 3
        std_vel = [self._std_weight_vel * scale_xyz] * 3
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])) * self.process_noise_scale

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance, measurement_conf=1.0):
        mean = np.asarray(mean, dtype=np.float64)
        covariance = np.asarray(covariance, dtype=np.float64)

        conf = float(np.clip(measurement_conf, 1e-3, 1.0))
        scale_xyz = max(np.linalg.norm(mean[:3]), 1e-5)

        # Higher confidence means smaller measurement noise.
        adaptive_scale = self.measurement_noise_scale / conf
        std = [adaptive_scale * self._std_weight_pos * scale_xyz] * 3
        innovation_cov = np.diag(np.square(std))

        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return projected_mean, projected_cov

    def update(self, mean, covariance, measurement, measurement_conf=1.0):
        measurement = np.asarray(measurement, dtype=np.float64).reshape(3)
        projected_mean, projected_cov = self.project(mean, covariance, measurement_conf=measurement_conf)

        # Equivalent to the scipy Cholesky solve in KalmanFilter6D, but avoids a
        # hard dependency here. The matrix is only 3x3.
        PHt = np.dot(covariance, self._update_mat.T)
        kalman_gain = np.linalg.solve(projected_cov.T, PHt.T).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


class JointKalmanManager(object):
    """
    Keep one Kalman state for each joint across frames.

    Usage:
        kalman_manager = JointKalmanManager(n_joints)
        keypoints3d = kalman_manager.update(keypoints3d)

    This object must be created once before the frame loop, exactly like the
    pose_kalman_states in predict_pose_refine.py are kept inside the predictor.
    """

    def __init__(
        self,
        n_joints,
        measurement_noise_scale=5.0,
        process_noise_scale=1.0,
        init_cov_scale=1.0,
        max_jump=None,
        reset_after_missing=10,
    ):
        self.n_joints = int(n_joints)
        self.kf = KalmanFilter3D(
            measurement_noise_scale=measurement_noise_scale,
            process_noise_scale=process_noise_scale,
            init_cov_scale=init_cov_scale,
        )
        self.states = [None] * self.n_joints
        self.missing_counts = np.zeros(self.n_joints, dtype=np.int32)
        self.max_jump = max_jump
        self.reset_after_missing = int(reset_after_missing)

    def reset(self):
        self.states = [None] * self.n_joints
        self.missing_counts[:] = 0

    def update(self, keypoints3d):
        """
        Args:
            keypoints3d: (nJoints, 4), [x, y, z, conf]

        Returns:
            filtered: (nJoints, 4)
        """
        keypoints3d = np.asarray(keypoints3d, dtype=np.float64)
        if keypoints3d.shape[0] != self.n_joints:
            self.n_joints = keypoints3d.shape[0]
            self.states = [None] * self.n_joints
            self.missing_counts = np.zeros(self.n_joints, dtype=np.int32)

        filtered = keypoints3d.copy()

        for j in range(self.n_joints):
            conf = float(keypoints3d[j, 3])

            if conf <= 0:
                self.missing_counts[j] += 1
                if self.missing_counts[j] >= self.reset_after_missing:
                    self.states[j] = None
                continue

            self.missing_counts[j] = 0
            z = keypoints3d[j, :3]

            if self.states[j] is None:
                mean, covariance = self.kf.initiate(z)
            else:
                mean, covariance = self.states[j]
                mean, covariance = self.kf.predict(mean, covariance)

                # Optional hard reset for obviously wrong jumps.
                if self.max_jump is not None:
                    jump = np.linalg.norm(z - mean[:3])
                    if jump > self.max_jump:
                        mean, covariance = self.kf.initiate(z)
                    else:
                        mean, covariance = self.kf.update(mean, covariance, z, measurement_conf=conf)
                else:
                    mean, covariance = self.kf.update(mean, covariance, z, measurement_conf=conf)

            self.states[j] = (mean, covariance)
            filtered[j, :3] = mean[:3]
            filtered[j, 3] = conf

        return filtered.astype(keypoints3d.dtype, copy=False)


def apply_kalman_filter_to_keypoints3d(keypoints3d, kalman_manager=None):
    if kalman_manager is None:
        return keypoints3d
    return kalman_manager.update(keypoints3d)


def project_point(P, X):
    """
    Project a 3D point in homogeneous form to image plane.
    """
    x = P @ X
    z = x[2]
    if abs(z) < 1e-12:
        z = 1e-12
    return x[0] / z, x[1] / z


def triangulate_dlt_one_joint(keypoints, Pall, prior=None, lamb=1e3):
    """
    Linear triangulation for one joint using weighted DLT.
    """
    valid = keypoints[:, 2] > 0
    if valid.sum() < 2:
        return None

    kp = keypoints[valid]
    P = Pall[valid]

    A = []
    for i in range(kp.shape[0]):
        u, v, c = kp[i]
        Pi = P[i]
        A.append(c * (u * Pi[2] - Pi[0]))
        A.append(c * (v * Pi[2] - Pi[1]))

    A = np.asarray(A, dtype=np.float64)

    if prior is not None and prior[3] > 0:
        x0, y0, z0, c0 = prior
        w = lamb * c0
        B = np.array([
            [1.0, 0.0, 0.0, -x0],
            [0.0, 1.0, 0.0, -y0],
            [0.0, 0.0, 1.0, -z0],
        ], dtype=np.float64) * w
        A = np.vstack([A, B])

    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    if abs(X[3]) < 1e-12:
        return None
    X = X / X[3]
    return X


def reprojection_residual_xyz(
    xyz,
    keypoints,
    Pall,
    prior=None,
    lamb=1e3,
    use_sqrt_conf=True,
):
    """
    Residual for nonlinear optimization on XYZ.
    """
    X = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)

    residuals = []
    valid = keypoints[:, 2] > 0

    for i in range(keypoints.shape[0]):
        if not valid[i]:
            continue

        u_obs, v_obs, conf = keypoints[i]
        u_pred, v_pred = project_point(Pall[i], X)

        w = np.sqrt(conf) if use_sqrt_conf else conf
        residuals.append(w * (u_pred - u_obs))
        residuals.append(w * (v_pred - v_obs))

    if prior is not None and prior[3] > 0:
        w_prior = np.sqrt(lamb * prior[3]) if use_sqrt_conf else (lamb * prior[3])
        residuals.append(w_prior * (xyz[0] - prior[0]))
        residuals.append(w_prior * (xyz[1] - prior[1]))
        residuals.append(w_prior * (xyz[2] - prior[2]))

    return np.asarray(residuals, dtype=np.float64)


def refine_one_joint_nonlinear(
    X_init,
    keypoints,
    Pall,
    prior=None,
    lamb=1e3,
    max_nfev=50,
    method='lm',
):
    """
    Nonlinear refinement by minimizing weighted reprojection error.
    """
    if X_init is None or least_squares is None:
        return X_init

    valid = keypoints[:, 2] > 0
    n_obs = int(valid.sum())

    if method == 'lm':
        n_res = 2 * n_obs + (3 if (prior is not None and prior[3] > 0) else 0)
        if n_res < 3:
            return X_init

    x0 = X_init[:3].copy()

    try:
        res = least_squares(
            reprojection_residual_xyz,
            x0,
            method=method,
            max_nfev=max_nfev,
            args=(keypoints, Pall, prior, lamb, True),
        )
        xyz = res.x
        return np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
    except Exception:
        return X_init


def batch_triangulate(keypoints_, Pall, min_view=2, kalman_manager=None):
    """Triangulate the keypoints of whole body.

    Args:
        keypoints_ (nViews, nJoints, 3): 2D detections
        Pall (nViews, 3, 4) | (nViews, nJoints, 3, 4): projection matrix
        min_view (int): min views for visible points
        kalman_manager (JointKalmanManager | None): optional temporal filter

    Returns:
        keypoints3d: (nJoints, 4)
    """
    v = (keypoints_[:, :, -1] > 0).sum(axis=0)
    valid_joint = np.where(v >= min_view)[0]

    result = np.zeros((keypoints_.shape[1], 4), dtype=np.float64)
    if len(valid_joint) == 0:
        return apply_kalman_filter_to_keypoints3d(result, kalman_manager)

    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0) / v[valid_joint]

    if len(Pall.shape) == 3:
        P0 = Pall[None, :, 0, :]
        P1 = Pall[None, :, 1, :]
        P2 = Pall[None, :, 2, :]
    else:
        P0 = Pall[:, :, 0, :].swapaxes(0, 1)
        P1 = Pall[:, :, 1, :].swapaxes(0, 1)
        P2 = Pall[:, :, 2, :].swapaxes(0, 1)

    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = np.hstack([Au, Av])

    u, s, v = np.linalg.svd(A)
    X = v[:, -1, :]
    valid_h = np.abs(X[:, 3]) > 1e-12
    X[valid_h] = X[valid_h] / X[valid_h, 3:]

    good_joint = valid_joint[valid_h]
    result[good_joint, :3] = X[valid_h, :3]
    result[valid_joint, 3] = conf3d
    result[valid_joint[~valid_h], 3] = 0.0

    return apply_kalman_filter_to_keypoints3d(result, kalman_manager)


eps = 0.01


def simple_recon_person(keypoints_use, Puse, kalman_manager=None):
    out = batch_triangulate(keypoints_use, Puse, kalman_manager=kalman_manager)
    kpts_repro = projectN3(out, Puse)
    conf = np.repeat(out[None, :, -1:], len(Puse), 0)
    kpts_repro = np.concatenate((kpts_repro, conf), axis=2)
    return out, kpts_repro


def check_limb(keypoints3d, limb_means, thres=0.5):
    # keypoints3d: (nJ, 4)
    valid = True
    cnt = 0
    for (src, dst), val in limb_means.items():
        if not (keypoints3d[src, 3] > 0 and keypoints3d[dst, 3] > 0):
            continue
        cnt += 1
        l_est = np.linalg.norm(keypoints3d[src, :3] - keypoints3d[dst, :3])
        if abs(l_est - val['mean']) / val['mean'] / val['std'] > thres:
            valid = False
            break
    valid = valid and cnt > 2
    return valid
