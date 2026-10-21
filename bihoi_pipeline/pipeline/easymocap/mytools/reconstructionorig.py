'''
 * @ Date: 2020-09-14 11:01:52
 * @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-04-13 20:31:34
  @ FilePath: /EasyMocapRelease/media/qing/Project/mirror/EasyMocap/easymocap/mytools/reconstruction.py
'''

import numpy as np

def solveZ(A):
    u, s, v = np.linalg.svd(A)
    X = v[-1, :]
    X = X / X[3]
    return X[:3]

def projectN3(kpts3d, Pall):
    # kpts3d: (N, 3)
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
    error = np.mean((kpts1[:, :2] - kpts1_proj[:, :2])**2)
    return error
    
def simple_triangulate(kpts, Pall):
    # kpts: (nViews, 3)
    # Pall: (nViews, 3, 4)
    #   return: kpts3d(3,), conf: float
    nViews = len(kpts)
    A = np.zeros((nViews*2, 4), dtype=np.float)
    result = np.zeros(4)
    result[3] = kpts[:, 2].sum()/(kpts[:, 2]>0).sum()
    for i in range(nViews):
        P = Pall[i]
        A[i*2, :] = kpts[i, 2]*(kpts[i, 0]*P[2:3,:] - P[0:1,:])
        A[i*2 + 1, :] = kpts[i, 2]*(kpts[i, 1]*P[2:3,:] - P[1:2,:])
    result[:3] = solveZ(A)

    return result

import numpy as np
from scipy.optimize import least_squares


def project_point(P, X):
    """
    Project a 3D point in homogeneous form to image plane.

    Args:
        P: (3, 4) projection matrix
        X: (4,) homogeneous 3D point

    Returns:
        (u, v): projected 2D coordinates
    """
    x = P @ X
    z = x[2]
    if abs(z) < 1e-12:
        z = 1e-12
    return x[0] / z, x[1] / z


def triangulate_dlt_one_joint(keypoints, Pall, prior=None, lamb=1e3):
    """
    Linear triangulation for one joint using weighted DLT.

    Args:
        keypoints: (nViews, 3), each row is [u, v, conf]
        Pall: (nViews, 3, 4)
        prior: (4,) or None, previous 3D point [x, y, z, conf]
        lamb: prior weight

    Returns:
        X: (4,), homogeneous-normalized 3D point [x, y, z, 1]
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

    A = np.asarray(A)  # (2*nValid, 4)

    if prior is not None and prior[3] > 0:
        # Soft prior on XYZ only
        x0, y0, z0, c0 = prior
        w = lamb * c0
        B = np.array([
            [1.0, 0.0, 0.0, -x0],
            [0.0, 1.0, 0.0, -y0],
            [0.0, 0.0, 1.0, -z0],
        ]) * w
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
    use_sqrt_conf=True
):
    """
    Residual for nonlinear optimization on XYZ.

    Args:
        xyz: (3,), current 3D point
        keypoints: (nViews, 3), [u, v, conf]
        Pall: (nViews, 3, 4)
        prior: (4,) or None, previous 3D point [x, y, z, conf]
        lamb: prior weight
        use_sqrt_conf: whether to use sqrt(conf) as weight

    Returns:
        residuals: (2*nValid [+3],)
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
        # Add soft prior on XYZ
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
    method='lm'
):
    """
    Nonlinear refinement by minimizing weighted reprojection error.

    Args:
        X_init: (4,), initial homogeneous 3D point
        keypoints: (nViews, 3)
        Pall: (nViews, 3, 4)
        prior: (4,) or None
        lamb: prior weight
        max_nfev: max function evaluations
        method: 'lm' or 'trf'

    Returns:
        X_refined: (4,), refined homogeneous-normalized 3D point
    """
    if X_init is None:
        return None

    valid = keypoints[:, 2] > 0
    n_obs = int(valid.sum())

    # LM requires number of residuals >= number of variables
    # Here variables are 3 (X, Y, Z)
    # residuals are 2*n_obs (+3 if prior exists)
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
            args=(keypoints, Pall, prior, lamb, True)
        )
        xyz = res.x
        return np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
    except Exception:
        # Fallback to linear result if optimization fails
        return X_init


def batch_triangulate(keypoints_, Pall, min_view=2):
    """ triangulate the keypoints of whole body

    Args:
        keypoints_ (nViews, nJoints, 3): 2D detections
        Pall (nViews, 3, 4) | (nViews, nJoints, 3, 4): projection matrix of each view
        min_view (int, optional): min view for visible points. Defaults to 2.

    Returns:
        keypoints3d: (nJoints, 4)
    """
    # keypoints: (nViews, nJoints, 3)
    # Pall: (nViews, 3, 4)
    # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
    v = (keypoints_[:, :, -1]>0).sum(axis=0)
    valid_joint = np.where(v >= min_view)[0]
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0)/v[valid_joint]
    # P2: P矩阵的最后一行：(1, nViews, 1, 4)
    if len(Pall.shape) == 3:
        P0 = Pall[None, :, 0, :]
        P1 = Pall[None, :, 1, :]
        P2 = Pall[None, :, 2, :]
    else:
        P0 = Pall[:, :, 0, :].swapaxes(0, 1)
        P1 = Pall[:, :, 1, :].swapaxes(0, 1)
        P2 = Pall[:, :, 2, :].swapaxes(0, 1)
    # uP2: x坐标乘上P2: (nJoints, nViews, 1, 4)
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = np.hstack([Au, Av])
    u, s, v = np.linalg.svd(A)
    X = v[:, -1, :]
    X = X / X[:, 3:]
    # out: (nJoints, 4)
    result = np.zeros((keypoints_.shape[1], 4))
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d #* (conf[..., 0].sum(axis=-1)>min_view)
    return result
    
eps = 0.01
def simple_recon_person(keypoints_use, Puse):
    out = batch_triangulate(keypoints_use, Puse)
    # compute reprojection error
    kpts_repro = projectN3(out, Puse)
    square_diff = (keypoints_use[:, :, :2] - kpts_repro[:, :, :2])**2 
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
        # 计算骨长
        l_est = np.linalg.norm(keypoints3d[src, :3] - keypoints3d[dst, :3])
        if abs(l_est - val['mean'])/val['mean']/val['std'] > thres:
            valid = False
            break
    # 至少两段骨头可以使用
    valid = valid and cnt > 2
    return valid
