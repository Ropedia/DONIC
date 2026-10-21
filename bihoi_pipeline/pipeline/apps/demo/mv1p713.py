'''
  @ Date: 2021-04-13 19:46:51
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-13 17:56:25
  @ FilePath: /EasyMocap/apps/demo/mv1p.py

This version adds joint-level Kalman filtering after triangulation.
The Kalman manager is created once before the frame loop, so its state is
kept across frames, similar to PoseRefinePredictor.pose_kalman_states.
'''
from tqdm import tqdm
from easymocap.smplmodel import check_keypoints, load_model, select_nf
from easymocap.mytools import (
    simple_recon_person,
    Timer,
    projectN3,
    JointKalmanManager,
)
from easymocap.pipeline import smpl_from_keypoints3d2d,smpl_from_keypoints3d
import os
import json
from glob import glob
from os.path import join, isabs
import numpy as np

import numpy as np


# MediaPipe / common 21-hand format:
# 0 wrist
# thumb:  1-2-3-4
# index:  5-6-7-8
# middle: 9-10-11-12
# ring:   13-14-15-16
# pinky:  17-18-19-20
HAND21_PARENTS = np.array([
    -1,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19,
], dtype=np.int64)


def estimate_canonical_bone_lengths(
    kp3ds,
    parents=HAND21_PARENTS,
    conf_thres=0.1,
    min_samples=5,
):
    """
    Estimate fixed bone lengths from a sequence of 3D keypoints.

    Args:
        kp3ds: list of (nJoints, 4), xyz + confidence
        parents: (nJoints,), parent index for each joint, -1 for root
        conf_thres: only use joints with confidence larger than this
        min_samples: minimum valid samples required for one bone

    Returns:
        target_lengths: (nJoints,), target_lengths[j] is length(parent[j] -> j)
    """
    n_joints = len(parents)
    all_lengths = [[] for _ in range(n_joints)]

    for keypoints3d in kp3ds:
        xyz = keypoints3d[:, :3]
        conf = keypoints3d[:, 3]

        for child in range(n_joints):
            parent = parents[child]
            if parent < 0:
                continue

            if conf[parent] <= conf_thres or conf[child] <= conf_thres:
                continue

            length = np.linalg.norm(xyz[child] - xyz[parent])
            if np.isfinite(length) and length > 1e-8:
                all_lengths[child].append(length)

    target_lengths = np.full((n_joints,), np.nan, dtype=np.float64)

    for child in range(n_joints):
        if len(all_lengths[child]) >= min_samples:
            target_lengths[child] = np.median(all_lengths[child])

    return target_lengths


def enforce_bone_lengths_one_hand(
    keypoints3d,
    target_lengths,
    parents=HAND21_PARENTS,
    conf_thres=0.1,
    blend=1.0,
):
    """
    Enforce fixed parent-child bone lengths for one 21-joint hand.

    Args:
        keypoints3d: (21, 4), xyz + confidence
        target_lengths: (21,), fixed bone lengths
        parents: (21,), parent index for each joint
        conf_thres: only constrain valid parent-child pairs
        blend:
            1.0 means hard constraint;
            0.5 means halfway between original and constrained position.

    Returns:
        constrained: (21, 4)
    """
    constrained = keypoints3d.copy()
    raw_xyz = keypoints3d[:, :3].copy()
    conf = keypoints3d[:, 3]

    for child in range(len(parents)):
        parent = parents[child]
        if parent < 0:
            continue

        target_len = target_lengths[child]
        if not np.isfinite(target_len) or target_len <= 1e-8:
            continue

        if conf[parent] <= conf_thres or conf[child] <= conf_thres:
            continue

        direction = raw_xyz[child] - raw_xyz[parent]
        norm = np.linalg.norm(direction)

        if not np.isfinite(norm) or norm <= 1e-8:
            continue

        direction = direction / norm

        parent_pos = constrained[parent, :3]
        constrained_pos = parent_pos + direction * target_len

        constrained[child, :3] = (
            (1.0 - blend) * constrained[child, :3]
            + blend * constrained_pos
        )

    return constrained

HAND21_PARENTS = np.array([
    -1,  # 0 wrist
     0, 1, 2, 3,        # thumb: 0-1-2-3-4
     0, 5, 6, 7,        # index: 0-5-6-7-8
     0, 9, 10, 11,      # middle: 0-9-10-11-12
     0, 13, 14, 15,     # ring: 0-13-14-15-16
     0, 17, 18, 19      # pinky: 0-17-18-19-20
], dtype=np.int64)


# Packaging: private measured bone-length constants omitted. The original calibration
# execution has use_bone_length_constraint=False; active triangulation is unchanged.
FIXED_HAND21_BONE_LENGTHS = None


def apply_hand_bone_length_constraint_sequence(
    kp3ds,
    hand_indices=None,
    parents=HAND21_PARENTS,
    target_lengths=FIXED_HAND21_BONE_LENGTHS,
    conf_thres=0.1,
    blend=1.0,
):
    """
    Apply fixed hand bone length constraint to a full triangulated 3D keypoint sequence.

    This version does NOT estimate bone lengths from the sequence.
    It directly uses fixed target bone lengths computed from your reference 3D hand file.

    Args:
        kp3ds:
            list of arrays, each shape (nJoints, 4).
            Each keypoint is [x, y, z, confidence].

        hand_indices:
            None:
                assume kp3ds are exactly 21 hand joints.
            list/array of length 21:
                constrain only this hand subset.
            list of lists:
                constrain multiple hands, e.g. [left_indices, right_indices].

        parents:
            local 21-joint hand parent array.

        target_lengths:
            fixed bone lengths, shape (21,).
            target_lengths[child] is the length of parent -> child.

        conf_thres:
            confidence threshold.
            If parent or child confidence is too low, this bone can be skipped
            depending on enforce_bone_lengths_one_hand implementation.

        blend:
            1.0 means hard constraint.
            0.5 means move halfway toward fixed bone length.

    Returns:
        constrained_kp3ds:
            list of arrays, same shape as input.
    """

    if len(kp3ds) == 0:
        return kp3ds

    n_joints = kp3ds[0].shape[0]

    if hand_indices is None:
        if n_joints != 21:
            raise ValueError(
                f"hand_indices is None, but n_joints={n_joints}. "
                f"Please provide 21 hand joint indices."
            )
        hand_groups = [np.arange(21)]
    else:
        if isinstance(hand_indices, (list, tuple)) and len(hand_indices) > 0:
            first = hand_indices[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                hand_groups = [np.asarray(x, dtype=np.int64) for x in hand_indices]
            else:
                hand_groups = [np.asarray(hand_indices, dtype=np.int64)]
        else:
            hand_groups = [np.asarray(hand_indices, dtype=np.int64)]

    target_lengths = np.asarray(target_lengths, dtype=np.float32)

    if target_lengths.shape[0] != 21:
        raise ValueError(
            f"target_lengths must have shape (21,), got {target_lengths.shape}"
        )

    constrained_kp3ds = [x.copy() for x in kp3ds]

    print("[BoneLength] using fixed target lengths:")
    for child in range(len(parents)):
        parent = parents[child]
        if parent >= 0:
            print(
                f"  parent {parent:02d} -> child {child:02d}: "
                f"{target_lengths[child]:.10f}"
            )

    for group in hand_groups:
        if len(group) != 21:
            raise ValueError(f"Each hand group must have 21 joints, got {len(group)}.")

        for i in range(len(kp3ds)):
            hand_constrained = enforce_bone_lengths_one_hand(
                constrained_kp3ds[i][group],
                target_lengths=target_lengths,
                parents=parents,
                conf_thres=conf_thres,
                blend=blend,
            )

            constrained_kp3ds[i][group] = hand_constrained

    return constrained_kp3ds
    
    
    
    
    
    
def check_repro_error(keypoints3d, kpts_repro, keypoints2d, P, MAX_REPRO_ERROR):
    square_diff = (keypoints2d[:, :, :2] - kpts_repro[:, :, :2])**2 
    conf = keypoints3d[None, :, -1:]
    conf = (keypoints3d[None, :, -1:] > 0) * (keypoints2d[:, :, -1:] > 0)
    dist = np.sqrt((((kpts_repro[..., :2] - keypoints2d[..., :2])*conf)**2).sum(axis=-1))
    vv, jj = np.where(dist > MAX_REPRO_ERROR)
    if vv.shape[0] > 0:
        keypoints2d[vv, jj, -1] = 0.
        keypoints3d, kpts_repro = simple_recon_person(keypoints2d, P)
    return keypoints3d, kpts_repro
import numpy as np
from itertools import combinations


def _get_P_for_joint(Pall, joint_id):
    """
    Support:
        Pall: (nViews, 3, 4)
        Pall: (nViews, nJoints, 3, 4)

    Return:
        Ps: (nViews, 3, 4)
    """
    Pall = np.asarray(Pall)

    if Pall.ndim == 3:
        return Pall
    elif Pall.ndim == 4:
        return Pall[:, joint_id]
    else:
        raise ValueError(f"Unsupported Pall shape: {Pall.shape}")


def _triangulate_one_point_dlt(kpts2d, Ps):
    """
    Temporary weighted DLT triangulation.

    Args:
        kpts2d: (nUsedViews, 3), [u, v, conf]
        Ps: (nUsedViews, 3, 4)

    Returns:
        X: (3,) or None
    """
    A = []

    for kpt, P in zip(kpts2d, Ps):
        u, v, c = kpt

        if c <= 0:
            continue

        # Use confidence as weight.
        w = float(c)

        A.append(w * (u * P[2, :] - P[0, :]))
        A.append(w * (v * P[2, :] - P[1, :]))

    if len(A) < 4:
        return None

    A = np.asarray(A, dtype=np.float64)

    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None

    X_h = Vt[-1]

    if abs(X_h[3]) < 1e-12:
        return None

    X = X_h[:3] / X_h[3]
    return X


def _project_point(P, X):
    """
    Project 3D point to one camera.

    Args:
        P: (3, 4)
        X: (3,)

    Returns:
        uv: (2,) or None
    """
    X_h = np.array([X[0], X[1], X[2], 1.0], dtype=np.float64)
    x = P @ X_h

    if abs(x[2]) < 1e-12:
        return None

    uv = x[:2] / x[2]
    return uv


def _triangulate_joint_all_views_remove_outliers(
    kpts_j,
    Ps,
    valid_views,
    repro_thres=25.0,
    min_view=2,
):
    """
    Rescue one joint when no 2-view pair/group reaches the requested number
    of inliers.

    Procedure:
      1. triangulate this joint once from all currently valid views;
      2. reproject the temporary 3D point to all those views;
      3. remove views whose reprojection error is larger than repro_thres;
      4. triangulate again from the remaining views;
      5. do one final reprojection check and optionally prune once more.

    Args:
        kpts_j: (nViews, 3), [u, v, conf] for one joint.
        Ps: (nViews, 3, 4), camera projection matrices for this joint.
        valid_views: view ids whose confidence is already valid.
        repro_thres: maximum allowed reprojection error in pixels.
        min_view: minimum number of views required for DLT.

    Returns:
        dict with X, inliers, mean_err, max_err, conf_sum; or None.
    """
    valid_views = np.asarray(valid_views, dtype=np.int64)
    if len(valid_views) < int(min_view):
        return None

    # 1) Coarse triangulation from all valid views.
    X0 = _triangulate_one_point_dlt(
        kpts2d=kpts_j[valid_views],
        Ps=Ps[valid_views],
    )
    if X0 is None:
        return None

    # 2) Reproject to all valid views and remove outliers.
    inliers = []
    for v in valid_views:
        uv_proj = _project_point(Ps[v], X0)
        if uv_proj is None:
            continue
        uv_obs = kpts_j[v, :2]
        err = float(np.linalg.norm(uv_proj - uv_obs))
        if np.isfinite(err) and err <= float(repro_thres):
            inliers.append(int(v))

    inliers = np.asarray(inliers, dtype=np.int64)
    if len(inliers) < int(min_view):
        return None

    # 3) Refine triangulation from surviving views.
    X1 = _triangulate_one_point_dlt(
        kpts2d=kpts_j[inliers],
        Ps=Ps[inliers],
    )
    if X1 is None:
        return None

    # 4) Final reprojection check. After re-triangulation, a view can become
    # bad again; prune once more and re-triangulate if possible.
    final_inliers = []
    final_errors = []
    for v in inliers:
        uv_proj = _project_point(Ps[v], X1)
        if uv_proj is None:
            continue
        uv_obs = kpts_j[v, :2]
        err = float(np.linalg.norm(uv_proj - uv_obs))
        if np.isfinite(err) and err <= float(repro_thres):
            final_inliers.append(int(v))
            final_errors.append(err)

    final_inliers = np.asarray(final_inliers, dtype=np.int64)
    if len(final_inliers) < int(min_view):
        return None

    # Optional second refinement after final pruning.
    if len(final_inliers) != len(inliers) or not np.array_equal(final_inliers, inliers):
        X2 = _triangulate_one_point_dlt(
            kpts2d=kpts_j[final_inliers],
            Ps=Ps[final_inliers],
        )
        if X2 is not None:
            checked_errors = []
            checked_inliers = []
            for v in final_inliers:
                uv_proj = _project_point(Ps[v], X2)
                if uv_proj is None:
                    continue
                uv_obs = kpts_j[v, :2]
                err = float(np.linalg.norm(uv_proj - uv_obs))
                if np.isfinite(err) and err <= float(repro_thres):
                    checked_inliers.append(int(v))
                    checked_errors.append(err)
            checked_inliers = np.asarray(checked_inliers, dtype=np.int64)
            if len(checked_inliers) >= int(min_view):
                X1 = X2
                final_inliers = checked_inliers
                final_errors = checked_errors

    if len(final_errors) == 0:
        # This can happen only in a rare numerical branch; recompute for safety.
        final_errors = []
        for v in final_inliers:
            uv_proj = _project_point(Ps[v], X1)
            if uv_proj is None:
                continue
            final_errors.append(float(np.linalg.norm(uv_proj - kpts_j[v, :2])))
        if len(final_errors) == 0:
            return None

    return {
        "X": np.asarray(X1, dtype=np.float64),
        "inliers": final_inliers.copy(),
        "mean_err": float(np.mean(final_errors)),
        "max_err": float(np.max(final_errors)),
        "conf_sum": float(np.sum(kpts_j[final_inliers, 2])),
        "conf_mean": float(np.mean(kpts_j[final_inliers, 2])),
    }


def select_effective_2d_points_by_best_pairs(
    keypoints2d,
    Pall,
    conf_thres=0.7,
    repro_thres=25.0,
    min_view=2,
    prefer_min_inliers=3,
    max_candidate_views=8,
    print_each_joint=True,
    frame_id=None,
):
    """
    Select effective 2D keypoints before final triangulation.

    Normal path:
        1. collect views whose confidence > conf_thres
        2. enumerate all 2-view pairs among top-confidence candidate views
        3. temporarily triangulate one 3D point from each pair
        4. project this temporary 3D point back to candidate views
        5. count inliers whose reprojection error <= repro_thres
        6. choose the best pair/group only if it has >= prefer_min_inliers
        7. keep only inlier views, set other views' confidence to 0

    Rescue path when every pair/group has < prefer_min_inliers:
        1. triangulate this joint using all valid views of the current frame
        2. reproject the temporary 3D point to all valid views
        3. remove 2D keypoints whose reprojection error exceeds repro_thres
        4. triangulate again using the remaining views
        5. store this 3D rescue measurement in info['fallback_measurements_3d']
           so the caller can update this joint with Kalman for the current frame

    Returns:
        filtered_keypoints2d:
            same shape as input. Bad 2D points have confidence set to 0.
        info:
            dict containing selected pair/inliers and possible all-view rescue
            3D measurements for each joint.
    """

    keypoints2d = np.asarray(keypoints2d, dtype=np.float64)
    filtered = keypoints2d.copy()

    n_views, n_joints, _ = filtered.shape

    info = {
        "selected_pairs": {},
        "inlier_views": {},
        "num_inliers": {},
        "mean_repro_error": {},
        "max_repro_error": {},
        "selection_method": {},
        "fallback_measurements_3d": {},
        "fallback_confidence": {},
    }

    prefix = f"[frame {frame_id}] " if frame_id is not None else ""
    required_inliers = max(int(min_view), int(prefer_min_inliers))

    for j in range(n_joints):
        Ps = _get_P_for_joint(Pall, j)
        kpts_j = filtered[:, j, :]
        conf = kpts_j[:, 2]

        # All candidate views after check_keypoints/confidence threshold.
        # This set is used by the all-view rescue branch.
        valid_views_all = np.where(conf > conf_thres)[0]
        valid_views_all = valid_views_all[np.argsort(conf[valid_views_all])[::-1]]

        if len(valid_views_all) < int(min_view):
            filtered[:, j, 2] = 0.0

            info["selected_pairs"][j] = None
            info["inlier_views"][j] = []
            info["num_inliers"][j] = 0
            info["mean_repro_error"][j] = None
            info["max_repro_error"][j] = None
            info["selection_method"][j] = "invalid_not_enough_views"
            info["fallback_measurements_3d"][j] = None
            info["fallback_confidence"][j] = 0.0

            if print_each_joint:
                print(
                    f"{prefix}[select2d] joint {j:02d}: "
                    f"not enough candidate views, valid={valid_views_all.tolist()}"
                )
            continue

        # Use only top-N confidence views to avoid too many pair combinations.
        pair_candidate_views = valid_views_all[:int(max_candidate_views)]

        best = None

        for pair in combinations(pair_candidate_views, 2):
            pair = np.asarray(pair, dtype=np.int64)

            # Temporary 2-view triangulation hypothesis.
            X = _triangulate_one_point_dlt(
                kpts2d=kpts_j[pair],
                Ps=Ps[pair],
            )
            if X is None:
                continue

            inliers = []

            # Test this temporary 3D point against candidate views.
            for v in pair_candidate_views:
                uv_proj = _project_point(Ps[v], X)
                if uv_proj is None:
                    continue

                uv_obs = kpts_j[v, :2]
                err = float(np.linalg.norm(uv_proj - uv_obs))
                if np.isfinite(err) and err <= float(repro_thres):
                    inliers.append(int(v))

            if len(inliers) < required_inliers:
                continue

            inliers = np.asarray(inliers, dtype=np.int64)

            # Refine temporary 3D point using all inliers.
            X_refined = _triangulate_one_point_dlt(
                kpts2d=kpts_j[inliers],
                Ps=Ps[inliers],
            )
            if X_refined is None:
                continue

            final_errors = []
            final_inliers = []
            for v in inliers:
                uv_proj = _project_point(Ps[v], X_refined)
                if uv_proj is None:
                    continue

                uv_obs = kpts_j[v, :2]
                err = float(np.linalg.norm(uv_proj - uv_obs))
                if np.isfinite(err) and err <= float(repro_thres):
                    final_inliers.append(int(v))
                    final_errors.append(err)

            if len(final_inliers) < required_inliers or len(final_errors) == 0:
                continue

            final_inliers = np.asarray(final_inliers, dtype=np.int64)
            mean_err = float(np.mean(final_errors))
            max_err = float(np.max(final_errors))
            conf_sum = float(np.sum(conf[final_inliers]))

            # Scoring rule:
            # 1. Prefer more inliers.
            # 2. Prefer smaller mean reprojection error.
            # 3. Prefer smaller max reprojection error.
            # 4. Prefer larger confidence sum.
            score = (
                len(final_inliers),
                -mean_err,
                -max_err,
                conf_sum,
            )

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "pair": pair.copy(),
                    "inliers": final_inliers.copy(),
                    "mean_err": mean_err,
                    "max_err": max_err,
                    "conf_sum": conf_sum,
                    "conf_mean": float(np.mean(conf[final_inliers])),
                    "X": X_refined,
                }

        if best is None:
            # No pair/group reaches required_inliers. Do not immediately clear
            # this joint. First try an all-view rescue measurement for this
            # current frame.
            rescue = _triangulate_joint_all_views_remove_outliers(
                kpts_j=kpts_j,
                Ps=Ps,
                valid_views=valid_views_all,
                repro_thres=repro_thres,
                min_view=min_view,
            )

            if rescue is None:
                filtered[:, j, 2] = 0.0

                info["selected_pairs"][j] = None
                info["inlier_views"][j] = []
                info["num_inliers"][j] = 0
                info["mean_repro_error"][j] = None
                info["max_repro_error"][j] = None
                info["selection_method"][j] = "invalid_no_pair_no_rescue"
                info["fallback_measurements_3d"][j] = None
                info["fallback_confidence"][j] = 0.0

                if print_each_joint:
                    print(
                        f"{prefix}[select2d] joint {j:02d}: "
                        f"no reliable pair with >= {required_inliers} inliers, "
                        f"all-view rescue failed, candidates={valid_views_all.tolist()}"
                    )
                continue

            keep_views = rescue["inliers"]
            filtered[:, j, 2] = 0.0
            filtered[keep_views, j, 2] = keypoints2d[keep_views, j, 2]

            info["selected_pairs"][j] = None
            info["inlier_views"][j] = keep_views.tolist()
            info["num_inliers"][j] = int(len(keep_views))
            info["mean_repro_error"][j] = float(rescue["mean_err"])
            info["max_repro_error"][j] = float(rescue["max_err"])
            info["selection_method"][j] = "all_view_rescue"
            info["fallback_measurements_3d"][j] = rescue["X"].copy()
            info["fallback_confidence"][j] = float(rescue["conf_mean"])

            if print_each_joint:
                print(
                    f"{prefix}[select2d] joint {j:02d}: "
                    f"no pair >= {required_inliers}; all-view rescue used, "
                    f"inliers={len(keep_views)}, "
                    f"inlier_views={keep_views.tolist()}, "
                    f"mean_err={rescue['mean_err']:.2f}px, "
                    f"max_err={rescue['max_err']:.2f}px"
                )
            continue

        keep_views = best["inliers"]

        # Clear all views for this joint and keep only selected inlier views.
        filtered[:, j, 2] = 0.0
        filtered[keep_views, j, 2] = keypoints2d[keep_views, j, 2]

        info["selected_pairs"][j] = best["pair"].tolist()
        info["inlier_views"][j] = keep_views.tolist()
        info["num_inliers"][j] = int(len(keep_views))
        info["mean_repro_error"][j] = float(best["mean_err"])
        info["max_repro_error"][j] = float(best["max_err"])
        info["selection_method"][j] = "best_pair"
        info["fallback_measurements_3d"][j] = None
        info["fallback_confidence"][j] = 0.0

        if print_each_joint:
            print(
                f"{prefix}[select2d] joint {j:02d}: "
                f"best_pair={best['pair'].tolist()}, "
                f"inliers={len(keep_views)}, "
                f"inlier_views={keep_views.tolist()}, "
                f"mean_err={best['mean_err']:.2f}px, "
                f"max_err={best['max_err']:.2f}px"
            )

    return filtered, info


class RawPredictionJointKalmanManager:
    """
    Per-joint constant-velocity Kalman filter for fallback-only usage.

    Important behavior for this file:
      - If a joint has a reliable triangulated measurement in the current frame,
        the returned 3D result keeps that raw triangulated measurement unchanged.
        The filter is only updated internally for future tracking.
      - If a joint has no selected clean 2D pair/inliers in the current frame,
        the returned 3D result uses the filter's raw prediction, if that joint
        has already been initialized by previous valid measurements.

    State per joint: [x, y, z, vx, vy, vz].
    """

    def __init__(
        self,
        n_joints,
        measurement_noise_scale=5.0,
        process_noise_scale=1.0,
        init_cov_scale=1.0,
        max_jump=None,
        reset_after_missing=10,
        prediction_conf=0.05,
    ):
        self.n_joints = int(n_joints)
        self.measurement_noise_scale = float(measurement_noise_scale)
        self.process_noise_scale = float(process_noise_scale)
        self.init_cov_scale = float(init_cov_scale)
        self.max_jump = max_jump
        self.reset_after_missing = int(reset_after_missing)
        self.prediction_conf = float(prediction_conf)

        self.x = np.zeros((self.n_joints, 6), dtype=np.float64)
        self.P = np.zeros((self.n_joints, 6, 6), dtype=np.float64)
        self.initialized = np.zeros((self.n_joints,), dtype=bool)
        self.missing_count = np.zeros((self.n_joints,), dtype=np.int64)

        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = 1.0
        self.F[1, 4] = 1.0
        self.F[2, 5] = 1.0

        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # These are deliberately simple/conservative. If your 3D unit is meter,
        # you may want smaller values, e.g. measurement_noise=0.005~0.02.
        q = max(self.process_noise_scale, 1e-12)
        r = max(self.measurement_noise_scale, 1e-12)
        self.Q = (q ** 2) * np.eye(6, dtype=np.float64)
        self.R = (r ** 2) * np.eye(3, dtype=np.float64)

    def _is_valid_xyz(self, xyz):
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        return np.all(np.isfinite(xyz))

    def _init_joint(self, j, xyz):
        self.x[j, :3] = xyz
        self.x[j, 3:] = 0.0
        self.P[j] = self.init_cov_scale * np.eye(6, dtype=np.float64)
        self.initialized[j] = True
        self.missing_count[j] = 0

    def predict_only(self, j):
        """Advance one frame and return raw predicted xyz for joint j."""
        j = int(j)
        if not self.initialized[j]:
            return None

        self.x[j] = self.F @ self.x[j]
        self.P[j] = self.F @ self.P[j] @ self.F.T + self.Q
        self.missing_count[j] += 1

        if self.reset_after_missing is not None and self.missing_count[j] > self.reset_after_missing:
            self.initialized[j] = False
            return None

        return self.x[j, :3].copy()

    def update_with_measurement(self, j, xyz):
        """
        Advance one frame and update with the raw triangulated measurement.
        Return the filtered estimate, but mv1pmf_skel intentionally does NOT use
        this value as output when measurement is valid.
        """
        j = int(j)
        z = np.asarray(xyz, dtype=np.float64).reshape(3)
        if not self._is_valid_xyz(z):
            return None

        if not self.initialized[j]:
            self._init_joint(j, z)
            return self.x[j, :3].copy()

        # Optional jump gate. If the measurement suddenly jumps too far, keep
        # the raw output for this frame but do not corrupt the filter state.
        if self.max_jump is not None:
            pred_for_gate = self.F @ self.x[j]
            jump = float(np.linalg.norm(z - pred_for_gate[:3]))
            if jump > float(self.max_jump):
                # Advance as missing, but let caller still keep raw measurement.
                return self.predict_only(j)

        # Predict.
        self.x[j] = self.F @ self.x[j]
        self.P[j] = self.F @ self.P[j] @ self.F.T + self.Q

        # Update.
        y = z - (self.H @ self.x[j])
        S = self.H @ self.P[j] @ self.H.T + self.R
        try:
            K = self.P[j] @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P[j] @ self.H.T @ np.linalg.pinv(S)
        self.x[j] = self.x[j] + K @ y
        I = np.eye(6, dtype=np.float64)
        self.P[j] = (I - K @ self.H) @ self.P[j]
        self.missing_count[j] = 0
        return self.x[j, :3].copy()


def _selected_pair_valid_mask(select_info, n_joints, min_inliers=2):
    """
    Convert select_effective_2d_points_by_best_pairs() info to a per-joint mask.

    True means this joint has a valid selected clean 2D pair/inlier set in the
    current frame, so the triangulated 3D measurement should be trusted as the
    output and the Kalman filter should only be updated internally.
    """
    mask = np.zeros((n_joints,), dtype=bool)
    for j in range(n_joints):
        pair = select_info.get('selected_pairs', {}).get(j, None)
        num_inliers = int(select_info.get('num_inliers', {}).get(j, 0) or 0)
        if pair is not None and num_inliers >= int(min_inliers):
            mask[j] = True
    return mask


def _project_keypoints3d_to_views(keypoints3d, Pall):
    """
    Project final 3D keypoints to all camera views for visualization.

    Args:
        keypoints3d: (nJoints, 4), [x, y, z, conf]
        Pall: (nViews, 3, 4) or (nViews, nJoints, 3, 4)

    Returns:
        kpts_repro: (nViews, nJoints, 3), [u, v, conf]
    """
    keypoints3d = np.asarray(keypoints3d, dtype=np.float64)
    Pall = np.asarray(Pall, dtype=np.float64)
    n_joints = keypoints3d.shape[0]
    n_views = Pall.shape[0]
    out = np.zeros((n_views, n_joints, 3), dtype=np.float64)

    for v in range(n_views):
        for j in range(n_joints):
            if keypoints3d[j, 3] <= 0:
                continue
            P = Pall[v] if Pall.ndim == 3 else Pall[v, j]
            X = np.array([
                keypoints3d[j, 0],
                keypoints3d[j, 1],
                keypoints3d[j, 2],
                1.0,
            ], dtype=np.float64)
            x = P @ X
            if abs(x[2]) < 1e-12:
                continue
            out[v, j, :2] = x[:2] / x[2]
            out[v, j, 2] = keypoints3d[j, 3]
    return out


def apply_kalman_prediction_fallback_per_joint(
    keypoints3d,
    select_info,
    kalman_manager,
    min_inliers=2,
    prediction_conf=0.05,
    print_summary=True,
    frame_id=None,
):
    """
    Per-joint Kalman handling after 2D filtering and triangulation.

    Rules:
      1. best_pair joint + valid triangulation:
           keep raw triangulated output, only update Kalman internally.
      2. all_view_rescue joint:
           use the all-view rescue 3D measurement to update Kalman, and write
           the Kalman-updated estimate back to this frame's keypoint result.
           If the filter is not initialized yet, this equals the measurement.
      3. invalid joint:
           use Kalman raw prediction if available; otherwise set confidence 0.
    """
    if kalman_manager is None:
        return keypoints3d

    out = np.asarray(keypoints3d, dtype=np.float64).copy()
    n_joints = out.shape[0]
    selected_mask = _selected_pair_valid_mask(select_info, n_joints, min_inliers=min_inliers)
    triangulated_valid = out[:, 3] > 0
    measurement_mask = selected_mask & triangulated_valid

    updated_joints = []
    rescue_updated_joints = []
    predicted_joints = []
    invalid_joints = []

    methods = select_info.get("selection_method", {})
    rescue_measurements = select_info.get("fallback_measurements_3d", {})
    rescue_confidence = select_info.get("fallback_confidence", {})

    for j in range(n_joints):
        method = methods.get(j, None)

        if measurement_mask[j]:
            # Clean best-pair measurement: keep raw triangulated output, only
            # update Kalman state internally for future frames.
            kalman_manager.update_with_measurement(j, out[j, :3])
            updated_joints.append(j)
            continue

        if method == "all_view_rescue":
            # Rescue measurement: update Kalman and write the filtered/current
            # estimate back to this frame. This is the branch requested for
            # joints whose every pair/group has fewer than min_inliers.
            rescue_xyz = rescue_measurements.get(j, None)
            if rescue_xyz is None:
                # Fall back to simple_recon_person result if it succeeded from
                # the rescued 2D inlier views.
                rescue_xyz = out[j, :3] if out[j, 3] > 0 else None

            if rescue_xyz is not None and np.all(np.isfinite(rescue_xyz)):
                filtered_xyz = kalman_manager.update_with_measurement(j, rescue_xyz)
                if filtered_xyz is not None and np.all(np.isfinite(filtered_xyz)):
                    out[j, :3] = filtered_xyz
                else:
                    out[j, :3] = np.asarray(rescue_xyz, dtype=np.float64).reshape(3)

                # Keep a meaningful confidence. It is still a measurement, not
                # a pure prediction, so do not force it to prediction_conf.
                conf = float(rescue_confidence.get(j, 0.0) or 0.0)
                if conf <= 0 and out[j, 3] > 0:
                    conf = float(out[j, 3])
                if conf <= 0:
                    conf = float(prediction_conf)
                out[j, 3] = conf
                rescue_updated_joints.append(j)
                continue

        # No reliable current measurement: use Kalman prediction if possible.
        pred = kalman_manager.predict_only(j)
        if pred is not None and np.all(np.isfinite(pred)):
            out[j, :3] = pred
            out[j, 3] = float(prediction_conf)
            predicted_joints.append(j)
        else:
            out[j, :] = 0.0
            invalid_joints.append(j)

    if print_summary:
        prefix = f"[frame {frame_id}] " if frame_id is not None else ""
        print(
            f"{prefix}[kalman-fallback] "
            f"raw_update={len(updated_joints)} joints, "
            f"all_view_rescue_update={len(rescue_updated_joints)} joints, "
            f"prediction_used={len(predicted_joints)} joints, "
            f"invalid_no_state={len(invalid_joints)} joints"
        )
        if len(rescue_updated_joints) > 0:
            print(f"{prefix}[kalman-fallback] all-view rescue joints: {rescue_updated_joints}")
        if len(predicted_joints) > 0:
            print(f"{prefix}[kalman-fallback] prediction joints: {predicted_joints}")
        if len(invalid_joints) > 0:
            print(f"{prefix}[kalman-fallback] invalid joints: {invalid_joints}")

    return out



def _fallback_annotation_root(dataset, args):
    """Return the absolute root of ``annots_fallback`` for this dataset."""
    fallback_annot = getattr(args, 'fallback2d_annot', 'annots_fallback')
    if isabs(fallback_annot):
        return fallback_annot
    dataset_root = getattr(dataset, 'root', None)
    if dataset_root is None:
        dataset_root = getattr(args, 'path', None)
    if dataset_root is None:
        raise RuntimeError(
            'Cannot resolve fallback annotation root: dataset.root and args.path are both unavailable.'
        )
    return join(dataset_root, fallback_annot)


def _fallback_json_index_for_camera(fallback_root, cam_id, index_cache):
    """
    Build one {frame_number: json_path} index per fallback camera folder.

    Besides avoiding a glob for every frame, this supports normal EasyMocap
    names such as ``000123.json`` and unpadded names such as ``123.json``.
    """
    cam_id = str(cam_id)
    cache_key = (fallback_root, cam_id)
    if cache_key in index_cache:
        return index_cache[cache_key]

    cam_dir = join(fallback_root, cam_id)
    frame_index = {}
    for filename in sorted(glob(join(cam_dir, '*.json'))):
        stem = os.path.splitext(os.path.basename(filename))[0]
        try:
            frame_index[int(stem)] = filename
        except ValueError:
            # Ignore non-numeric sidecar JSON files.
            continue
    index_cache[cache_key] = frame_index
    return frame_index


def _choose_fallback_person(record, pid):
    """Choose the same person id as MV1PMF from a fallback JSON payload."""
    if isinstance(record, list):
        if len(record) == 0:
            return None
        for item in record:
            if isinstance(item, dict):
                for id_key in ('id', 'personID', 'person_id', 'pid'):
                    if id_key in item:
                        try:
                            if int(item[id_key]) == int(pid):
                                return item
                        except (TypeError, ValueError):
                            pass
        # For a one-person hand annotation this is the standard case.
        if len(record) == 1:
            return record[0]
        # If no explicit id exists, MV1PMF's pid is also a reasonable list index.
        if isinstance(pid, (int, np.integer)) and 0 <= int(pid) < len(record):
            return record[int(pid)]
        return record[0]

    if not isinstance(record, dict):
        return record

    # Some exporters wrap detections inside one of these container keys.
    for container_key in ('annots', 'annotations', 'people', 'persons', 'instances', 'data'):
        value = record.get(container_key, None)
        if isinstance(value, list):
            return _choose_fallback_person(value, pid)

    return record


def _fallback_hand_field_from_args(args):
    """Return the correct EasyMocap 2D-hand field for the requested hand."""
    explicit = getattr(args, 'fallback2d_hand_field', None)
    if explicit not in (None, '', 'auto'):
        explicit = str(explicit)
        if explicit not in ('handl2d', 'handr2d'):
            raise ValueError(
                f'--fallback2d_hand_field must be handl2d, handr2d, or auto; got {explicit!r}'
            )
        return explicit

    body = str(getattr(args, 'body', '')).lower()
    model = str(getattr(args, 'model', '')).lower()
    tokens = f'{body} {model}'
    if 'handl' in tokens or 'manol' in tokens or 'left' in tokens:
        return 'handl2d'
    if 'handr' in tokens or 'manor' in tokens or 'right' in tokens:
        return 'handr2d'
    return None


def _extract_fallback_keypoints2d(payload, pid, n_joints, hand_field=None):
    """
    Extract a (n_joints, 3) [u, v, confidence] array from common JSON layouts.

    In particular, the normal EasyMocap hand JSON layout is::

        {"annots": [{"personID": 0, "handl2d": [...], "handr2d": [...]}]}

    ``hand_field`` selects the correct hand explicitly.  With ``--body handr``
    it is ``handr2d``; with ``--body handl`` it is ``handl2d``.
    """
    record = _choose_fallback_person(payload, pid)
    if record is None:
        return None

    candidate = None
    if isinstance(record, dict):
        candidate_keys = []
        if hand_field is not None:
            candidate_keys.append(hand_field)
        candidate_keys += [
            'keypoints', 'keypoints2d', 'keypoints_2d',
            'hand_keypoints', 'hand_keypoints2d', 'hand_keypoints_2d',
        ]

        # For unknown body/model settings, allow an unambiguous hand payload.
        if hand_field is None:
            candidate_keys += ['handr2d', 'handl2d']

        for key in candidate_keys:
            if key in record:
                candidate = record[key]
                break
    else:
        candidate = record

    if candidate is None:
        return None

    try:
        points = np.asarray(candidate, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    # Allow one extra singleton person dimension from some annotation exporters.
    if points.ndim == 3 and points.shape[0] == 1:
        points = points[0]

    if points.ndim != 2 or points.shape[0] < n_joints or points.shape[1] < 2:
        return None

    out = np.zeros((n_joints, 3), dtype=np.float64)
    out[:, :2] = points[:n_joints, :2]
    if points.shape[1] >= 3:
        out[:, 2] = points[:n_joints, 2]
    else:
        out[:, 2] = 1.0

    invalid = (
        ~np.isfinite(out[:, 0]) |
        ~np.isfinite(out[:, 1]) |
        ~np.isfinite(out[:, 2]) |
        (out[:, 2] <= 0)
    )
    out[invalid] = 0.0
    return out

def load_fallback_hand_keypoints2d(dataset, frame_id, n_joints, args, index_cache):
    """
    Load alternate hand 2D keypoints for **all active camera views**.

    Expected layout:
        <dataset_root>/annots_fallback/<cam_id>/<frame_id>.json

    ``frame_id`` is matched numerically, so both ``000123.json`` and
    ``123.json`` work. The returned array is aligned with ``dataset.Pall`` and
    therefore can be passed directly to ``simple_recon_person``.
    """
    fallback_root = _fallback_annotation_root(dataset, args)
    n_views = int(np.asarray(dataset.Pall).shape[0])
    cams = list(getattr(dataset, 'cams', []))
    if len(cams) != n_views:
        raise RuntimeError(
            f'Fallback 2D loader needs one camera id per projection matrix, '
            f'but got len(dataset.cams)={len(cams)} and n_views={n_views}.'
        )

    fallback2d = np.zeros((n_views, n_joints, 3), dtype=np.float64)
    hand_field = _fallback_hand_field_from_args(args)
    loaded_cams = []
    missing_files = []
    invalid_payloads = []

    for view_id, cam_id in enumerate(cams):
        json_index = _fallback_json_index_for_camera(fallback_root, cam_id, index_cache)
        filename = json_index.get(int(frame_id), None)
        if filename is None:
            missing_files.append(str(cam_id))
            continue

        try:
            with open(filename, 'r') as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            invalid_payloads.append(f'{cam_id} ({exc})')
            continue

        points = _extract_fallback_keypoints2d(
            payload=payload,
            pid=getattr(dataset, 'pid', 0),
            n_joints=n_joints,
            hand_field=hand_field,
        )
        if points is None:
            invalid_payloads.append(str(cam_id))
            continue

        fallback2d[view_id] = points
        loaded_cams.append(str(cam_id))

    info = {
        'fallback_root': fallback_root,
        'hand_field': hand_field,
        'loaded_cams': loaded_cams,
        'missing_files': missing_files,
        'invalid_payloads': invalid_payloads,
    }
    return fallback2d, info


def fill_missing_hand_joints_from_fallback2d(
    keypoints3d,
    keypoints2d_clean,
    dataset,
    frame_id,
    args,
    index_cache,
):
    """
    Re-triangulate only missing hand joints from ``annots_fallback``.

    A joint is considered missing when it has fewer than ``select2d_min_view``
    surviving cleaned 2D observations, or when the primary triangulation output
    itself is invalid. The fallback annotations are loaded from every camera and
    triangulated once. Valid fallback 3D measurements overwrite **only** the
    missing joints; valid primary measurements remain untouched.

    When ``--fallback2d_require_complete`` is active (the default), this raises
    a clear error instead of silently writing a frame with fewer than 21 valid
    hand joints.
    """
    out = np.asarray(keypoints3d, dtype=np.float64).copy()
    n_joints = out.shape[0]

    expected_joints = int(getattr(args, 'fallback2d_num_joints', 21))
    if n_joints != expected_joints:
        raise RuntimeError(
            f'Fallback hand reconstruction expects {expected_joints} joints, '
            f'but the primary annotations contain {n_joints}. '
            f'Use --fallback2d_num_joints {n_joints} only when the fallback JSON '
            f'uses exactly that same joint order and count.'
        )

    min_view = int(getattr(args, 'select2d_min_view', 2))
    clean_valid_count = (np.asarray(keypoints2d_clean)[..., 2] > 0).sum(axis=0)
    primary_valid = (
        np.isfinite(out[:, :3]).all(axis=1) &
        np.isfinite(out[:, 3]) &
        (out[:, 3] > 0)
    )
    missing_mask = (clean_valid_count < min_view) | (~primary_valid)
    missing_joint_ids = np.where(missing_mask)[0]

    if len(missing_joint_ids) == 0:
        return out, {
            'requested_joints': [],
            'recovered_joints': [],
            'still_missing_joints': [],
            'fallback_info': None,
        }

    fallback2d, fallback_info = load_fallback_hand_keypoints2d(
        dataset=dataset,
        frame_id=frame_id,
        n_joints=n_joints,
        args=args,
        index_cache=index_cache,
    )

    fallback_conf_thres = float(getattr(args, 'fallback2d_conf_thres', 0.0))
    if fallback_conf_thres > 0:
        fallback2d[fallback2d[..., 2] < fallback_conf_thres, 2] = 0.0

    # This is intentionally direct triangulation of the alternate points.
    # We do not run the strict best-pair cleaner again: the point of this branch
    # is to recover joints that the primary cleaner rejected.
    fallback3d, _ = simple_recon_person(
        fallback2d,
        dataset.Pall,
        kalman_manager=None,
    )
    fallback3d = np.asarray(fallback3d, dtype=np.float64)
    fallback_valid = (
        np.isfinite(fallback3d[:, :3]).all(axis=1) &
        np.isfinite(fallback3d[:, 3]) &
        (fallback3d[:, 3] > 0)
    )

    recovered_joint_ids = []
    for j in missing_joint_ids:
        if fallback_valid[j]:
            out[j] = fallback3d[j]
            recovered_joint_ids.append(int(j))

    still_missing = [int(j) for j in missing_joint_ids if not fallback_valid[j]]
    return out, {
        'requested_joints': [int(j) for j in missing_joint_ids],
        'recovered_joints': recovered_joint_ids,
        'still_missing_joints': still_missing,
        'fallback_info': fallback_info,
    }


def mv1pmf_skel(dataset, check_repro=True, args=None):
    no_img = not (args.vis_det or args.vis_repro)
    dataset.no_img = no_img
    kp3ds = []
    start, end = args.start, min(args.end, len(dataset))
    kpts_repro = None

    # Create a per-joint Kalman manager once before the frame loop.
    # It is used ONLY as fallback when a joint has no selected clean 2D pair.
    kalman_manager = None
    use_joint_kalman = getattr(args, 'use_joint_kalman', True)
    if use_joint_kalman:
        n_joints = dataset.config['nJoints'] if isinstance(dataset.config, dict) and 'nJoints' in dataset.config else None
        if n_joints is None:
            _, annots0 = dataset[start]
            n_joints = annots0['keypoints'].shape[1]
        kalman_manager = RawPredictionJointKalmanManager(
            n_joints=n_joints,
            measurement_noise_scale=getattr(args, 'joint_kalman_measurement_noise', 5.0),
            process_noise_scale=getattr(args, 'joint_kalman_process_noise', 1.0),
            init_cov_scale=getattr(args, 'joint_kalman_init_cov', 1.0),
            max_jump=getattr(args, 'joint_kalman_max_jump', None),
            reset_after_missing=getattr(args, 'joint_kalman_reset_after_missing', 10),
            prediction_conf=getattr(args, 'joint_kalman_prediction_conf', 0.05),
        )

    # Cache <frame number -> JSON file> maps once per fallback camera folder.
    fallback2d_index_cache = {}

    for nf in tqdm(range(start, end), desc='triangulation'):
        images, annots = dataset[nf]

        # 1) Confidence thresholding / hand confidence processing.
        check_keypoints(
            annots['keypoints'],
            WEIGHT_DEBUFF=1,
            min_conf=getattr(args, 'select2d_conf_thres', 0.),
        )

        # 2) Before final triangulation, select reliable 2D points by best pair
        # consensus. Joints with no reliable pair will later use Kalman prediction.
        keypoints2d_clean, select_info = select_effective_2d_points_by_best_pairs(
            annots['keypoints'],
            dataset.Pall,
            conf_thres=getattr(args, 'select2d_conf_thres', 0.4),
            repro_thres=getattr(args, 'select2d_repro_thres', 50.0),
            min_view=getattr(args, 'select2d_min_view', 2),
            prefer_min_inliers=getattr(args, 'select2d_prefer_min_inliers', 4),
            max_candidate_views=getattr(args, 'select2d_max_candidate_views', 16),
            print_each_joint=getattr(args, 'select2d_print_each_joint', True),
            frame_id=nf,
        )

        # 3) Final triangulation uses cleaned 2D points only.
        # Do NOT pass Kalman here, because valid joints must keep raw triangulated
        # measurements instead of smoothed Kalman outputs.
        keypoints3d, kpts_repro = simple_recon_person(
            keypoints2d_clean,
            dataset.Pall,
            kalman_manager=None,
        )

        # 4) Missing-joint fallback from alternate 2D hand annotations.
        # Only joints that cannot be triangulated from keypoints2d_clean are
        # replaced. All valid primary triangulations stay exactly unchanged.
        if getattr(args, 'use_fallback2d', True):
            keypoints3d, fallback_result = fill_missing_hand_joints_from_fallback2d(
                keypoints3d=keypoints3d,
                keypoints2d_clean=keypoints2d_clean,
                dataset=dataset,
                frame_id=nf,
                args=args,
                index_cache=fallback2d_index_cache,
            )

            requested = fallback_result['requested_joints']
            recovered = fallback_result['recovered_joints']
            still_missing = fallback_result['still_missing_joints']
            if len(requested) > 0:
                fallback_info = fallback_result['fallback_info']
                print(
                    f"[frame {nf}] [fallback2d] requested={requested}, "
                    f"recovered={recovered}, still_missing={still_missing}, "
                    f"loaded_cams={fallback_info['loaded_cams']}"
                )
                if len(fallback_info['missing_files']) > 0:
                    print(
                        f"[frame {nf}] [fallback2d] no fallback JSON for cams: "
                        f"{fallback_info['missing_files']}"
                    )
                if len(fallback_info['invalid_payloads']) > 0:
                    print(
                        f"[frame {nf}] [fallback2d] invalid fallback payloads: "
                        f"{fallback_info['invalid_payloads']}"
                    )

            if len(still_missing) > 0 and getattr(args, 'fallback2d_require_complete', True):
                fallback_root = _fallback_annotation_root(dataset, args)
                raise RuntimeError(
                    f'[frame {nf}] fallback triangulation still has missing hand joints '
                    f'{still_missing}. Expected valid 2D points in at least '
                    f'{getattr(args, "select2d_min_view", 2)} views under '
                    f'{fallback_root}/<cam_id>/{nf:06d}.json. '
                    f'Use --allow_fallback2d_missing only for debugging; it will write invalid joints.'
                )

        # 5) Per-joint Kalman fallback is intentionally not used here: this
        # pipeline must output true current-frame triangulations, not predicted
        # joints. The fallback2d branch above is the only recovery mechanism.

        # Recompute reprojection from the final result, because some joints may
        # have been recovered from annots_fallback after primary triangulation.
        kpts_repro = _project_keypoints3d_to_views(keypoints3d, dataset.Pall)

        kp3ds.append(keypoints3d)

        if args.vis_det:
            dataset.vis_detections(images, annots, nf, sub_vis=args.sub_vis)
        if args.vis_repro:
            dataset.vis_repro(images, kpts_repro, nf=nf, sub_vis=args.sub_vis)

    use_bone_length_constraint = False

    if use_bone_length_constraint:
        bone_length_conf_thres = getattr(args, 'bone_length_conf_thres', 0.1)
        bone_length_blend = 1

        # Case 1: your keypoints3d is exactly one 21-joint hand.
        hand_indices = None

        kp3ds = apply_hand_bone_length_constraint_sequence(
            kp3ds,
            hand_indices=hand_indices,
            parents=HAND21_PARENTS,
            conf_thres=bone_length_conf_thres,
            blend=bone_length_blend,
        )

    for nf in tqdm(range(len(kp3ds)), desc='dump'):
        dataset.write_keypoints3d(kp3ds[nf], nf + start)


def mv1pmf_smpl(dataset, args, weight_pose=None, weight_shape=None):
    dataset.skel_path = args.skel
    kp3ds = []
    start, end = args.start, min(args.end, len(dataset))
    keypoints2d, bboxes = [], []
    dataset.no_img = True
    for nf in tqdm(range(start, end), desc='loading'):
        images, annots = dataset[nf]
        keypoints2d.append(annots['keypoints'])
        bboxes.append(annots['bbox'])
    kp3ds = dataset.read_skeleton(start, end)
    keypoints2d = np.stack(keypoints2d)
    bboxes = np.stack(bboxes)
    kp3ds = check_keypoints(kp3ds, 1)
    # optimize the human shape
    with Timer('Loading {}, {}'.format(args.model, args.gender), not args.verbose):
        body_model = load_model(gender=args.gender, model_type=args.model)
    params = smpl_from_keypoints3d2d(body_model, kp3ds, keypoints2d, bboxes, 
        dataset.Pall, config=dataset.config, args=args,
        weight_shape=weight_shape, weight_pose=weight_pose)
    # write out the results
    dataset.no_img = not (args.vis_smpl or args.vis_repro)
    for nf in tqdm(range(start, end), desc='render'):
        images, annots = dataset[nf]
        param = select_nf(params, nf-start)
        dataset.write_smpl(param, nf)
        if args.write_smpl_full:
            param_full = param.copy()
            param_full['poses'] = body_model.full_poses(param['poses'])
            dataset.write_smpl(param_full, nf, mode='smpl_full')
        if args.write_vertices:
            vertices = body_model(return_verts=True, return_tensor=False, **param)
            write_data = [{'id': 0, 'vertices': vertices[0]}]
            dataset.write_vertices(write_data, nf)
        if args.vis_smpl:
            vertices = body_model(return_verts=True, return_tensor=False, **param)
            dataset.vis_smpl(vertices=vertices[0], faces=body_model.faces, images=images, nf=nf, sub_vis=args.sub_vis, add_back=True)
        if args.vis_repro:
            keypoints = body_model(return_verts=False, return_tensor=False, **param)[0]
            kpts_repro = projectN3(keypoints, dataset.Pall)
            dataset.vis_repro(images, kpts_repro, nf=nf, sub_vis=args.sub_vis, mode='repro_smpl')
    # compose rendered images into video
    if args.vis_smpl:
        compose_video(join(args.out, 'smpl'), join(args.out, 'smpl.mp4'))
    if args.vis_repro:
        compose_video(join(args.out, 'repro_smpl'), join(args.out, 'repro_smpl.mp4'))


def convert_subs_to_rpizero(subs):
    """
    Convert args.sub like ['1', '2', '4', '35'].
    Current rule keeps EasyMocap camera folder names as string indices.
    """
    converted = []
    for s in subs:
        idx = int(s) - 1
        if idx < 0:
            print(f'[WARN] invalid sub index: {s}, skipped.')
            continue
        converted.append(str(idx+1))
    return converted


if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser
    from easymocap.dataset import CONFIG, MV1PMF

    parser = load_parser()
    parser.add_argument('--skel', action='store_true')
    parser.add_argument('--name', default=None)
    # Kalman options. Defaults are enabled and conservative.
    parser.add_argument('--use_joint_kalman', action='store_true', default=True)
    parser.add_argument('--disable_joint_kalman', dest='use_joint_kalman', action='store_false')
    parser.add_argument('--joint_kalman_measurement_noise', type=float, default=5)
    parser.add_argument('--joint_kalman_process_noise', type=float, default=1)
    parser.add_argument('--joint_kalman_init_cov', type=float, default=1.0)
    parser.add_argument('--joint_kalman_max_jump', type=float, default=None)
    parser.add_argument('--joint_kalman_reset_after_missing', type=int, default=10)

    # Kalman filter on optimized MANO/SMPL global RT parameters.
    # This is applied inside easymocap.pipeline.basic.multi_stage_optimize.
    parser.add_argument('--use_param_kalman', action='store_true', default=True)
    parser.add_argument('--disable_param_kalman', dest='use_param_kalman', action='store_false')
    parser.add_argument('--param_kalman_measurement_noise', type=float, default=5)
    parser.add_argument('--param_kalman_process_noise', type=float, default=1)
    parser.add_argument('--param_kalman_init_cov', type=float, default=1.0)
    parser.add_argument('--param_kalman_max_jump', type=float, default=None)

    # 2D best-pair prefilter options.
    parser.add_argument('--select2d_conf_thres', type=float, default=0.4)
    parser.add_argument('--select2d_repro_thres', type=float, default=50)
    parser.add_argument('--select2d_min_view', type=int, default=3)
    parser.add_argument('--select2d_prefer_min_inliers', type=int, default=4)
    parser.add_argument('--select2d_max_candidate_views', type=int, default=36)
    parser.add_argument('--select2d_print_each_joint', action='store_true', default=True)
    parser.add_argument('--disable_select2d_print_each_joint', dest='select2d_print_each_joint', action='store_false')

    # Fallback 2D annotations for joints that primary clean 2D cannot triangulate.
    # Layout: <args.path>/<fallback2d_annot>/<cam_id>/<frame:06d>.json
    parser.add_argument('--use_fallback2d', action='store_true', default=True)
    parser.add_argument('--disable_fallback2d', dest='use_fallback2d', action='store_false')
    parser.add_argument('--fallback2d_annot', type=str, default='annots_fallback')
    parser.add_argument(
        '--fallback2d_hand_field', type=str, default='auto',
        help='Fallback 2D field: auto, handr2d, or handl2d. auto follows --body/--model.'
    )
    parser.add_argument('--fallback2d_num_joints', type=int, default=21)
    parser.add_argument('--fallback2d_conf_thres', type=float, default=0.0)
    parser.add_argument('--fallback2d_require_complete', action='store_true', default=True)
    parser.add_argument('--allow_fallback2d_missing', dest='fallback2d_require_complete', action='store_false')

    # Joint Kalman fallback options.
    parser.add_argument('--joint_kalman_prediction_conf', type=float, default=0.05)
    parser.add_argument('--joint_kalman_print_summary', action='store_true', default=True)
    parser.add_argument('--disable_joint_kalman_print_summary', dest='joint_kalman_print_summary', action='store_false')

    args = parse_parser(parser)
    help = """
  Demo code for multiple views and one person:

    - Input : {} => {}
    - Output: {}
    - Body  : {}=>{}, {}
    - Joint Kalman: {}
    - Param Kalman: {}
""".format(
        args.path,
        ', '.join(args.sub),
        args.out,
        args.model,
        args.gender,
        args.body,
        args.use_joint_kalman,
        args.use_param_kalman,
    )
    print(help)

    skel_path = join(args.out, 'keypoints3d')
    sub = convert_subs_to_rpizero(args.sub)
    dataset = MV1PMF(
        args.path,
        annot_root=args.annot,
        cams=sub,
        out=args.out,
        config=CONFIG[args.body],
        kpts_type=args.body,
        undis=args.undis,
        no_img=False,
        verbose=args.verbose,
    )
    dataset.writer.save_origin = args.save_origin

    if args.skel or not os.path.exists(skel_path):
        mv1pmf_skel(dataset, check_repro=True, args=args)

