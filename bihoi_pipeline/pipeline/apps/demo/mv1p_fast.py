'''
  @ Date: 2021-04-13 19:46:51
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-13 17:56:25
  @ FilePath: /EasyMocap/apps/demo/mv1p.py

This version uses annots_fallback as the only 2D source, performs
GPU-batched all-pairs per-joint RANSAC triangulation across all frames first,
then applies a six-state constant-velocity Kalman filter [x, y, z, vx, vy, vz]
sequentially across frames.
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
import multiprocessing as mp
import traceback
import time
import torch
from pathlib import Path

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


# Fixed bone lengths from your uploaded 3D keypoints file.
# Unit follows your input keypoints, most likely meters.
FIXED_HAND21_BONE_LENGTHS = np.array([
    0.0,                  # 0 wrist, no parent

    0.0403522956,         # 0 -> 1
    0.0317298877,         # 1 -> 2
    0.0256056534,         # 2 -> 3
    0.0252380995,         # 3 -> 4

    0.0778951084,         # 0 -> 5
    0.0366046181,         # 5 -> 6
    0.0195334323,         # 6 -> 7
    0.0194314754,         # 7 -> 8

    0.0749317618,         # 0 -> 9
    0.0388034843,         # 9 -> 10
    0.0236226696,         # 10 -> 11
    0.0208810735,         # 11 -> 12

    0.0712808481,         # 0 -> 13
    0.0352158183,         # 13 -> 14
    0.0221516755,         # 14 -> 15
    0.0199436905,         # 15 -> 16

    0.0701994748,         # 0 -> 17
    0.0282052967,         # 17 -> 18
    0.0161373083,         # 18 -> 19
    0.0179026640,         # 19 -> 20
], dtype=np.float32)





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


def ransac_triangulate_all_pairs(
    keypoints2d, Pall, conf_thres=0.3, repro_thres=50.0,
    min_view=2, print_each_joint=True, frame_id=None,
):
    """Pure all-pairs RANSAC triangulation for one multi-view hand frame."""
    keypoints2d = np.asarray(keypoints2d, dtype=np.float64)
    n_views, n_joints, _ = keypoints2d.shape
    keypoints3d = np.zeros((n_joints, 4), dtype=np.float64)
    keypoints2d_inliers = np.zeros_like(keypoints2d)
    info = {"selected_pairs": {}, "inlier_views": {}, "num_inliers": {},
            "mean_repro_error": {}, "max_repro_error": {}}
    prefix = f"[frame {frame_id}] " if frame_id is not None else ""

    for joint_id in range(n_joints):
        Ps = _get_P_for_joint(Pall, joint_id)
        kpts_j = keypoints2d[:, joint_id, :]
        conf = kpts_j[:, 2]
        valid_views = np.where(np.isfinite(conf) & (conf > float(conf_thres)))[0]

        if len(valid_views) < int(min_view):
            info["selected_pairs"][joint_id] = None
            info["inlier_views"][joint_id] = []
            info["num_inliers"][joint_id] = 0
            info["mean_repro_error"][joint_id] = None
            info["max_repro_error"][joint_id] = None
            if print_each_joint:
                print(f"{prefix}[ransac] joint {joint_id:02d}: blank, valid_views={valid_views.tolist()}")
            continue

        best = None
        for pair_tuple in combinations(valid_views.tolist(), 2):
            pair = np.asarray(pair_tuple, dtype=np.int64)
            X_pair = _triangulate_one_point_dlt(kpts_j[pair], Ps[pair])
            if X_pair is None or not np.all(np.isfinite(X_pair)):
                continue

            inliers = []
            for view_id in valid_views:
                uv_proj = _project_point(Ps[view_id], X_pair)
                if uv_proj is None or not np.all(np.isfinite(uv_proj)):
                    continue
                err = float(np.linalg.norm(uv_proj - kpts_j[view_id, :2]))
                if np.isfinite(err) and err <= float(repro_thres):
                    inliers.append(int(view_id))

            if len(inliers) < int(min_view):
                continue
            inliers = np.asarray(inliers, dtype=np.int64)
            X_refined = _triangulate_one_point_dlt(kpts_j[inliers], Ps[inliers])
            if X_refined is None or not np.all(np.isfinite(X_refined)):
                continue

            final_inliers, final_errors = [], []
            for view_id in valid_views:
                uv_proj = _project_point(Ps[view_id], X_refined)
                if uv_proj is None or not np.all(np.isfinite(uv_proj)):
                    continue
                err = float(np.linalg.norm(uv_proj - kpts_j[view_id, :2]))
                if np.isfinite(err) and err <= float(repro_thres):
                    final_inliers.append(int(view_id))
                    final_errors.append(err)

            if len(final_inliers) < int(min_view):
                continue
            final_inliers = np.asarray(final_inliers, dtype=np.int64)
            X_final = _triangulate_one_point_dlt(kpts_j[final_inliers], Ps[final_inliers])
            if X_final is None or not np.all(np.isfinite(X_final)):
                continue

            scored_errors = []
            for view_id in final_inliers:
                uv_proj = _project_point(Ps[view_id], X_final)
                if uv_proj is not None:
                    scored_errors.append(float(np.linalg.norm(uv_proj - kpts_j[view_id, :2])))
            if len(scored_errors) < int(min_view):
                continue

            mean_err = float(np.mean(scored_errors))
            max_err = float(np.max(scored_errors))
            conf_sum = float(np.sum(conf[final_inliers]))
            score = (len(final_inliers), -mean_err, -max_err, conf_sum)
            if best is None or score > best["score"]:
                best = {"score": score, "pair": pair.copy(), "inliers": final_inliers.copy(),
                        "X": np.asarray(X_final), "mean_err": mean_err,
                        "max_err": max_err, "conf_mean": float(np.mean(conf[final_inliers]))}

        if best is None:
            info["selected_pairs"][joint_id] = None
            info["inlier_views"][joint_id] = []
            info["num_inliers"][joint_id] = 0
            info["mean_repro_error"][joint_id] = None
            info["max_repro_error"][joint_id] = None
            if print_each_joint:
                print(f"{prefix}[ransac] joint {joint_id:02d}: blank, no valid pair")
            continue

        inliers = best["inliers"]
        keypoints3d[joint_id, :3] = best["X"]
        keypoints3d[joint_id, 3] = best["conf_mean"]
        keypoints2d_inliers[inliers, joint_id, :] = keypoints2d[inliers, joint_id, :]
        info["selected_pairs"][joint_id] = best["pair"].tolist()
        info["inlier_views"][joint_id] = inliers.tolist()
        info["num_inliers"][joint_id] = int(len(inliers))
        info["mean_repro_error"][joint_id] = best["mean_err"]
        info["max_repro_error"][joint_id] = best["max_err"]
        if print_each_joint:
            print(f"{prefix}[ransac] joint {joint_id:02d}: best_pair={best['pair'].tolist()}, "
                  f"inliers={len(inliers)}, inlier_views={inliers.tolist()}, "
                  f"mean_err={best['mean_err']:.2f}px, max_err={best['max_err']:.2f}px")

    return keypoints3d, keypoints2d_inliers, info


# =============================================================================
# Multi-GPU triangulation support
# =============================================================================

def _visible_physical_gpu_label(logical_gpu_id):
    """Best-effort logical->physical label for logs under CUDA_VISIBLE_DEVICES."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if visible:
        tokens = [x.strip() for x in visible.split(',') if x.strip()]
        if 0 <= int(logical_gpu_id) < len(tokens):
            return tokens[int(logical_gpu_id)]
    return str(logical_gpu_id)


def _select_visible_triangulation_gpus(gpu_spec='auto', min_free_vram_mb=32768):
    """
    Select logical CUDA devices visible to this mv1p process.

    Important: run_extract...sh may give right/left hands disjoint physical GPU
    sets through CUDA_VISIBLE_DEVICES.  Therefore this function intentionally
    works with *logical* ids inside the current process and never reaches outside
    the visible set.
    """
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for multi-GPU triangulation, but torch.cuda.is_available() is False.')

    n_visible = int(torch.cuda.device_count())
    if n_visible <= 0:
        raise RuntimeError('No CUDA devices are visible to mv1p.py.')

    spec = str(gpu_spec or 'auto').strip().lower()
    if spec == 'auto':
        requested = list(range(n_visible))
    else:
        try:
            requested = [int(x.strip()) for x in spec.split(',') if x.strip()]
        except ValueError as exc:
            raise ValueError(
                f'Invalid --triangulation_gpus={gpu_spec!r}; use auto or logical ids such as 0,1,2.'
            ) from exc
        if not requested:
            raise ValueError('--triangulation_gpus selected no logical CUDA ids.')

    invalid = [idx for idx in requested if idx < 0 or idx >= n_visible]
    if invalid:
        raise ValueError(
            f'Logical CUDA ids {invalid} are not visible; current visible count is {n_visible}.'
        )

    eligible = []
    snapshot = []
    threshold = int(min_free_vram_mb)
    for logical_id in requested:
        # mem_get_info(device) returns current free/total memory for that visible device.
        free_bytes, total_bytes = torch.cuda.mem_get_info(logical_id)
        free_mb = int(free_bytes // (1024 * 1024))
        total_mb = int(total_bytes // (1024 * 1024))
        physical = _visible_physical_gpu_label(logical_id)
        snapshot.append((logical_id, physical, free_mb, total_mb))
        if free_mb >= threshold:
            eligible.append((logical_id, physical, free_mb, total_mb))

    # Prefer the currently emptier cards. All eligible cards are retained.
    eligible.sort(key=lambda x: (-x[2], x[0]))
    if not eligible:
        status = ', '.join(
            f'logical cuda:{lid}/physical {phys}: {free} MiB free'
            for lid, phys, free, _ in snapshot
        )
        raise RuntimeError(
            f'No visible GPU has at least {threshold} MiB free VRAM for triangulation. '
            f'Visible GPU state: {status}'
        )
    return eligible


def _batched_weighted_dlt_from_view_masks(kpts_t, Ps_t, masks):
    """
    GPU weighted DLT for many hypotheses at once.

    kpts_t: (V,3) [u,v,conf]
    Ps_t:   (V,3,4)
    masks:  (H,V) bool, selecting the views used by each hypothesis

    Returns:
      X:     (H,3)
      valid: (H,) numerical-validity mask
    """
    dtype = Ps_t.dtype
    u = kpts_t[:, 0]
    v = kpts_t[:, 1]
    c = kpts_t[:, 2]

    row0 = c[:, None] * (u[:, None] * Ps_t[:, 2, :] - Ps_t[:, 0, :])
    row1 = c[:, None] * (v[:, None] * Ps_t[:, 2, :] - Ps_t[:, 1, :])
    rows = torch.stack([row0, row1], dim=1)  # (V,2,4)

    # Per-view A^T A contribution, then sum only selected views per hypothesis.
    per_view_ata = torch.einsum('vri,vrj->vij', rows, rows)  # (V,4,4)
    ata = torch.einsum('hv,vij->hij', masks.to(dtype), per_view_ata)

    # The eigenvector associated with the smallest eigenvalue solves homogeneous DLT.
    _, eigvecs = torch.linalg.eigh(ata)
    xh = eigvecs[:, :, 0]
    denom = xh[:, 3]
    valid = torch.isfinite(xh).all(dim=1) & (torch.abs(denom) > 1e-12)

    safe_denom = torch.where(valid, denom, torch.ones_like(denom))
    X = xh[:, :3] / safe_denom[:, None]
    valid = valid & torch.isfinite(X).all(dim=1)
    return X, valid


def _project_hypotheses_gpu(Ps_t, X):
    """Project H 3D hypotheses into V views -> uv(H,V,2), valid(H,V)."""
    ones = torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)
    Xh = torch.cat([X, ones], dim=1)
    proj = torch.einsum('vij,hj->hvi', Ps_t, Xh)
    z = proj[:, :, 2]
    valid = torch.isfinite(proj).all(dim=2) & (torch.abs(z) > 1e-12)
    safe_z = torch.where(valid, z, torch.ones_like(z))
    uv = proj[:, :, :2] / safe_z[:, :, None]
    valid = valid & torch.isfinite(uv).all(dim=2)
    return uv, valid


def ransac_triangulate_all_pairs_gpu(
    keypoints2d, Pall, device,
    conf_thres=0.3, repro_thres=50.0,
    min_view=2, print_each_joint=True, frame_id=None,
    Pall_gpu=None, pair_cache=None,
):
    """
    GPU-batched version of all-pairs per-joint RANSAC.

    The expensive pair-hypothesis DLT/refinement/reprojection work for a joint is
    vectorized on one GPU. Multi-GPU parallelism is added at the frame level by
    mv1pmf_skel(): independent frames are distributed across all eligible GPUs.
    """
    keypoints2d = np.asarray(keypoints2d, dtype=np.float64)
    Pall = np.asarray(Pall, dtype=np.float64)
    n_views, n_joints, _ = keypoints2d.shape
    keypoints3d = np.zeros((n_joints, 4), dtype=np.float64)
    keypoints2d_inliers = np.zeros_like(keypoints2d)
    info = {
        'selected_pairs': {}, 'inlier_views': {}, 'num_inliers': {},
        'mean_repro_error': {}, 'max_repro_error': {},
    }
    prefix = f'[frame {frame_id}] ' if frame_id is not None else ''
    device = torch.device(device)

    # Upload each frame's complete 2D array once, rather than 21 separate
    # host->device copies.  Pall_gpu is uploaded once per worker and reused
    # across every frame.
    kpts_full_t = torch.as_tensor(keypoints2d, dtype=torch.float64, device=device)
    if Pall_gpu is None:
        Pall_gpu = torch.as_tensor(Pall, dtype=torch.float64, device=device)
    if pair_cache is None:
        pair_cache = {}

    for joint_id in range(n_joints):
        kpts_j_np = keypoints2d[:, joint_id, :]
        conf_np = kpts_j_np[:, 2]
        valid_views = np.where(np.isfinite(conf_np) & (conf_np > float(conf_thres)))[0]

        if len(valid_views) < int(min_view):
            info['selected_pairs'][joint_id] = None
            info['inlier_views'][joint_id] = []
            info['num_inliers'][joint_id] = 0
            info['mean_repro_error'][joint_id] = None
            info['max_repro_error'][joint_id] = None
            if print_each_joint:
                print(f'{prefix}[gpu-ransac] joint {joint_id:02d}: blank, valid_views={valid_views.tolist()}')
            continue

        valid_idx_t = torch.as_tensor(valid_views, dtype=torch.long, device=device)
        kpts_t = kpts_full_t[:, joint_id, :].index_select(0, valid_idx_t)

        if Pall_gpu.ndim == 3:
            Ps_all_t = Pall_gpu
        elif Pall_gpu.ndim == 4:
            Ps_all_t = Pall_gpu[:, joint_id, :, :]
        else:
            raise ValueError(f"Unsupported Pall_gpu shape: {tuple(Pall_gpu.shape)}")
        Ps_t = Ps_all_t.index_select(0, valid_idx_t)
        vcount = int(len(valid_views))

        # Pair topology depends only on number of valid views. Cache it on the
        # worker GPU instead of rebuilding torch.combinations + masks 21 times/frame.
        cached = pair_cache.get(vcount)
        if cached is None:
            local_ids = torch.arange(vcount, device=device, dtype=torch.long)
            pairs = torch.combinations(local_ids, r=2)
            hcount = int(pairs.shape[0])
            if hcount == 0:
                continue
            pair_masks = torch.zeros((hcount, vcount), dtype=torch.bool, device=device)
            row_ids = torch.arange(hcount, device=device)
            pair_masks[row_ids, pairs[:, 0]] = True
            pair_masks[row_ids, pairs[:, 1]] = True
            pair_cache[vcount] = (pairs, pair_masks)
        else:
            pairs, pair_masks = cached
            hcount = int(pairs.shape[0])

        X_pair, valid_pair = _batched_weighted_dlt_from_view_masks(kpts_t, Ps_t, pair_masks)
        uv_pair, proj_valid = _project_hypotheses_gpu(Ps_t, X_pair)
        obs = kpts_t[None, :, :2]
        err_pair = torch.linalg.norm(uv_pair - obs, dim=2)
        inlier1 = proj_valid & torch.isfinite(err_pair) & (err_pair <= float(repro_thres))
        valid_pair = valid_pair & (inlier1.sum(dim=1) >= int(min_view))

        # First refinement using all inliers from each pair hypothesis.
        X_ref, valid_ref = _batched_weighted_dlt_from_view_masks(kpts_t, Ps_t, inlier1)
        uv_ref, proj_valid_ref = _project_hypotheses_gpu(Ps_t, X_ref)
        err_ref = torch.linalg.norm(uv_ref - obs, dim=2)
        inlier2 = proj_valid_ref & torch.isfinite(err_ref) & (err_ref <= float(repro_thres))
        valid_ref = valid_ref & valid_pair & (inlier2.sum(dim=1) >= int(min_view))

        # Final DLT after the second inlier check, matching the CPU pipeline's
        # final refinement semantics.
        X_final, valid_final = _batched_weighted_dlt_from_view_masks(kpts_t, Ps_t, inlier2)
        uv_final, proj_valid_final = _project_hypotheses_gpu(Ps_t, X_final)
        err_final = torch.linalg.norm(uv_final - obs, dim=2)

        # Match the original CPU semantics: X_final is triangulated from the
        # inlier set produced by X_ref.  The CPU code then SCORES those same
        # inliers after final DLT; it does not apply a third reprojection-threshold
        # pruning pass. We only drop numerically unprojectable views here.
        final_inliers = inlier2 & proj_valid_final & torch.isfinite(err_final)
        counts = final_inliers.sum(dim=1)
        candidate_valid = valid_final & valid_ref & (counts >= int(min_view))

        if not bool(candidate_valid.any().item()):
            info['selected_pairs'][joint_id] = None
            info['inlier_views'][joint_id] = []
            info['num_inliers'][joint_id] = 0
            info['mean_repro_error'][joint_id] = None
            info['max_repro_error'][joint_id] = None
            if print_each_joint:
                print(f'{prefix}[gpu-ransac] joint {joint_id:02d}: blank, no valid pair')
            continue

        # Score exactly in the same lexicographic order as the CPU code:
        # (num_inliers, -mean_err, -max_err, conf_sum).
        mask_f = final_inliers.to(err_final.dtype)
        count_f = counts.clamp_min(1).to(err_final.dtype)
        mean_err = (torch.where(final_inliers, err_final, torch.zeros_like(err_final)).sum(dim=1) / count_f)
        max_err = torch.where(
            final_inliers,
            err_final,
            torch.full_like(err_final, float('-inf')),
        ).max(dim=1).values
        conf_sum = (mask_f * kpts_t[None, :, 2]).sum(dim=1)

        valid_ids = torch.where(candidate_valid)[0].detach().cpu().tolist()
        counts_cpu = counts.detach().cpu().numpy()
        mean_cpu = mean_err.detach().cpu().numpy()
        max_cpu = max_err.detach().cpu().numpy()
        confsum_cpu = conf_sum.detach().cpu().numpy()
        best_idx = max(
            valid_ids,
            key=lambda i: (
                int(counts_cpu[i]),
                -float(mean_cpu[i]),
                -float(max_cpu[i]),
                float(confsum_cpu[i]),
            ),
        )

        best_local_inliers = torch.where(final_inliers[best_idx])[0].detach().cpu().numpy()
        best_inliers = valid_views[best_local_inliers]
        best_pair_local = pairs[best_idx].detach().cpu().numpy()
        best_pair = valid_views[best_pair_local]
        best_X = X_final[best_idx].detach().cpu().numpy()
        best_mean = float(mean_cpu[best_idx])
        best_max = float(max_cpu[best_idx])
        best_conf_mean = float(np.mean(conf_np[best_inliers]))

        keypoints3d[joint_id, :3] = best_X
        keypoints3d[joint_id, 3] = best_conf_mean
        keypoints2d_inliers[best_inliers, joint_id, :] = keypoints2d[best_inliers, joint_id, :]
        info['selected_pairs'][joint_id] = best_pair.tolist()
        info['inlier_views'][joint_id] = best_inliers.tolist()
        info['num_inliers'][joint_id] = int(len(best_inliers))
        info['mean_repro_error'][joint_id] = best_mean
        info['max_repro_error'][joint_id] = best_max

        if print_each_joint:
            print(
                f'{prefix}[gpu-ransac] joint {joint_id:02d}: '
                f'best_pair={best_pair.tolist()}, inliers={len(best_inliers)}, '
                f'inlier_views={best_inliers.tolist()}, '
                f'mean_err={best_mean:.2f}px, max_err={best_max:.2f}px'
            )

    return keypoints3d, keypoints2d_inliers, info


def _raw_triangulation_path(raw_root, frame_id):
    return Path(raw_root) / f'{int(frame_id):06d}.npz'


def _save_raw_triangulation(raw_root, frame_id, keypoints3d, keypoints2d_inliers):
    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    final_path = _raw_triangulation_path(raw_root, frame_id)
    tmp_path = raw_root / f'.{int(frame_id):06d}.{os.getpid()}.tmp.npz'
    np.savez(
        tmp_path,
        keypoints3d=np.asarray(keypoints3d, dtype=np.float64),
        keypoints2d_inliers=np.asarray(keypoints2d_inliers, dtype=np.float64),
    )
    os.replace(tmp_path, final_path)


def _load_raw_triangulation(raw_root, frame_id):
    path = _raw_triangulation_path(raw_root, frame_id)
    if not path.is_file():
        raise FileNotFoundError(f'Missing raw triangulation result for frame {frame_id}: {path}')
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data['keypoints3d'], dtype=np.float64),
            np.asarray(data['keypoints2d_inliers'], dtype=np.float64),
        )


def _triangulation_gpu_worker(
    worker_rank,
    logical_gpu_id,
    physical_label,
    tasks,
    Pall,
    raw_root,
    conf_thres,
    repro_thres,
    min_view,
    print_each_joint,
    error_queue,
):
    """One spawned worker owns one logical GPU and a disjoint set of frames."""
    try:
        torch.cuda.set_device(int(logical_gpu_id))
        device = torch.device(f'cuda:{int(logical_gpu_id)}')
        free_b, total_b = torch.cuda.mem_get_info(int(logical_gpu_id))
        print(
            f'[Triangulation W{worker_rank}] logical cuda:{logical_gpu_id} '
            f'physical={physical_label}, frames={len(tasks)}, '
            f'free={free_b / 1024**3:.2f} GiB / total={total_b / 1024**3:.2f} GiB',
            flush=True,
        )

        # Static cameras are identical for every frame handled by this worker.
        # Upload once, then reuse. Pair masks are also cached by valid-view count.
        Pall_gpu = torch.as_tensor(
            np.asarray(Pall, dtype=np.float64),
            dtype=torch.float64,
            device=device,
        )
        pair_cache = {}
        total_tasks = len(tasks)

        for local_i, (frame_id, keypoints2d) in enumerate(tasks, start=1):
            raw3d, inliers2d, _ = ransac_triangulate_all_pairs_gpu(
                keypoints2d=keypoints2d,
                Pall=Pall,
                device=device,
                conf_thres=conf_thres,
                repro_thres=repro_thres,
                min_view=min_view,
                print_each_joint=print_each_joint,
                frame_id=frame_id,
                Pall_gpu=Pall_gpu,
                pair_cache=pair_cache,
            )
            _save_raw_triangulation(raw_root, frame_id, raw3d, inliers2d)
            if local_i == 1 or local_i % 25 == 0 or local_i == total_tasks:
                print(
                    f'[Triangulation W{worker_rank} GPU {physical_label}] '
                    f'{local_i}/{total_tasks} frames done (latest={frame_id})',
                    flush=True,
                )

        torch.cuda.synchronize(device)
    except Exception:
        error_queue.put((int(worker_rank), str(physical_label), traceback.format_exc()))
        raise


def _run_multigpu_triangulation_phase(tasks, Pall, raw_root, args):
    """Distribute independent frame triangulation across every eligible visible GPU."""
    eligible = _select_visible_triangulation_gpus(
        gpu_spec=getattr(args, 'triangulation_gpus', 'auto'),
        min_free_vram_mb=int(getattr(args, 'triangulation_min_free_vram_mb', 32768)),
    )

    if len(tasks) == 0:
        return eligible

    n_workers = min(len(eligible), len(tasks))
    eligible = eligible[:n_workers]
    buckets = [[] for _ in range(n_workers)]
    for i, task in enumerate(tasks):
        buckets[i % n_workers].append(task)

    print('[Triangulation GPU scheduler]')
    print(f'  frames              = {len(tasks)}')
    print(f'  min free VRAM       = {int(getattr(args, "triangulation_min_free_vram_mb", 32768))} MiB')
    print(f'  GPU workers         = {n_workers}')
    print(
        '  selected GPUs       = ' + ', '.join(
            f'logical cuda:{lid}/physical {phys} ({free} MiB free)'
            for lid, phys, free, _ in eligible
        )
    )

    ctx = mp.get_context('spawn')
    error_queue = ctx.Queue()
    procs = []
    for rank, (logical_id, physical, _, _) in enumerate(eligible):
        proc = ctx.Process(
            target=_triangulation_gpu_worker,
            args=(
                rank,
                logical_id,
                physical,
                buckets[rank],
                np.asarray(Pall, dtype=np.float64),
                str(raw_root),
                float(getattr(args, 'select2d_conf_thres', 0.3)),
                float(getattr(args, 'select2d_repro_thres', 50.0)),
                int(getattr(args, 'select2d_min_view', 2)),
                bool(getattr(args, 'select2d_print_each_joint', True)),
                error_queue,
            ),
        )
        proc.start()
        procs.append(proc)

    failed = []
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            failed.append(proc.exitcode)

    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())

    if failed or errors:
        detail = '\n'.join(
            f'[worker {rank} physical GPU {phys}]\n{tb}'
            for rank, phys, tb in errors
        )
        raise RuntimeError(
            f'Multi-GPU triangulation failed; worker exit codes={failed}.\n{detail}'
        )

    return eligible


class RawPredictionJointKalmanManager:
    """
    Independent constant-velocity Kalman filter for every hand joint.

    State per joint:
        [x, y, z, vx, vy, vz]

    Every frame advances every initialized joint exactly once:
      - valid triangulated joint: predict + measurement update;
      - missing triangulated joint: prediction only;
      - never-observed joint: remains invalid until its first measurement.

    ``dt`` is the frame interval in seconds, so velocities are expressed in
    input-distance-units per second (normally metres/second).

    Default preset is tuned for 30 FPS with stronger anti-jitter smoothing:
      dt=1/30 s, measurement sigma=0.015 m, acceleration sigma=0.30 m/s^2.
    """

    def __init__(
        self,
        n_joints,
        measurement_noise_scale=0.015,
        process_noise_scale=0.30,
        init_cov_scale=1.0,
        max_jump=None,
        reset_after_missing=30,
        prediction_conf=0.05,
        dt=1.0 / 30.0,
    ):
        self.n_joints = int(n_joints)
        self.measurement_noise_scale = float(measurement_noise_scale)
        self.process_noise_scale = float(process_noise_scale)
        self.init_cov_scale = float(init_cov_scale)
        self.max_jump = None if max_jump is None else float(max_jump)
        if reset_after_missing is None or int(reset_after_missing) < 0:
            self.reset_after_missing = None
        else:
            self.reset_after_missing = int(reset_after_missing)
        self.prediction_conf = float(prediction_conf)
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError(f"Kalman dt must be > 0, got {self.dt}.")

        self.x = np.zeros((self.n_joints, 6), dtype=np.float64)
        self.P = np.zeros((self.n_joints, 6, 6), dtype=np.float64)
        self.initialized = np.zeros((self.n_joints,), dtype=bool)
        self.missing_count = np.zeros((self.n_joints,), dtype=np.int64)
        self.last_update_used_measurement = np.zeros(
            (self.n_joints,), dtype=bool
        )

        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self._build_system_matrices()

    def _build_system_matrices(self):
        dt = self.dt
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # White-acceleration constant-velocity process model.  The supplied
        # process_noise_scale is acceleration standard deviation.
        q = max(self.process_noise_scale, 1e-12) ** 2
        q_pos = 0.25 * dt ** 4 * q
        q_cross = 0.5 * dt ** 3 * q
        q_vel = dt ** 2 * q
        self.Q = np.zeros((6, 6), dtype=np.float64)
        for pos_axis, vel_axis in ((0, 3), (1, 4), (2, 5)):
            self.Q[pos_axis, pos_axis] = q_pos
            self.Q[pos_axis, vel_axis] = q_cross
            self.Q[vel_axis, pos_axis] = q_cross
            self.Q[vel_axis, vel_axis] = q_vel

        r = max(self.measurement_noise_scale, 1e-12) ** 2
        self.R = r * np.eye(3, dtype=np.float64)

    @staticmethod
    def _is_valid_xyz(xyz):
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        return bool(np.all(np.isfinite(xyz)))

    def _init_joint(self, joint_id, xyz):
        j = int(joint_id)
        self.x[j, :3] = np.asarray(xyz, dtype=np.float64).reshape(3)
        self.x[j, 3:] = 0.0

        # Position and velocity uncertainty are kept separate.  A larger
        # initial velocity covariance lets the filter learn motion quickly.
        pos_var = max(self.init_cov_scale, 1e-12)
        vel_var = pos_var / max(self.dt ** 2, 1e-12)
        self.P[j] = np.diag([
            pos_var, pos_var, pos_var,
            vel_var, vel_var, vel_var,
        ]).astype(np.float64)
        self.initialized[j] = True
        self.missing_count[j] = 0
        self.last_update_used_measurement[j] = True

    def _predict_in_place(self, joint_id):
        j = int(joint_id)
        self.x[j] = self.F @ self.x[j]
        self.P[j] = self.F @ self.P[j] @ self.F.T + self.Q

    def predict_only(self, joint_id):
        """Advance one frame without measurement and return predicted xyz."""
        j = int(joint_id)
        self.last_update_used_measurement[j] = False
        if not self.initialized[j]:
            return None

        self._predict_in_place(j)
        self.missing_count[j] += 1

        if (
            self.reset_after_missing is not None
            and self.missing_count[j] > self.reset_after_missing
        ):
            self.initialized[j] = False
            return None

        return self.x[j, :3].copy()

    def update_with_measurement(self, joint_id, xyz):
        """Advance one frame, update from xyz, and return filtered xyz."""
        j = int(joint_id)
        z = np.asarray(xyz, dtype=np.float64).reshape(3)
        if not self._is_valid_xyz(z):
            return self.predict_only(j)

        if not self.initialized[j]:
            self._init_joint(j, z)
            return self.x[j, :3].copy()

        # Predict exactly once for this frame.
        self._predict_in_place(j)

        # Optional innovation-distance gate.  Rejected measurements are treated
        # as missing, so a sudden outlier cannot corrupt position or velocity.
        if self.max_jump is not None:
            jump = float(np.linalg.norm(z - self.x[j, :3]))
            if jump > self.max_jump:
                self.last_update_used_measurement[j] = False
                self.missing_count[j] += 1
                if (
                    self.reset_after_missing is not None
                    and self.missing_count[j] > self.reset_after_missing
                ):
                    self.initialized[j] = False
                    return None
                return self.x[j, :3].copy()

        innovation = z - self.H @ self.x[j]
        S = self.H @ self.P[j] @ self.H.T + self.R
        try:
            K = self.P[j] @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P[j] @ self.H.T @ np.linalg.pinv(S)

        self.x[j] = self.x[j] + K @ innovation
        I = np.eye(6, dtype=np.float64)
        # Joseph form is more numerically stable and preserves symmetry/PSD.
        IKH = I - K @ self.H
        self.P[j] = IKH @ self.P[j] @ IKH.T + K @ self.R @ K.T
        self.P[j] = 0.5 * (self.P[j] + self.P[j].T)
        self.missing_count[j] = 0
        self.last_update_used_measurement[j] = True
        return self.x[j, :3].copy()

    def step_frame(self, keypoints3d, prediction_conf=None):
        """
        Filter every joint in one frame and return filtered keypoints + summary.
        """
        measurements = np.asarray(keypoints3d, dtype=np.float64)
        if measurements.shape != (self.n_joints, 4):
            raise ValueError(
                f"Expected ({self.n_joints}, 4) keypoints, "
                f"got {measurements.shape}."
            )

        prediction_conf = (
            self.prediction_conf
            if prediction_conf is None
            else float(prediction_conf)
        )
        out = measurements.copy()
        measured, predicted, uninitialized, rejected = [], [], [], []

        valid_measurement = (
            np.isfinite(measurements[:, :3]).all(axis=1)
            & np.isfinite(measurements[:, 3])
            & (measurements[:, 3] > 0)
        )

        for j in range(self.n_joints):
            if valid_measurement[j]:
                xyz = self.update_with_measurement(j, measurements[j, :3])
                if xyz is None:
                    out[j] = 0.0
                    uninitialized.append(j)
                    continue
                out[j, :3] = xyz
                if self.last_update_used_measurement[j]:
                    out[j, 3] = measurements[j, 3]
                    measured.append(j)
                else:
                    out[j, 3] = prediction_conf
                    predicted.append(j)
                    rejected.append(j)
            else:
                xyz = self.predict_only(j)
                if xyz is None:
                    out[j] = 0.0
                    uninitialized.append(j)
                else:
                    out[j, :3] = xyz
                    out[j, 3] = prediction_conf
                    predicted.append(j)

        summary = {
            'measured_joints': measured,
            'predicted_joints': predicted,
            'uninitialized_joints': uninitialized,
            'rejected_measurement_joints': rejected,
        }
        return out, summary

    def sync_constrained_positions(self, keypoints3d, previous_keypoints3d=None):
        """
        Write hard bone-length-constrained positions back into Kalman states.

        Velocity is recomputed from consecutive constrained positions, making
        [x,y,z,vx,vy,vz] internally consistent with the saved skeleton.
        """
        current = np.asarray(keypoints3d, dtype=np.float64)
        if current.shape != (self.n_joints, 4):
            raise ValueError(
                f"Expected ({self.n_joints}, 4), got {current.shape}."
            )
        previous = None
        if previous_keypoints3d is not None:
            previous = np.asarray(previous_keypoints3d, dtype=np.float64)
            if previous.shape != current.shape:
                raise ValueError(
                    f"previous_keypoints3d shape {previous.shape} does not "
                    f"match current shape {current.shape}."
                )

        current_valid = (
            np.isfinite(current[:, :3]).all(axis=1)
            & np.isfinite(current[:, 3])
            & (current[:, 3] > 0)
        )
        previous_valid = np.zeros((self.n_joints,), dtype=bool)
        if previous is not None:
            previous_valid = (
                np.isfinite(previous[:, :3]).all(axis=1)
                & np.isfinite(previous[:, 3])
                & (previous[:, 3] > 0)
            )

        for j in range(self.n_joints):
            if not current_valid[j]:
                continue
            if not self.initialized[j]:
                self._init_joint(j, current[j, :3])
            if previous is not None and previous_valid[j]:
                self.x[j, 3:] = (
                    current[j, :3] - previous[j, :3]
                ) / self.dt
            self.x[j, :3] = current[j, :3]

    def get_states(self):
        """Return state [x,y,z,vx,vy,vz] and initialization mask."""
        return self.x.copy(), self.initialized.copy()


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
    """Return and create the absolute ``annots_fallback`` root directory."""
    fallback_annot = getattr(args, 'fallback2d_annot', 'annots_fallback')
    if isabs(fallback_annot):
        fallback_root = fallback_annot
    else:
        dataset_root = getattr(dataset, 'root', None)
        if dataset_root is None:
            dataset_root = getattr(args, 'path', None)
        if dataset_root is None:
            raise RuntimeError(
                'Cannot resolve fallback annotation root: dataset.root and args.path are both unavailable.'
            )
        fallback_root = join(dataset_root, fallback_annot)

    # Create the root automatically. Existing annotations are left untouched.
    os.makedirs(fallback_root, exist_ok=True)
    return fallback_root


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



def _extract_saved_keypoints3d(payload):
    """Best-effort reader for EasyMocap keypoints3d JSON payloads."""
    if isinstance(payload, dict):
        if 'keypoints3d' in payload:
            try:
                arr = np.asarray(payload['keypoints3d'], dtype=np.float64)
            except (TypeError, ValueError):
                arr = None
            if arr is not None and arr.ndim == 2 and arr.shape[1] >= 4:
                return arr[:, :4].copy()

        # Common wrappers used by EasyMocap/export scripts.
        for key in ('annots', 'annotations', 'people', 'persons', 'instances', 'data'):
            if key in payload:
                found = _extract_saved_keypoints3d(payload[key])
                if found is not None:
                    return found
        return None

    if isinstance(payload, list):
        # Some files store the keypoint matrix directly as a list of rows.
        try:
            arr = np.asarray(payload, dtype=np.float64)
        except (TypeError, ValueError):
            arr = None
        if arr is not None and arr.ndim == 2 and arr.shape[1] >= 4:
            return arr[:, :4].copy()

        # Standard EasyMocap output is commonly a list of person dictionaries.
        for item in payload:
            found = _extract_saved_keypoints3d(item)
            if found is not None:
                return found
    return None


def _load_saved_keypoints3d(filename):
    """Load one completed keypoints3d JSON; return None for corrupt/incomplete files."""
    try:
        with open(filename, 'r') as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    keypoints3d = _extract_saved_keypoints3d(payload)
    if keypoints3d is None or keypoints3d.ndim != 2 or keypoints3d.shape[1] < 4:
        return None
    if keypoints3d.shape[0] == 0:
        return None
    return np.asarray(keypoints3d[:, :4], dtype=np.float64)


def _existing_keypoints3d_index(output_root):
    """Return {frame_id: json_path} for valid, already-triangulated frames."""
    frame_index = {}
    if not os.path.isdir(output_root):
        return frame_index

    for filename in sorted(glob(join(output_root, '*.json'))):
        stem = os.path.splitext(os.path.basename(filename))[0]
        try:
            frame_id = int(stem)
        except ValueError:
            continue
        if _load_saved_keypoints3d(filename) is not None:
            frame_index[frame_id] = filename
    return frame_index


def _latest_fallback_annotation_mtime(fallback_root, frame_id):
    """Newest mtime of ``<fallback_root>/<cam>/<frame_id>.json`` over all cameras, or None."""
    if not fallback_root or not os.path.isdir(fallback_root):
        return None
    latest = None
    for cam_dir in os.listdir(fallback_root):
        for name in (f'{frame_id:06d}.json', f'{frame_id}.json'):
            path = join(fallback_root, cam_dir, name)
            if os.path.isfile(path):
                mtime = os.path.getmtime(path)
                latest = mtime if latest is None else max(latest, mtime)
                break
    return latest


def _find_triangulation_resume_frame(output_root, start, end, fallback_root=None):
    """
    Scan [start, end) and return the first frame without a valid keypoints3d file.

    A keypoints3d file older than the 2D annotations of the same frame is treated
    as missing: after a re-extraction (e.g. 30 fps -> 10 fps) the frame ids are
    reused with different content, so stale results must not be resumed from.

    If every requested frame is already present, resume_frame == end.
    """
    existing = _existing_keypoints3d_index(output_root)
    if fallback_root:
        stale = []
        for frame_id, path in existing.items():
            annot_mtime = _latest_fallback_annotation_mtime(fallback_root, frame_id)
            if annot_mtime is not None and os.path.getmtime(path) < annot_mtime:
                stale.append(frame_id)
        for frame_id in stale:
            del existing[frame_id]
        if stale:
            print(
                f'[Resume] Ignoring {len(stale)} keypoints3d file(s) older than their '
                f'2D annotations (first: {min(stale)}); they will be recomputed.'
            )
    resume_frame = end
    for frame_id in range(start, end):
        if frame_id not in existing:
            resume_frame = frame_id
            break
    return resume_frame, existing


def _save_joint_kalman_resume_checkpoint(kalman_manager, frame_id, args):
    """Atomically save the full Kalman state needed for exact continuation."""
    if kalman_manager is None:
        return
    os.makedirs(args.out, exist_ok=True)
    path = join(args.out, 'joint_kalman_resume_state.npz')
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez_compressed(
            f,
            frame_id=np.asarray(int(frame_id), dtype=np.int64),
            x=kalman_manager.x,
            P=kalman_manager.P,
            initialized=kalman_manager.initialized,
            missing_count=kalman_manager.missing_count,
            last_update_used_measurement=kalman_manager.last_update_used_measurement,
            dt=np.asarray(kalman_manager.dt, dtype=np.float64),
            n_joints=np.asarray(kalman_manager.n_joints, dtype=np.int64),
        )
    os.replace(tmp, path)


def _restore_joint_kalman_resume_checkpoint(kalman_manager, expected_frame_id, args):
    """Restore a checkpoint only when it exactly matches the frame before resume."""
    path = join(args.out, 'joint_kalman_resume_state.npz')
    if not os.path.isfile(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as ckpt:
            frame_id = int(np.asarray(ckpt['frame_id']).item())
            n_joints = int(np.asarray(ckpt['n_joints']).item())
            dt = float(np.asarray(ckpt['dt']).item())
            if frame_id != int(expected_frame_id):
                return False
            if n_joints != kalman_manager.n_joints:
                return False
            if not np.isclose(dt, kalman_manager.dt):
                return False

            x = np.asarray(ckpt['x'], dtype=np.float64)
            P = np.asarray(ckpt['P'], dtype=np.float64)
            initialized = np.asarray(ckpt['initialized'], dtype=bool)
            missing_count = np.asarray(ckpt['missing_count'], dtype=np.int64)
            last_used = np.asarray(ckpt['last_update_used_measurement'], dtype=bool)

            if x.shape != kalman_manager.x.shape or P.shape != kalman_manager.P.shape:
                return False
            if initialized.shape != kalman_manager.initialized.shape:
                return False

            kalman_manager.x[...] = x
            kalman_manager.P[...] = P
            kalman_manager.initialized[...] = initialized
            kalman_manager.missing_count[...] = missing_count
            kalman_manager.last_update_used_measurement[...] = last_used
        print(f'[Resume] Restored exact joint Kalman checkpoint at frame {expected_frame_id}.')
        return True
    except (OSError, ValueError, KeyError, EOFError) as exc:
        print(f'[Resume] Ignoring invalid Kalman checkpoint {path}: {exc}')
        return False


def _warm_start_joint_kalman_from_saved_3d(
    kalman_manager,
    existing_index,
    resume_frame,
    start,
):
    """
    Warm-start Kalman from the last one/two saved 3D frames without re-triangulation.

    This is used for partial outputs produced by older code that did not save a
    full Kalman checkpoint. Position comes from frame resume_frame-1 and velocity
    is estimated from the previous saved frame when available.
    """
    last_id = int(resume_frame) - 1
    if last_id < int(start) or last_id not in existing_index:
        return False

    last = _load_saved_keypoints3d(existing_index[last_id])
    if last is None or last.shape[0] != kalman_manager.n_joints:
        return False

    prev = None
    prev_id = last_id - 1
    if prev_id >= int(start) and prev_id in existing_index:
        prev = _load_saved_keypoints3d(existing_index[prev_id])
        if prev is not None and prev.shape != last.shape:
            prev = None

    last_valid = (
        np.isfinite(last[:, :3]).all(axis=1)
        & np.isfinite(last[:, 3])
        & (last[:, 3] > 0)
    )
    prev_valid = np.zeros((kalman_manager.n_joints,), dtype=bool)
    if prev is not None:
        prev_valid = (
            np.isfinite(prev[:, :3]).all(axis=1)
            & np.isfinite(prev[:, 3])
            & (prev[:, 3] > 0)
        )

    initialized_count = 0
    velocity_count = 0
    for j in range(kalman_manager.n_joints):
        if not last_valid[j]:
            continue
        kalman_manager._init_joint(j, last[j, :3])
        initialized_count += 1
        if prev is not None and prev_valid[j]:
            kalman_manager.x[j, 3:] = (last[j, :3] - prev[j, :3]) / kalman_manager.dt
            velocity_count += 1

    if initialized_count == 0:
        return False

    print(
        f'[Resume] Warm-started joint Kalman from saved keypoints3d frame {last_id}: '
        f'initialized={initialized_count}, velocity_from_two_frames={velocity_count}. '
        f'(No exact checkpoint was available.)'
    )
    return True


def mv1pmf_skel(dataset, check_repro=True, args=None):
    """
    Two-phase hand reconstruction:

      Phase A (parallel, multi-GPU): triangulate every requested frame first.
      Phase B (sequential): run the temporal 6D joint Kalman filter strictly in
      frame order over the completed raw 3D measurements, then write keypoints3d.

    This ordering is deliberate: Kalman never runs while triangulation is still
    being produced for later frames.
    """
    if args is None:
        raise ValueError('mv1pmf_skel requires parsed args.')

    want_images = bool(args.vis_det or args.vis_repro)
    dataset.no_img = True  # Phase A only needs annotations/cameras.
    requested_start, end = args.start, min(args.end, len(dataset))
    keypoints3d_root = join(args.out, 'keypoints3d')
    resume_start, existing_index = _find_triangulation_resume_frame(
        keypoints3d_root,
        requested_start,
        end,
        fallback_root=_fallback_annotation_root(dataset, args),
    )

    completed_requested = sum(
        1 for frame_id in range(requested_start, end)
        if frame_id in existing_index
    )
    total_requested = max(0, end - requested_start)

    if resume_start >= end:
        print(
            f'[Resume] keypoints3d already complete: '
            f'{completed_requested}/{total_requested} frames in '
            f'[{requested_start}, {end}). Skip triangulation + Kalman.'
        )
        return

    later_existing = [
        frame_id for frame_id in existing_index
        if resume_start < frame_id < end
    ]
    print(
        f'[Resume] keypoints3d complete before frame {resume_start}: '
        f'{resume_start - requested_start}/{total_requested} contiguous frames. '
        f'Recompute raw triangulation and Kalman from frame {resume_start}.'
    )
    if later_existing:
        print(
            f'[Resume] Found {len(later_existing)} existing frame(s) after the first gap; '
            f'they will be overwritten from frame {resume_start} onward to preserve '
            f'temporal/Kalman consistency.'
        )

    start = resume_start
    conf_thres = float(getattr(args, 'select2d_conf_thres', 0.3))
    repro_thres = float(getattr(args, 'select2d_repro_thres', 50.0))
    min_view = int(getattr(args, 'select2d_min_view', 2))
    use_joint_kalman = bool(getattr(args, 'use_joint_kalman', True))
    prediction_conf = float(getattr(args, 'joint_kalman_prediction_conf', 0.05))
    save_kalman_states = bool(getattr(args, 'save_kalman_states', True))

    if min_view < 2:
        raise ValueError(f'--select2d_min_view must be >= 2, got {min_view}.')
    if conf_thres < 0:
        raise ValueError(f'--select2d_conf_thres must be >= 0, got {conf_thres}.')
    if repro_thres <= 0:
        raise ValueError(f'--select2d_repro_thres must be > 0, got {repro_thres}.')

    print(
        f'[2D source] ONLY {getattr(args, "fallback2d_annot", "annots_fallback")} '
        f'(body={args.body})'
    )
    print(
        f'[RANSAC] GPU-batched all 2-view pairs, conf>{conf_thres}, '
        f'repro<={repro_thres}px, min_view={min_view}'
    )
    print(
        f'[Joint Kalman] runs AFTER ALL raw triangulation is complete; '
        f'enabled={use_joint_kalman}, state=[x,y,z,vx,vy,vz], '
        f'dt={float(getattr(args, "joint_kalman_dt", 0.1)):.6f}s, '
        f'measurement_sigma={float(getattr(args, "joint_kalman_measurement_noise", 0.01))}, '
        f'acceleration_sigma={float(getattr(args, "joint_kalman_process_noise", 1.0))}'
    )

    # ------------------------------------------------------------------
    # Phase A: load 2D for every frame, then triangulate ALL frames first.
    # ------------------------------------------------------------------
    raw_root = Path(args.out) / 'raw_triangulation_gpu'
    raw_root.mkdir(parents=True, exist_ok=True)

    # Remove stale raw results for the segment that will be recomputed. This
    # guarantees parameter changes cannot accidentally reuse an older RANSAC result.
    for nf in range(start, end):
        stale = _raw_triangulation_path(raw_root, nf)
        if stale.exists():
            stale.unlink()

    tasks = []
    for nf in tqdm(range(start, end), desc='phase A: load fallback 2D'):
        _, annots = dataset[nf]
        if 'keypoints' not in annots:
            raise KeyError(
                f'[frame {nf}] annotation loaded from '
                f'{getattr(args, "fallback2d_annot", "annots_fallback")} '
                f'has no "keypoints" field after EasyMocap parsing.'
            )

        keypoints2d = np.asarray(annots['keypoints'], dtype=np.float64).copy()
        if keypoints2d.ndim != 3 or keypoints2d.shape[-1] < 3:
            raise RuntimeError(
                f'[frame {nf}] expected keypoints shape (nViews,nJoints,3+), '
                f'got {keypoints2d.shape}.'
            )
        keypoints2d = keypoints2d[..., :3]
        invalid = (
            ~np.isfinite(keypoints2d).all(axis=-1)
            | (keypoints2d[..., 2] <= conf_thres)
        )
        keypoints2d[invalid] = 0.0
        tasks.append((int(nf), keypoints2d))

    tri_t0 = time.perf_counter()
    selected_gpu_info = _run_multigpu_triangulation_phase(
        tasks=tasks,
        Pall=dataset.Pall,
        raw_root=raw_root,
        args=args,
    )
    print(
        f'[Timing] Phase A GPU triangulation: {time.perf_counter() - tri_t0:.2f}s '
        f'for {len(tasks)} frame(s)'
    )

    missing_raw = [
        nf for nf in range(start, end)
        if not _raw_triangulation_path(raw_root, nf).is_file()
    ]
    if missing_raw:
        raise RuntimeError(
            f'Phase A finished but raw triangulation is missing {len(missing_raw)} frame(s): '
            f'{missing_raw[:20]}'
        )
    print(
        f'[Phase A complete] triangulated {end - start} frame(s) before Kalman, '
        f'using {len(selected_gpu_info)} eligible GPU worker(s).'
    )

    # ------------------------------------------------------------------
    # Phase B: now, and only now, run Kalman sequentially in time order.
    # ------------------------------------------------------------------
    kalman_manager = None
    kalman_states_per_frame = []
    kalman_initialized_per_frame = []
    kalman_frame_ids = []
    processed_count = 0

    dataset.no_img = not want_images
    for nf in tqdm(range(start, end), desc='phase B: sequential Kalman'):
        raw_keypoints3d, keypoints2d_inliers = _load_raw_triangulation(raw_root, nf)
        keypoints3d = np.asarray(raw_keypoints3d, dtype=np.float64).copy()
        n_joints = keypoints3d.shape[0]

        if use_joint_kalman:
            if kalman_manager is None:
                kalman_manager = RawPredictionJointKalmanManager(
                    n_joints=n_joints,
                    measurement_noise_scale=float(getattr(
                        args, 'joint_kalman_measurement_noise', 0.015
                    )),
                    process_noise_scale=float(getattr(
                        args, 'joint_kalman_process_noise', 0.30
                    )),
                    init_cov_scale=float(getattr(
                        args, 'joint_kalman_init_cov', 1.0
                    )),
                    max_jump=getattr(args, 'joint_kalman_max_jump', None),
                    reset_after_missing=getattr(
                        args, 'joint_kalman_reset_after_missing', -1
                    ),
                    prediction_conf=prediction_conf,
                    dt=float(getattr(args, 'joint_kalman_dt', 1.0 / 30.0)),
                )

                if nf > requested_start:
                    restored = _restore_joint_kalman_resume_checkpoint(
                        kalman_manager,
                        expected_frame_id=nf - 1,
                        args=args,
                    )
                    if not restored:
                        _warm_start_joint_kalman_from_saved_3d(
                            kalman_manager,
                            existing_index=existing_index,
                            resume_frame=nf,
                            start=requested_start,
                        )
            elif kalman_manager.n_joints != n_joints:
                raise RuntimeError(
                    f'Joint count changed from {kalman_manager.n_joints} '
                    f'to {n_joints} at frame {nf}.'
                )

            keypoints3d, kalman_summary = kalman_manager.step_frame(
                keypoints3d,
                prediction_conf=prediction_conf,
            )

            if bool(getattr(args, 'joint_kalman_print_summary', True)):
                print(
                    f'[frame {nf}] [kalman-after-triangulation] '
                    f'measurement={len(kalman_summary["measured_joints"])}, '
                    f'prediction={len(kalman_summary["predicted_joints"])}, '
                    f'uninitialized={len(kalman_summary["uninitialized_joints"])}, '
                    f'rejected={len(kalman_summary["rejected_measurement_joints"])}'
                )

            states6d, initialized_mask = kalman_manager.get_states()
            kalman_states_per_frame.append(states6d)
            kalman_initialized_per_frame.append(initialized_mask)
            kalman_frame_ids.append(nf)

        dataset.write_keypoints3d(keypoints3d, nf)
        processed_count += 1

        if use_joint_kalman:
            _save_joint_kalman_resume_checkpoint(kalman_manager, nf, args)

        if want_images:
            images, annots = dataset[nf]
            if args.vis_det:
                annots_inliers = dict(annots)
                annots_inliers['keypoints'] = keypoints2d_inliers
                dataset.vis_detections(
                    images, annots_inliers, nf, sub_vis=args.sub_vis
                )
            if args.vis_repro:
                kpts_repro = _project_keypoints3d_to_views(
                    keypoints3d,
                    dataset.Pall,
                )
                dataset.vis_repro(
                    images,
                    kpts_repro,
                    nf=nf,
                    sub_vis=args.sub_vis,
                    mode='repro',
                )

    if (
        save_kalman_states
        and use_joint_kalman
        and len(kalman_states_per_frame) > 0
    ):
        os.makedirs(args.out, exist_ok=True)
        state_path = join(args.out, 'kalman_states_xyz_vxyz.npz')
        np.savez_compressed(
            state_path,
            states=np.stack(kalman_states_per_frame, axis=0),
            initialized=np.stack(kalman_initialized_per_frame, axis=0),
            frame_ids=np.asarray(kalman_frame_ids, dtype=np.int64),
            dt=np.asarray(
                float(getattr(args, 'joint_kalman_dt', 1.0 / 30.0)),
                dtype=np.float64,
            ),
            state_order=np.asarray(['x', 'y', 'z', 'vx', 'vy', 'vz']),
        )
        print(f'[Done] Saved Kalman states for resumed segment: {state_path}')

    print(
        f'[Done] Phase A triangulated {processed_count} frame(s) first; '
        f'Phase B then applied sequential Kalman and saved keypoints3d '
        f'from frame {start} through {end - 1}, using only '
        f'{getattr(args, "fallback2d_annot", "annots_fallback")}.'
    )


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
    # Kalman options. Defaults use a 30 FPS strong anti-jitter preset.
    parser.add_argument('--use_joint_kalman', action='store_true', default=True)
    parser.add_argument('--disable_joint_kalman', dest='use_joint_kalman', action='store_false')
    parser.add_argument('--joint_kalman_dt', type=float, default=1.0/30.0, help='Frame interval in seconds; 30 FPS -> 1/30.')
    parser.add_argument('--joint_kalman_measurement_noise', type=float, default=0.015, help='3D measurement standard deviation in metres. 30 FPS anti-jitter preset: 0.015 m.')
    parser.add_argument('--joint_kalman_process_noise', type=float, default=0.30, help='Acceleration standard deviation in m/s^2. Lower = stronger smoothing; 30 FPS anti-jitter preset: 0.30.')
    parser.add_argument('--joint_kalman_init_cov', type=float, default=1.0)
    parser.add_argument('--joint_kalman_max_jump', type=float, default=None)
    parser.add_argument('--joint_kalman_reset_after_missing', type=int, default=-1, help='Negative means never reset an initialized joint.')
    parser.add_argument('--save_kalman_states', action='store_true', default=True)
    parser.add_argument('--disable_save_kalman_states', dest='save_kalman_states', action='store_false')

    # Kalman filter on optimized MANO/SMPL global RT parameters.
    # This is applied inside easymocap.pipeline.basic.multi_stage_optimize.
    parser.add_argument('--use_param_kalman', action='store_true', default=True)
    parser.add_argument('--disable_param_kalman', dest='use_param_kalman', action='store_false')
    parser.add_argument('--param_kalman_measurement_noise', type=float, default=5)
    parser.add_argument('--param_kalman_process_noise', type=float, default=1)
    parser.add_argument('--param_kalman_init_cov', type=float, default=1.0)
    parser.add_argument('--param_kalman_max_jump', type=float, default=None)

    # All-pairs RANSAC triangulation options.
    parser.add_argument('--select2d_conf_thres', type=float, default=0.3)
    parser.add_argument('--select2d_repro_thres', type=float, default=50)
    parser.add_argument('--select2d_min_view', type=int, default=3)
    parser.add_argument('--select2d_prefer_min_inliers', type=int, default=4)
    parser.add_argument('--select2d_max_candidate_views', type=int, default=32)
    parser.add_argument('--select2d_print_each_joint', action='store_true', default=True)
    parser.add_argument('--disable_select2d_print_each_joint', dest='select2d_print_each_joint', action='store_false')


    # Multi-GPU triangulation. IDs are logical CUDA ids *inside the current
    # CUDA_VISIBLE_DEVICES set*. run_extract...sh uses this to give the right and
    # left hands disjoint GPU groups when running bimanual reconstruction.
    parser.add_argument(
        '--triangulation_gpus', type=str, default='auto',
        help='Logical CUDA ids for triangulation, e.g. 0,1,2; auto uses every visible GPU with enough free VRAM.'
    )
    parser.add_argument(
        '--triangulation_min_free_vram_mb', type=int, default=32768,
        help='Use a visible GPU for triangulation only when free VRAM is >= this many MiB (default 32768 = 32 GiB).'
    )

    # The following directory is the ONLY 2D annotation source.
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

    # This modified pipeline intentionally ignores the primary annotation
    # directory and uses annots_fallback as its only 2D input.
    args.annot = args.fallback2d_annot
    fallback_root = join(args.path, args.annot)
    fallback_root_existed = os.path.isdir(fallback_root)
    os.makedirs(fallback_root, exist_ok=True)
    if fallback_root_existed:
        print(f'[Annotation source] ONLY: {fallback_root}')
    else:
        print(f'[Annotation source] Created empty fallback directory: {fallback_root}')

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

    # Always let mv1pmf_skel perform its cheap completeness scan.  It returns
    # immediately when every requested keypoints3d frame already exists, and
    # resumes from the first missing frame otherwise.
    mv1pmf_skel(dataset, check_repro=True, args=args)
    mv1pmf_smpl(dataset, args)
