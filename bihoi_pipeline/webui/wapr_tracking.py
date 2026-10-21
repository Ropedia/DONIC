#!/usr/bin/env python3
"""WAPR temporal tracking for the calibrated, metric RGB-D capture pipeline.

Resident estimators/meshes per sequence. Initialize once with 240
FoundationPose hypotheses followed by one full six-candidate tracking update;
track with three center-update candidates, and select a consistent group of
camera estimates. On consensus failure, register 96 WAPR hypotheses in the
largest-mask view, seeded by the multi-view triangulated center. Outputs
use ORIGINAL mesh coordinates in metres, matching the existing geometry and
hand/object optimizers. Input images are already sampled by Stage 1.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np

try:
    from .masked_depth import masked_depth_world_points, remove_statistical_outliers
except ImportError:
    from masked_depth import masked_depth_world_points, remove_statistical_outliers

try:
    from .settings import PIPELINE_ROOT, PROJECT_ROOT, MODEL_ROOT, python_for
except ImportError:
    from settings import PIPELINE_ROOT, PROJECT_ROOT, MODEL_ROOT, python_for

ROOT = PIPELINE_ROOT
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
INITIALIZATION_METHOD = "foundationpose_240_refine5_then_tracking6"
REINITIALIZATION_METHOD = "wapr96_sapr_wbps"
REINITIALIZATION_VIEWS = 12
REINITIALIZATION_INPLANE = 8
REINITIALIZATION_HYPOTHESES = REINITIALIZATION_VIEWS * REINITIALIZATION_INPLANE
SIX_HYPOTHESES = ["previous", "depth_center", "wapr_center", "sapr_center",
                  "sapr_wapr_center", "sapr_wapr_previous"]


def natural_key(value):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(value))]


def frame_files(folder, extensions):
    files = {}
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in extensions:
                if path.stem in files:
                    raise ValueError(f"Duplicate frame id {path.stem} in {folder}")
                files[path.stem] = path
    return files


def original_pose_at_center(pose, center_xyz, mesh_center):
    """Move the mesh's bounding-box center, preserving its original-coordinate R."""
    out = np.asarray(pose, np.float64).copy()
    out[:3, 3] = center_xyz - out[:3, :3] @ mesh_center
    return out


def triangulated_center_in_camera(center_world, world_to_camera):
    """Validate a triangulated seed before calling single-view registration."""
    if center_world is None:
        return None
    center_world = np.asarray(center_world, dtype=np.float64)
    if center_world.shape != (3,) or not np.isfinite(center_world).all():
        return None
    center_cam = (world_to_camera @ np.r_[center_world, 1])[:3]
    if not np.isfinite(center_cam).all() or center_cam[2] <= 0:
        return None
    return center_cam


def triangulate_center(observations, max_error=30.0):
    """RANSAC over mask-center rays; return None if fewer than 3 views agree."""
    if len(observations) < 3:
        return None
    origins, rays, projections, pixels = [], [], [], []
    for obs in observations:
        ext, K = obs["ext"], obs["K"]
        uv = obs["center_uv"]
        ray = ext[:3, :3].T @ np.linalg.solve(K, [uv[0], uv[1], 1])
        rays.append(ray / np.linalg.norm(ray))
        origins.append(-ext[:3, :3].T @ ext[:3, 3])
        projections.append(K @ ext[:3])
        pixels.append(uv)
    rays, origins, pixels = map(np.asarray, (rays, origins, pixels))
    projectors = np.eye(3)[None] - rays[:, :, None] * rays[:, None, :]

    def solve(ids):
        A = projectors[ids].sum(axis=0)
        if np.linalg.cond(A) > 1e8:
            return None
        return np.linalg.solve(A, np.einsum("nij,nj->ni", projectors[ids], origins[ids]).sum(axis=0))

    best_ids, best_error = [], np.inf
    for i, j in itertools.combinations(range(len(observations)), 2):
        if abs(rays[i] @ rays[j]) > np.cos(np.deg2rad(2)):
            continue
        xyz = solve([i, j])
        if xyz is None:
            continue
        p = np.asarray(projections) @ np.r_[xyz, 1]
        errors = np.linalg.norm(p[:, :2] / np.maximum(p[:, 2:], 1e-8) - pixels, axis=1)
        ids = np.flatnonzero((errors < max_error) & (p[:, 2] > 0))
        error = errors[ids].sum()
        if len(ids) > len(best_ids) or (len(ids) == len(best_ids) and error < best_error):
            best_ids, best_error = ids, error
    return solve(best_ids) if len(best_ids) >= 3 else None


def consensus(candidates, mesh_center, min_views=3, rotation_deg=30., translation_m=.05):
    """Largest all-pairs-consistent group; keep its best-view R and average centers."""
    n = len(candidates)
    if n < min_views:
        return None
    poses = np.asarray([c["pose"] for c in candidates])
    centers = poses[:, :3, 3] + poses[:, :3, :3] @ mesh_center
    compatible = np.eye(n, dtype=bool)
    for i, j in itertools.combinations(range(n), 2):
        cosine = (np.trace(poses[i, :3, :3].T @ poses[j, :3, :3]) - 1) / 2
        angle = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
        compatible[i, j] = compatible[j, i] = (
            angle <= rotation_deg and np.linalg.norm(centers[i] - centers[j]) <= translation_m)
    for size in range(n, min_views - 1, -1):
        groups = [ids for ids in itertools.combinations(range(n), size)
                  if compatible[np.ix_(ids, ids)].all()]
        if groups:
            ids = max(groups, key=lambda ids: sum(candidates[i]["area"] for i in ids))
            best = max(ids, key=lambda i: candidates[i]["area"])
            xyz = np.average(centers[list(ids)], axis=0, weights=[candidates[i]["area"] for i in ids])
            pose = original_pose_at_center(poses[best], xyz, mesh_center)
            return pose, candidates[best], [candidates[i]["cam"] for i in ids]
    return None


class Capture:
    def __init__(self, args):
        # These IO helpers have no dependency on FoundationPose's neural networks.
        sys.path.insert(0, str(ROOT / "FoundationPose"))
        from postopt_geometry_global_pose import load_extrinsics_world_to_cam, load_intrinsics, read_depth_m
        self.read_depth = read_depth_m
        self.args = args
        scene = args.scene_dir
        self.extrinsics = load_extrinsics_world_to_cam(scene / "extri.yml")
        self.images = {cam: frame_files(scene / "images" / cam, IMAGE_EXTS) for cam in self.extrinsics}
        self.cams = sorted([cam for cam, files in self.images.items() if files], key=natural_key)
        if not self.cams:
            raise ValueError("No calibrated images/<camera>/<frame> files")
        self.main_cam = args.main_cam or ("1" if "1" in self.cams else self.cams[0])
        if self.main_cam not in self.cams:
            raise ValueError(f"Main camera {self.main_cam} has no images")
        self.frames = sorted(self.images[self.main_cam], key=natural_key)
        if args.max_frames:
            self.frames = self.frames[:args.max_frames]
        # Stage 1 writes intri.yml for the prepared image resolution. Use the same
        # per-camera K as the geometry optimizer; do not infer sensor size from cx.
        self.Ks = load_intrinsics(scene / "intri.yml", self.cams, scene / "cam_K.txt")
        for cam, K in self.Ks.items():
            if not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
                raise ValueError(f"Invalid intrinsics for camera {cam}")
        self.masks = {cam: frame_files(scene / "masks" / cam, IMAGE_EXTS) for cam in self.cams}
        self.depths = {cam: frame_files(scene / "depth" / cam, IMAGE_EXTS | {".npy"}) for cam in self.cams}
        self.shape = (args.height, args.width)

    def observation(self, cam, frame):
        if frame not in self.images[cam] or frame not in self.masks[cam]:
            return None
        mask = cv2.imread(str(self.masks[cam][frame]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unreadable mask: {self.masks[cam][frame]}")
        mask = cv2.resize(mask, (self.args.width, self.args.height), interpolation=cv2.INTER_NEAREST) > 0
        area = int(mask.sum())
        if area < self.args.min_mask_area:
            return None
        y, x = np.nonzero(mask)
        uv = cv2.minAreaRect(np.column_stack((x, y)).astype(np.float32))[0]
        return dict(cam=cam, mask=mask, area=area, center_uv=uv, K=self.Ks[cam],
                    ext=self.extrinsics[cam], frame=frame)

    def observations(self, frame, pool=None):
        read = lambda cam: self.observation(cam, frame)
        out = [obs for obs in (pool.map(read, self.cams) if pool else map(read, self.cams)) if obs is not None]
        return sorted(out, key=lambda obs: obs["area"], reverse=True)

    def load_depth(self, obs):
        cam, frame = obs["cam"], obs["frame"]
        path = self.depths[cam].get(frame)
        if path is None:
            return None
        depth = self.read_depth(path, self.args.depth_scale, 1.)
        if depth.shape != self.shape:
            depth = cv2.resize(depth, (self.args.width, self.args.height), interpolation=cv2.INTER_NEAREST)
        if np.count_nonzero(np.isfinite(depth) & (depth > 0) & obs["mask"]) < self.args.min_mask_area:
            return None
        return depth

    def hydrate(self, obs):
        depth = self.load_depth(obs)
        if depth is None:
            return False
        cam, frame = obs["cam"], obs["frame"]
        rgb = cv2.imread(str(self.images[cam][frame]))
        if rgb is None or rgb.shape[:2] != self.shape:
            raise ValueError(f"Expected prepared RGB {self.args.width}x{self.args.height}: {self.images[cam][frame]}")
        obs.update(rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth=depth)
        return True

    def fused_depth_cloud(self, frame, hydrated=()):
        """Merge and filter this frame's masked depths from ALL calibrated cameras.

        The inference top-k and preview camera count do not limit fusion. Cached
        inference depths avoid repeat decoding during first-frame registration.
        Returns retained world points, contributing cameras and filter statistics.
        """
        cached = {obs["cam"]: obs for obs in hydrated if obs["frame"] == frame}

        def cloud_for_camera(cam):
            obs = cached.get(cam)
            if obs is None:
                obs = self.observation(cam, frame)
            if obs is None:
                return None
            depth = obs.get("depth")
            if depth is None:
                depth = self.load_depth(obs)
            if depth is None:
                return None
            points = masked_depth_world_points(depth, obs["mask"], obs["K"], obs["ext"])
            return (cam, points) if len(points) >= self.args.min_mask_area else None

        with ThreadPoolExecutor(max_workers=self.args.io_workers) as pool:
            clouds = [item for item in pool.map(cloud_for_camera, self.cams) if item is not None]
        points = np.concatenate([points for _, points in clouds]) if clouds else np.empty((0, 3))
        points, stats = remove_statistical_outliers(
            points, neighbors=self.args.cloud_outlier_neighbors, std_ratio=self.args.cloud_outlier_std_ratio,
            workers=self.args.io_workers, enabled=not self.args.no_cloud_outlier_removal)
        return points, [cam for cam, _ in clouds], stats

    def prepare_frame(self, frame, pool):
        observations = self.observations(frame, pool)
        center = triangulate_center(observations)
        usable = []
        # Preserve the exact ranked top-k selection, including missing-depth fallback.
        for start in range(0, len(observations), self.args.topk):
            batch = observations[start:start + self.args.topk]
            usable.extend(obs for obs, valid in zip(batch, pool.map(self.hydrate, batch)) if valid)
            if len(usable) >= self.args.topk:
                break
        return usable[:self.args.topk], center

    def prepared_frames(self):
        # A single-frame lookahead overlaps CPU decoding/triangulation with GPU work.
        # Neither frame order nor the number of inference views changes.
        with ThreadPoolExecutor(max_workers=self.args.io_workers) as pool, ThreadPoolExecutor(max_workers=1) as prefetch:
            future = prefetch.submit(self.prepare_frame, self.frames[0], pool) if self.frames else None
            for index, frame in enumerate(self.frames):
                usable, center = future.result()
                if index + 1 < len(self.frames):
                    future = prefetch.submit(self.prepare_frame, self.frames[index + 1], pool)
                yield frame, usable, center


class WAPRBackend:
    def __init__(self, args):
        # Calling a venv's Python by absolute path does not add its build tools
        # (or those inherited from its base environment) to PATH.
        bins = [str(Path(sys.executable).parent), str(Path(sys.base_prefix) / "bin")]
        os.environ["PATH"] = os.pathsep.join(bins + [os.environ.get("PATH", "")])
        os.environ["WAPR_BACKEND"] = args.backend
        os.environ["WAPR_RENDERER"] = args.renderer
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("WAPR_CACHE_DIR", str(MODEL_ROOT / "wapr"))
        import pybind11
        prefixes = [str(PROJECT_ROOT / "web_demos/.native/usr"), pybind11.get_cmake_dir()]
        if os.environ.get("CMAKE_PREFIX_PATH"):
            prefixes.append(os.environ["CMAKE_PREFIX_PATH"])
        os.environ["CMAKE_PREFIX_PATH"] = os.pathsep.join(prefixes)
        sys.path.insert(0, str(args.wapr_repo))
        import torch
        import trimesh
        from wapr import WAPREstimator, recipe
        from wapr.tracking import track_many_categories_many_instances
        if not torch.cuda.is_available():
            raise RuntimeError("WAPR requires a CUDA GPU")
        torch.cuda.set_device(args.device)
        torch.set_num_threads(args.cpu_threads)
        torch.manual_seed(0)
        recipe.visualize = False
        mesh = trimesh.load(args.mesh_file or args.scene_dir / "meshes/compressed.ply", force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError("Expected a finite triangular mesh in metres")
        self.center = np.asarray(mesh.bounds.mean(axis=0), np.float64)
        self.diameter = float(np.linalg.norm(mesh.extents))
        if self.diameter <= 0:
            raise ValueError("Mesh has zero diameter")
        self.est = WAPREstimator(device=args.device)
        self.mesh = self.est.prepare_meshes([mesh])[0]
        self.track_fn = track_many_categories_many_instances
        self.args = args
        self.foundationpose = None
        self.view_workers = args.view_workers
        self.parallel = None
        if self.view_workers > 1:
            try:
                from .wapr_parallel import ParallelViews
            except ImportError:
                from wapr_parallel import ParallelViews
            self.parallel = ParallelViews(args)

    def estimate(self, obs, previous_world=None, center_world=None):
        if previous_world is None:
            if center_world is None:
                raise ValueError("Registration requires the fused masked depth world centroid")
            return self.initialize(obs, center_world)
        result = self._track_camera_pose(obs, obs["ext"] @ previous_world, center_world,
                                         center_only=True)
        return self.world_result(obs, result)

    def _track_camera_pose(self, obs, pose_camera, center_world, *, center_only):
        """Share the same center update between normal tracking and registration."""
        center_pose = np.asarray(pose_camera).copy()
        if center_world is not None:
            center_cam = (obs["ext"] @ np.r_[center_world, 1])[:3]
            center_pose = original_pose_at_center(pose_camera, center_cam, self.center)
        return self.track_fn(
            self.est, rgb=obs["rgb"], depth_m=obs["depth"], K=obs["K"],
            instances=[dict(track_id="object", mesh=self.mesh, diameter_m=self.diameter,
                            pose_4x4=pose_camera, center_pose_4x4=center_pose)],
            center_only=center_only)[0]

    def world_result(self, obs, result):
        pose = np.asarray(result["pose_4x4"], np.float64)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all() or pose[2, 3] <= 0
                or not np.isfinite(result["score_6d"])):
            raise RuntimeError(f"WAPR returned an invalid pose for {obs['cam']}/{obs['frame']}")
        return dict(pose=np.linalg.inv(obs["ext"]) @ pose, cam=obs["cam"],
                    area=obs["area"], score=float(result["score_6d"]),
                    hypothesis=result.get("hypothesis", "registration"))

    def initialize(self, obs, center_world, tracking_center_world=None):
        """First-frame registration: FP winner -> six candidates -> WBPS winner."""
        try:
            from .foundationpose_reinit import FoundationPoseClient
        except ImportError:
            from foundationpose_reinit import FoundationPoseClient
        center_cam = (obs["ext"] @ np.r_[center_world, 1])[:3]
        if not np.isfinite(center_cam).all() or center_cam[2] <= 0:
            raise ValueError("Fused depth centroid must be finite and in front of the registration camera")
        print(f"[WAPR registration] {obs['frame']} cam={obs['cam']}: FoundationPose 240 hypotheses, "
              "5 refinement iterations, scorer selects one pose", flush=True)
        if self.foundationpose is None:
            self.foundationpose = FoundationPoseClient(self.args)
        registration = self.foundationpose.register(obs, center_cam)
        print(f"[WAPR registration] FoundationPose winner={registration['best_id']}; "
              "running full six-candidate tracking and WBPS selection", flush=True)
        result = self.refine_initialization(obs, registration["pose_4x4"], tracking_center_world)
        best = self.world_result(obs, result)
        best.update(initialization_method=INITIALIZATION_METHOD,
                    foundationpose_hypotheses=registration["hypotheses"],
                    foundationpose_refine_iterations=registration["refine_iterations"],
                    foundationpose_score=registration["score"])
        return best

    def refine_initialization(self, obs, pose_camera, tracking_center_world=None):
        """Use the FP winner as P; WBPS may choose any of the full six candidates."""
        return self._track_camera_pose(obs, pose_camera, tracking_center_world, center_only=False)

    def register_best_mask(self, obs, center_world):
        """WAPR's masked registration recipe, with 96 rotations and an XYZ seed.

        The official estimator converts the mesh-center seed to each hypothesis's
        original-mesh translation, runs WAPR -> SAPR -> WBPS, and returns the
        winner with its refined translation. Reuse this resident model/mesh.
        """
        center_cam = triangulated_center_in_camera(center_world, obs["ext"])
        if center_cam is None:
            raise ValueError("WAPR96 registration requires a finite triangulated center in front of the selected camera")
        print(f"[WAPR reinitialization] {obs['frame']} cam={obs['cam']}: triangulated XYZ, "
              f"{REINITIALIZATION_HYPOTHESES} hypotheses "
              f"({REINITIALIZATION_VIEWS} directions x {REINITIALIZATION_INPLANE} in-plane) -> WAPR -> SAPR -> WBPS", flush=True)
        result = self.est.estimate_one_category_one_instance(
            rgb=obs["rgb"], depth_m=obs["depth"], K=obs["K"], mesh=self.mesh,
            diameter_m=self.diameter, mask=obs["mask"],
            n_view=REINITIALIZATION_VIEWS, n_inplane=REINITIALIZATION_INPLANE,
            center_xyz_m=center_cam)
        best = self.world_result(obs, result)
        best.update(hypothesis=REINITIALIZATION_METHOD,
                    reinitialization_method=REINITIALIZATION_METHOD,
                    reinitialization_hypotheses=REINITIALIZATION_HYPOTHESES,
                    reinitialization_xyz="multiview_mask_triangulation",
                    reinitialization_center_world=np.asarray(center_world).tolist())
        return best

    def estimate_views(self, observations, previous_world, center_world):
        if self.parallel is not None:
            return self.parallel.estimate(observations, previous_world, center_world)
        return [self.estimate(obs, previous_world, center_world) for obs in observations]

    def close(self):
        try:
            if self.parallel is not None:
                self.parallel.close()
        finally:
            if self.foundationpose is not None:
                self.foundationpose.close()


def track_capture(args, capture, backend, preview=None):
    output = args.output_dir or args.scene_dir / "debug_depth"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("ob_in_world", "ob_in_cam", "ob_in_selected_cam"):
        folder = output / name
        if folder.exists() and any(folder.glob("*.txt")):
            raise ValueError(f"Output already contains poses: {folder}. Use a fresh --output_dir or Stage 3 in the WebUI.")
    # Commit only a complete sequence. An OOM/failed run can be retried by the
    # scheduler without mixing partial poses from two different attempts.
    with tempfile.TemporaryDirectory(prefix=".wapr-pending-", dir=output) as pending:
        work = Path(pending)
        try:
            report = _track_frames(args, capture, backend, work, preview)
        except BaseException:
            log = work / "wapr_tracking_log.csv"
            if log.exists():
                shutil.copy2(log, output / "wapr_tracking_failed.csv")
            raise
        for path in work.iterdir():
            dest = output / path.name
            if dest.is_dir():
                dest.rmdir()  # only empty folders are allowed by the preflight above
            os.replace(path, dest)
    return report


def _track_frames(args, capture, backend, output, preview=None):
    for name in ("ob_in_world", "ob_in_cam", "ob_in_selected_cam"):
        (output / name).mkdir()
    previous = None
    records = []
    started = time.perf_counter()
    with (output / "wapr_tracking_log.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame", "mode", "camera", "views", "consensus_views", "score", "seconds", "initialization_xyz",
                                                   "initialization_views", "initialization_points", "initialization_center_world",
                                                   "initialization_raw_points", "initialization_removed_points", "initialization_outlier_threshold_m",
                                                   "initialization_method", "foundationpose_hypotheses",
                                                   "foundationpose_refine_iterations", "foundationpose_score", "hypothesis",
                                                   "consensus_failed", "reinitialization_method", "reinitialization_hypotheses",
                                                   "reinitialization_xyz", "reinitialization_center_world", "reinitialization_skip_reason"])
        writer.writeheader()
        def serial_frames():
            for frame in capture.frames:
                observations = capture.observations(frame)
                usable = []
                for obs in observations:
                    if capture.hydrate(obs):
                        usable.append(obs)
                    if len(usable) == args.topk:
                        break
                yield frame, usable, triangulate_center(observations)

        frames = capture.prepared_frames() if hasattr(capture, "prepared_frames") else serial_frames()
        tick = time.perf_counter()
        for idx, (frame, usable, center_world) in enumerate(frames):
            candidates, accepted = [], None
            consensus_failed = False
            cloud_cameras, cloud_count, depth_center = [], 0, None
            cloud_stats = {}
            if usable and previous is not None:
                if hasattr(backend, "estimate_views"):
                    candidates = backend.estimate_views(usable, previous, center_world)
                else:
                    candidates = [backend.estimate(obs, previous, center_world) for obs in usable]
                accepted = consensus(candidates, backend.center, min(args.min_views, args.topk),
                                     args.rotation_deg, args.translation_m)
            if accepted is not None:
                pose, best, views = accepted
                mode = "tracking"
            elif usable and previous is not None:
                consensus_failed = True
                best_obs = max(usable, key=lambda obs: obs["area"])
                if triangulated_center_in_camera(center_world, best_obs["ext"]) is not None:
                    best = backend.register_best_mask(best_obs, center_world)
                    mode = "reinit_wapr96"
                else:
                    # A missing/invalid triangulation must never silently become
                    # a single-view depth seed. Preserve the best-mask tracking
                    # result for this frame and report why registration skipped.
                    best = dict(next(c for c in candidates if c["cam"] == best_obs["cam"]))
                    best["reinitialization_skip_reason"] = (
                        "triangulation_unavailable" if center_world is None else "triangulation_invalid_for_best_mask")
                    mode = "tracking_best_mask"
                    logging.warning("Frame %s: WAPR96 skipped (%s); keeping best-mask tracking cam=%s",
                                    frame, best["reinitialization_skip_reason"], best["cam"])
                pose, views = best["pose"], []
            elif usable:
                points, cloud_cameras, cloud_stats = capture.fused_depth_cloud(frame, hydrated=usable)
                cloud_count = len(points)
                if not cloud_count:
                    raise ValueError(f"Frame {frame} has no multi-view masked depth points after outlier filtering")
                print(f"[WAPR cloud] {frame}: {cloud_stats['raw_points']} -> {cloud_count} points "
                      f"from {len(cloud_cameras)} views ({cloud_stats['method']}: {cloud_stats['status']})", flush=True)
                depth_center = points.mean(axis=0)
                best = backend.initialize(usable[0], depth_center, tracking_center_world=center_world)
                pose, views = best["pose"], [best["cam"]]
                mode = "initialization"
            elif previous is not None:
                # Keep the frame timeline intact through temporary full occlusion.
                pose, views = previous.copy(), []
                best = dict(cam=capture.main_cam, score=None)
                mode = "held_no_rgbd"
                logging.warning("Frame %s has no valid masked RGB-D; holding last pose, will resume tracking", frame)
            else:
                raise ValueError(f"First frame {frame} has no usable object mask and metric depth")
            previous = pose.copy()
            for name, transform in (("ob_in_world", np.eye(4)),
                                    ("ob_in_cam", capture.extrinsics[capture.main_cam]),
                                    ("ob_in_selected_cam", capture.extrinsics[best["cam"]])):
                np.savetxt(output / name / f"{frame}.txt", transform @ pose, fmt="%.9f")
            row = dict(frame=frame, mode=mode, camera=best["cam"], views=len(usable),
                       consensus_failed=int(consensus_failed),
                       consensus_views=";".join(views), score=best["score"], hypothesis=best.get("hypothesis", ""),
                       seconds=time.perf_counter() - tick,
                       initialization_xyz="multiview_masked_depth_pointcloud_centroid" if cloud_count else "",
                       initialization_views=";".join(cloud_cameras), initialization_points=cloud_count or "",
                       initialization_center_world=json.dumps(depth_center.tolist()) if depth_center is not None else "",
                       initialization_raw_points=cloud_stats.get("raw_points", ""),
                       initialization_removed_points=cloud_stats.get("removed_points", ""),
                       initialization_outlier_threshold_m=cloud_stats.get("threshold_m", ""),
                       reinitialization_center_world=(json.dumps(best["reinitialization_center_world"])
                                                      if "reinitialization_center_world" in best else ""),
                       **{key: best.get(key, "") for key in ("initialization_method", "foundationpose_hypotheses",
                                                           "foundationpose_refine_iterations", "foundationpose_score",
                                                           "reinitialization_method", "reinitialization_hypotheses",
                                                           "reinitialization_xyz", "reinitialization_skip_reason")})
            writer.writerow(row)
            stream.flush()
            records.append(row)
            print(f"[WAPR {idx+1}/{len(capture.frames)}] {frame} {mode} cam={best['cam']} "
                  f"consensus={len(views)}/{len(usable)} {row['seconds']:.3f}s", flush=True)
            if preview is not None:
                preview.publish(frame, pose, [obs["cam"] for obs in usable], idx + 1, len(capture.frames), mode)
            tick = time.perf_counter()
    registration_recipe = dict(method=INITIALIZATION_METHOD, hypotheses=240,
                               foundationpose_refine_iterations=5,
                               camera_policy="largest_valid_mask",
                               tracking_hypotheses=SIX_HYPOTHESES,
                               tracking_center="mask_triangulation_or_previous_pose",
                               preserve_refined_translation=True)
    report = dict(method="wapr", backend=args.backend, renderer=args.renderer, fps=args.fps, frames=len(records),
                  pose_convention="original_mesh_to_world", translation_unit="metre",
                  mesh_center_m=backend.center.tolist(), diameter_m=backend.diameter,
                  topk=args.topk, min_views=args.min_views, depth_scale=args.depth_scale,
                  view_workers=getattr(backend, "view_workers", 1), initialization_xyz="multiview_masked_depth_pointcloud_centroid",
                  initialization_camera_policy="all_valid_calibrated_cameras",
                  tracking_hypotheses=["WAPR(C)", "SAPR(C)", "SAPR(WAPR(C))"],
                  initialization=registration_recipe,
                  reinitialization=dict(enabled=True, trigger="no_multiview_consensus",
                                        method=REINITIALIZATION_METHOD, camera_policy="largest_valid_mask",
                                        xyz="multiview_mask_triangulation", hypotheses=REINITIALIZATION_HYPOTHESES,
                                        n_view=REINITIALIZATION_VIEWS, n_inplane=REINITIALIZATION_INPLANE,
                                        invalid_triangulation="keep_best_mask_tracking_result",
                                        preserve_refined_translation=True),
                  consensus_failure="largest_valid_mask_wapr_registration_96",
                  consensus_failed_frames=sum(r["consensus_failed"] for r in records),
                  rgbd_recovery="track_from_last_pose",
                  cloud_outlier_removal=dict(method="none" if args.no_cloud_outlier_removal else "statistical",
                                             neighbors=args.cloud_outlier_neighbors, std_ratio=args.cloud_outlier_std_ratio),
                  elapsed_seconds=time.perf_counter() - started,
                  modes={mode: sum(r["mode"] == mode for r in records) for mode in sorted({r["mode"] for r in records})})
    (output / "wapr_tracking.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--mesh_file", type=Path)
    parser.add_argument("--wapr_repo", type=Path, default=PROJECT_ROOT / "web_demos/third_party/WAPR")
    parser.add_argument("--foundationpose_python", type=Path,
                        default=python_for("foundationpose"),
                        help="FoundationPose environment used by the initialization worker")
    parser.add_argument("--backend", choices=("torch", "trt"), default="torch")
    parser.add_argument("--renderer", choices=("nvdiffrast", "ogl"), default="nvdiffrast",
                        help="CUDA rasterizer avoids OpenGL interop stalls on shared GPUs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=10.)
    parser.add_argument("--depth_scale", type=float, default=1000.)
    parser.add_argument("--main_cam")
    parser.add_argument("--topk", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--min_views", type=int, default=3)
    parser.add_argument("--rotation_deg", type=float, default=30.)
    parser.add_argument("--translation_m", type=float, default=.05)
    parser.add_argument("--min_mask_area", type=int, default=16)
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--io_workers", type=int, default=4)
    parser.add_argument("--cloud_outlier_neighbors", type=int, default=20,
                        help="Number of nearest neighbors for fused point-cloud statistical outlier removal")
    parser.add_argument("--cloud_outlier_std_ratio", type=float, default=2.,
                        help="Keep mean neighbor distances <= cloud mean + this multiple of std")
    parser.add_argument("--no_cloud_outlier_removal", action="store_true",
                        help="Disable statistical filtering for comparison; nonfinite points are still rejected")
    parser.add_argument("--view_workers", type=int, choices=range(1, 5), default=4,
                        help="Resident per-camera inference workers on the selected GPU; 1 is sequential")
    parser.add_argument("--no_live_preview", action="store_true")
    parser.add_argument("--max_frames", type=int, default=0, help="Debug only; 0 processes the entire sampled sequence")
    return parser


def main():
    args = build_parser().parse_args()
    numeric = [args.width, args.height, args.fps, args.depth_scale, args.min_views,
               args.cpu_threads, args.io_workers, args.min_mask_area, args.rotation_deg, args.translation_m,
               args.cloud_outlier_neighbors, args.cloud_outlier_std_ratio]
    if not np.isfinite(numeric).all() or min(numeric) <= 0 or args.max_frames < 0 or args.cloud_outlier_neighbors < 2:
        raise ValueError("Dimensions, FPS, depth scale and counts must be positive; max_frames >= 0; cloud_outlier_neighbors >= 2")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cv2.setNumThreads(1)
    capture = Capture(args)
    print(f"[WAPR] {len(capture.frames)} frames at {args.fps:g} FPS, {len(capture.cams)} cameras, "
          f"up to {args.topk} views per frame; {args.view_workers} resident view worker(s)", flush=True)
    print("[WAPR] tracking: WAPR(C), SAPR(C), SAPR(WAPR(C)); initialization: "
          "best mask -> FoundationPose 240 hypotheses x 5 updates -> winner -> tracking six -> WBPS winner", flush=True)
    print("[WAPR] on consensus failure: best mask + triangulated XYZ -> WAPR 96 hypotheses "
          "-> SAPR -> WBPS; unavailable triangulation keeps best-mask tracking", flush=True)
    # Legacy renderer uses this shared K; the tracker and optimizer use per-camera K.
    np.savetxt(args.scene_dir / "cam_K.txt", capture.Ks[capture.main_cam], fmt="%.9f")
    if args.no_live_preview:
        backend = WAPRBackend(args)
        print("[WAPR] Model and mesh ready; starting temporal tracking", flush=True)
        try:
            track_capture(args, capture, backend)
        finally:
            backend.close()
    else:
        try:
            from .tracking_preview import PreviewPublisher
        except ImportError:
            from tracking_preview import PreviewPublisher
        with PreviewPublisher(args) as preview:
            backend = WAPRBackend(args)
            print("[WAPR] Model and mesh ready; starting temporal tracking", flush=True)
            try:
                track_capture(args, capture, backend, preview)
            finally:
                backend.close()


if __name__ == "__main__":
    main()
