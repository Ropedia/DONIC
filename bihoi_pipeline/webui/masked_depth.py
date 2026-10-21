"""Masked metric depth geometry shared by WAPR initialization and its preview."""
import numpy as np


def masked_depth_points(depth_m, mask, K):
    """Backproject every finite, positive masked depth to camera-space XYZ metres.

    K corresponds to the depth/mask resolution; depth stores camera Z, not range.
    No subsampling is applied here, including when computing the initial center.
    """
    depth = np.asarray(depth_m)
    mask = np.asarray(mask, dtype=bool)
    K = np.asarray(K, dtype=np.float64)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("Depth and mask must have the same 2D shape")
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("Expected finite 3x3 camera intrinsics")
    y, x = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    pixels = np.column_stack((x, y, np.ones(len(x))))
    rays = pixels @ np.linalg.inv(K).T
    if len(rays) and (not np.isfinite(rays).all() or np.any(rays[:, 2] <= 0)):
        raise ValueError("Camera intrinsics must produce finite forward-facing rays")
    return rays * (depth[y, x] / rays[:, 2])[:, None]


def masked_depth_center(depth_m, mask, K):
    """Arithmetic centroid of all valid masked 3D points in camera coordinates."""
    points = masked_depth_points(depth_m, mask, K)
    if not len(points):
        raise ValueError("Cannot initialize XYZ without valid masked depth points")
    return points.mean(axis=0)


def masked_depth_world_points(depth_m, mask, K, world_to_camera):
    """Place one camera's masked depth cloud in the common world frame."""
    points = masked_depth_points(depth_m, mask, K)
    transform = np.asarray(world_to_camera, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Expected a finite 4x4 world-to-camera transform")
    camera_to_world = np.linalg.inv(transform)
    return points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def remove_statistical_outliers(points, neighbors=20, std_ratio=2., workers=4, enabled=True):
    """Exact k-NN statistical filtering, retaining original point coordinates/order.

    A point is an outlier when its mean distance to its k nearest OTHER points
    exceeds the cloud-wide mean + std_ratio * standard deviation. Chunked queries
    bound temporary memory without subsampling or approximating neighbor search.
    Small clouds cannot support the requested neighborhood and are left intact.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an Nx3 point cloud")
    if (not isinstance(neighbors, (int, np.integer)) or neighbors < 2 or
            not np.isfinite(std_ratio) or std_ratio <= 0 or workers < 1):
        raise ValueError("Outlier filtering requires >= 2 neighbors, positive std_ratio and workers")
    finite = np.isfinite(points).all(axis=1)
    valid = points if finite.all() else points[finite]
    stats = dict(method="statistical" if enabled else "none", neighbors=int(neighbors),
                 std_ratio=float(std_ratio), raw_points=len(points), finite_points=len(valid),
                 kept_points=len(valid), removed_points=len(points) - len(valid), threshold_m=None,
                 status="too_few_points" if enabled else "disabled")
    if not enabled or len(valid) <= neighbors:
        return valid, stats

    from scipy.spatial import cKDTree
    tree = cKDTree(valid)
    mean_distances = np.empty(len(valid), dtype=np.float64)
    for start in range(0, len(valid), 32768):
        distances, _ = tree.query(valid[start:start + 32768], k=neighbors + 1, workers=workers)
        mean_distances[start:start + len(distances)] = distances[:, 1:].mean(axis=1)
    cutoff = float(mean_distances.mean() + std_ratio * mean_distances.std())
    keep = mean_distances <= cutoff
    stats.update(status="filtered", threshold_m=cutoff, kept_points=int(keep.sum()),
                 removed_points=len(points) - int(keep.sum()))
    return valid[keep], stats
