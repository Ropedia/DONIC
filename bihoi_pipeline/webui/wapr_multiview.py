"""Batch independent camera views using WAPR's unchanged six-pose recipe.

Camera views share GPU launches, not attention: each camera retains its own
six-candidate WBPS group, intrinsics, RGB-D image and pose. All computations use
the existing torch model precision, crop size and refinement iteration counts.
"""
import numpy as np
import torch

from wapr import recipe
from wapr.export_tracking_engine import tracking_hypotheses
from wapr.ogl import depth2xyzmap_batch


@torch.inference_mode()
def track_views(estimator, mesh, mesh_center, diameter, observations, previous, centered):
    count = len(observations)
    if not count:
        return []
    if any(estimator.nets[name].engine is not None for name in ("wapr_wo_mask", "sapr", "wbps")):
        raise ValueError("Multi-camera batching requires the torch backend")
    device = estimator.device
    cameras = np.stack([obs["K"] for obs in observations]).astype(np.float32)
    rgb = torch.as_tensor(np.stack([obs["rgb"] for obs in observations]), device=device, dtype=torch.float32)
    depth = torch.as_tensor(np.stack([obs["depth"] for obs in observations]), device=device, dtype=torch.float32)
    xyz = depth2xyzmap_batch(depth, torch.as_tensor(cameras, device=device)).permute(0, 3, 1, 2)
    previous = torch.as_tensor(np.asarray(previous, np.float32), device=device)
    centered = torch.as_tensor(np.asarray(centered, np.float32), device=device)
    mesh_id = estimator.renderer.cached_mesh_id(mesh)

    def inputs(group):
        # Keep each camera's candidates adjacent, exactly as in official tracking.
        return dict(rgb=rgb.repeat_interleave(group, dim=0),
                    depth=depth.repeat_interleave(group, dim=0), mask=None,
                    K=np.repeat(cameras, group, axis=0), mesh=mesh,
                    center=mesh_center, diameter=diameter,
                    observed_xyz=xyz.repeat_interleave(group, dim=0),
                    mesh_ids=[mesh_id] * (count * group))

    wapr_in = torch.stack((centered, previous), dim=1).reshape(count * 2, 4, 4)
    wapr = estimator._refine(
        estimator.nets["wapr_wo_mask"], wapr_in, **inputs(2),
        iters=recipe.wapr_iters, max_rot=recipe.wapr_max_rot_rad,
        crop_ratio=recipe.refine_crop_ratio, use_mask=False, group_size=2,
    ).reshape(count, 2, 4, 4)
    sapr_in = torch.stack((centered, wapr[:, 0], wapr[:, 1]), dim=1).reshape(count * 3, 4, 4)
    sapr = estimator._refine(
        estimator.nets["sapr"], sapr_in, **inputs(3),
        iters=recipe.sapr_iters, max_rot=recipe.sapr_max_rot_rad,
        crop_ratio=recipe.refine_crop_ratio, use_mask=False, group_size=3,
    ).reshape(count, 3, 4, 4)
    candidates = torch.stack((previous, centered, wapr[:, 0], sapr[:, 0], sapr[:, 1], sapr[:, 2]), dim=1)
    group = len(tracking_hypotheses)
    within, between = estimator._wbps(candidates.reshape(count * group, 4, 4), **inputs(group), L=group)
    within, between = within.reshape(count, group), between.reshape(count, group)
    best = within.argmax(dim=1)
    selected = candidates[torch.arange(count, device=device), best].cpu().numpy()
    scores = ((100.0 - between.max(dim=1).values) / 200.0).cpu().numpy()
    choices = best.cpu().numpy()
    return [dict(pose_4x4=selected[i], score_6d=float(scores[i]),
                 hypothesis=tracking_hypotheses[int(choices[i])]) for i in range(count)]
