"""
Pose Alignment Module

This module provides pose optimization functionality to refine SAM3D's Stage 1
pose output by aligning with DA3 point clouds.

Main components:
- PoseOptimizer: Core optimizer class
- extract_object_pointcloud_from_scene: Extract object points from DA3 scene
- run_pose_optimization: Command-line interface

Usage:
    python -m sam3d_objects.pose_align.run_pose_optimization \
        --sam3d_result <result_dir> \
        --da3_output <da3_output.npz> \
        --mask_dir <mask_dir> \
        --num_views 8
"""

from .pose_optimization import (
    PoseOptimizer,
    extract_object_pointcloud_from_scene,
    project_points_to_frame,
)

__all__ = [
    'PoseOptimizer',
    'extract_object_pointcloud_from_scene',
    'project_points_to_frame',
]
