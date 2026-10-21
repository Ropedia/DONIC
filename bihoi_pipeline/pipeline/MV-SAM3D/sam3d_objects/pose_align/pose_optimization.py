"""
Pose Optimization: Direct optimization in aligned space

This module optimizes SAM3D pose (scale, rotation, translation) to align with
DA3 point clouds using the complete transformation chain from canonical to aligned space.

Key features:
- Mask-based object extraction from DA3 scene
- Chamfer Distance optimization with regularization
- Early stopping and adaptive learning rates
- Optional mask erosion to remove edge artifacts
"""
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix
from typing import Dict, List, Optional, Tuple
from loguru import logger
import trimesh

# Check for opencv (required for mask erosion)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("opencv-python not found. Mask erosion will be disabled.")
    logger.warning("Install with: pip install opencv-python")


# =============================================================================
# Utility Functions
# =============================================================================

def project_points_to_frame(points_world: np.ndarray, K: np.ndarray, w2c: np.ndarray, H: int, W: int):
    """
    Project 3D points (world space) to image plane.
    
    Args:
        points_world: (N, 3)
        K: (3, 3) intrinsics
        w2c: (4, 4) world-to-camera
        H, W: image dimensions
        
    Returns:
        uv: (N, 2) pixel coordinates
        valid: (N,) boolean mask
        depth: (N,) depth values
    """
    N = len(points_world)
    
    # To camera space
    points_homo = np.hstack([points_world, np.ones((N, 1))])
    points_cam = (w2c @ points_homo.T).T[:, :3]
    
    # Project to image
    points_img = (K @ points_cam.T).T
    depth = points_img[:, 2]
    uv = points_img[:, :2] / (depth[:, None] + 1e-8)
    
    # Check validity
    valid = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    
    return uv, valid, depth


def extract_object_pointcloud_from_scene(
    scene_glb_path: str,
    da3_npz_path: str,
    masks: List[np.ndarray],
    mask_threshold: int = 128,
    depth_tolerance: float = 0.1,
    max_points: int = 100000,
    mask_erosion_kernel: int = 3,
) -> np.ndarray:
    """
    Extract object points from DA3 scene.glb using masks (back-projection method).
    
    This function uses mask back-projection to identify which points in the DA3 scene
    belong to the target object. It projects each scene point to all views, checks if
    it falls within the mask, and verifies depth consistency.
    
    Args:
        scene_glb_path: Path to DA3 scene.glb
        da3_npz_path: Path to DA3 output npz
        masks: List of binary masks (one per view)
        mask_threshold: Threshold for object detection (0-255)
        depth_tolerance: Relative depth matching tolerance (e.g., 0.1 = 10%)
        max_points: Maximum points to return (downsampling if exceeded)
        mask_erosion_kernel: Erosion kernel size (pixels), removes mask edge errors
            - 0: No erosion
            - 3: Recommended (removes 1-2px edge errors)
            - 5: More conservative (removes 2-3px edge errors)
        
    Returns:
        object_points: (N, 3) in aligned space
    """
    from tqdm import tqdm
    
    logger.info("Extracting object pointcloud from scene.glb...")
    
    # Load scene
    scene = trimesh.load(scene_glb_path)
    
    # Find point cloud in scene
    point_cloud = None
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.points.PointCloud) or (
            hasattr(geom, 'vertices') and len(geom.vertices) > 100000
        ):
            point_cloud = geom
            break
    
    if point_cloud is None:
        raise ValueError("No point cloud found in scene.glb")
    
    points_aligned = point_cloud.vertices.copy()
    logger.info(f"  Total points in scene: {len(points_aligned)}")
    
    # Get alignment matrix
    alignment_matrix = None
    if hasattr(scene, 'metadata') and scene.metadata:
        alignment_matrix = scene.metadata.get('hf_alignment')
    
    if alignment_matrix is None:
        raise ValueError("No hf_alignment found in scene metadata")
    
    alignment_matrix = np.array(alignment_matrix)
    
    # Transform to world space
    alignment_inv = np.linalg.inv(alignment_matrix)
    points_world = trimesh.transform_points(points_aligned, alignment_inv)
    
    # Load DA3 data
    da3_data = np.load(da3_npz_path)
    depth = da3_data['depth']
    intrinsics = da3_data['intrinsics']
    extrinsics = da3_data['extrinsics']
    
    N, H, W = depth.shape
    
    # Check mask count vs frame count
    num_masks = len(masks) if masks else 0
    if num_masks < N:
        logger.warning(f"  Only {num_masks} masks provided for {N} DA3 frames. "
                      f"Only frames with masks will be used for point extraction.")
    
    # Back-project to identify object points
    is_object = np.zeros(len(points_world), dtype=bool)
    
    batch_size = 50000
    num_batches = (len(points_world) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Extracting object points"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(points_world))
        batch_points = points_world[start_idx:end_idx]
        
        for frame_idx in range(N):
            K = intrinsics[frame_idx]
            ext = extrinsics[frame_idx]
            depth_map = depth[frame_idx]
            
            # Handle case where masks list is shorter than DA3 frames
            # This happens when user only uses a subset of images for inference
            if frame_idx >= len(masks):
                continue
            
            mask = masks[frame_idx]
            
            # Skip if no mask for this frame
            if mask is None:
                continue
            
            # Convert to 4x4
            if ext.shape == (3, 4):
                w2c = np.eye(4)
                w2c[:3, :4] = ext
            else:
                w2c = ext
            
            # Resize mask if needed
            if mask.shape != (H, W):
                from PIL import Image
                mask = np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST))
            
            # Apply mask erosion to remove edge errors
            if mask_erosion_kernel > 0 and HAS_CV2:
                kernel = np.ones((mask_erosion_kernel, mask_erosion_kernel), np.uint8)
                mask_binary = (mask > mask_threshold).astype(np.uint8)
                mask_eroded = cv2.erode(mask_binary, kernel, iterations=1)
                mask_for_check = mask_eroded * 255
            else:
                mask_for_check = mask
            
            # Project
            uv, valid, depth_proj = project_points_to_frame(batch_points, K, w2c, H, W)
            
            if np.any(valid):
                u = np.clip(uv[valid, 0].astype(int), 0, W-1)
                v = np.clip(uv[valid, 1].astype(int), 0, H-1)
                
                # Depth verification
                actual_depth = depth_map[v, u]
                projected_depth = depth_proj[valid]
                depth_match = np.abs(actual_depth - projected_depth) < (depth_tolerance * actual_depth)
                
                # Check mask
                valid_indices = np.where(valid)[0]
                depth_matched_indices = valid_indices[depth_match]
                
                if len(depth_matched_indices) > 0:
                    u_matched = u[depth_match]
                    v_matched = v[depth_match]
                    mask_values = mask_for_check[v_matched, u_matched]
                    is_object_in_frame = mask_values > mask_threshold
                    
                    batch_is_object = np.zeros(len(batch_points), dtype=bool)
                    batch_is_object[depth_matched_indices] = is_object_in_frame
                    is_object[start_idx:end_idx] |= batch_is_object
    
    # Filter object points (in aligned space)
    object_points = points_aligned[is_object]
    
    logger.info(f"  Object points extracted: {len(object_points)} ({len(object_points)/len(points_aligned)*100:.1f}%)")
    
    # Downsample if needed
    if len(object_points) > max_points:
        indices = np.random.choice(len(object_points), max_points, replace=False)
        object_points = object_points[indices]
        logger.info(f"  Downsampled to {max_points} points")
    
    return object_points


# =============================================================================
# Pose Optimizer
# =============================================================================

class PoseOptimizer:
    """
    Pose optimizer that works in aligned space.
    
    This optimizer transforms the canonical mesh through the complete SAM3D
    transformation chain to aligned space, then optimizes the pose parameters
    (scale, rotation, translation) to minimize Chamfer Distance with target points.
    
    Transformation chain:
        canonical (Z-up) 
        → Y-up rotation
        → scale
        → rotation (quaternion)
        → translation
        → PyTorch3D to CV
        → alignment matrix
        → aligned space
    """
    
    def __init__(
        self,
        canonical_mesh_path: str,
        initial_pose: Dict[str, np.ndarray],
        target_points: np.ndarray,
        alignment_matrix: np.ndarray,
        device: str = 'cuda',
        optimize_scale: bool = False,
    ):
        """
        Args:
            canonical_mesh_path: Path to canonical mesh (result.glb)
            initial_pose: Initial pose with keys 'scale', 'rotation', 'translation'
            target_points: (N, 3) target point cloud in aligned space
            alignment_matrix: (4, 4) hf_alignment matrix from DA3
            device: 'cuda' or 'cpu'
            optimize_scale: If True, optimize scale; if False, keep scale fixed (default: False)
        """
        self.optimize_scale = optimize_scale
        self.device = device
        
        # Load canonical mesh
        logger.info("Loading canonical mesh...")
        sam3d_scene = trimesh.load(canonical_mesh_path)
        
        # Extract canonical vertices from GLB
        if isinstance(sam3d_scene, trimesh.Scene):
            source_vertices = None
            for name, geom in sam3d_scene.geometry.items():
                if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                    source_vertices = geom.vertices
                    logger.info(f"  Found mesh: {name} ({len(source_vertices)} vertices)")
                    break
            if source_vertices is None:
                raise ValueError("No mesh found in canonical GLB")
        else:
            source_vertices = sam3d_scene.vertices
        
        # Sample points from mesh
        if len(source_vertices) > 50000:
            indices = np.random.choice(len(source_vertices), 50000, replace=False)
            source_points = source_vertices[indices]
        else:
            source_points = source_vertices
        
        self.source_canonical = torch.from_numpy(source_points).float().to(device)
        logger.info(f"  Source points (canonical): {len(self.source_canonical)}")
        
        # Target points
        self.target_points = torch.from_numpy(target_points).float().to(device)
        logger.info(f"  Target points (aligned): {len(self.target_points)}")
        
        # Alignment matrix
        self.alignment_matrix = torch.from_numpy(alignment_matrix).float().to(device)
        
        # Z-up to Y-up rotation
        self.R_zup_to_yup = torch.tensor([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1,  0]
        ], dtype=torch.float32, device=device)
        
        # PyTorch3D to CV
        self.p3d_to_cv = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device))
        
        # Initialize optimizable parameters
        scale = initial_pose['scale']
        rotation = initial_pose['rotation']
        translation = initial_pose['translation']
        
        if len(scale.shape) > 1:
            scale = scale.flatten()
        if len(rotation.shape) > 1:
            rotation = rotation.flatten()
        if len(translation.shape) > 1:
            translation = translation.flatten()
        
        # Use log scale for better optimization
        # If optimize_scale=False, this will be a fixed tensor (not Parameter)
        scale_value = torch.log(torch.tensor(scale[0], dtype=torch.float32, device=device))
        if optimize_scale:
            self.log_scale = torch.nn.Parameter(scale_value)
        else:
            # Keep as regular tensor (not Parameter), so it won't be optimized
            self.log_scale = scale_value
        
        # Quaternion (wxyz)
        self.quat = torch.nn.Parameter(
            torch.tensor(rotation, dtype=torch.float32, device=device)
        )
        
        # Translation
        self.translation = torch.nn.Parameter(
            torch.tensor(translation, dtype=torch.float32, device=device)
        )
        
        # Store initial values for regularization
        self.initial_log_scale = self.log_scale.data.clone()
        self.initial_quat = self.quat.data.clone()
        self.initial_translation = self.translation.data.clone()
        
        logger.info("[PoseOptimizer] Initialized:")
        logger.info(f"  Initial scale: {scale[0]:.4f} {'(optimizable)' if optimize_scale else '(fixed)'}")
        logger.info(f"  Initial rotation (wxyz): {rotation}")
        logger.info(f"  Initial translation: {translation}")
    
    def transform_to_aligned_space(self) -> torch.Tensor:
        """
        Apply complete transformation chain: canonical → aligned space.
        
        Returns:
            points: (N, 3) in aligned space
        """
        points = self.source_canonical.clone()
        
        # 1. Z-up to Y-up
        points = points @ self.R_zup_to_yup.T
        
        # 2. Scale
        scale = torch.exp(self.log_scale)
        points = points * scale
        
        # 3. Rotation
        quat_normalized = F.normalize(self.quat, p=2, dim=0)
        R = quaternion_to_matrix(quat_normalized.unsqueeze(0))[0]
        points = points @ R
        
        # 4. Translation
        points = points + self.translation
        
        # 5. PyTorch3D to CV
        points = points @ self.p3d_to_cv.T
        
        # 6. Apply alignment
        points_homo = torch.cat([points, torch.ones((len(points), 1), device=self.device)], dim=1)
        points = (self.alignment_matrix @ points_homo.T).T[:, :3]
        
        return points
    
    def compute_loss(self) -> Tuple[torch.Tensor, float]:
        """
        Compute Chamfer Distance loss with batching to avoid OOM.
        
        Returns:
            total_loss: Loss value for backprop
            cd: Chamfer distance value
        """
        source_aligned = self.transform_to_aligned_space()
        
        # One-sided CD: target → source (with batching)
        batch_size = 5000
        num_batches = (len(self.target_points) + batch_size - 1) // batch_size
        
        all_min_dists = []
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(self.target_points))
            target_batch = self.target_points[start:end]
            
            dists = torch.cdist(target_batch, source_aligned)
            min_dists, _ = torch.min(dists, dim=1)
            all_min_dists.append(min_dists)
        
        all_min_dists = torch.cat(all_min_dists)
        cd_loss = all_min_dists.mean()
        
        # Regularization to prevent large deviations
        quat_reg = 0.001 * (self.quat - self.initial_quat).pow(2).sum()
        trans_reg = 0.001 * (self.translation - self.initial_translation).pow(2).sum()
        
        # Only add scale regularization if we're optimizing scale
        if self.optimize_scale:
            scale_reg = 0.001 * (self.log_scale - self.initial_log_scale).pow(2)
            total_loss = cd_loss + scale_reg + quat_reg + trans_reg
        else:
            total_loss = cd_loss + quat_reg + trans_reg
        
        return total_loss, cd_loss.item()
    
    def optimize(
        self, 
        num_iterations: int = 300, 
        lr: float = 0.01, 
        early_stopping: bool = True, 
        patience: int = 50
    ) -> Dict:
        """
        Run optimization.
        
        Args:
            num_iterations: Maximum number of iterations
            lr: Base learning rate
            early_stopping: Enable early stopping if CD stops improving
            patience: Number of iterations to wait before stopping
        
        Returns:
            history: Dictionary with optimization history
        """
        # Adaptive learning rate based on initial CD
        initial_loss, initial_cd = self.compute_loss()
        logger.info(f"  Initial CD: {initial_cd:.6f}")
        
        # If initial CD is very small, use smaller learning rates
        if initial_cd < 0.01:
            lr_scale_factor = 0.5
            logger.warning(f"  Initial CD is very small ({initial_cd:.6f}), reducing learning rates by 50%")
        else:
            lr_scale_factor = 1.0
        
        # Build parameter groups based on what we're optimizing
        param_groups = [
            {'params': [self.quat], 'lr': lr * lr_scale_factor},
            {'params': [self.translation], 'lr': lr * 5 * lr_scale_factor},
        ]
        
        # Only add scale to optimizer if optimize_scale is True
        if self.optimize_scale:
            param_groups.insert(0, {'params': [self.log_scale], 'lr': lr * 10 * lr_scale_factor})
            logger.info("  Optimizing: scale + rotation + translation")
        else:
            logger.info("  Optimizing: rotation + translation (scale fixed)")
        
        optimizer = torch.optim.Adam(param_groups)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        
        history = {
            'loss': [],
            'cd': [],
            'scale': [],
        }
        
        # Early stopping variables
        best_cd = float('inf')
        best_params = None
        patience_counter = 0
        
        from tqdm import tqdm
        
        with tqdm(total=num_iterations, desc="Optimizing pose") as pbar:
            for i in range(num_iterations):
                optimizer.zero_grad()
                loss, cd = self.compute_loss()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Record
                scale = torch.exp(self.log_scale).item()
                history['loss'].append(loss.item())
                history['cd'].append(cd)
                history['scale'].append(scale)
                
                # Early stopping logic
                if early_stopping:
                    if cd < best_cd:
                        best_cd = cd
                        best_params = {
                            'log_scale': self.log_scale.data.clone(),
                            'quat': self.quat.data.clone(),
                            'translation': self.translation.data.clone(),
                        }
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        logger.warning(f"  Early stopping at iteration {i} (no improvement for {patience} steps)")
                        # Restore best parameters
                        if best_params is not None:
                            self.log_scale.data = best_params['log_scale']
                            self.quat.data = best_params['quat']
                            self.translation.data = best_params['translation']
                            logger.info(f"  Restored best parameters (CD={best_cd:.6f})")
                        pbar.update(num_iterations - i)
                        break
                
                if i % 10 == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.6f}',
                        'cd': f'{cd:.6f}',
                        'scale': f'{scale:.4f}'
                    })
                    pbar.update(10)
        
        logger.info(f"✓ Optimization finished. Final loss: {history['loss'][-1]:.6f}, Best CD: {best_cd:.6f}")
        
        return history
    
    def get_optimized_pose(self) -> Dict[str, np.ndarray]:
        """
        Get optimized pose parameters.
        
        Returns:
            Dictionary with 'scale', 'rotation', 'translation'
        """
        with torch.no_grad():
            scale = torch.exp(self.log_scale).item()
            quat = F.normalize(self.quat, p=2, dim=0).cpu().numpy()
            trans = self.translation.cpu().numpy()
        
        return {
            'scale': np.array([scale, scale, scale]),
            'rotation': quat,
            'translation': trans,
        }
