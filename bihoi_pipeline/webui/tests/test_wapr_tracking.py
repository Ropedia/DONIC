from pathlib import Path
import json
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from webui import tracking_run, wapr_tracking as tracking
from webui import interaction_capture_webui as ui
from webui.masked_depth import (
    masked_depth_center, masked_depth_points, masked_depth_world_points, remove_statistical_outliers)
from webui.tracking_preview import overlay_depth_points
from webui import foundationpose_reinit


class GeometryTests(unittest.TestCase):
    def test_foundationpose_grid_has_exactly_240_valid_distinct_rotations(self):
        # 42 input directions, matching the minimum-40 FoundationPose icosphere.
        ids = np.arange(42)
        z = 1 - 2 * (ids + .5) / len(ids)
        theta = ids * np.pi * (3 - np.sqrt(5))
        directions = np.column_stack((np.sqrt(1-z*z)*np.cos(theta), np.sqrt(1-z*z)*np.sin(theta), z))
        views = np.repeat(np.eye(4)[None], len(ids), axis=0)
        views[:, :3, 3] = directions
        views[:, :3, 2] = -directions
        x = np.cross([0, 0, 1], -directions)
        x /= np.linalg.norm(x, axis=1)[:, None]
        views[:, :3, 0] = x
        views[:, :3, 1] = np.cross(-directions, x)
        original = views.copy()
        poses = foundationpose_reinit.rotation_grid_240(views)
        self.assertEqual(poses.shape, (240, 4, 4))
        rotations = poses[:, :3, :3]
        np.testing.assert_allclose(rotations @ rotations.transpose(0, 2, 1),
                                   np.repeat(np.eye(3)[None], 240, axis=0), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(rotations), 1, atol=1e-6)
        np.testing.assert_array_equal(poses[:, :3, 3], 0)
        self.assertEqual(len(np.unique(np.round(rotations.reshape(240, -1), 5), axis=0)), 240)
        np.testing.assert_array_equal(views, original)
        np.testing.assert_array_equal(poses, foundationpose_reinit.rotation_grid_240(views))

    def test_initialization_refines_fp_winner_in_original_mesh_coordinates(self):
        backend = tracking.WAPRBackend.__new__(tracking.WAPRBackend)
        ext = np.eye(4)
        ext[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        ext[:3, 3] = [.1, .2, .3]
        center_world = np.array([.2, -.1, 1.5])
        tracking_center = np.array([.3, -.2, 1.6])
        winner = np.eye(4)
        winner[:3, 3] = [.4, .5, 1.9]
        refined = winner.copy()
        refined[0, 3] += .03
        seen = []

        def register(obs, center):
            np.testing.assert_allclose(center, (ext @ np.r_[center_world, 1])[:3])
            seen.append('foundationpose')
            return dict(pose_4x4=winner, best_id=17, score=2.4, hypotheses=240, refine_iterations=5)

        def refine(obs, seed, center):
            np.testing.assert_array_equal(seed, winner)
            np.testing.assert_array_equal(center, tracking_center)
            seen.append('tracking6')
            return dict(pose_4x4=refined, score_6d=.4, hypothesis='sapr_wapr_previous')

        backend.foundationpose = SimpleNamespace(register=register)
        backend.refine_initialization = refine
        obs = dict(ext=ext, cam='2', frame='3', area=20)
        result = backend.initialize(obs, center_world, tracking_center)
        self.assertEqual(seen, ['foundationpose', 'tracking6'])
        np.testing.assert_allclose(result['pose'], np.linalg.inv(ext) @ refined)
        self.assertEqual(result['hypothesis'], 'sapr_wapr_previous')
        self.assertEqual(result['initialization_method'], tracking.INITIALIZATION_METHOD)
        self.assertEqual(result['foundationpose_hypotheses'], 240)
        self.assertEqual(result['foundationpose_refine_iterations'], 5)
        # Failure must not silently fall back to the old WAPR-only registration.
        backend.foundationpose.register = lambda *a: (_ for _ in ()).throw(RuntimeError('FP failed'))
        with self.assertRaisesRegex(RuntimeError, 'FP failed'):
            backend.initialize(obs, center_world, tracking_center)
        self.assertEqual(seen, ['foundationpose', 'tracking6'])

    def test_initialization_runs_tracking_six_and_can_select_the_fp_seed(self):
        backend = tracking.WAPRBackend.__new__(tracking.WAPRBackend)
        backend.center, backend.diameter, backend.mesh = np.array([.1, .2, .3]), .2, object()
        backend.est = object()
        calls = []
        seed = np.eye(4)
        seed[2, 3] = 1.
        center = np.array([.2, -.1, 2.])
        ext = np.eye(4)
        ext[0, 3] = .3

        def track_fn(est, **kwargs):
            self.assertFalse(kwargs['center_only'])
            item = kwargs['instances'][0]
            np.testing.assert_array_equal(item['pose_4x4'], seed)
            calls.append(item['center_pose_4x4'])
            return [dict(pose_4x4=seed, score_6d=.8, hypothesis='previous')]

        backend.track_fn = track_fn
        obs = dict(ext=ext, rgb=None, depth=None, K=None)
        result = backend.refine_initialization(obs, seed, center)
        np.testing.assert_allclose(calls[0] @ np.r_[backend.center, 1], ext @ np.r_[center, 1])
        np.testing.assert_array_equal(result['pose_4x4'], seed)
        self.assertEqual(result['hypothesis'], 'previous')
        backend.refine_initialization(obs, seed)
        np.testing.assert_array_equal(calls[1], seed)  # No triangulation: C=P.

    def test_pose_recentering_preserves_original_mesh_frame(self):
        center = np.array([.12, -.08, .04])
        pose = np.eye(4)
        pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        xyz = np.array([.5, .3, 1.2])
        result = tracking.original_pose_at_center(pose, xyz, center)
        np.testing.assert_allclose((result @ np.r_[center, 1])[:3], xyz)
        np.testing.assert_array_equal(result[:3, :3], pose[:3, :3])

    def test_triangulation_rejects_bad_mask_center(self):
        K = np.array([[500., 0, 320], [0, 500, 180], [0, 0, 1]])
        target = np.array([.03, -.02, 1.5])
        observations = []
        for i, x in enumerate([-.3, -.1, .1, .3]):
            ext = np.eye(4)
            ext[0, 3] = -x
            point = K @ (ext @ np.r_[target, 1])[:3]
            uv = point[:2] / point[2]
            if i == 3:
                uv += [200, 100]
            observations.append(dict(ext=ext, K=K, center_uv=uv))
        np.testing.assert_allclose(tracking.triangulate_center(observations), target, atol=1e-8)

    def test_consensus_compares_mesh_centers_and_rejects_outlier(self):
        center = np.array([.2, -.1, .05])
        candidates = []
        for i, degrees in enumerate([0, 10, 20, 150]):
            pose = np.eye(4)
            a = np.deg2rad(degrees)
            pose[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
            pose = tracking.original_pose_at_center(pose, np.array([.1, .2, 1.]), center)
            candidates.append(dict(pose=pose, cam=str(i), area=10-i))
        pose, best, views = tracking.consensus(candidates, center)
        self.assertEqual(views, ['0', '1', '2'])
        self.assertEqual(best['cam'], '0')
        np.testing.assert_allclose((pose @ np.r_[center, 1])[:3], [.1, .2, 1.])
        self.assertIsNone(tracking.consensus(candidates[:2], center))

    def test_backend_transforms_world_prior_and_back(self):
        backend = tracking.WAPRBackend.__new__(tracking.WAPRBackend)
        backend.center = np.array([.1, .2, 0.])
        backend.diameter, backend.mesh, backend.est = .5, object(), object()
        previous = np.eye(4)
        previous[2, 3] = 2.
        ext = np.eye(4)
        ext[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        ext[0, 3] = .3
        center = np.array([.2, -.1, 2.1])

        def track_fn(estimator, **kwargs):
            self.assertTrue(kwargs['center_only'])
            item = kwargs['instances'][0]
            np.testing.assert_allclose(item['pose_4x4'], ext @ previous)
            np.testing.assert_allclose(item['center_pose_4x4'] @ np.r_[backend.center, 1], ext @ np.r_[center, 1])
            return [dict(pose_4x4=ext @ previous, score_6d=.1)]

        backend.track_fn = track_fn
        obs = dict(ext=ext, cam='2', frame='0', area=20, rgb=None, depth=None, K=None)
        np.testing.assert_allclose(backend.estimate(obs, previous, center)['pose'], previous)

    def test_registration_transforms_fused_world_center_to_selected_camera(self):
        backend = tracking.WAPRBackend.__new__(tracking.WAPRBackend)
        backend.center = np.array([.1, .2, 0.])
        backend.diameter, backend.mesh = .5, object()
        ext = np.eye(4)
        ext[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        ext[0, 3] = .3
        fused_center = np.array([.2, -.1, 2.1])
        calls = []

        def register(obs, center_world):
            calls.append((obs['ext'] @ np.r_[center_world, 1])[:3])
            pose = tracking.original_pose_at_center(np.eye(4), center_world, backend.center)
            return dict(pose=pose, score=.3)

        backend.initialize = register
        K = np.array([[2., 0, 1.], [0, 4., 1.], [0, 0, 1.]])
        obs = dict(ext=ext, cam='2', frame='0', area=20, rgb=None,
                   depth=np.array([[2., 100.], [0., 4.]]), K=K,
                   mask=np.array([[1, 0], [1, 1]]))
        result = backend.estimate(obs, None, fused_center)
        center_cam = (ext @ np.r_[fused_center, 1])[:3]
        np.testing.assert_allclose(calls[0], center_cam)
        np.testing.assert_allclose((result['pose'] @ np.r_[backend.center, 1])[:3], fused_center)
        with self.assertRaisesRegex(ValueError, 'fused masked depth'):
            backend.estimate(obs)

    def test_wapr96_registration_uses_mask_and_triangulated_seed_preserving_refined_pose(self):
        backend = tracking.WAPRBackend.__new__(tracking.WAPRBackend)
        backend.mesh, backend.diameter = object(), .3
        ext = np.eye(4)
        ext[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        ext[:3, 3] = [.1, .2, .3]
        center = np.array([.3, -.2, 1.4])
        winner = np.eye(4)
        winner[:3, 3] = [.5, .6, 2.]
        obs = dict(ext=ext, cam='7', frame='24', area=100,
                   rgb=object(), depth=object(), K=object(), mask=object())
        calls = []

        def register(**kwargs):
            calls.append(kwargs)
            np.testing.assert_allclose(kwargs['center_xyz_m'], (ext @ np.r_[center, 1])[:3])
            self.assertEqual((kwargs['n_view'], kwargs['n_inplane']), (12, 8))
            self.assertIs(kwargs['mask'], obs['mask'])
            self.assertIs(kwargs['depth_m'], obs['depth'])
            self.assertIs(kwargs['mesh'], backend.mesh)
            return dict(pose_4x4=winner, score_6d=.4)

        backend.est = SimpleNamespace(estimate_one_category_one_instance=register)
        result = backend.register_best_mask(obs, center)
        np.testing.assert_allclose(result['pose'], np.linalg.inv(ext) @ winner)
        self.assertEqual(result['cam'], '7')
        self.assertEqual(result['reinitialization_hypotheses'], 96)
        self.assertEqual(result['reinitialization_xyz'], 'multiview_mask_triangulation')
        np.testing.assert_array_equal(result['reinitialization_center_world'], center)
        for invalid in (None, [0, 0, -1], [0, np.nan, 1], [0, 1]):
            with self.assertRaisesRegex(ValueError, 'triangulated center'):
                backend.register_best_mask(obs, invalid)
        self.assertEqual(len(calls), 1)

    def test_fusion_uses_extrinsics_and_all_valid_cameras_beyond_topk(self):
        capture = tracking.Capture.__new__(tracking.Capture)
        capture.args = SimpleNamespace(topk=1, min_mask_area=1, io_workers=2,
                                       cloud_outlier_neighbors=20, cloud_outlier_std_ratio=2., no_cloud_outlier_removal=False)
        capture.cams = ['a', 'b', 'missing', 'no_depth', 'invalid_depth']
        ext = np.eye(4)
        ext[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        ext[:3, 3] = [1, -2, 0]
        observations = {
            cam: dict(cam=cam, frame='0', K=np.eye(3), ext=ext if cam == 'b' else np.eye(4),
                      mask=np.array([[1, 1]]) if cam == 'a' else np.array([[1, 0]]))
            for cam in capture.cams if cam != 'missing'}
        depths = dict(a=np.array([[2., 2.]]), b=np.array([[4., 99.]]),
                      no_depth=None, invalid_depth=np.array([[np.nan, np.inf]]))
        capture.observation = lambda cam, frame: observations.get(cam)
        loaded = []

        def load(obs):
            loaded.append(obs['cam'])
            return depths[obs['cam']]

        capture.load_depth = load
        cached = dict(observations['a'], depth=depths['a'])
        points, cameras, stats = capture.fused_depth_cloud('0', hydrated=[cached])
        self.assertEqual(cameras, ['a', 'b'])
        self.assertEqual(stats['raw_points'], 3)
        self.assertEqual(stats['status'], 'too_few_points')
        self.assertNotIn('a', loaded)
        expected = np.array([[0., 0., 2.], [2., 0., 2.], [2., 1., 4.]])
        np.testing.assert_allclose(points, expected)
        np.testing.assert_allclose(points.mean(axis=0), [4/3, 1/3, 8/3])
        # A mean of per-camera means would incorrectly give extra weight to b.
        self.assertFalse(np.allclose(points.mean(axis=0), [1.5, .5, 3.]))

    def test_fused_point_projects_into_a_different_camera(self):
        source_ext = np.eye(4)
        source_ext[0, 3] = -1.
        points_world = masked_depth_world_points(np.array([[2.]]), np.ones((1, 1)),
                                                np.eye(3), source_ext)
        np.testing.assert_allclose(points_world, [[1., 0., 2.]])
        target_ext = np.eye(4)
        target_ext[1, 3] = 1.
        target_points = points_world @ target_ext[:3, :3].T + target_ext[:3, 3]
        image = np.zeros((30, 30, 3), np.uint8)
        overlay_depth_points(image, target_points, np.array([[20., 0, 5.], [0, 20., 5.], [0, 0, 1.]]), opacity=1.)
        self.assertGreater(int(image[15, 15, 0]), 200)
        self.assertLess(int(image[15, 15, 1]), 60)

    def test_depth_centroid_uses_all_valid_masked_points(self):
        depth = np.array([[2., 100., np.nan, np.inf], [0., -1., 4., 6.]])
        mask = np.ones(depth.shape, bool)
        mask[0, 1] = False
        K = np.array([[2., 0, 1.], [0, 4., 1.], [0, 0, 1.]])
        expected = np.array([[-1., -.5, 2.], [2., 0., 4.], [6., 0., 6.]])
        np.testing.assert_allclose(masked_depth_points(depth, mask, K), expected)
        np.testing.assert_allclose(masked_depth_center(depth, mask, K), expected.mean(axis=0))
        # The centroid weights every measured 3D point, rather than multiplying
        # the mean pixel by a mean/median depth (which differs for slanted surfaces).
        self.assertNotAlmostEqual(masked_depth_center(depth, mask, K)[0], 4 / 3)

    def test_empty_depth_cannot_initialize(self):
        depth = np.array([[0., np.nan], [-1., np.inf]])
        points = masked_depth_points(depth, np.ones((2, 2)), np.eye(3))
        self.assertEqual(points.shape, (0, 3))
        with self.assertRaisesRegex(ValueError, 'valid masked depth'):
            masked_depth_center(depth, np.ones((2, 2)), np.eye(3))

    def test_red_points_project_with_scaled_intrinsics_and_ignore_invalid_depth(self):
        K = np.array([[20., 0, 20.], [0, 20., 10.], [0, 0, 1.]])
        points = np.array([[1., 1., 2.], [0, 0, -1.], [np.nan, 0, 1.], [1e9, 0, 1.]])
        original = points.copy()
        # Half-resolution preview: the valid point projects to (15, 10).
        preview_K = K.copy()
        preview_K[:2] *= .5
        image = np.zeros((20, 30, 3), np.uint8)
        overlay_depth_points(image, points, preview_K, opacity=1.)
        self.assertGreater(int(image[10, 15, 0]), 200)
        self.assertLess(int(image[10, 15, 1]), 60)
        np.testing.assert_array_equal(image[:5], 0)
        np.testing.assert_array_equal(points, original)
        unchanged = image.copy()
        overlay_depth_points(image, np.empty((0, 3)), preview_K)
        np.testing.assert_array_equal(image, unchanged)


class OutlierTests(unittest.TestCase):
    def test_removes_isolated_points_preserving_surface_and_attached_detail(self):
        x, y = np.meshgrid(np.linspace(-.05, .05, 20), np.linspace(-.05, .05, 20))
        surface = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        detail = np.column_stack((np.linspace(.055, .1, 20), np.zeros(20), np.zeros(20)))
        expected = np.concatenate((surface, detail))
        cloud = np.concatenate((expected, [[.7, .7, .7], [-1., 1., 1.], [1., -1., 1.]]))
        original = cloud.copy()
        filtered, stats = remove_statistical_outliers(cloud)
        np.testing.assert_array_equal(filtered, expected)
        np.testing.assert_array_equal(cloud, original)
        self.assertEqual(stats['removed_points'], 3)
        self.assertEqual(stats['status'], 'filtered')

    def test_filter_matches_brute_force_neighbor_distances(self):
        points = np.random.default_rng(7).normal(size=(90, 3)) * .02
        points[-3:] += 1.
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        means = np.sort(distances, axis=1)[:, :8].mean(axis=1)
        cutoff = means.mean() + 2 * means.std()
        filtered, stats = remove_statistical_outliers(points, neighbors=8)
        np.testing.assert_array_equal(filtered, points[means <= cutoff])
        self.assertAlmostEqual(stats['threshold_m'], cutoff)

    def test_small_empty_and_duplicate_clouds_are_safe(self):
        for points in (np.empty((0, 3)), np.array([[0., 0., 0.], [1., 1., 1.]]), np.ones((50, 3))):
            with self.subTest(count=len(points)):
                filtered, stats = remove_statistical_outliers(points)
                np.testing.assert_array_equal(filtered, points)
                self.assertEqual(stats['removed_points'], 0)
        filtered, stats = remove_statistical_outliers(np.array([[np.nan, 0, 0], [0, np.inf, 0], [1., 2., 3.]]))
        np.testing.assert_array_equal(filtered, [[1., 2., 3.]])
        self.assertEqual(stats['removed_points'], 2)

    def test_disabled_filter_preserves_finite_points(self):
        points = np.random.default_rng(9).normal(size=(50, 3))
        filtered, stats = remove_statistical_outliers(points, enabled=False)
        np.testing.assert_array_equal(filtered, points)
        self.assertEqual(stats['status'], 'disabled')
        self.assertIsNone(stats['threshold_m'])


class TemporalTests(unittest.TestCase):
    def test_parallel_dispatch_keeps_camera_order_and_official_inputs(self):
        from webui.wapr_parallel import ParallelViews
        pool = ParallelViews.__new__(ParallelViews)
        tasks_seen = []

        def mapped(fn, tasks):
            tasks_seen.extend(tasks)
            return [task[0]['cam'] for task in tasks]

        pool.executor = SimpleNamespace(map=mapped)
        observations = [dict(cam=cam, rgb='rgb', depth='depth', K='K', ext='ext',
                             frame='3', area=10, mask='large unused mask') for cam in ['8', '2', '5']]
        result = pool.estimate(observations, 'previous', 'center')
        self.assertEqual(result, ['8', '2', '5'])
        self.assertEqual(tasks_seen[0][1:], ('previous', 'center'))
        self.assertNotIn('mask', tasks_seen[0][0])
        self.assertEqual(tasks_seen[0][0]['depth'], 'depth')

    def make_capture(self):
        def observations(frame):
            if frame == '3':
                return []
            return [dict(cam=str(i), frame=frame, area=100-i, ext=np.eye(4)) for i in range(3)]

        return SimpleNamespace(frames=list('01234'), main_cam='0',
                               extrinsics={str(i): np.eye(4) for i in range(3)},
                               observations=observations, hydrate=lambda obs: True,
                               fused_depth_cloud=lambda frame, hydrated=(): (
                                   np.array([[.1 + int(frame), .3, 1.], [.3 + int(frame), .3, 1.]]), ['0', '1'],
                                   dict(raw_points=3, kept_points=2, removed_points=1, method='statistical',
                                        status='filtered', threshold_m=.01)))

    def make_backend(self):
        backend = SimpleNamespace(center=np.zeros(3), diameter=.2, calls=[], initializations=[], priors=[],
                                  reinitializations=[])

        def estimate(obs, previous=None, center=None):
            backend.calls.append((obs['frame'], previous is not None))
            pose = np.eye(4) if previous is None else previous.copy()
            if previous is None:
                pose[2, 3] = 1.
            else:
                backend.priors.append((obs['frame'], previous.copy()))
            if obs['frame'] == '2' and previous is not None:
                pose[0, 3] = .25 + int(obs['cam'])  # no cross-camera consensus
            return dict(pose=pose, cam=obs['cam'], area=obs['area'], score=.3 + int(obs['cam']),
                        hypothesis='WAPR(C)' if previous is not None else 'previous')

        backend.estimate = estimate

        def initialize(obs, center, tracking_center_world=None):
            backend.initializations.append(obs['frame'])
            return dict(estimate(obs, None, center),
                        initialization_method=tracking.INITIALIZATION_METHOD,
                        foundationpose_hypotheses=240, foundationpose_refine_iterations=5,
                        foundationpose_score=2.4)

        backend.initialize = initialize

        def register_best_mask(obs, center):
            backend.reinitializations.append((obs['frame'], obs['cam'], center.copy()))
            pose = tracking.original_pose_at_center(np.eye(4), center, backend.center)
            return dict(pose=pose, cam=obs['cam'], area=obs['area'], score=.8,
                        hypothesis=tracking.REINITIALIZATION_METHOD,
                        reinitialization_method=tracking.REINITIALIZATION_METHOD,
                        reinitialization_hypotheses=96, reinitialization_xyz='multiview_mask_triangulation',
                        reinitialization_center_world=center.tolist())

        backend.register_best_mask = register_best_mask
        return backend

    def test_no_triangulation_reuses_best_mask_and_resumes_tracking_after_occlusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = tracking.build_parser().parse_args(['--scene_dir', tmp])
            backend = self.make_backend()
            # Candidate arrival order and scores must not override mask area.
            backend.estimate_views = lambda observations, previous, center: [
                backend.estimate(obs, previous, center) for obs in reversed(observations)]
            capture = self.make_capture()
            with patch.object(tracking, 'triangulate_center', return_value=None), \
                 patch.object(capture, 'fused_depth_cloud', wraps=capture.fused_depth_cloud) as cloud:
                result = tracking.track_capture(args, capture, backend)
            cloud.assert_called_once()
            self.assertEqual(result['frames'], 5)
            self.assertEqual(result['modes'], dict(initialization=1, tracking=2, tracking_best_mask=1, held_no_rgbd=1))
            self.assertEqual(sum(is_tracking for _, is_tracking in backend.calls), 9)
            self.assertEqual(backend.initializations, ['0'])
            self.assertEqual(backend.reinitializations, [])
            folder = Path(tmp)/'debug_depth/ob_in_world'
            self.assertEqual(len(list(folder.glob('*.txt'))), 5)
            for frame in ('2', '3', '4'):
                self.assertAlmostEqual(np.loadtxt(folder/f'{frame}.txt')[0, 3], .25)
            for frame, prior in backend.priors:
                if frame == '4':
                    self.assertAlmostEqual(prior[0, 3], .25)
            import csv
            with (Path(tmp)/'debug_depth/wapr_tracking_log.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[2]['camera'], '0')
            self.assertEqual(rows[2]['hypothesis'], 'WAPR(C)')
            self.assertEqual(rows[2]['consensus_views'], '')
            self.assertEqual(rows[2]['initialization_method'], '')
            self.assertEqual(rows[2]['consensus_failed'], '1')
            self.assertEqual(rows[2]['reinitialization_skip_reason'], 'triangulation_unavailable')

    def test_no_consensus_registers_best_mask_using_triangulated_center_and_updates_prior(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = tracking.build_parser().parse_args(['--scene_dir', tmp])
            backend, capture = self.make_backend(), self.make_capture()
            backend.estimate_views = lambda obs, prior, center: [
                backend.estimate(o, prior, center) for o in reversed(obs)]
            xyz = np.array([.4, .5, 1.2])
            emitted = []
            with patch.object(tracking, 'triangulate_center', return_value=xyz), \
                 patch.object(capture, 'fused_depth_cloud', wraps=capture.fused_depth_cloud) as cloud:
                report = tracking.track_capture(args, capture, backend,
                                               SimpleNamespace(publish=lambda *a: emitted.append(a)))
            cloud.assert_called_once()  # Fused depth remains only the first-frame seed.
            self.assertEqual(report['modes'], dict(initialization=1, tracking=2, reinit_wapr96=1, held_no_rgbd=1))
            self.assertEqual(backend.initializations, ['0'])
            self.assertEqual(len(backend.reinitializations), 1)
            frame, camera, center = backend.reinitializations[0]
            self.assertEqual((frame, camera), ('2', '0'))  # Mask area, not score/order.
            np.testing.assert_array_equal(center, xyz)
            for frame in ('2', '3', '4'):
                saved = np.loadtxt(Path(tmp)/'debug_depth/ob_in_world'/f'{frame}.txt')
                np.testing.assert_allclose(saved[:3, 3], xyz)
            for frame, prior in backend.priors:
                if frame == '4':
                    np.testing.assert_allclose(prior[:3, 3], xyz)
            import csv
            with (Path(tmp)/'debug_depth/wapr_tracking_log.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[2]['reinitialization_hypotheses'], '96')
            self.assertEqual(rows[2]['reinitialization_xyz'], 'multiview_mask_triangulation')
            self.assertEqual(rows[2]['reinitialization_skip_reason'], '')
            self.assertEqual(rows[2]['consensus_views'], '')
            self.assertEqual(report['consensus_failed_frames'], 1)
            self.assertEqual(emitted[2][-1], 'reinit_wapr96')

    def test_invalid_triangulation_skips_registration_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = tracking.build_parser().parse_args(['--scene_dir', tmp])
            backend = self.make_backend()
            with patch.object(tracking, 'triangulate_center', return_value=np.array([0., 0., -1.])):
                tracking.track_capture(args, self.make_capture(), backend)
            self.assertEqual(backend.reinitializations, [])
            import csv
            with (Path(tmp)/'debug_depth/wapr_tracking_log.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[2]['mode'], 'tracking_best_mask')
            self.assertEqual(rows[2]['reinitialization_skip_reason'], 'triangulation_invalid_for_best_mask')

    def test_wapr96_failure_does_not_silently_keep_old_tracking_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = tracking.build_parser().parse_args(['--scene_dir', tmp])
            backend = self.make_backend()
            backend.register_best_mask = lambda *a: (_ for _ in ()).throw(RuntimeError('WAPR96 failed'))
            with patch.object(tracking, 'triangulate_center', return_value=np.array([0., 0., 1.])):
                with self.assertRaisesRegex(RuntimeError, 'WAPR96 failed'):
                    tracking.track_capture(args, self.make_capture(), backend)
            self.assertFalse((Path(tmp)/'debug_depth/ob_in_world').exists())
            self.assertTrue((Path(tmp)/'debug_depth/wapr_tracking_failed.csv').is_file())

    def test_fewer_views_than_consensus_requires_use_best_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = tracking.build_parser().parse_args(['--scene_dir', tmp])
            backend, capture = self.make_backend(), self.make_capture()
            observations = capture.observations
            capture.observations = lambda frame: observations(frame)[:1]
            with patch.object(tracking, 'triangulate_center', return_value=None):
                result = tracking.track_capture(args, capture, backend)
            self.assertEqual(result['modes'], dict(initialization=1, tracking_best_mask=3, held_no_rgbd=1))
            self.assertEqual(backend.initializations, ['0'])

    def test_failed_attempt_is_retryable_without_partial_poses(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = tracking.build_parser().parse_args(['--scene_dir', tmp])
            backend = self.make_backend()
            real_estimate = backend.estimate

            def fail(obs, *a):
                if obs['frame'] == '1':
                    raise RuntimeError('simulated CUDA failure')
                return real_estimate(obs, *a)

            backend.estimate = fail
            with patch.object(tracking, 'triangulate_center', return_value=None):
                with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                    tracking.track_capture(args, self.make_capture(), backend)
                self.assertFalse((Path(tmp)/'debug_depth/ob_in_world').exists())
                tracking.track_capture(args, self.make_capture(), self.make_backend())

    def test_initialization_runs_once_and_all_frames_emit_live_previews(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = tracking.build_parser().parse_args(['--scene_dir', tmp])
            backend = self.make_backend()
            estimate = backend.estimate
            centers, emitted = [], []
            xyz = np.array([-9., -9., 9.])

            def recording_estimate(obs, previous=None, center=None):
                if previous is None:
                    centers.append(center)
                return estimate(obs, previous, center)

            backend.estimate = recording_estimate
            initialize = backend.initialize

            def recording_initialize(obs, center, tracking_center_world=None):
                centers.append(center)
                self.assertEqual(obs['cam'], '0')  # Largest valid mask.
                np.testing.assert_array_equal(tracking_center_world, xyz)
                return initialize(obs, center, tracking_center_world)

            backend.initialize = recording_initialize
            preview = SimpleNamespace(publish=lambda *args: emitted.append(args))
            with patch.object(tracking, 'triangulate_center', return_value=xyz):
                report = tracking.track_capture(args, self.make_capture(), backend, preview)
            self.assertEqual(len(centers), 1)
            np.testing.assert_allclose(centers[0], [.2, .3, 1.])
            self.assertEqual(report['initialization_xyz'], 'multiview_masked_depth_pointcloud_centroid')
            import csv
            with (Path(tmp)/'debug_depth/wapr_tracking_log.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r['initialization_xyz'] for r in rows if 'initialization' in r['mode']],
                             ['multiview_masked_depth_pointcloud_centroid'])
            self.assertEqual([r['initialization_views'] for r in rows if 'initialization' in r['mode']], ['0;1'])
            self.assertEqual([r['initialization_points'] for r in rows if 'initialization' in r['mode']], ['2'])
            self.assertEqual([r['initialization_raw_points'] for r in rows if 'initialization' in r['mode']], ['3'])
            self.assertEqual([r['initialization_removed_points'] for r in rows if 'initialization' in r['mode']], ['1'])
            self.assertEqual(report['cloud_outlier_removal']['method'], 'statistical')
            self.assertEqual([r['foundationpose_hypotheses'] for r in rows], ['240', '', '', '', ''])
            self.assertEqual([r['foundationpose_refine_iterations'] for r in rows], ['5', '', '', '', ''])
            self.assertEqual(report['initialization']['method'], tracking.INITIALIZATION_METHOD)
            self.assertTrue(report['reinitialization']['enabled'])
            self.assertEqual(report['reinitialization']['method'], tracking.REINITIALIZATION_METHOD)
            self.assertEqual(report['reinitialization']['hypotheses'], 96)
            self.assertEqual(report['consensus_failure'], 'largest_valid_mask_wapr_registration_96')
            self.assertEqual(report['initialization']['tracking_hypotheses'], tracking.SIX_HYPOTHESES)
            self.assertEqual(report['tracking_hypotheses'], ['WAPR(C)', 'SAPR(C)', 'SAPR(WAPR(C))'])
            self.assertEqual([item[0] for item in emitted], list('01234'))
            self.assertEqual(emitted[-1][3:5], (5, 5))

    def test_prefetch_preserves_frame_order_and_ranked_valid_view_selection(self):
        capture = tracking.Capture.__new__(tracking.Capture)
        capture.args = SimpleNamespace(topk=3, io_workers=2)
        capture.frames = list('012')
        capture.cams = list('012345')
        capture.observation = lambda cam, frame: dict(cam=cam, frame=frame, area=100-int(cam))
        capture.hydrate = lambda obs: obs['cam'] not in ('0', '2')
        with patch.object(tracking, 'triangulate_center', return_value=None):
            output = list(capture.prepared_frames())
        self.assertEqual([item[0] for item in output], list('012'))
        for _, usable, _ in output:
            self.assertEqual([obs['cam'] for obs in usable], ['1', '3', '4'])


class PipelineTests(unittest.TestCase):
    def completed_legacy(self, root):
        poses = root/'debug_depth/geometry_global_opt/ob_in_world'
        poses.mkdir(parents=True)
        (poses/'000000.txt').write_text('old pose')
        video = root/'mesh_overlay_allcams_fast_green/mesh_overlay_allcams_fast_green.mp4'
        video.parent.mkdir()
        video.write_bytes(b'old video')

    def test_switch_backend_archives_outputs_and_does_not_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.completed_legacy(root)
            self.assertTrue(ui._stage_complete(root, 'tracking', 'foundationpose'))
            self.assertFalse(ui._stage_complete(root, 'tracking', 'wapr'))
            tracking_run.prepare(root, 'wapr')
            state = tracking_run.read_state(root)
            self.assertEqual((Path(state['previous_outputs'])/'debug_depth/geometry_global_opt/ob_in_world/000000.txt').read_text(), 'old pose')
            self.assertFalse(ui._stage_complete(root, 'tracking', 'wapr'))
            with self.assertRaises(ValueError):
                tracking_run.complete(root, 'wapr')
            self.completed_legacy(root)
            tracking_run.complete(root, 'wapr')
            self.assertTrue(ui._stage_complete(root, 'tracking', 'wapr'))
            self.assertFalse(ui._stage_complete(root, 'tracking', 'foundationpose'))

    def test_stage_commands_route_backend_and_fix_output_path(self):
        root = ui.EASYMOCAP_ROOT
        for method in tracking_run.METHODS:
            with self.subTest(method=method), patch.object(ui, '_ensure_pipeline_data_name', return_value='rigid/collector01/test'):
                steps = ui._tracking_steps(ui.Job('test', 'collector01'), root, root, 'auto', 'auto', 640, 360, 10, 1000, method)
            tracker = steps[2]
            self.assertEqual(tracker.gpu, 'single')
            self.assertIn('"$GPU_PICK"', tracker.command)
            self.assertIn('wapr_tracking.py' if method == 'wapr' else 'run_resize_camk_foundationpose_args_local.sh', tracker.command)
            self.assertIn('--pose_dir', steps[4].command)
            self.assertIn('debug_depth/geometry_global_opt/ob_in_world', steps[4].command)
            self.assertIn('--fps 10.0', steps[4].command)
            self.assertIn(' complete ', steps[-1].command)

    def test_final_export_becomes_stale_after_new_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root/'outputmano/grid/final_hand_object_interaction.mp4'
            video.parent.mkdir(parents=True)
            video.write_bytes(b'old video')
            (root/ui.EXPORT_STANDARD_DIR).mkdir(parents=True)
            self.assertTrue(ui._stage_complete(root, 'final'))
            tracking_run.write_state(root, dict(method='wapr', status='complete', completed=video.stat().st_mtime+1))
            self.assertFalse(ui._stage_complete(root, 'final'))

    def test_plan_preserves_selected_method_with_skip_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = root/'rigid/collector01/test'
            (ds/'images').mkdir(parents=True)
            self.completed_legacy(ds)
            args = dict(stage_sequence=['tracking'], hand_prior_name='collector01', selected_datasets=['test'],
                        prompt_table=[], hand_side='bimanual', data_root_str=str(root), start_sec=0, end_sec=0,
                        fps=10, width=640, height=360, ffmpeg_threads=2, da3_gpus='auto', sam3d_gpu='auto',
                        tracking_gpu='auto', postopt_gpu='auto', depth_unit='mm', pixel_stride=2, mask_erode=1,
                        max_points_per_cam=50000, max_size_mb=9.8, save_debug_pointcloud=False,
                        depth_scale=1000, max_frames=500, all_takes=False, skip_done=True)
            with patch.object(ui, '_ensure_pipeline_data_name', return_value='rigid/collector01/test'), \
                 patch.object(ui, 'python_for', return_value=Path(sys.executable)):
                with self.assertRaisesRegex(ValueError, 'Nothing to run'):
                    ui._plan_steps(**args, tracking_method='foundationpose')
                _, _, records, _ = ui._plan_steps(**args, tracking_method='wapr')
            self.assertTrue(all(r['tracking_method'] == 'wapr' for r in records))
            self.assertTrue(any('wapr_tracking.py' in r['command'] for r in records))

    def test_live_preview_uses_published_frame_and_rejects_another_run(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'debug_depth/live_tracking'
            folder.mkdir(parents=True)
            Image.new('RGB', (12, 8), (0, 255, 0)).save(folder/'test.jpg')
            request = dict(session='session-a', started=12, completed=4, total=8)
            (folder/'request.json').write_text(json.dumps(request))
            (folder/'latest.json').write_text(json.dumps(dict(request, frame='000003', image='test.jpg', mode='tracking')))
            meta = dict(steps=[dict(title='WAPR object tracking', dataset_path=str(root))],
                        state=dict(steps={'0':dict(started=10, finished=20)}))
            with patch.object(ui, '_find_run', return_value=meta):
                images, status, key = ui.live_tracking_view('a')
                self.assertEqual(len(images), 1)
                self.assertIn('4/8', status)
                unchanged, _, _ = ui.live_tracking_view('a', key)
                self.assertEqual(unchanged, ui.gr.skip())
                meta['state']['steps']['0']['started'] = 15
                images, status, _ = ui.live_tracking_view('b')
                self.assertEqual(images, [])
                self.assertIn('waiting', status)


if __name__ == '__main__':
    unittest.main()
