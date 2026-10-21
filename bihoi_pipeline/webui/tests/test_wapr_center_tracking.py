"""Verify candidate data flow and WBPS grouping without loading GPU models."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


class CenterTrackingTests(unittest.TestCase):
    def setUp(self):
        repo = Path(__file__).resolve().parents[2] / "web_demos/third_party/WAPR"
        with patch.object(sys, "path", [str(repo), *sys.path]):
            from wapr import tracking
        self.tracking = tracking

    def run_candidates(self, center_only, engine=False):
        tracker = self.tracking
        calls, scored, score_groups = [], [], []
        profile = SimpleNamespace(min_batch=2, max_batch=120) if engine else None

        def refine(net, poses, *args, **kwargs):
            calls.append((net.name, poses.clone(), kwargs['iters']))
            out = poses.clone()
            out[:, 0, 3] += 1 if net.name == 'wapr' else 3
            return out

        def crops(rgb, depth, mask, poses, *args, **kwargs):
            scored.append(poses.clone())
            return poses, poses, None, None

        class Score:
            def __init__(self):
                self.engine = profile

            def __call__(self, A, B, L):
                score_groups.append(L)
                groups = len(A) // L
                within = torch.zeros(groups, L)
                # Distinct winners per object exercise independent attention groups.
                for i, pose in enumerate(A.reshape(groups, L, 4, 4)[:, 0]):
                    choice = (int(pose[0, 3]) // 10) % L if center_only else 4
                    within[i, choice] = 1
                return dict(within_group=within, between_group=torch.arange(groups * L).reshape(groups, L).float())

        mesh = SimpleNamespace(metadata={})
        est = SimpleNamespace(device='cpu',
            nets=dict(wapr_wo_mask=SimpleNamespace(name='wapr', engine=profile),
                      sapr=SimpleNamespace(name='sapr', engine=profile), wbps=Score()),
            _pose_meshes=lambda meshes: meshes,
            renderer=SimpleNamespace(cached_mesh_id=lambda mesh: 0), _refine=refine)
        instances = []
        for index in range(2):
            previous = np.eye(4, dtype=np.float32)
            previous[:3, 3] = [-10 * (index + 1), 0, 1]
            center = previous.copy()
            center[0, 3] *= -1
            instances.append(dict(track_id=index, mesh=mesh, diameter_m=.2,
                                  pose_4x4=previous, center_pose_4x4=center))
        with patch.object(tracker, 'make_crop_pair', side_effect=crops), \
             patch.object(tracker.os.path, 'isfile', return_value=False), \
             patch.object(tracker, '_Engine', side_effect=AssertionError('Must not load fixed six-pose engine')):
            result = tracker.track_many_categories_many_instances(
                est, np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), np.float32),
                np.eye(3), instances, center_only=center_only)
        return result, calls, torch.cat(scored), score_groups

    def test_only_three_center_branches_are_computed_and_scored(self):
        result, calls, scored, groups = self.run_candidates(True)
        self.assertEqual([(name, len(poses), n) for name, poses, n in calls], [('wapr', 2, 3), ('sapr', 4, 2)])
        torch.testing.assert_close(calls[0][1][:, 0, 3], torch.tensor([10., 20.]))
        torch.testing.assert_close(calls[1][1][:, 0, 3], torch.tensor([10., 11., 20., 21.]))
        torch.testing.assert_close(scored[:, 0, 3], torch.tensor([11., 13., 14., 21., 23., 24.]))
        self.assertEqual(groups, [3])
        self.assertEqual([r['hypothesis'] for r in result], ['SAPR(C)', 'SAPR(WAPR(C))'])
        np.testing.assert_allclose([r['pose_4x4'][0, 3] for r in result], [13., 24.])
        self.assertEqual([r['track_id'] for r in result], [0, 1])

    def test_three_candidates_use_general_trt_scorer_without_six_pose_engine(self):
        result, calls, scored, groups = self.run_candidates(True, engine=True)
        self.assertEqual(groups, [3, 3])
        self.assertEqual([len(poses) for _, poses, _ in calls], [1, 2, 1, 2])
        self.assertEqual(len(scored), 6)
        np.testing.assert_allclose([r['pose_4x4'][0, 3] for r in result], [13., 24.])

    def test_default_api_preserves_six_candidate_demo(self):
        result, calls, scored, groups = self.run_candidates(False)
        self.assertEqual([(name, len(poses)) for name, poses, _ in calls], [('wapr', 4), ('sapr', 6)])
        self.assertEqual(groups, [6])
        torch.testing.assert_close(scored[:, 0, 3], torch.tensor([-10., 10., 11., 13., 14., -6.,
                                                                -20., 20., 21., 23., 24., -16.]))
        self.assertEqual([r['hypothesis'] for r in result], ['sapr_wapr_center'] * 2)


class Registration96Tests(unittest.TestCase):
    def test_native_registration_refines_and_scores_one_group_of_96_seeded_rotations(self):
        repo = Path(__file__).resolve().parents[2] / 'web_demos/third_party/WAPR'
        with patch.object(sys, 'path', [str(repo), *sys.path]):
            from wapr import estimator
        rotations = estimator.make_view_rots(12, 8)
        self.assertEqual(rotations.shape, (96, 3, 3))
        self.assertEqual(len(np.unique(np.round(rotations.reshape(96, -1), 5), axis=0)), 96)
        np.testing.assert_allclose(rotations @ rotations.transpose(0, 2, 1),
                                   np.repeat(np.eye(3)[None], 96, axis=0), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(rotations), 1, atol=1e-6)

        center = np.array([.2, -.1, .03], np.float32)
        xyz = np.array([.4, .3, 2.], np.float32)
        calls, groups = [], []
        mesh = SimpleNamespace(metadata={'model_center':center})
        est = estimator.WAPREstimator.__new__(estimator.WAPREstimator)
        est.device = 'cpu'
        est._pose_meshes = lambda meshes: meshes
        est.renderer = SimpleNamespace(cached_mesh_id=lambda mesh:0)

        class Score:
            engine = None

            def __call__(self, A, B, L):
                groups.append(L)
                within = torch.zeros(1, L)
                within[0, 73] = 1
                return dict(within_group=within, between_group=torch.zeros(1, L))

        est.nets = dict(wapr_w_mask=SimpleNamespace(name='masked', engine=None),
                        sapr=SimpleNamespace(name='sapr', engine=None), wbps=Score())

        def refine(net, poses, *args, **kwargs):
            calls.append((net.name, poses.clone(), kwargs))
            result = poses.clone()
            result[:, 0, 3] += .01
            return result

        def crops(rgb, depth, mask, poses, *args, **kwargs):
            return poses, poses, None, None

        est._refine = refine
        with patch.object(estimator, 'make_crop_pair', side_effect=crops), \
             patch.object(estimator, 'guess_translations', side_effect=AssertionError('Must use triangulated XYZ')):
            result = est.estimate_one_category_one_instance(
                rgb=np.zeros((4, 4, 3), np.uint8), depth_m=np.ones((4, 4), np.float32),
                K=np.eye(3), mesh=mesh, diameter_m=.2, mask=np.ones((4, 4), np.uint8),
                n_view=12, n_inplane=8, center_xyz_m=xyz)
        self.assertEqual([(name, len(poses), kw['iters']) for name, poses, kw in calls],
                         [('masked', 96, 3), ('sapr', 96, 2)])
        self.assertTrue(calls[0][2]['use_mask'])
        self.assertFalse(calls[1][2]['use_mask'])
        seeds = calls[0][1].numpy()
        np.testing.assert_allclose(seeds[:, :3, :3] @ center + seeds[:, :3, 3],
                                   np.repeat(xyz[None], 96, axis=0), atol=1e-6)
        self.assertEqual(groups, [96])  # All candidates compete in the same group.
        expected = seeds[73].copy()
        expected[0, 3] += .02
        np.testing.assert_allclose(result['pose_4x4'], expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
