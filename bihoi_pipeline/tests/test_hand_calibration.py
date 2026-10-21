import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'webui'), str(ROOT / 'pipeline')]
import hand_calibration as calibration
import calibration_panel
from publish_hand_calibration import publish, validate


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


class CalibrationTests(unittest.TestCase):
    def args(self, *extra):
        return calibration.parser().parse_args(['--data_name', 'hands_collector01', '--start_sec', '0',
            '--end_sec', '0.1', '--hand_side', 'right', *extra])

    def test_reference_mapping_and_exact_interval(self):
        p = calibration.plan(self.args(), 'test')
        self.assertEqual(p['person'], 'collector01')
        self.assertEqual(p['frames'], 1)
        self.assertEqual(p['target'].parts[-4:], ('rigid', 'collector01', 'calib', 'restored_flat_real_scale.pt'))
        renamer = p['commands'][0][2]
        self.assertIn('--strict-calibration', renamer)
        self.assertEqual(Path(renamer[1]).name, 'rename_videos_31cams.py')
        extract = p['commands'][1][2]
        self.assertIn('--skip-rename', extract)
        self.assertEqual(extract[extract.index('--start-sec') + 1], '0.0')
        self.assertEqual(extract[extract.index('--end-sec') + 1], '0.1')
        self.assertEqual(extract[extract.index('--fps') + 1], '10')

    def test_argument_propagation_both_sides(self):
        for side, body, model in [('right', 'handr', 'manor'), ('left', 'handl', 'manol')]:
            p = calibration.plan(self.args('--hand_side', side, '--person_name', 'another_person'), 'test')
            tri, gcn, restore, publication = [x[2] for x in p['commands'][3:]]
            self.assertEqual(tri[tri.index('--body') + 1], body)
            self.assertEqual(tri[tri.index('--model') + 1], model)
            for cmd in [gcn, restore]:
                self.assertEqual(cmd[cmd.index('--name') + 1], 'hands_collector01')
                self.assertEqual(cmd[cmd.index('--hand_side') + 1], side)
            self.assertEqual(restore[restore.index('--real-kpts-json') + 1], str(p['work_data'] / f'output/{model}/keypoints3d/000000.json'))
            self.assertEqual(publication[publication.index('--person-name') + 1], 'another_person')
            self.assertEqual(publication[publication.index('--end-sec') + 1], '0.1')

    def test_unmapped_capture_requires_explicit_person(self):
        with self.assertRaisesRegex(ValueError, 'No verified person mapping'):
            calibration.plan(self.args('--data_name', 'unknown_capture'))
        self.assertEqual(calibration.plan(self.args('--data_name', 'unknown_capture', '--person_name', 'collector'))['person'], 'collector')

    def test_invalid_interval_and_non10fps_rejected(self):
        for flags in [('--end_sec', '0'), ('--end_sec', '0.09'), ('--start_sec', '-1'),
                      ('--start_sec', 'nan'), ('--end_sec', 'inf'), ('--fps', '30')]:
            with self.assertRaises(ValueError):
                calibration.plan(self.args(*flags))

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'root'; root.mkdir()
            (root / 'hands_collector01').symlink_to(Path(directory), target_is_directory=True)
            with self.assertRaises(ValueError):
                calibration.plan(self.args('--calibration-root', str(root)))

    def run_renamer(self, videos, output, *options):
        rename = module(ROOT / 'pipeline/rename_videos_31cams.py')
        args = ['rename_videos_31cams.py', '--input_dir', str(videos), '--output_dir', str(output),
                '--strict-calibration', *options]
        with patch.object(sys, 'argv', args):
            rename.main()

    def test_calibration_camera_order_and_immutable_inputs(self):
        order = module(ROOT / 'pipeline/rename_videos_31cams.py').CALIB_CAMERA_ORDER
        self.assertEqual(len(order), 31)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input/hands_collector01'; (source / 'videos').mkdir(parents=True)
            for camera in reversed(order):
                (source / 'videos' / (camera + '.mp4')).write_bytes(camera.encode())
            for filename in ['intri.yml', 'extri.yml']:
                (source / filename).write_text('synthetic calibration')
            with patch.dict(os.environ, {'WEBUI_RUNS_DIR': str(root / 'runs')}):
                p = calibration.plan(self.args('--calibration-root', str(root / 'input'), '--data-root', str(root / 'output')), 'isolated')
                calibration.prepare_workspace(p)
            self.assertEqual(list((p['work_data'] / 'videos').iterdir()), [])
            self.run_renamer(source / 'videos', p['work_data'] / 'videos')
            for index, camera in enumerate(order, 1):
                copied = p['work_data'] / 'videos' / f'{index}.mp4'
                self.assertEqual(copied.read_bytes(), camera.encode())
                self.assertFalse(copied.is_symlink())
            (p['work_data'] / 'videos/1.mp4').write_bytes(b'changed copy')
            self.assertEqual((source / 'videos' / (order[0] + '.mp4')).read_bytes(), order[0].encode())

    def test_strict_calibration_rejects_ambiguous_and_missing_cameras(self):
        order = module(ROOT / 'pipeline/rename_videos_31cams.py').CALIB_CAMERA_ORDER
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); videos = root / 'videos'; videos.mkdir()
            for camera in order[:-1]:
                (videos / (camera + '.mp4')).write_bytes(b'test')
            with self.assertRaisesRegex(SystemExit, 'requires all 31'):
                self.run_renamer(videos, root / 'out')
            (videos / 'other_cam.mp4').write_bytes(b'test')
            with self.assertRaisesRegex(SystemExit, 'only part'):
                self.run_renamer(videos, root / 'out')
            self.assertFalse((root / 'out').exists())

    def test_strict_calibration_preserves_legacy_and_numeric_indices(self):
        for style in ('numeric', 'rpizero'):
            with self.subTest(style=style), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); videos = root / 'videos'; videos.mkdir()
                for index in reversed(range(31)):
                    filename = f'{index+1}.mp4' if style == 'numeric' else f'rpizero{index:03d}_test0000.mp4'
                    (videos / filename).write_bytes(str(index + 1).encode())
                self.run_renamer(videos, root / 'out')
                for index in range(1, 32):
                    self.assertEqual((root / 'out' / f'{index}.mp4').read_bytes(), str(index).encode())

    def test_frame_count_failure_never_extends_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'images/1').mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, 'interval was not extended'):
                calibration.verify_frames({'work_data': root, 'frames': 1})

    def test_actual_ffmpeg_one_frame_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); video = root / '1.mp4'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                'color=c=red:s=64x64:r=30:d=1', '-c:v', 'mpeg4', str(video)], check=True)
            extractor = module(ROOT / 'pipeline/rename_videos_extract_frames_ffmpeg_pts_compat_resize.py')
            count = extractor.extract_frames_from_video(video, root / 'images/1', 'ffmpeg',
                start_sec=0, end_sec=0.1, sample_fps=10, execute=True, verify_count=True)
            self.assertEqual(count, 1)
            self.assertEqual([f.name for f in (root / 'images/1').iterdir()], ['000000.jpg'])

    def result(self):
        return {'real_scale_betas10_lstsq': torch.arange(10, dtype=torch.float32),
                'original_betas10': torch.zeros(10), 'real_scale': 1.2}

    def test_beta_validation(self):
        for value in [torch.zeros(9), torch.zeros(1, 10), torch.full((10,), float('nan'))]:
            data = self.result(); data['real_scale_betas10_lstsq'] = value
            with self.assertRaises(ValueError):
                validate(data)
        for scale in [0, -1, float('nan'), float('inf')]:
            data = self.result(); data['real_scale'] = scale
            with self.assertRaises(ValueError):
                validate(data)

    def test_publish_creates_person_contract_and_retains_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'result.pt'; torch.save(self.result(), source)
            target = root / 'rigid/collector/calib/restored_flat_real_scale.pt'
            publish(source, target, {'person_name': 'collector'})
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                publish(source, target, {})
            publish(source, target, {'person_name': 'collector'}, replace=True)
            backups = list(target.parent.glob('*.backup'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)

    def test_existing_hands_loader_and_final_export_accept_result(self):
        from easymocap.pipeline.basic import load_betas_from_restored_pt, set_fixed_betas_to_params
        exporter = module(ROOT / 'pipeline/FoundationPose/export_standardized_mano_object_table_4rings_first500.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source.pt'; torch.save(self.result(), source)
            target = root / 'rigid/person/calib/restored_flat_real_scale.pt'; publish(source, target, {})
            old = Path.cwd()
            try:
                os.chdir(root)
                betas = load_betas_from_restored_pt('rigid/person')
                params = set_fixed_betas_to_params({'shapes': np.zeros((2, 10))}, 'rigid/person')
            finally:
                os.chdir(old)
            np.testing.assert_array_equal(betas, np.arange(10))
            np.testing.assert_array_equal(params['shapes'], np.tile(np.arange(10), (2, 1)))
            frame = root / 'frame.json'
            frame.write_text(json.dumps([{'poses': [0] * 48, 'shapes': params['shapes'][0].tolist(), 'Rh': [0] * 3, 'Th': [0] * 3}]))
            np.testing.assert_array_equal(exporter.read_hand_json(frame)['betas'], betas)

    def test_original_scale_computation(self):
        restore = module(ROOT / 'pipeline/FMHR/restore_gcn_mano_flat_real_scale_v3_all_bones.py')
        joints = np.arange(63).reshape(21, 3) / 100
        chains = restore.parse_finger_chains('0,1,2,3,4;0,5,6,7,8;0,9,10,11,12;0,13,14,15,16;0,17,18,19,20')
        result = restore.compute_scale_from_chains(joints * 2, joints, chains)
        self.assertAlmostEqual(result['scale_sum_ratio'], 2)

    def test_webui_command_is_separate_and_safe(self):
        cmd = calibration_panel.command('/tmp/calibration test', 'person', 'capture', 0, 0.1, 'right', '0', dry_run=True)
        self.assertEqual(cmd[cmd.index('--person_name') + 1], 'person')
        self.assertIn('--dry-run', cmd)
        with self.assertRaises(ValueError):
            calibration_panel.command('/tmp/test', '../person', 'capture', 0, 0.1, 'right', '0')

    def test_webui_preview_runs_launcher_without_inference(self):
        # A real shell/Python round trip, rather than merely checking command text.
        with patch.dict(os.environ, {'HOI_ENV_EASYMOCAP': str(Path(sys.executable).resolve().parents[1])}):
            results = list(calibration_panel.run('/tmp/calibration preview', 'collector01', 'hands_collector01', 0, 0.1, 'right', '0', False, True))
        self.assertEqual(results[-1][0], 'Calibration plan ready.')
        self.assertIn('1 frame(s)/camera', results[-1][1])
        self.assertIn('real-scale MANO beta fitting', results[-1][1])

    def test_two_runs_do_not_share_gcn_outputs(self):
        a, b = calibration.plan(self.args()), calibration.plan(self.args())
        self.assertNotEqual(a['workspace'], b['workspace'])
        for p in [a, b]:
            self.assertIn(str(p['workspace'] / 'gcn_out'), p['commands'][4][2])

    def test_fmhr_glut_import_uses_glx_without_changing_other_stages(self):
        with patch.dict(os.environ, {'PYOPENGL_PLATFORM': 'egl'}):
            self.assertEqual(calibration.stage_environment(self.args(), 4)['PYOPENGL_PLATFORM'], 'glx')
            self.assertEqual(calibration.stage_environment(self.args(), 3)['PYOPENGL_PLATFORM'], 'egl')


if __name__ == '__main__':
    unittest.main()
