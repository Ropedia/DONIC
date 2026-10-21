"""Run the existing WiLoR/EasyMocap/FMHR hand calibration in an isolated workspace."""
import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'webui'))
from settings import PIPELINE_ROOT, MODEL_ROOT, python_for


def name(value):
    if not value or value in {'.', '..'} or not re.fullmatch(r'[A-Za-z0-9_.-]+', value):
        raise ValueError('Dataset/person names must be single folder names: letters, digits, _, -, .')
    return value


def under(root, relative):
    root = Path(root).expanduser().resolve()
    result = (root / relative).resolve()
    if result != root and root not in result.parents:
        raise ValueError('Path escapes configured root: ' + str(relative))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_name', '--data-name', required=True)
    p.add_argument('--person_name', '--person-name', default=None)
    p.add_argument('--start_sec', '--start-sec', type=float, default=0)
    p.add_argument('--end_sec', '--end-sec', type=float, required=True)
    p.add_argument('--hand_side', '--hand-side', choices=['left', 'right'], default='right')
    p.add_argument('--fps', type=int, default=10)
    p.add_argument('--data-root', type=Path, default=Path(os.getenv('HOI_DATA_ROOT', ROOT / 'data')))
    p.add_argument('--calibration-root', type=Path, default=None)
    p.add_argument('--mapping', type=Path, default=ROOT / 'configs/hand_calibration.json')
    p.add_argument('--gpus', default=os.getenv('CUDA_VISIBLE_DEVICES') or os.getenv('WEBUI_ALLOWED_GPUS', 'auto'))
    p.add_argument('--min-free-vram-mb', type=int, default=32768)
    p.add_argument('--replace-prior', action='store_true', help='Back up an existing prior before replacing it')
    p.add_argument('--dry-run', action='store_true', help='Print exact stage commands without writing or inference')
    p.add_argument('--check', action='store_true', help='Check inputs/assets without running inference')
    return p


def plan(args, run_id=None):
    name(args.data_name)
    if args.gpus.strip().lower() in {'all', '*', ''}:
        args.gpus = 'auto'
    elif args.gpus != 'auto' and not re.fullmatch(r'\d+(?:,\d+)*', args.gpus):
        raise ValueError('--gpus must be auto, all, or comma-separated physical GPU IDs')
    if args.min_free_vram_mb < 1:
        raise ValueError('--min-free-vram-mb must be positive')
    if not math.isfinite(args.start_sec) or not math.isfinite(args.end_sec) or args.start_sec < 0 or args.end_sec <= args.start_sec:
        raise ValueError('Require finite 0 <= start_sec < end_sec; the interval is never extended')
    if args.fps != 10:
        raise ValueError('This pipeline is the 10 FPS version; --fps must be 10')
    frames = math.floor((args.end_sec - args.start_sec) * args.fps + 1e-9)
    if frames < 1:
        raise ValueError('The requested interval contains fewer than one 10 FPS sample; choose an interval explicitly')
    mappings = json.loads(args.mapping.read_text()).get('datasets', {})
    person = args.person_name or mappings.get(args.data_name, {}).get('person_name')
    if not person:
        raise ValueError('No verified person mapping: set --person_name or add this dataset to --mapping')
    name(person)
    data_root = args.data_root.expanduser().resolve()
    input_root = args.calibration_root or Path(os.getenv('HOI_HAND_CALIBRATION_ROOT', data_root / 'hand_calibration'))
    dataset = under(input_root, args.data_name)
    target = under(data_root, f'rigid/{person}/calib/restored_flat_real_scale.pt')
    runs = Path(os.getenv('WEBUI_RUNS_DIR', ROOT / '.webui_runs')).expanduser().resolve()
    workspace = under(runs, 'hand_calibration/' + (run_id or uuid.uuid4().hex))
    if dataset == workspace or dataset in workspace.parents or workspace in dataset.parents:
        raise ValueError('Calibration input and workspace must be separate')
    work_data = workspace / 'dataset'
    fmhr_models = Path(os.getenv('HOI_FMHR_MODEL_DIR', MODEL_ROOT / 'fmhr')).expanduser().resolve()
    mano = Path(os.getenv('HOI_MANO_DIR', MODEL_ROOT / 'mano')).expanduser().resolve()
    wilor = Path(os.getenv('HOI_EXTERNAL_ROOT', ROOT / '.external')).expanduser().resolve() / 'WiLoR'
    cpu, detector = str(python_for('easymocap')), str(python_for('wilor'))
    side = args.hand_side
    body, model = ('handr', 'manor') if side == 'right' else ('handl', 'manol')
    keypoints = work_data / f'output/{model}/keypoints3d/000000.json'
    commands = [
        ('Copy videos in calibrated 31-camera order', PIPELINE_ROOT, [cpu, str(PIPELINE_ROOT / 'rename_videos_31cams.py'), '--input_dir', str(dataset / 'videos'), '--output_dir', str(work_data / 'videos'), '--strict-calibration']),
        ('Extract exact 10 FPS interval', PIPELINE_ROOT, [cpu, str(PIPELINE_ROOT / 'rename_videos_extract_frames_ffmpeg_pts_compat_resize.py'), '--data-name', str(work_data), '--start-sec', str(args.start_sec), '--end-sec', str(args.end_sec), '--fps', str(args.fps), '--execute', '--verify-count', '--skip-rename']),
        ('WiLoR hand annotations', wilor, [detector, str(wilor / 'demo.py'), '--image_root', str(work_data / 'images'), '--annots_root', str(work_data / 'annots_fallback'), '--gpus', args.gpus, '--min_free_vram_mb', str(args.min_free_vram_mb)]),
        ('Original EasyMocap triangulation', PIPELINE_ROOT, [cpu, str(PIPELINE_ROOT / 'apps/demo/mv1p713.py'), str(work_data), '--name', args.data_name, '--out', str(work_data / f'output/{model}'), '--vis_det', '--vis_repro', '--undis', '--body', body, '--model', model, '--gender', 'male', '--vis_smpl']),
        ('Original FMHR GCN inference', workspace, [cpu, str(PIPELINE_ROOT / 'FMHR/multihands_mano.py'), '--name', args.data_name, '--hand_side', side, '--data-dir', str(work_data), '--model-dir', str(fmhr_models), '--out-dir', str(workspace / 'gcn_out')]),
        ('Original real-scale MANO beta fitting', workspace, [cpu, str(PIPELINE_ROOT / 'FMHR/restore_gcn_mano_flat_real_scale_v3_all_bones.py'), '--name', args.data_name, '--hand_side', side, '--gcn-pt', str(workspace / f'gcn_out/0_{side}.pt'), '--template-obj', str(fmhr_models / f'TEMPLATE_{side.upper()}.obj'), '--mano-model-path', str(mano / f'MANO_{side.upper()}.pkl'), '--real-kpts-json', str(keypoints), '--out-dir', str(workspace / 'result')]),
        ('Validate and publish person prior', workspace, [cpu, str(ROOT / 'scripts/publish_hand_calibration.py'), '--source', str(workspace / 'result/restored_flat_real_scale.pt'), '--target', str(target), '--person-name', person, '--data-name', args.data_name, '--hand-side', side, '--start-sec', str(args.start_sec), '--end-sec', str(args.end_sec), *(['--replace-prior'] if args.replace_prior else [])]),
    ]
    return dict(person=person, dataset=dataset, target=target, workspace=workspace, work_data=work_data, fmhr_models=fmhr_models, mano=mano, wilor=wilor, frames=frames, commands=commands)


def check_inputs(args, p):
    videos = p['dataset'] / 'videos'
    if not videos.is_dir() or not any(f.is_file() and f.suffix.lower() == '.mp4' for f in videos.iterdir()):
        raise FileNotFoundError('Expected 31 synchronized .mp4 calibration videos directly under ' + str(videos))
    required = [p['dataset'] / f for f in ['intri.yml', 'extri.yml']]
    required += [p['fmhr_models'] / f for f in ['gcn_left.pth', 'gcn_right.pth', 'TEMPLATE_LEFT.obj', 'TEMPLATE_RIGHT.obj']]
    required += [p['mano'] / f'MANO_{side}.pkl' for side in ['LEFT', 'RIGHT']]
    required += [p['wilor'] / f for f in ['demo.py', 'pretrained_models/wilor_final.ckpt', 'pretrained_models/model_config.yaml', 'pretrained_models/detector.pt', 'mano_data/MANO_RIGHT.pkl', 'mano_data/mano_mean_params.npz']]
    required += [Path(c[2][0]) for c in p['commands']]
    missing = sorted({str(f) for f in required if not f.is_file()})
    if missing:
        raise FileNotFoundError('Missing calibration dependencies:\n' + '\n'.join(missing))
    manifest = json.loads((ROOT / 'provenance/restricted-wilor.json').read_text())
    import hashlib
    for record in manifest['required_files']:
        f = p['wilor'] / record['path']
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != record['sha256']:
            raise RuntimeError('Licensed WiLoR variant is missing or different: ' + record['path'])
    if p['target'].exists() and not args.replace_prior:
        raise FileExistsError('Prior already exists; use --replace-prior to retain a backup and recalibrate')
    for exe in ['ffmpeg', 'ffprobe']:
        if not shutil.which(exe):
            raise RuntimeError('Missing executable: ' + exe)
    # Preview the strict camera mapping without modifying the input capture.
    subprocess.run([*p['commands'][0][2], '--dry-run'], cwd=PIPELINE_ROOT, check=True)


def prepare_workspace(p):
    p['workspace'].mkdir(parents=True, exist_ok=False)
    videos = p['work_data'] / 'videos'
    videos.mkdir(parents=True)
    # Stage 0 copies videos directly from the original capture and renames them
    # in this isolated workspace; source videos are never modified.
    for filename in ['intri.yml', 'extri.yml']:
        shutil.copy2(p['dataset'] / filename, p['work_data'] / filename)
    # FMHR's original import-time MANO layers resolve ./mano from its working directory.
    (p['workspace'] / 'mano').symlink_to(p['mano'], target_is_directory=True)


def verify_frames(p):
    folders = [f for f in (p['work_data'] / 'images').iterdir() if f.is_dir()]
    if not folders:
        raise RuntimeError('No extracted camera frames')
    for folder in folders:
        count = len(list(folder.glob('*.jpg')))
        if count != p['frames']:
            raise RuntimeError(f'{folder.name}: expected {p["frames"]} exact-interval frames, got {count}; interval was not extended')


def stage_environment(args, index):
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(PIPELINE_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    # psbody's import-only MeshViewer requires GLX GLUT font symbols. FMHR does
    # no OpenGL rendering; the other stages retain EGL for headless rendering.
    env['PYOPENGL_PLATFORM'] = 'glx' if index == 4 else env.get('PYOPENGL_PLATFORM', 'egl')
    if args.gpus not in {'all', 'auto', '*', ''}:
        env['CUDA_VISIBLE_DEVICES'] = args.gpus
    return env


def main(argv=None):
    args = parser().parse_args(argv)
    p = plan(args)
    print(f'Calibration: {args.data_name} → person {p["person"]}; {args.hand_side}; [{args.start_sec}, {args.end_sec}) at 10 FPS, {p["frames"]} frame(s)/camera', flush=True)
    print('Person prior:', p['target'], flush=True)
    if args.dry_run:
        for title, cwd, command in p['commands']:
            print(title + '\n  cwd: ' + str(cwd) + '\n  ' + shlex.join(command))
        return 0
    check_inputs(args, p)
    if args.check:
        print('Input and asset paths passed. GPU/native imports are checked when stages run.')
        return 0
    prepare_workspace(p)
    for index, (title, cwd, command) in enumerate(p['commands']):
        print('\n[Stage] ' + title, flush=True)
        subprocess.run(command, cwd=cwd, env=stage_environment(args, index), check=True)
        if index == 1:
            verify_frames(p)
    print('[Done] Calibration prior: ' + str(p['target']), flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as error:
        print('[Calibration error] ' + str(error), file=sys.stderr)
        sys.exit(1)
