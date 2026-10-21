"""Explicit per-person calibration action, separate from dataset processing jobs."""
import os
import re
import subprocess
from pathlib import Path

from settings import PROJECT_ROOT


def command(data_root, person, dataset, start, end, side, gpus, replace=False, dry_run=False):
    for label, value in [('person', person), ('dataset', dataset)]:
        if not value or value in {'.', '..'} or not re.fullmatch(r'[A-Za-z0-9_.-]+', str(value)):
            raise ValueError('Select a valid ' + label + ' folder name')
    if side not in {'left', 'right'}:
        raise ValueError('Choose left or right for this calibration capture')
    root = Path(data_root).expanduser().resolve()
    target = (root / 'rigid' / person).resolve()
    if root not in target.parents:
        raise ValueError('Person folder escapes the data root')
    args = ['bash', str(PROJECT_ROOT / 'run_hand_calibration_args.sh'),
        '--data-root', str(root), '--person_name', person, '--data_name', dataset,
        '--start_sec', str(start), '--end_sec', str(end), '--hand_side', side,
        '--gpus', str(gpus or 'auto')]
    if replace:
        args.append('--replace-prior')
    if dry_run:
        args.append('--dry-run')
    return args


def run(data_root, person, dataset, start, end, side, gpus, replace, dry_run):
    process = None
    try:
        args = command(data_root, person, dataset, start, end, side, gpus, replace, dry_run)
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        process = subprocess.Popen(args, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, start_new_session=True)
        log = ''
        yield 'Calibration started separately from the processing chain.', log
        for line in process.stdout:
            log = (log + line)[-20000:]
            yield 'Running calibration; requested interval is unchanged.', log
        code = process.wait()
        yield ('Calibration plan ready.' if dry_run and code == 0 else
               'Calibration complete; the person prior is ready.' if code == 0 else
               f'Calibration failed (exit {code}); inspect the log.'), log
    except (ValueError, OSError) as error:
        yield 'Calibration could not start: ' + str(error), ''
    finally:
        if process is not None and process.poll() is None:
            import signal
            os.killpg(process.pid, signal.SIGTERM)
