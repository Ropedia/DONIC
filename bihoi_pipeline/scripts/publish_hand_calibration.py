"""Validate the original FMHR beta contract before publishing a private person prior."""
import argparse
import datetime
import os
import shutil
import uuid
from pathlib import Path

import torch


def validate(data):
    if not isinstance(data, dict):
        raise ValueError('Expected the original restoration result dictionary')
    for key in ['real_scale_betas10_lstsq', 'original_betas10']:
        value = data.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (10,) or not torch.isfinite(value).all():
            raise ValueError(key + ' must be a finite tensor with shape (10,)')
    scale = float(data.get('real_scale', 0))
    if not 0 < scale < float('inf'):
        raise ValueError('real_scale must be finite and positive')
    return data


def publish(source, target, metadata, replace=False):
    data = validate(torch.load(source, map_location='cpu', weights_only=True))
    data['calibration'] = metadata
    target = Path(target)
    if target.exists() and not replace:
        raise FileExistsError('Person prior exists; replacement requires --replace-prior')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        backup = target.with_name(target.name + '.' + uuid.uuid4().hex + '.backup')
        shutil.copy2(target, backup)
        print('Existing prior retained at:', backup)
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.pending')
    torch.save(data, temporary)
    # Validate serialized bytes before the atomic publication.
    validate(torch.load(temporary, map_location='cpu', weights_only=True))
    os.replace(temporary, target)
    print('Published ten MANO betas:', target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    for flag in ['person-name', 'data-name', 'hand-side']:
        p.add_argument('--' + flag, required=True)
    for flag in ['start-sec', 'end-sec']:
        p.add_argument('--' + flag, type=float, required=True)
    p.add_argument('--replace-prior', action='store_true')
    a = p.parse_args()
    publish(a.source, a.target, {'person_name': a.person_name, 'data_name': a.data_name,
        'hand_side': a.hand_side, 'start_sec': a.start_sec, 'end_sec': a.end_sec,
        'fps': 10, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}, a.replace_prior)


if __name__ == '__main__':
    main()
