"""Portable runtime settings. Credentials are read from environment only."""
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = Path(os.getenv('HOI_PIPELINE_ROOT', PROJECT_ROOT / 'pipeline')).expanduser().resolve()
MODEL_ROOT = Path(os.getenv('HOI_MODEL_ROOT', PROJECT_ROOT / '.models')).expanduser().resolve()
CALIBRATION_DIR = Path(os.getenv('HOI_CALIBRATION_DIR', PROJECT_ROOT / '.calibration')).expanduser().resolve()
def environment(name):
    key = 'HOI_ENV_' + name.upper().replace('-', '_')
    return os.getenv(key, str(Path(os.getenv('HOI_ENV_ROOT', PROJECT_ROOT / '.environments')).expanduser().resolve() / name))
def python_for(name):
    value = environment(name)
    if Path(value).is_absolute():
        return Path(value) / 'bin/python'
    # Explicit named-environment override remains supported.
    import shutil, subprocess
    conda = os.getenv('CONDA_EXE') or shutil.which('conda')
    if not conda:
        raise RuntimeError('Conda is required to resolve a named environment: ' + value)
    base = subprocess.check_output([conda, 'info', '--base'], text=True).strip()
    return Path(base) / 'envs' / value / 'bin/python'
