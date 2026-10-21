#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
python="${HOI_DOWNLOAD_PYTHON:-$(hoi_python handobj_webui)}"
exec "$python" "$HOI_PROJECT_ROOT/scripts/download_models.py" "$@"
