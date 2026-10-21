#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
python="${HOI_CHECK_PYTHON:-$(hoi_python handobj_webui)}"
[[ -x "$python" ]] || python="$(command -v python3)"
exec "$python" "$HOI_PROJECT_ROOT/scripts/check_env.py" "$@"
