#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec "$(hoi_python easymocap)" "$HOI_PROJECT_ROOT/scripts/prepare_mano_regressors.py" "$@"
