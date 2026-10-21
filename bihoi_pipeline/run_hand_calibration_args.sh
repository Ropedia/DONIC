#!/usr/bin/env bash
# Separate per-person setup; never invoked by the normal five-stage processing chain.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
exec "$(hoi_python easymocap)" "$HOI_PROJECT_ROOT/scripts/hand_calibration.py" "$@"
