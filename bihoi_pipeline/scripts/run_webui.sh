#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
python="$(hoi_python handobj_webui)"
[[ -x "$python" ]] || { echo "Missing WebUI environment: $python. Run scripts/install.sh --component handobj_webui" >&2; exit 1; }
export CONDA_EXE="$(hoi_conda)"
export CONDA_SH="${CONDA_SH:-$("$CONDA_EXE" info --base)/etc/profile.d/conda.sh}"
args=(--data-root "$HOI_DATA_ROOT" --host "${WEBUI_HOST:-127.0.0.1}" --port "${WEBUI_PORT:-7860}")
[[ "${WEBUI_SHARE:-0}" == 1 ]] && args+=(--share)
[[ "${WEBUI_NO_AUTH:-0}" == 1 ]] && args+=(--no-auth)
if [[ "${WEBUI_NO_AUTH:-0}" == 1 && ( "${WEBUI_SHARE:-0}" == 1 || "${WEBUI_HOST:-127.0.0.1}" != 127.0.0.1 ) ]]; then
  echo 'Public/bound-to-network access requires authentication. Set WEBUI_NO_AUTH=0.' >&2; exit 2
fi
cd -- "$HOI_PROJECT_ROOT"
exec "$python" "$HOI_PROJECT_ROOT/webui/interaction_capture_webui.py" "${args[@]}" "$@"
