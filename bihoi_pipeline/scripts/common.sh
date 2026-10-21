#!/usr/bin/env bash
# Shared portable settings; this file is sourced, not executed.
HOI_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HOI_PROJECT_ROOT
if [[ -f "${HOI_CONFIG_FILE:-$HOI_PROJECT_ROOT/.env}" ]]; then
  set -a
  source "${HOI_CONFIG_FILE:-$HOI_PROJECT_ROOT/.env}"
  set +a
fi
export HOI_PIPELINE_ROOT="${HOI_PIPELINE_ROOT:-$HOI_PROJECT_ROOT/pipeline}"
export HOI_DATA_ROOT="${HOI_DATA_ROOT:-$HOI_PROJECT_ROOT/data}"
export HOI_MODEL_ROOT="${HOI_MODEL_ROOT:-$HOI_PROJECT_ROOT/.models}"
export HOI_CALIBRATION_DIR="${HOI_CALIBRATION_DIR:-$HOI_PROJECT_ROOT/.calibration}"
export HOI_ENV_ROOT="${HOI_ENV_ROOT:-$HOI_PROJECT_ROOT/.environments}"
export HOI_EXTERNAL_ROOT="${HOI_EXTERNAL_ROOT:-$HOI_PROJECT_ROOT/.external}"
export HOI_MOGE_MODEL="${HOI_MOGE_MODEL:-$HOI_MODEL_ROOT/moge/model.pt}"
export HOI_MANO_DIR="${HOI_MANO_DIR:-$HOI_MODEL_ROOT/mano}"
export HOI_SMPL_MODEL_DIR="${HOI_SMPL_MODEL_DIR:-$HOI_PIPELINE_ROOT/data/smplx}"
export WEBUI_RUNS_DIR="${WEBUI_RUNS_DIR:-$HOI_PROJECT_ROOT/.webui_runs}"
export PYTHONPATH="$HOI_PROJECT_ROOT/webui:$HOI_PIPELINE_ROOT:$HOI_EXTERNAL_ROOT/HO-Cap-Annotation${PYTHONPATH:+:$PYTHONPATH}"
export WAPR_CACHE_DIR="${WAPR_CACHE_DIR:-$HOI_MODEL_ROOT/wapr}"
export GRADIO_ANALYTICS_ENABLED=False
hoi_conda() {
  local executable="${CONDA_EXE:-}"
  [[ -n "$executable" && -x "$executable" ]] || executable="$(command -v conda || true)"
  if [[ -z "$executable" ]]; then
    for candidate in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" /opt/conda/bin/conda; do
      if [[ -x "$candidate" ]]; then executable="$candidate"; break; fi
    done
  fi
  [[ -n "$executable" ]] || { echo 'Conda missing: set CONDA_EXE or install Miniforge.' >&2; return 127; }
  printf '%s\n' "$executable"
}
hoi_env() {
  local name="$1" key value executable
  key="HOI_ENV_${name^^}"; key="${key//-/_}"; value="${!key:-$HOI_ENV_ROOT/$name}"
  if [[ "$value" != /* ]]; then
    executable="$(hoi_conda)" || return
    value="$("$executable" info --base)/envs/$value"
  fi
  printf '%s\n' "$value"
}
hoi_python() { printf '%s/bin/python\n' "$(hoi_env "$1")"; }
