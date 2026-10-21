#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/../common.sh"
if [[ -n "${HOI_RENDER_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$HOI_RENDER_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
fi
export PYOPENGL_PLATFORM=egl NUMPY_MADVISE_HUGEPAGE=0
export OMP_NUM_THREADS="${HOI_RENDER_THREADS:-2}" OPENBLAS_NUM_THREADS="${HOI_RENDER_THREADS:-2}"
exec "$(hoi_python easymocap)" "$script_dir/single_sequence.py" "$@"
