#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
prefix="$(hoi_env easymocap)"; python="$(hoi_python easymocap)"
[[ -f "$prefix/.hoi-managed" ]] || { echo 'Install calibration dependencies only into a package-managed easymocap prefix.' >&2; exit 1; }
[[ "${HOI_ACCEPT_PSBODY_MESH_LICENSE:-0}" == 1 ]] || { echo 'Read licenses/psbody-mesh.txt; set HOI_ACCEPT_PSBODY_MESH_LICENSE=1 after accepting its terms.' >&2; exit 1; }
# psbody-mesh compiles native mesh helpers and may require system Boost/CGAL tools.
# Preserve the environment's NumPy/Torch versions during optional installation.
export PIP_CONSTRAINT="$prefix/.hoi-torch-constraints.txt"
"$python" -m pip install --no-build-isolation --no-deps -r "$HOI_PROJECT_ROOT/environments/hand-calibration.requirements.txt"
"$python" -m pip check
PYOPENGL_PLATFORM=glx "$python" -c 'from psbody.mesh import Mesh; import plyfile; from FMHR.models.gcn import GCNDecoder; print("FMHR calibration dependencies imported")'
