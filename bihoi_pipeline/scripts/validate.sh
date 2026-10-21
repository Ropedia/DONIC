#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
cd -- "$HOI_PROJECT_ROOT"
ui="$(hoi_python handobj_webui)";cpu="$(hoi_python easymocap)"
"$ui" -m unittest discover -s tests -p test_packaging.py -v
"$cpu" -m unittest discover -s tests -p test_cpu_stages.py -v
"$cpu" -m unittest discover -s tests -p test_hand_calibration.py -v
"$ui" -m unittest discover -s scripts/hoi_qualitative/tests -p test_web_panel.py -v
"$cpu" -m unittest discover -s scripts/hoi_qualitative/tests -p test_single_sequence.py -v
"$ui" -m unittest discover -s scripts/hoi_qualitative/tests -p test_render_hoi.py -v
"$ui" "$HOI_PROJECT_ROOT/scripts/audit_source.py" --syntax-only

# Candidate routing, cloud geometry, recovery, HTTP and video timing regressions.
wapr="$(hoi_python wapr)"
if [[ -x "$wapr" ]]; then
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$wapr" -m unittest discover -s webui/tests -v
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$wapr" -m unittest discover -s web_demos/tests -p 'test_*.py' -v
else
  echo 'WAPR environment absent; install --component wapr to run tracking/demo tests.'
fi
