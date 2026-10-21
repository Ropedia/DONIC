#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
component=all; dry=0; native=1
while [[ $# -gt 0 ]]; do
 case "$1" in
  --component) component="${2:?component required}";shift 2;;
  --dry-run) dry=1;shift;;
  --skip-native) native=0;shift;;
  -h|--help) echo 'install.sh [--component all|handobj_webui|easymocap|sam3|depthanything3|sam3d-objects|foundationpose|wapr|wilor|hocap-annotation] [--skip-native] [--dry-run]';exit 0;;
  *) echo "Unknown argument $1" >&2;exit 2;;
 esac
done
[[ "$(uname -s)" == Linux ]] || { echo 'The CUDA/EGL pipeline requires Linux.' >&2;exit 1; }
conda="$(hoi_conda)";export CONDA_EXE="$conda"
all=(handobj_webui easymocap sam3 depthanything3 sam3d-objects foundationpose wapr wilor hocap-annotation)
if [[ "$component" == all ]]; then selected=("${all[@]}");else
 found=0;for item in "${all[@]}";do [[ "$item" == "$component" ]] && found=1;done
 [[ "$found" == 1 ]] || { echo "Unknown component $component" >&2;exit 2; }
 selected=("$component")
fi
for executable in git ffmpeg ffprobe;do command -v "$executable" >/dev/null || { echo "Required executable absent: $executable (install with system package manager or Conda)." >&2;exit 1; };done
if command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=driver_version,memory.total --format=csv,noheader;then :;else
 [[ "$component" == handobj_webui || "$dry" == 1 ]] || { echo 'NVIDIA driver/GPU inaccessible; processing environment installation stopped.' >&2;exit 1; }
fi
for name in "${selected[@]}";do
 prefix="$(hoi_env "$name")";python="$prefix/bin/python"
 echo "Component: $name -> $prefix"
 if [[ "$dry" == 1 ]];then echo "PLAN conda env create; pinned PyTorch/direct packages; editable sources; native=$native. No models downloaded.";continue;fi
 if [[ -d "$prefix/conda-meta" ]];then
  [[ -f "$prefix/.hoi-managed" ]] || { echo "Refusing to modify unmanaged environment $prefix. Use a new HOI_ENV_ROOT." >&2;exit 1; }
 else
  [[ ! -e "$prefix" ]] || { echo "Existing incomplete/conflicting prefix $prefix; no files removed." >&2;exit 1; }
  "$conda" env create --prefix "$prefix" --file "$HOI_PROJECT_ROOT/environments/$name.yml"
  touch "$prefix/.hoi-managed"
 fi
 if [[ "$name" != handobj_webui ]];then
  read -r torch torchvision cuda < <("$python" - "$HOI_PROJECT_ROOT/environments/observed.json" "$name" <<'VERSIONS'
import json,sys
x=json.load(open(sys.argv[1]))[sys.argv[2]]
t=x['torch']['version'];print(t,x['torchvision']['version'],t.split('+cu')[1])
VERSIONS
)
  "$python" -m pip install "torch==$torch" "torchvision==$torchvision" --index-url "https://download.pytorch.org/whl/cu$cuda"
  "$python" - <<'GPUCHECK'
import torch
if not torch.cuda.is_available():raise SystemExit('Installed Torch cannot initialize CUDA; check NVIDIA driver compatibility before continuing.')
x=torch.zeros(1,device='cuda');torch.cuda.synchronize();print('CUDA runtime initialization passed:',torch.version.cuda)
GPUCHECK
  # Keep subsequent dependency resolution on the observed Torch build.
  printf 'torch==%s\ntorchvision==%s\n' "$torch" "$torchvision" > "$prefix/.hoi-torch-constraints.txt"
  export PIP_CONSTRAINT="$prefix/.hoi-torch-constraints.txt"
 fi
 requirements="$HOI_PROJECT_ROOT/environments/$name.requirements.txt"
 if [[ "$name" == handobj_webui ]]; then requirements="$HOI_PROJECT_ROOT/environments/handobj_webui.lock.txt"; fi
 "$python" -m pip install --no-build-isolation -r "$requirements"
 "$python" -m pip install --no-deps --no-build-isolation -e "$HOI_PIPELINE_ROOT"
 case "$name" in
  sam3)
   "$python" -m pip install --no-deps --no-build-isolation -e "$HOI_PIPELINE_ROOT/sam3"
   SAM2_BUILD_CUDA=0 "$python" -m pip install --no-deps --no-build-isolation -e "$HOI_PIPELINE_ROOT/sam3/sam2";;
  depthanything3) "$python" -m pip install --no-deps --no-build-isolation -e "$HOI_PIPELINE_ROOT/Depth-Anything-3";;
  sam3d-objects)
   "$python" -m pip install --no-deps --no-build-isolation -e "$HOI_PIPELINE_ROOT/MV-SAM3D";;
 esac
 case "$name" in
  foundationpose|wapr|sam3d-objects|hocap-annotation)
   if [[ "$native" == 1 ]];then "$HOI_PROJECT_ROOT/scripts/build_native.sh" "$name";else echo "Native dependencies for $name remain unbuilt.";fi;;
 esac
 "$python" -m pip check
 unset PIP_CONSTRAINT
 # Record exact installed metadata locally, not in Git.
 "$python" -m pip freeze > "$prefix/.hoi-installed.txt"
done
if [[ "$dry" == 0 ]];then
 "$(hoi_python "${selected[0]}")" "$HOI_PROJECT_ROOT/scripts/setup_assets.py"
 echo 'Installation entrypoint completed. Supply calibration, licensed models, and restricted WiLoR source, then run scripts/check_env.sh.'
fi
