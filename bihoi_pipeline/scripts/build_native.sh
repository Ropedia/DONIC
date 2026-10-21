#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
component="${1:?usage: build_native.sh foundationpose|wapr|sam3d-objects|depthanything3|hocap-annotation}"
python="$(hoi_python "$component")"; prefix="$(hoi_env "$component")"
[[ -x "$python" ]] || { echo "Missing environment $component" >&2; exit 1; }
[[ -f "$prefix/.hoi-managed" || "${HOI_ALLOW_EXISTING_NATIVE_BUILD:-0}" == 1 ]] || { echo 'Build only into package-managed environments by default.' >&2; exit 1; }
key="HOI_CUDA_HOME_${component^^}";key="${key//-/_}"
export CUDA_HOME="${!key:-${CUDA_HOME:-$prefix}}"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "No nvcc at $CUDA_HOME/bin/nvcc; set $key to a matching toolkit." >&2; exit 1; }
export PATH="$prefix/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${HOI_BUILD_JOBS:-2}"
"$python" - <<'CHECK'
import os,re,subprocess,torch
v=subprocess.check_output([os.environ['CUDA_HOME']+'/bin/nvcc','--version'],text=True)
built=re.search(r'release (\d+\.\d+)',v).group(1)
if built!=torch.version.cuda:raise SystemExit(f'CUDA toolkit {built} differs from torch CUDA {torch.version.cuda}; choose the matching toolkit.')
if not torch.cuda.is_available():raise SystemExit('CUDA unavailable; native builds have not been attempted.')
print('Torch/CUDA:',torch.__version__,built,'GPU:',torch.cuda.get_device_name(0))
CHECK
pip=("$python" -m pip install --no-build-isolation --no-deps)
case "$component" in
 foundationpose)
  "${pip[@]}" 'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@e73a7e7bfc580d1f9225ad9ff7d2c753c320aabd'
  "${pip[@]}" 'nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae'
  cmake -S "$HOI_PIPELINE_ROOT/FoundationPose/mycpp" -B "$HOI_PIPELINE_ROOT/FoundationPose/mycpp/build" -DCMAKE_PREFIX_PATH="$prefix" -DPYTHON_EXECUTABLE="$python" -Dpybind11_DIR="$("$python" -m pybind11 --cmakedir)"
  cmake --build "$HOI_PIPELINE_ROOT/FoundationPose/mycpp/build" -j "$MAX_JOBS"
  ;;
 wapr)
  "${pip[@]}" 'nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae'
  # The alternative OGL renderer compiles lazily; no weights are fetched here.
  ;;
 sam3d-objects)
  "${pip[@]}" 'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47'
  "${pip[@]}" 'nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae'
  "${pip[@]}" 'gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70'
  "${pip[@]}" kaolin==0.17.0 --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html
  # The original default attention backend is PyTorch SDPA. FlashAttention is optional.
  if [[ "${HOI_BUILD_FLASH_ATTENTION:-0}" == 1 ]]; then "${pip[@]}" flash-attn==2.8.3; fi
  ;;
 depthanything3)
  "${pip[@]}" 'gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70'
  ;;
 hocap-annotation)
  "$python" "$HOI_PROJECT_ROOT/scripts/fetch_external.py" hocap
  "${pip[@]}" -e "$HOI_EXTERNAL_ROOT/HO-Cap-Annotation"
  "${pip[@]}" "$HOI_EXTERNAL_ROOT/HO-Cap-Annotation/hocap_annotation/loss/meshsdf_loss"
  ;;
 *) echo "Unknown native component $component" >&2;exit 2;;
esac
