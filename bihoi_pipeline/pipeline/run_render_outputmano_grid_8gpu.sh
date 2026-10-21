#!/usr/bin/env bash
# Render the optimized hand+object result of Stage 5 (outputmano/) as per-camera
# overlays + one 31-camera grid image per frame, split across several GPUs.
#
# The JointPoseSolver script (run_joint_pose_solver_only_and_export_*.sh) runs with
# --skip_render so an EGL/pyrender problem can never fail the optimization.  This
# script is the separate, render-only pass over its outputs:
#   object poses : outputmano/smoothed_ob_in_world/*.txt
#   hands        : outputmano/contact_mano/{left,right}/smpl/*.json
#   mesh         : outputmano/object_mesh_scaled.ply (falls back to meshes/compressed.ply)
# Output: outputmano/grid/<frame>.jpg (+ outputmano/by_cam/<cam>/overlay/<frame>.png)
set -Eeuo pipefail

EASYMOCAP_ROOT="${HOI_PIPELINE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
DATA_NAME=""
GPU_IDS="0"
DOWNSCALE="1"   # K scale factor: images/ and cam_K.txt / intri.yml are already at the same resolution
GRID_FPS="10"

usage() {
  cat <<'USAGE'
Usage:
  bash run_render_outputmano_grid_8gpu.sh --data_name NAME [--gpus 0,1,2,3] [--downscale 1] [--grid_fps 10]

  --gpus IDS      comma-separated PHYSICAL GPU ids (one render worker per GPU)
USAGE
}

die() { echo "[ERROR] $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_name|--data-name) DATA_NAME="${2:?}"; shift 2 ;;
    --gpus) GPU_IDS="${2:?}"; shift 2 ;;
    --downscale) DOWNSCALE="${2:?}"; shift 2 ;;
    --grid_fps|--grid-fps) GRID_FPS="${2:?}"; shift 2 ;;
    # Accepted for CLI compatibility with the solver script; not needed here.
    --hand|--hand_side|--min_free_vram_mb|--max_gpus) shift 2 ;;
    left|right|bimanual) shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "$DATA_NAME" ]] || die "--data_name is required"
[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "Invalid --gpus '$GPU_IDS' (need comma-separated ids)"

DATA_DIR="${EASYMOCAP_ROOT}/${DATA_NAME}"
FOUNDATIONPOSE_DIR="${EASYMOCAP_ROOT}/FoundationPose"
RENDER_SCRIPT="${FOUNDATIONPOSE_DIR}/render_object_both_hands_joint_pose_solver_only_10fps_8gpu_hocap_meshsdf_31cam_fastgrid.py"
OUTPUT_DIR="${DATA_DIR}/outputmano"
OBJECT_POSE_DIR="${OUTPUT_DIR}/smoothed_ob_in_world"
MANO_DIR="${HOI_MANO_DIR:-${FOUNDATIONPOSE_DIR}/mycpp}"

MESH_FILE="${OUTPUT_DIR}/object_mesh_scaled.ply"
[[ -f "$MESH_FILE" ]] || MESH_FILE="${DATA_DIR}/meshes/compressed.ply"

[[ -f "$RENDER_SCRIPT" ]] || die "Missing renderer: $RENDER_SCRIPT"
[[ -f "$MESH_FILE" ]] || die "Missing mesh: $MESH_FILE"
[[ -d "$OBJECT_POSE_DIR" ]] || die "Missing optimized object poses (run the MANO refinement step first): $OBJECT_POSE_DIR"
[[ -d "$MANO_DIR" ]] || die "Missing MANO models: $MANO_DIR"

HAND_ARGS=()
for side in left right; do
  d="${OUTPUT_DIR}/contact_mano/${side}/smpl"
  if [[ -d "$d" ]] && find "$d" -type f -iname '*.json' -print -quit | grep -q .; then
    HAND_ARGS+=("--${side}_hand_json_dir" "$d")
  fi
done
(( ${#HAND_ARGS[@]} > 0 )) || die "No optimized MANO json under ${OUTPUT_DIR}/contact_mano"

if [[ -n "${CONDA_EXE:-}" && -f "${CONDA_EXE}" ]]; then
  CONDA_BASE="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  die "conda not found"
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${HOI_ENV_HOCAP_ANNOTATION:-hocap-annotation}"
echo "[Env][Render] conda=${CONDA_DEFAULT_ENV:-unknown} python=$(which python)"

IFS=',' read -r -a _G <<< "$GPU_IDS"
NG="${#_G[@]}"
LOGICAL_GPU_IDS="$(seq -s, 0 $((NG - 1)))"
echo "[Render] physical GPUs=${GPU_IDS}; logical GPUs=${LOGICAL_GPU_IDS}; workers=${NG}"
echo "[Render] mesh=${MESH_FILE}"
echo "[Render] object poses=${OBJECT_POSE_DIR}"

# Stale frames from an earlier run must not end up in the video.
rm -rf "${OUTPUT_DIR}/grid"
# final-grid mode intentionally does not create outputmano/by_cam/

cd "$FOUNDATIONPOSE_DIR"
export PYOPENGL_PLATFORM=egl
# The conda env ships its own libGLdispatch/libOpenGL (glvnd); mixed with the system
# NVIDIA libEGL they leave every GL entry point unresolved (glGetString -> None, then
# PyOpenGL raises "startswith first arg must be bytes"). Preload the system pair so
# EGL and GL dispatch through the same glvnd instance.
GLVND_PRELOAD=""
for lib in /usr/lib/x86_64-linux-gnu/libGLdispatch.so.0 /usr/lib/x86_64-linux-gnu/libOpenGL.so.0; do
  [[ -f "$lib" ]] && GLVND_PRELOAD="${GLVND_PRELOAD:+$GLVND_PRELOAD:}$lib"
done
[[ -n "$GLVND_PRELOAD" ]] || echo "[WARN] system glvnd libraries not found; EGL rendering may fail" >&2

LD_PRELOAD="${GLVND_PRELOAD}${LD_PRELOAD:+:$LD_PRELOAD}" \
CUDA_VISIBLE_DEVICES="${GPU_IDS}" python "$(basename "$RENDER_SCRIPT")" \
  --mesh_file "$MESH_FILE" \
  --test_scene_dir "$DATA_DIR" \
  --ob_in_world_dir "$OBJECT_POSE_DIR" \
  "${HAND_ARGS[@]}" \
  --mano_dir "$MANO_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --device cuda \
  --joint_solver_enable 0 \
  --multi_gpu 1 \
  --gpu_ids "$LOGICAL_GPU_IDS" \
  --num_gpu_workers "$NG" \
  --downscale "$DOWNSCALE" \
  --white_background 0 \
  --grid_video_fps "$GRID_FPS" \
  --grid_only 1 \
  --skip_grid_video \
  --grid_jpeg_quality 90 \
  --render_log_every 10

n_grid="$(find "${OUTPUT_DIR}/grid" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)"
(( n_grid > 0 )) || die "Renderer finished but produced no grid frames under ${OUTPUT_DIR}/grid"
echo "[Done] ${n_grid} grid frames: ${OUTPUT_DIR}/grid"
