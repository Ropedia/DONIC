#!/usr/bin/env bash
# 10 FPS - JointPoseSolver-only pipeline with exact HO-Cap MeshSDF and all 31 depth cameras.
# NO contact/collision/penetration/snap/push-pull optimization is used.
#
# Stage 1:
#   jointly optimize hand MANO + object pose using only:
#     SDF observation + hand/object pose regularization + hand/object temporal smoothness
#   then render with multiple GPUs.
# Stage 2:
#   export optimized MANO to Do-As-I-Do raw format.
set -Eeuo pipefail

EASYMOCAP_ROOT="${HOI_PIPELINE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
DATA_NAME="my_sequence"
DAI_HAND_SIDE="bimanual"
GPU_IDS="auto"
MIN_FREE_VRAM_MB="49152"
MAX_GPUS="0"
MANO_BATCH_SIZE="256"
STAGE1_MAX_ATTEMPTS="4"
STAGE2_MAX_ATTEMPTS="4"
RETRY_VRAM_STEP_MB="16384"
HOCAP_ROOT="${HOCAP_ANNOTATION_ROOT:-${EASYMOCAP_ROOT}/HO-Cap-Annotation}"

usage() {
  cat <<'USAGE'
Usage:
  bash run_joint_pose_solver_only_and_export_bimanual_10fps_8gpu.sh [options] [left|right|bimanual]

Options:
  --data_name NAME
  --hand SIDE                 left | right | bimanual
  --gpus IDS                  auto or comma-separated physical GPU ids
  --min_free_vram_mb N        default 49152
  --max_gpus N                0 = all qualifying GPUs
  --mano_batch_size N         exporter batch size, default 256
  --hocap_root PATH           local HO-Cap-Annotation repo (optional if package already installed)
  -h, --help

Example:
  bash run_joint_pose_solver_only_and_export_bimanual_10fps_8gpu.sh \
    --data_name my_sequence \
    --hand bimanual \
    --gpus auto
USAGE
}

die() { echo "[ERROR] $*" >&2; exit 1; }

hand_dir_has_json() {
  local d="$1"
  [[ -d "$d" ]] && find "$d" -type f -iname '*.json' -print -quit | grep -q .
}

select_auto_gpus() {
  local stage_label="${1:-stage}"
  local min_free_mb="${2:-$MIN_FREE_VRAM_MB}"
  local max_gpus="${3:-$MAX_GPUS}"
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  local rows
  rows="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)" || return 1
  local candidates=()
  local idx free_mb
  while IFS=',' read -r idx free_mb; do
    idx="${idx//[[:space:]]/}"
    free_mb="${free_mb//[[:space:]]/}"
    [[ "$idx" =~ ^[0-9]+$ && "$free_mb" =~ ^[0-9]+$ ]] || continue
    if (( free_mb >= min_free_mb )); then
      echo "[GPU auto][${stage_label}] GPU ${idx}: ${free_mb} MiB free -> eligible" >&2
      candidates+=("${free_mb}:${idx}")
    else
      echo "[GPU auto][${stage_label}] GPU ${idx}: ${free_mb} MiB free -> skip" >&2
    fi
  done <<< "$rows"
  (( ${#candidates[@]} > 0 )) || return 1
  mapfile -t candidates < <(printf '%s\n' "${candidates[@]}" | sort -t: -k1,1nr -k2,2n)
  local selected=() item
  for item in "${candidates[@]}"; do
    selected+=("${item#*:}")
    if (( max_gpus > 0 && ${#selected[@]} >= max_gpus )); then break; fi
  done
  (IFS=,; echo "${selected[*]}")
}

validate_manual_gpus() {
  [[ "$1" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "Invalid --gpus '$1'"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_name|--data-name) DATA_NAME="${2:?}"; shift 2 ;;
    --hand|--hand_side|--hand-side|--dai_hand_side|--dai-hand-side) DAI_HAND_SIDE="${2:?}"; shift 2 ;;
    --gpus) GPU_IDS="${2:?}"; shift 2 ;;
    --min_free_vram_mb|--min-free-vram-mb) MIN_FREE_VRAM_MB="${2:?}"; shift 2 ;;
    --max_gpus|--max-gpus) MAX_GPUS="${2:?}"; shift 2 ;;
    --mano_batch_size|--mano-batch-size) MANO_BATCH_SIZE="${2:?}"; shift 2 ;;
    --hocap_root|--hocap-root) HOCAP_ROOT="${2:?}"; shift 2 ;;
    left|right|bimanual) DAI_HAND_SIDE="$1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

case "$DAI_HAND_SIDE" in left|right|bimanual) ;; *) die "Invalid hand side: $DAI_HAND_SIDE" ;; esac
GPU_REQUEST="$GPU_IDS"
if [[ "${GPU_REQUEST,,}" == "auto" ]]; then
  GPU_SELECTION_MODE="auto"
else
  validate_manual_gpus "$GPU_REQUEST"
  GPU_SELECTION_MODE="manual"
fi

DATA_DIR="${EASYMOCAP_ROOT}/${DATA_NAME}"
FOUNDATIONPOSE_DIR="${EASYMOCAP_ROOT}/FoundationPose"
SOLVER_SCRIPT="${FOUNDATIONPOSE_DIR}/render_object_both_hands_joint_pose_solver_only_10fps_8gpu_hocap_meshsdf_31cam.py"

if [[ -f "${FOUNDATIONPOSE_DIR}/export_outputmano_to_do_as_i_do_raw_bimanual_8gpu.py" ]]; then
  EXPORT_SCRIPT="${FOUNDATIONPOSE_DIR}/export_outputmano_to_do_as_i_do_raw_bimanual_8gpu.py"
elif [[ -f "${FOUNDATIONPOSE_DIR}/export_outputmano_to_do_as_i_do_raw_bimanual.py" ]]; then
  EXPORT_SCRIPT="${FOUNDATIONPOSE_DIR}/export_outputmano_to_do_as_i_do_raw_bimanual.py"
else
  EXPORT_SCRIPT="${FOUNDATIONPOSE_DIR}/export_outputmano_to_do_as_i_do_raw.py"
fi

MANO_DIR="${HOI_MANO_DIR:-${FOUNDATIONPOSE_DIR}/mycpp}"
MESH_FILE="${DATA_DIR}/meshes/compressed.ply"
OBJECT_POSE_DIR="${DATA_DIR}/debug_depth/geometry_global_opt/ob_in_world"
LEFT_HAND_DIR="${DATA_DIR}/output/manol/smpl"
RIGHT_HAND_DIR="${DATA_DIR}/output/manor/smpl"
DEPTH_ROOT="${DATA_DIR}/depth"
OUTPUT_DIR="${DATA_DIR}/outputmano"
DAI_RAW_DIR="${DATA_DIR}/do_as_i_do_raw"

if [[ -f "${DATA_DIR}/compressed.ply" ]]; then
  EXPORT_MESH_FILE="${DATA_DIR}/compressed.ply"
else
  EXPORT_MESH_FILE="$MESH_FILE"
fi

[[ -f "$SOLVER_SCRIPT" ]] || die "Missing solver: $SOLVER_SCRIPT"
[[ -f "$MESH_FILE" ]] || die "Missing mesh: $MESH_FILE"
[[ -d "$OBJECT_POSE_DIR" ]] || die "Missing object poses: $OBJECT_POSE_DIR"
[[ -d "$MANO_DIR" ]] || die "Missing MANO models: $MANO_DIR"
[[ -f "$EXPORT_SCRIPT" ]] || die "Missing exporter: $EXPORT_SCRIPT"

HAND_ARGS=()
AVAILABLE_HANDS=()
if hand_dir_has_json "$LEFT_HAND_DIR"; then
  HAND_ARGS+=(--left_hand_json_dir "$LEFT_HAND_DIR")
  AVAILABLE_HANDS+=(left)
fi
if hand_dir_has_json "$RIGHT_HAND_DIR"; then
  HAND_ARGS+=(--right_hand_json_dir "$RIGHT_HAND_DIR")
  AVAILABLE_HANDS+=(right)
fi
(( ${#HAND_ARGS[@]} > 0 )) || die "No MANO json found."

case "$DAI_HAND_SIDE" in
  left) hand_dir_has_json "$LEFT_HAND_DIR" || die "Left hand missing" ;;
  right) hand_dir_has_json "$RIGHT_HAND_DIR" || die "Right hand missing" ;;
  bimanual)
    hand_dir_has_json "$LEFT_HAND_DIR" || die "Left hand missing"
    hand_dir_has_json "$RIGHT_HAND_DIR" || die "Right hand missing"
    ;;
esac

if [[ -n "${CONDA_EXE:-}" && -f "${CONDA_EXE}" ]]; then
  CONDA_BASE="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  die "conda not found"
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Stage 1: exact HO-Cap MeshSDF JointPoseSolver
conda activate "${HOI_ENV_HOCAP_ANNOTATION:-hocap-annotation}"
echo "[Env][Stage1] conda=${CONDA_DEFAULT_ENV:-unknown}"
echo "[Env][Stage1] python=$(which python)"

# Exact HO-Cap MeshSDF is mandatory.  Add a local source checkout to PYTHONPATH if present,
# but DO NOT silently fall back to a different distance loss.
if [[ -d "$HOCAP_ROOT" ]]; then
  export PYTHONPATH="$HOCAP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi
if ! python - <<'PY_CHECK'
from hocap_annotation.loss import MeshSDFLoss
import meshsdf_loss_cuda
print('[HO-Cap] exact MeshSDFLoss + meshsdf_loss_cuda import: OK')
PY_CHECK
then
  cat >&2 <<EOF
[ERROR] Exact HO-Cap MeshSDFLoss is required and no proxy fallback is allowed.
Install/build it in the active hocap-annotation environment:
  cd "$HOCAP_ROOT"
  python -m pip install -e . --no-cache-dir
  cd hocap_annotation/loss/meshsdf_loss
  python -m pip install . --no-cache-dir
Then verify:
  python -c 'from hocap_annotation.loss import MeshSDFLoss; import meshsdf_loss_cuda; print("OK")'
EOF
  exit 1
fi

# Depth cameras actually present. A capture may miss a camera (30 of 31, see
# sync_calibration.py): use every camera that has a depth folder AND an entry in the
# dataset's (synced) intri.yml, and require all of those instead of a fixed 1..31.
SDF_CAMS=()
for _d in "$DEPTH_ROOT"/*/; do
  _c="$(basename "$_d")"
  [[ "$_c" =~ ^[0-9]+$ ]] || continue
  if grep -q "^K_${_c}:" "${DATA_DIR}/intri.yml" 2>/dev/null; then SDF_CAMS+=("$_c"); fi
done
mapfile -t SDF_CAMS < <(printf '%s\n' "${SDF_CAMS[@]}" | sort -n)
(( ${#SDF_CAMS[@]} >= 8 )) || die "Only ${#SDF_CAMS[@]} depth camera(s) with calibration under $DEPTH_ROOT (need >= 8)"
SDF_CAM_COUNT="${#SDF_CAMS[@]}"

cd "$FOUNDATIONPOSE_DIR"
mkdir -p "$OUTPUT_DIR" "$DAI_RAW_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_stage1_once() {
  local physical_gpu_ids="$1"
  IFS=',' read -r -a _G <<< "$physical_gpu_ids"
  local ng="${#_G[@]}"
  (( ng > 0 )) || return 2
  local logical_gpu_ids
  logical_gpu_ids="$(seq -s, 0 $((ng - 1)))"

  echo "[Stage1] JointPoseSolver ONLY"
  echo "[Stage1] physical GPUs=${physical_gpu_ids}; logical GPUs=${logical_gpu_ids}"
  echo "[Stage1] losses = exact HO-Cap MeshSDF + reg_hand + reg_object + smooth_hand + smooth_object"
  echo "[Stage1] depth observation = ALL ${SDF_CAM_COUNT} available cameras (${SDF_CAMS[*]}), fused in world coordinates"
  echo "[Stage1] NO contact/collision/penetration optimization"

  CUDA_VISIBLE_DEVICES="${physical_gpu_ids}" python "$(basename "$SOLVER_SCRIPT")" \
    --mesh_file "$MESH_FILE" \
    --test_scene_dir "$DATA_DIR" \
    --ob_in_world_dir "$OBJECT_POSE_DIR" \
    "${HAND_ARGS[@]}" \
    --mano_dir "$MANO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    --sequence_fps 10 \
    --temporal_reference_fps 30 \
    --joint_solver_enable 1 \
    --joint_lr 0.001 \
    --joint_total_steps 160 \
    --joint_sdf_steps 60 \
    --w_sdf 10.0 \
    --w_reg_m 1.0 \
    --w_reg_o 1.0 \
    --w_smooth_m 1.0 \
    --w_smooth_o 1.0 \
    --object_scale_enable 1 \
    --object_scale_lr 0.0005 \
    --object_scale_min 0.5 \
    --object_scale_max 2.0 \
    --object_scale_reg_scale 0.001 \
    --object_scale_epochs 2 \
    --object_scale_frame_batch 8 \
    --object_scale_select_thresh 0.015 \
    --object_scale_min_points 50 \
    --depth_root "$DEPTH_ROOT" \
    --sdf_cams "${SDF_CAMS[@]}" \
    --sdf_expected_cam_count "$SDF_CAM_COUNT" \
    --sdf_require_all_cams 1 \
    --sdf_cam_stride 1 \
    --sdf_pixel_stride 8 \
    --sdf_dist_thresh 0.01 \
    --sdf_voxel_size 0.003 \
    --sdf_max_points_per_frame 2000 \
    --sdf_frame_batch 4 \
    --sdf_io_workers 8 \
    --sdf_reuse_cache 1 \
    --multi_gpu 1 \
    --gpu_ids "$logical_gpu_ids" \
    --num_gpu_workers "$ng" \
    --grid_video_fps 10 \
    --skip_render
}

STAGE1_OK=0
PREV_COUNT=0
CAP="$MAX_GPUS"
for ((attempt=1; attempt<=STAGE1_MAX_ATTEMPTS; attempt++)); do
  threshold=$((MIN_FREE_VRAM_MB + (attempt - 1) * RETRY_VRAM_STEP_MB))
  if [[ "$GPU_SELECTION_MODE" == "auto" ]]; then
    if (( attempt > 1 && PREV_COUNT > 1 )); then CAP=$((PREV_COUNT - 1)); elif (( attempt > 1 )); then CAP=1; fi
    set +e
    S1_GPUS="$(select_auto_gpus "joint-solver-attempt-${attempt}" "$threshold" "$CAP")"
    rc=$?
    set -e
    (( rc == 0 )) && [[ -n "$S1_GPUS" ]] || continue
  else
    S1_GPUS="$GPU_REQUEST"
  fi
  IFS=',' read -r -a _TMP <<< "$S1_GPUS"
  PREV_COUNT="${#_TMP[@]}"
  set +e
  run_stage1_once "$S1_GPUS"
  rc=$?
  set -e
  if (( rc == 0 )); then STAGE1_OK=1; break; fi
  echo "[WARN] Stage1 attempt ${attempt} failed: ${rc}" >&2
  [[ "$GPU_SELECTION_MODE" == "auto" ]] || break
done
(( STAGE1_OK == 1 )) || die "JointPoseSolver stage failed."

echo "[Stage1] JointPoseSolver + all-frame object-scale fine-tune finished successfully."

# Prefer the globally scaled object mesh generated by Stage1 for downstream export.
SCALED_OBJECT_MESH="${OUTPUT_DIR}/object_mesh_scaled.ply"
if [[ -f "$SCALED_OBJECT_MESH" ]]; then
  EXPORT_MESH_FILE="$SCALED_OBJECT_MESH"
  echo "[ObjectScale] Stage2 will use scaled mesh: $EXPORT_MESH_FILE"
  if [[ -f "${OUTPUT_DIR}/object_scale.txt" ]]; then
    echo "[ObjectScale] final scale=$(cat "${OUTPUT_DIR}/object_scale.txt")"
  fi
else
  echo "[WARN] scaled object mesh not found; Stage2 falls back to: $EXPORT_MESH_FILE" >&2
fi

# Stage 2: Do-As-I-Do export/visualization in EasyMocap environment.
echo "[Env] switching hocap-annotation -> easymocap for Stage2"
conda activate "${HOI_ENV_EASYMOCAP:-easymocap}"
cd "$FOUNDATIONPOSE_DIR"
echo "[Env][Stage2] conda=${CONDA_DEFAULT_ENV:-unknown}"
echo "[Env][Stage2] python=$(which python)"

# Optimized MANO outputs from the clean solver.
EXPORT_HAND_ARGS=()
case "$DAI_HAND_SIDE" in
  left) EXPORT_HAND_ARGS+=(--left_hand_json_dir "${OUTPUT_DIR}/contact_mano/left/smpl") ;;
  right) EXPORT_HAND_ARGS+=(--right_hand_json_dir "${OUTPUT_DIR}/contact_mano/right/smpl") ;;
  bimanual)
    EXPORT_HAND_ARGS+=(
      --right_hand_json_dir "${OUTPUT_DIR}/contact_mano/right/smpl"
      --left_hand_json_dir "${OUTPUT_DIR}/contact_mano/left/smpl"
    )
    ;;
esac

run_stage2_once() {
  local physical_gpu_ids="$1"
  local batch_size="$2"
  IFS=',' read -r -a _G <<< "$physical_gpu_ids"
  local ng="${#_G[@]}"
  local logical_gpu_ids
  logical_gpu_ids="$(seq -s, 0 $((ng - 1)))"
  CUDA_VISIBLE_DEVICES="${physical_gpu_ids}" python "$(basename "$EXPORT_SCRIPT")" \
    --outputmano_dir "$OUTPUT_DIR" \
    --test_scene_dir "$DATA_DIR" \
    --mesh_file "$EXPORT_MESH_FILE" \
    --mano_dir "$MANO_DIR" \
    --dai_raw_dir "$DAI_RAW_DIR" \
    --dai_task "$DATA_NAME" \
    --dai_object_name object \
    --dai_hand_side "$DAI_HAND_SIDE" \
    "${EXPORT_HAND_ARGS[@]}" \
    --dai_ref_cam 1 \
    --dai_gravity_mode auto_three_ring \
    --gpus "$logical_gpu_ids" \
    --mano_batch_size "$batch_size" \
    --save_vis \
    --vis_stride 10 \
    --vis_max_frames 100 \
    --overwrite
}

STAGE2_OK=0
PREV_COUNT=0
CAP="$MAX_GPUS"
for ((attempt=1; attempt<=STAGE2_MAX_ATTEMPTS; attempt++)); do
  threshold=$((MIN_FREE_VRAM_MB + (attempt - 1) * RETRY_VRAM_STEP_MB))
  if [[ "$GPU_SELECTION_MODE" == "auto" ]]; then
    if (( attempt > 1 && PREV_COUNT > 1 )); then CAP=$((PREV_COUNT - 1)); elif (( attempt > 1 )); then CAP=1; fi
    set +e
    S2_GPUS="$(select_auto_gpus "export-attempt-${attempt}" "$threshold" "$CAP")"
    rc=$?
    set -e
    (( rc == 0 )) && [[ -n "$S2_GPUS" ]] || continue
  else
    S2_GPUS="$GPU_REQUEST"
  fi
  IFS=',' read -r -a _TMP <<< "$S2_GPUS"
  PREV_COUNT="${#_TMP[@]}"
  case "$attempt" in 1) batch="$MANO_BATCH_SIZE" ;; 2) batch=128 ;; 3) batch=64 ;; *) batch=32 ;; esac
  (( batch <= MANO_BATCH_SIZE )) || batch="$MANO_BATCH_SIZE"
  set +e
  run_stage2_once "$S2_GPUS" "$batch"
  rc=$?
  set -e
  if (( rc == 0 )); then STAGE2_OK=1; break; fi
  [[ "$GPU_SELECTION_MODE" == "auto" ]] || break
done
(( STAGE2_OK == 1 )) || die "Export stage failed."

echo "[Done] MANO (legacy-compatible path): ${OUTPUT_DIR}/contact_mano"
echo "[Done] Object poses (legacy-compatible path): ${OUTPUT_DIR}/smoothed_ob_in_world"
echo "[Done] Loss log: ${OUTPUT_DIR}/joint_pose_solver_loss.csv"
echo "[Done] Object scale: ${OUTPUT_DIR}/object_scale.txt"
echo "[Done] Scaled object mesh: ${OUTPUT_DIR}/object_mesh_scaled.ply"
echo "[Done] Object scale fine-tune log: ${OUTPUT_DIR}/object_scale_finetune.csv"
echo "[Done] Do-As-I-Do raw: ${DAI_RAW_DIR}"
