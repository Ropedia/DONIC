#!/usr/bin/env bash
# Resize images/masks -> regenerate cam_K.txt -> FoundationPose registration.
# Supports optional depth and optional --no-force.
set -Eeuo pipefail

EASYMOCAP_ROOT="${HOI_PIPELINE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
DATA_NAME="my_sequence"
W="640"
H="481"

# FoundationPose options
USE_DEPTH="0"
NO_FORCE="0"
DEPTH_DIR=""
DEBUG_DIR=""

usage() {
  cat <<'USAGE'
Usage:
  bash run_resize_camk_foundationpose_args.sh [options]

Options:
  --data_name NAME      Dataset directory under <pipeline-root>
                        (default: my_sequence)
  --w WIDTH             Target image/mask width
                        (default: 640)
  --h HEIGHT            Target image/mask height
                        (default: 481)

  --use_depth 0|1       Whether to use depth in FoundationPose
                        (default: 0)
  --depth_dir DIR       Depth directory
                        (default: <data_dir>/depth)

  --no_force 0|1        Whether to pass --no-force to FoundationPose
                        (default: 1)

  --debug_dir DIR       FoundationPose debug/output directory
                        default:
                          use_depth=1 -> <data_dir>/debug_depth
                          use_depth=0 -> <data_dir>/debug_register_best_mask

  -h, --help            Show this help

Both underscore and hyphen forms are accepted.

Examples:

  # 1) RGB only + no-force
  bash run_resize_camk_foundationpose_args.sh \
    --data_name my_sequence \
    --use_depth 0 \
    --no_force 1

  # 2) RGB + depth + no-force
  bash run_resize_camk_foundationpose_args.sh \
    --data_name my_sequence \
    --use_depth 1 \
    --depth_dir <pipeline-root>/my_sequence/depth \
    --no_force 1

  # 3) RGB + depth, but do NOT use --no-force
  bash run_resize_camk_foundationpose_args.sh \
    --data_name my_sequence \
    --use_depth 1 \
    --depth_dir <pipeline-root>/my_sequence/depth \
    --no_force 0
USAGE
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_name|--data-name)
      DATA_NAME="${2:?Missing value after $1}"
      shift 2
      ;;
    --w)
      W="${2:?Missing value after --w}"
      shift 2
      ;;
    --h)
      H="${2:?Missing value after --h}"
      shift 2
      ;;
    --use_depth|--use-depth)
      USE_DEPTH="${2:?Missing value after $1}"
      shift 2
      ;;
    --depth_dir|--depth-dir)
      DEPTH_DIR="${2:?Missing value after $1}"
      shift 2
      ;;
    --no_force|--no-force)
      NO_FORCE="${2:?Missing value after $1}"
      shift 2
      ;;
    --debug_dir|--debug-dir)
      DEBUG_DIR="${2:?Missing value after $1}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1 (use --help)"
      ;;
  esac
done

[[ "$W" =~ ^[1-9][0-9]*$ ]] || die "--w must be a positive integer."
[[ "$H" =~ ^[1-9][0-9]*$ ]] || die "--h must be a positive integer."
[[ "$USE_DEPTH" == "0" || "$USE_DEPTH" == "1" ]] || die "--use_depth must be 0 or 1."
[[ "$NO_FORCE" == "0" || "$NO_FORCE" == "1" ]] || die "--no_force must be 0 or 1."

DATA_DIR="${EASYMOCAP_ROOT}/${DATA_NAME}"
IMAGE_ROOT="${DATA_DIR}/images"
MASK_ROOT="${DATA_DIR}/masks"
INTRI_FILE="${DATA_DIR}/intri.yml"
CAM_K_FILE="${DATA_DIR}/cam_K.txt"
MESH_FILE="${DATA_DIR}/meshes/compressed.ply"

FOUNDATIONPOSE_DIR="${EASYMOCAP_ROOT}/FoundationPose"
FOUNDATIONPOSE_SCRIPT="${FOUNDATIONPOSE_DIR}/run_demo_d_working_depth_optional_no_force_bbox_center_fixed_fastio.py"

# Resolve defaults after DATA_DIR is known.
if [[ -z "$DEPTH_DIR" ]]; then
  DEPTH_DIR="${DATA_DIR}/depth"
fi

if [[ -z "$DEBUG_DIR" ]]; then
  if [[ "$USE_DEPTH" == "1" ]]; then
    DEBUG_DIR="${DATA_DIR}/debug_depth"
  else
    DEBUG_DIR="${DATA_DIR}/debug_register_best_mask"
  fi
fi

# Input checks.
[[ -d "$IMAGE_ROOT" ]] || die "Images directory not found: $IMAGE_ROOT"
[[ -d "$MASK_ROOT" ]] || die "Masks directory not found: $MASK_ROOT"
[[ -f "$INTRI_FILE" ]] || die "Missing intrinsics: $INTRI_FILE"
[[ -f "$MESH_FILE" ]] || die "Missing mesh: $MESH_FILE"
[[ -f "${EASYMOCAP_ROOT}/resize.py" ]] || die "Missing resize.py"
[[ -f "${EASYMOCAP_ROOT}/generate_cam_k.py" ]] || die "Missing generate_cam_k.py"
[[ -f "$FOUNDATIONPOSE_SCRIPT" ]] || die "Missing FoundationPose registration script: $FOUNDATIONPOSE_SCRIPT"

if [[ "$USE_DEPTH" == "1" ]]; then
  [[ -d "$DEPTH_DIR" ]] || die "Depth directory not found: $DEPTH_DIR"
fi

# Locate conda.
if [[ -n "${CONDA_EXE:-}" && -f "${CONDA_EXE}" ]]; then
  CONDA_BASE="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  die "conda was not found."
fi

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "[Config] data=${DATA_NAME}"
echo "[Config] resolution=${W}x${H}"
echo "[Config] mesh=${MESH_FILE}"
echo "[Config] use_depth=${USE_DEPTH}"
echo "[Config] no_force=${NO_FORCE}"
echo "[Config] debug_dir=${DEBUG_DIR}"

if [[ "$USE_DEPTH" == "1" ]]; then
  echo "[Config] depth_dir=${DEPTH_DIR}"
fi

# ------------------------------------------------------------
# 1. Resize images and masks
# ------------------------------------------------------------
conda activate "${HOI_ENV_EASYMOCAP:-easymocap}"
cd "$EASYMOCAP_ROOT"


# ------------------------------------------------------------
# 2. Regenerate cam_K.txt
# ------------------------------------------------------------
python generate_cam_k.py \
  --intri "$INTRI_FILE" \
  --out "$CAM_K_FILE" \
  --w "$W" \
  --h "$H"

# ------------------------------------------------------------
# 3. FoundationPose
# ------------------------------------------------------------
conda activate "${HOI_ENV_FOUNDATIONPOSE:-foundationpose}"
cd "$FOUNDATIONPOSE_DIR"

FOUNDATIONPOSE_ARGS=(
  --mesh_file "$MESH_FILE"
  --test_scene_dir "$DATA_DIR"
  --debug_dir "$DEBUG_DIR"
  --use_depth "$USE_DEPTH"
  --no_window
  --fast
  --simp_saving
  # Keep Python defaults unchanged:
  #   --fast_consensus_topk 10
  #   --fast_consensus_min_cluster 4
  # Tracking-pass visualization is disabled inside the FAST-IO Python script.
)

# Add depth directory only when depth is enabled.
if [[ "$USE_DEPTH" == "1" ]]; then
  FOUNDATIONPOSE_ARGS+=(
    --depth_dir "$DEPTH_DIR"
  )
fi

# Add --no-force only when requested.
if [[ "$NO_FORCE" == "1" ]]; then
  FOUNDATIONPOSE_ARGS+=(
    --no-force
  )
fi

echo
echo "[FoundationPose command]"
printf 'python %q ' "$(basename "$FOUNDATIONPOSE_SCRIPT")"
printf '%q ' "${FOUNDATIONPOSE_ARGS[@]}"
echo
echo

python "$(basename "$FOUNDATIONPOSE_SCRIPT")" "${FOUNDATIONPOSE_ARGS[@]}"

echo "[Done] FoundationPose results: $DEBUG_DIR"
