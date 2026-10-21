#!/usr/bin/env bash
# Pipeline:
#   1. Extract synchronized frames from the dataset videos.
#   2. Run WiLoR to generate annots_fallback for all camera views.
#   3. Use ONLY annots_fallback in mv1p.py.
#   4. GPU-parallel per-joint all-pairs RANSAC for ALL frames first.
#   5. Apply the 6D constant-velocity Kalman filter sequentially afterwards.
#   6. Fit the requested left/right MANO model.
set -Eeuo pipefail

EASYMOCAP_ROOT="${HOI_PIPELINE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
DATA_NAME="my_sequence"
HAND_PRIOR_NAME="collector01"
HAND_SIDE="right"
START_SEC="14"
END_SEC="21"
FPS="10"
GPU_MIN_FREE_VRAM_MB="${GPU_VRAM_MB:-32768}"

# RANSAC triangulation defaults.
RANSAC_CONF_THRES="0.3"
RANSAC_REPRO_THRES="50"
RANSAC_MIN_VIEW="3"

# Joint-level 3D Kalman defaults: tuned for 10 FPS.
#
# State is [x,y,z,vx,vy,vz] with dt=1/FPS.
# At 10 FPS, dt=0.1 s is 3x larger than at 30 FPS, so using the old
# process sigma=0.30 makes the per-step prediction much looser.
#
# This preset is intended for normal hand-object interaction without
# extremely fast hand motion:
#   measurement sigma = 0.010 m  -> trust triangulated 3D enough to avoid lag
#   process sigma     = 0.12 m/s^2 -> strong anti-jitter but still responsive
#   prediction conf   = 0.03      -> predicted-only joints are trusted less
#
# For faster hand motion, increase process noise to ~0.16-0.22.
# For even stronger anti-jitter, decrease it to ~0.08-0.10.
KALMAN_MEASUREMENT_NOISE="0.010"
KALMAN_PROCESS_NOISE="0.12"
KALMAN_PREDICTION_CONF="0.03"

usage() {
  cat <<'USAGE'
Usage:
  bash run_extract_sapiens_easymocap_10fps_kalman_multigpu.sh [options]

Main options:
  --data_name NAME                Dataset under <pipeline-root> (default: my_sequence)
  --hand_prior_name NAME          Calibrated hand prior/person name (default: collector01)
  --hand_side SIDE                Hand side: left, right, or bimanual (default: right)
  --start_sec SECONDS             Video start time (default: 14)
  --end_sec SECONDS               Video end time (default: 21)
  --fps N                         Extraction FPS (default: 10)

RANSAC options:
  --ransac_conf_thres VALUE       Minimum 2D confidence (default: 0.3)
  --ransac_repro_thres PIXELS     Maximum inlier reprojection error (default: 50)
  --ransac_min_view N             Minimum views for triangulation (default: 3)

Kalman options:
  --kalman_measurement_noise V    3D measurement sigma in metres (default: 0.010)
  --kalman_process_noise V        Acceleration sigma in m/s^2 (default: 0.12)
  --kalman_prediction_conf V      Confidence assigned to predictions (default: 0.03)

Other:
  -h, --help                      Show this help

Both underscore and hyphen forms are accepted.
USAGE
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

is_number() {
  [[ "$1" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]
}


# Return physical GPU ids sorted by current free VRAM (largest first), keeping
# every GPU with at least GPU_MIN_FREE_VRAM_MB free.
refresh_eligible_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 \
    || die "nvidia-smi not found; GPU scheduling is required."

  mapfile -t ELIGIBLE_GPUS < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F, -v min_mb="$GPU_MIN_FREE_VRAM_MB" '
          {
            gsub(/[[:space:]]/, "", $1);
            gsub(/[[:space:]]/, "", $2);
            # Respect the exact assignment made by the outer scheduler.
            allowed = ENVIRON["GPU_PICK"];
            if (allowed == "") allowed = ENVIRON["CUDA_VISIBLE_DEVICES"];
            if (allowed != "" && index("," allowed ",", "," $1 ",") == 0) next;
            if (($2 + 0) >= (min_mb + 0)) print ($2 + 0) "," $1;
          }
        ' \
      | sort -t, -k1,1nr \
      | cut -d, -f2
  )

  if (( ${#ELIGIBLE_GPUS[@]} == 0 )); then
    echo "[GPU scheduler] Current GPU state:" >&2
    nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader >&2 || true
    die "No GPU currently has >= ${GPU_MIN_FREE_VRAM_MB} MiB free VRAM."
  fi
}

join_csv() {
  local IFS=,
  echo "$*"
}

print_gpu_set() {
  local label="$1"; shift
  echo "[GPU scheduler] ${label}: $(join_csv "$@")"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_name|--data-name)
      DATA_NAME="${2:?Missing value after $1}"; shift 2 ;;
    --hand_prior_name|--hand-prior-name)
      HAND_PRIOR_NAME="${2:?Missing value after $1}"; shift 2 ;;
    --hand_side|--hand-side)
      HAND_SIDE="${2:?Missing value after $1}"; shift 2 ;;
    --start_sec|--start-sec)
      START_SEC="${2:?Missing value after $1}"; shift 2 ;;
    --end_sec|--end-sec)
      END_SEC="${2:?Missing value after $1}"; shift 2 ;;
    --fps)
      FPS="${2:?Missing value after --fps}"; shift 2 ;;
    --ransac_conf_thres|--ransac-conf-thres)
      RANSAC_CONF_THRES="${2:?Missing value after $1}"; shift 2 ;;
    --ransac_repro_thres|--ransac-repro-thres)
      RANSAC_REPRO_THRES="${2:?Missing value after $1}"; shift 2 ;;
    --ransac_min_view|--ransac-min-view)
      RANSAC_MIN_VIEW="${2:?Missing value after $1}"; shift 2 ;;
    --kalman_measurement_noise|--kalman-measurement-noise)
      KALMAN_MEASUREMENT_NOISE="${2:?Missing value after $1}"; shift 2 ;;
    --kalman_process_noise|--kalman-process-noise)
      KALMAN_PROCESS_NOISE="${2:?Missing value after $1}"; shift 2 ;;
    --kalman_prediction_conf|--kalman-prediction-conf)
      KALMAN_PREDICTION_CONF="${2:?Missing value after $1}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      die "Unknown argument: $1 (use --help)" ;;
  esac
done

HAND_SIDE="${HAND_SIDE,,}"
case "$HAND_SIDE" in
  right|left|bimanual)
    ;;
  *)
    die "--hand_side must be one of: left, right, bimanual."
    ;;
esac

[[ "$FPS" =~ ^[1-9][0-9]*$ ]] \
  || die "--fps must be a positive integer."
[[ "$RANSAC_MIN_VIEW" =~ ^[0-9]+$ ]] \
  || die "--ransac_min_view must be an integer >= 2."
(( RANSAC_MIN_VIEW >= 2 )) \
  || die "--ransac_min_view must be an integer >= 2."

is_number "$START_SEC" \
  || die "--start_sec must be numeric."
is_number "$END_SEC" \
  || die "--end_sec must be numeric."
is_number "$RANSAC_CONF_THRES" \
  || die "--ransac_conf_thres must be numeric."
is_number "$RANSAC_REPRO_THRES" \
  || die "--ransac_repro_thres must be numeric."
is_number "$KALMAN_MEASUREMENT_NOISE" \
  || die "--kalman_measurement_noise must be numeric."
is_number "$KALMAN_PROCESS_NOISE" \
  || die "--kalman_process_noise must be numeric."
is_number "$KALMAN_PREDICTION_CONF" \
  || die "--kalman_prediction_conf must be numeric."

# Match the Kalman state-transition interval to the extracted video FPS.
KALMAN_DT="$(LC_NUMERIC=C awk -v fps="$FPS" 'BEGIN {printf "%.10f", 1.0 / fps}')"

DATA_DIR="${EASYMOCAP_ROOT}/${DATA_NAME}"
WILOR_DIR="${EASYMOCAP_ROOT}/WiLoR"

[[ -d "$DATA_DIR" ]] \
  || die "Dataset directory not found: $DATA_DIR"
[[ -f "${EASYMOCAP_ROOT}/rename_videos_extract_frames.py" ]] \
  || die "Missing rename_videos_extract_frames.py"
[[ -f "${EASYMOCAP_ROOT}/apps/demo/mv1p_fast.py" ]] \
  || die "Missing apps/demo/mv1p_fast.py"
[[ -f "${WILOR_DIR}/demo_fast.py" ]] \
  || die "Missing WiLoR demo: ${WILOR_DIR}/demo_fast.py"

if [[ -n "${CONDA_EXE:-}" && -f "${CONDA_EXE}" ]]; then
  CONDA_BASE="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  die "conda was not found."
fi

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

cat <<CONFIG
[Config]
  data                     = ${DATA_NAME}
  hand_prior               = ${HAND_PRIOR_NAME}
  hand_side                = ${HAND_SIDE}
  interval                 = ${START_SEC}s - ${END_SEC}s
  fps                      = ${FPS}
  fallback annotations     = annots_fallback only
  RANSAC confidence        = ${RANSAC_CONF_THRES}
  RANSAC reprojection      = ${RANSAC_REPRO_THRES} px
  RANSAC minimum views     = ${RANSAC_MIN_VIEW}
  Kalman dt                = ${KALMAN_DT} s
  Kalman measurement sigma = ${KALMAN_MEASUREMENT_NOISE}
  Kalman process sigma     = ${KALMAN_PROCESS_NOISE}
  Kalman prediction conf   = ${KALMAN_PREDICTION_CONF}
  GPU min free VRAM       = ${GPU_MIN_FREE_VRAM_MB} MiB
CONFIG

# -----------------------------------------------------------------------------
# GPU schedule + WiLoR extraction.
# WiLoR uses EVERY currently eligible physical GPU (>=32 GiB free), splitting
# camera/image groups across independent model workers.
# -----------------------------------------------------------------------------
refresh_eligible_gpus
ALL_GPU_CSV="$(join_csv "${ELIGIBLE_GPUS[@]}")"
print_gpu_set "WiLoR physical GPUs" "${ELIGIBLE_GPUS[@]}"

cd "$WILOR_DIR"
conda activate "${HOI_ENV_WILOR:-wilor}"
python demo_fast.py \
  --image_root "../${DATA_NAME}/images" \
  --annots_root "../${DATA_NAME}/annots_fallback" \
  --gpus "$ALL_GPU_CSV" \
  --min_free_vram_mb "$GPU_MIN_FREE_VRAM_MB" \
  --max_gpus 0 \
  --workers_per_gpu 2 \
  --fast \
  --no_vis \
  --log_every 50

# -----------------------------------------------------------------------------
# mv1p: triangulate ALL frames first on as many eligible GPUs as possible, then
# run sequential Kalman. For bimanual reconstruction, split all eligible GPUs
# into two DISJOINT sets and run right/left mv1p.py concurrently when >=2 GPUs
# are available. Each hand's mv1p process uses all GPUs visible in its set.
# -----------------------------------------------------------------------------
conda activate "${HOI_ENV_EASYMOCAP:-easymocap}"
cd "$EASYMOCAP_ROOT"

run_mv1p_for_hand() {
  local side="$1"
  local body_name model_name

  case "$side" in
    right)
      body_name="handr"
      model_name="manor"
      ;;
    left)
      body_name="handl"
      model_name="manol"
      ;;
    *)
      die "Internal error: unsupported hand side: $side"
      ;;
  esac

  echo "[mv1p] Running ${side} hand -> body=${body_name}, model=${model_name}"
  echo "[mv1p] ${side} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

  python3 apps/demo/mv1p_fast.py "$DATA_DIR/" \
    --name "$HAND_PRIOR_NAME" \
    --out "$DATA_DIR/output/${model_name}" \
    --fallback2d_annot annots_fallback \
    --fallback2d_hand_field auto \
    --fallback2d_num_joints 21 \
    --select2d_conf_thres "$RANSAC_CONF_THRES" \
    --select2d_repro_thres "$RANSAC_REPRO_THRES" \
    --select2d_min_view "$RANSAC_MIN_VIEW" \
    --triangulation_gpus auto \
    --triangulation_min_free_vram_mb "$GPU_MIN_FREE_VRAM_MB" \
    --use_joint_kalman \
    --joint_kalman_dt "$KALMAN_DT" \
    --joint_kalman_measurement_noise "$KALMAN_MEASUREMENT_NOISE" \
    --joint_kalman_process_noise "$KALMAN_PROCESS_NOISE" \
    --joint_kalman_prediction_conf "$KALMAN_PREDICTION_CONF" \
    --disable_select2d_print_each_joint \
    --disable_joint_kalman_print_summary \
    --disable_param_kalman \
    --undis \
    --body "$body_name" \
    --model "$model_name" \
    --gender male

  echo "[Done] ${side^}-hand MANO output: ${DATA_DIR}/output/${model_name}"
  echo "[Done] ${side^}-hand Kalman states: ${DATA_DIR}/output/${model_name}/kalman_states_xyz_vxyz.npz"
}

# Re-query AFTER WiLoR has exited: its model workers have released VRAM and the
# global GPU availability may also have changed while extraction was running.
refresh_eligible_gpus

if [[ "$HAND_SIDE" == "bimanual" ]]; then
  if (( ${#ELIGIBLE_GPUS[@]} >= 2 )); then
    RIGHT_GPUS=()
    LEFT_GPUS=()

    # Alternating a free-VRAM-sorted list gives two reasonably balanced groups
    # and uses every eligible GPU globally.
    for i in "${!ELIGIBLE_GPUS[@]}"; do
      if (( i % 2 == 0 )); then
        RIGHT_GPUS+=("${ELIGIBLE_GPUS[$i]}")
      else
        LEFT_GPUS+=("${ELIGIBLE_GPUS[$i]}")
      fi
    done

    RIGHT_GPU_CSV="$(join_csv "${RIGHT_GPUS[@]}")"
    LEFT_GPU_CSV="$(join_csv "${LEFT_GPUS[@]}")"
    print_gpu_set "right-hand physical GPUs" "${RIGHT_GPUS[@]}"
    print_gpu_set "left-hand physical GPUs" "${LEFT_GPUS[@]}"
    echo "[mv1p] Bimanual mode: right and left run concurrently."

    (
      export CUDA_VISIBLE_DEVICES="$RIGHT_GPU_CSV"
      run_mv1p_for_hand right
    ) &
    PID_RIGHT=$!

    (
      export CUDA_VISIBLE_DEVICES="$LEFT_GPU_CSV"
      run_mv1p_for_hand left
    ) &
    PID_LEFT=$!

    RC_RIGHT=0
    RC_LEFT=0
    wait "$PID_RIGHT" || RC_RIGHT=$?
    wait "$PID_LEFT" || RC_LEFT=$?

    if (( RC_RIGHT != 0 || RC_LEFT != 0 )); then
      die "Bimanual mv1p failed: right exit=${RC_RIGHT}, left exit=${RC_LEFT}"
    fi
  else
    # Exactly one eligible card: both hands must share it sequentially to avoid OOM.
    ONLY_GPU_CSV="$(join_csv "${ELIGIBLE_GPUS[@]}")"
    print_gpu_set "single available physical GPU" "${ELIGIBLE_GPUS[@]}"
    echo "[mv1p] Only one eligible GPU; running right then left sequentially."
    (
      export CUDA_VISIBLE_DEVICES="$ONLY_GPU_CSV"
      run_mv1p_for_hand right
      run_mv1p_for_hand left
    )
  fi
else
  # Single-hand request: give it every eligible GPU so Phase A triangulation can
  # fan frames out across all of them.
  ALL_GPU_CSV="$(join_csv "${ELIGIBLE_GPUS[@]}")"
  print_gpu_set "${HAND_SIDE}-hand physical GPUs" "${ELIGIBLE_GPUS[@]}"
  (
    export CUDA_VISIBLE_DEVICES="$ALL_GPU_CSV"
    run_mv1p_for_hand "$HAND_SIDE"
  )
fi
