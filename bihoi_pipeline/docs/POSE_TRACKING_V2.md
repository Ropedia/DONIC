# 6D pose tracking in v2

Choose **Pose tracking method** above Advanced parameters: **FoundationPose tracking**
(default) or **WAPR tracking**. The selection applies to both **3 · Object tracking selected**
and the full pipeline. The WAPR interpreter comes from `HOI_ENV_WAPR` (default
`.environments/wapr`); FoundationPose initialization and preview rendering use
`HOI_ENV_FOUNDATIONPOSE` (default `.environments/foundationpose`). The modified WAPR
source is bundled at `web_demos/third_party/WAPR`. Its default capture renderer is
`nvdiffrast`, built in the WAPR environment with CUDA 12.8. See setup below.
`--renderer ogl` remains available for environments where OpenGL interop works reliably.

Both methods produce original-mesh-to-world poses in metres for the existing geometry
optimizer, all-camera preview, and Stage 5 hand/object optimization. Stage 3 preserves the
previous `debug_depth/` and preview directory under `tracking_history/<time>-<method>-.../`
before starting. `pose_tracking.json` records the selected method and completion state;
switching methods bypasses “Skip stages already completed” for tracking. A new tracking
result also invalidates the previous Stage 5 export for skip decisions.

WAPR tracking:

```bash
# Stage 1 images and Stage 2 metric depth/mesh must already exist.
CUDA_VISIBLE_DEVICES=0 .environments/wapr/bin/python \
  webui/wapr_tracking.py \
  --scene_dir <dataset> --output_dir <fresh-output-directory> \
  --width 640 --height 360 --fps 10 --depth_scale 1000
```

The model and mesh stay resident throughout the sequence. First-frame initialization
seeds **240 FoundationPose rotation hypotheses** with the
**fused multi-view masked depth point-cloud centroid**, then run the six-candidate
tracking selection described below.
For the same frame, every valid calibrated camera contributes its
finite, positive masked depth pixels, backprojected with its own intrinsics and transformed
into world coordinates by the inverse world-to-camera extrinsics. All these world-space
points are concatenated and statistically filtered; XYZ is the retained points' arithmetic
mean, without subsampling or averaging per-camera centroids equally. Fusion uses all valid cameras, independent of the 10-view
inference limit and six displayed cameras. This surface centroid is transformed into the
selected registration camera. Each original-mesh translation is
`center_camera - R @ mesh_center`; refinement remains free to update both rotation and translation.
FoundationPose's XYZ seed does not use the triangulated center. Its source is recorded as
`multiview_masked_depth_pointcloud_centroid` in the report and CSV's `initialization_xyz`
column, with contributing cameras, point count and world-space centroid in the CSV.

Statistical outlier removal (SOR) is enabled by default on the fused world cloud:
for each point, compute the mean distance to its 20 nearest other points, and reject
distances exceeding the cloud-wide mean plus 2 standard deviations. Searches are exact
and chunked to bound memory; retained coordinates are not smoothed or voxelized.
This same filter feeds first-frame registration and the red live preview.
Clouds with 20 or fewer valid points bypass SOR because they cannot support that
neighborhood. Tune `--cloud_outlier_neighbors` and `--cloud_outlier_std_ratio`, or use
`--no_cloud_outlier_removal` for comparison. CSV initialization fields record raw,
retained and removed point counts and the distance threshold; the JSON report records
filter settings. The original per-camera RGB-D inference inputs remain unchanged.

Subsequent frames compare exactly **WAPR(C), SAPR(C), SAPR(WAPR(C))**.
C keeps the previous rotation and moves the mesh center to the mask-triangulated center
(or retains the previous pose if triangulation is unavailable). WAPR(C) is computed once;
SAPR processes C and WAPR(C) together, and WBPS ranks only the three resulting poses.
Up to 10 views are selected by mask area. Their world-space estimates are tested for a
pairwise-consistent group (at least 3 views, 30 degrees, 5 cm); translation averages mesh centers
and rotation comes from the largest-mask view in that group. If consensus fails, run
**WAPR registration in the largest valid-mask camera with 96 hypotheses**, seeding XYZ
from the current frame's **multi-view mask triangulation**. Transform that world-space
center into the selected camera, then use WAPR's native 12-direction (icosahedron) ×
8-in-plane rotation grid. All 96 poses undergo masked WAPR (3 updates), SAPR (2 updates),
and compete in one WBPS group. The winning refined rotation and translation become
this frame's result and the next frame's prior; XYZ is a seed, not a constraint on the result.
The resident estimator and prepared mesh are reused. The WAPR candidate-count registry
includes 96 for this path; ordinary three-candidate tracking and the default demo grid
keep their existing sizes.

The CSV and live preview mark successful recovery as `reinit_wapr96`. CSV fields record
`consensus_failed`, `reinitialization_method`, `reinitialization_hypotheses`,
`reinitialization_xyz`, and `reinitialization_center_world`; `consensus_views` stays empty.
If triangulation is unavailable or invalid in the best-mask camera, retain that camera's
already-computed tracking result as `tracking_best_mask` and log
`reinitialization_skip_reason`. No single-view depth seed is substituted. Frames without
usable masked RGB-D hold the previous pose as `held_no_rgbd`; when observations return,
tracking resumes from that held pose and the same consensus recovery rule applies.
The JSON report records `reinitialization.enabled=true`, its 96-hypothesis recipe and the
total number of consensus-failure frames. Registration errors fail the attempt rather
than silently substituting a tracking result.

First-frame initialization uses **240 FoundationPose hypotheses
→ 5 FoundationPose refinement iterations → FoundationPose scorer selects one winner
→ one complete six-candidate tracking update → WBPS selects the final winner**, in the
largest valid-mask camera.
The 240 rotations use FoundationPose's
view/in-plane construction: deterministic farthest-point selection of 40 directions from
its 42-vertex icosphere, each with six 60-degree in-plane rotations, without symmetry pruning.
FoundationPose preserves the winner's refined rotation AND translation; the fused depth
centroid is only a seed. The FoundationPose winner becomes P in original-mesh coordinates.
C keeps P's rotation and updates its mesh center with the same current-frame mask
triangulation used in ordinary tracking (or C=P if unavailable). The candidates are
**P, C, WAPR(C), SAPR(C), SAPR(WAPR(C)), SAPR(WAPR(P))**. This is the complete tracking
recipe: unmasked WAPR (3 updates), SAPR (2 updates), and WBPS selection over all six poses.
Any candidate, including P, can win. The selected pose is converted to world coordinates
and becomes the tracking state. There is no separate forced final refinement after selection.
The tracking CSV records the selected hypothesis and FoundationPose metadata, including
on the first frame. The JSON report distinguishes the three ordinary tracking candidates
from the six candidates used only in initialization.

FoundationPose loads lazily in a persistent worker in the existing `foundationpose` environment
on the same selected GPU. Override that environment with `--foundationpose_python /path/to/python`
when needed. Errors stop the job. `foundationpose_reinit.log` contains the first-frame
refiner/scorer details, and the tracking CSV records initialization method, actual hypothesis count,
iteration count and FoundationPose score. The live preview continues to publish the final pose.

Camera inference runs in four persistent worker processes on the scheduler-selected GPU
(`--view_workers 4`; use `1` for the sequential reference). Each worker has its own resident
model and renderer and executes the three-candidate per-camera call. Model loading costs more at
startup and uses extra VRAM, then is reused across the sequence. Camera results are collected
in their original ranked order. Shared refinement prefixes are computed once: one WAPR
row, two SAPR rows and three scoring rows per tracking camera.
The 10-view selection, 160-pixel model crops, model precision, three WAPR updates,
two SAPR updates, and consensus thresholds are retained. End-to-end speed depends on
rendering and I/O; each consensus failure with valid triangulation adds one 96-hypothesis
registration in the best-mask camera, rather than registering independently in every view.
CPU image decoding runs in four workers and prefetches one frame while the GPU processes the
current one. No inference frames or views are dropped.

The **Live WAPR tracking preview** below the run logs refreshes automatically every second.
A separate renderer fuses the same frame's masked depths from all valid calibrated cameras
into one world-space cloud, removes statistical outliers, then projects that shared red cloud
alongside the green tracked mesh into up to six selected camera views. Its legend shows the contributing camera count
and retained/raw point counts. `live_tracking/latest.json` includes the filter statistics.
Fusion uses the same masks, units, intrinsics and extrinsics as initialization. Red points are thinned
only for display to keep the mesh visible; initialization uses every retained point.
The cloud is drawn as sparse small markers with a 35% red tint. Marker pixels use a
neutral brightness background so the green mesh cannot change their hue to green/brown.
The tracked mesh has a 65%
green fill and a bright green outline with a dark border so its pose remains prominent.
It consumes only the latest request, so slow preview
rendering does not build a queue or block inference. Preview files live under
`debug_depth/live_tracking/`. Final all-camera videos still include every tracked frame.
Use `--no_live_preview` to disable the renderer for a command-line benchmark.

WAPR reads every Stage 1 frame without further sampling. `--fps` records that sequence's FPS;
the WebUI uses the same FPS for the final preview. Integer depth is divided by `--depth_scale`
(default 1000 for millimetres); float depth is in metres. `intri.yml` must correspond to the
prepared RGB resolution, as it does for geometry post-optimization. WAPR outputs go directly
to geometry post-optimization; FoundationPose's optional internal filters are not applied.
`--topk 1` is available for single-camera debugging, and `--max_frames` can limit a manual
smoke test; the WebUI processes all frames with the default multi-view setting.

## Install and launch

Use the main README to install the upstream pipeline and obtain FoundationPose weights.
For an existing v1 installation, add the isolated WAPR environment:

```bash
bash scripts/install.sh --component wapr --skip-native
conda install -p .environments/wapr -c nvidia cuda-toolkit=12.8
bash scripts/build_native.sh wapr
bash scripts/check_env.sh --component wapr --skip-models
bash scripts/run_webui.sh
```

Open `http://127.0.0.1:7860`, select **WAPR tracking**, then run **3 · Object tracking selected**.
Select the GPUs in `.env`. Existing environments can be reused with `HOI_ENV_WAPR`
and `HOI_ENV_FOUNDATIONPOSE`; do not run the installer into an unmanaged environment.
The installer does not fetch weights. WAPR checks/downloads its four upstream weights
on first estimator construction, using `WAPR_CACHE_DIR` (default `.models/wapr`).
Review the bundled WAPR weight license before running. Do not replace the bundled
WAPR source with an unmodified pip release: v2 depends on its center-update and
96-candidate support.

The `--renderer ogl` alternative additionally needs OpenGL/EGL development libraries
and a working NVIDIA EGL device; its C++/CUDA renderer builds on first use. TensorRT
is optional and not the validated default. Use the PyTorch backend unless separately
prepared engines support the required 3-, 6- and 96-pose scoring groups.

For output inspection, read `debug_depth/wapr_tracking_log.csv`,
`debug_depth/wapr_tracking.json`, and `debug_depth/live_tracking/latest.json`.
Keep `tracking_history/` until previous results are no longer needed. Both tracking
methods feed the existing geometry post-optimization and hand/object refinement stages.
