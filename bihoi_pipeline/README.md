# BiHOI Pipeline

The multi-view hand–object interaction processing pipeline used to build the **BiHOI** dataset in [DONIC](../README.md). From synchronized multi-view videos it produces object masks, metric depth, an object mesh, 6D object poses, bimanual MANO hand reconstructions and a standardized export, driven from a Gradio WebUI.

> **Before you start**
>
> - **Not every stage runs with public assets.** Preprocessing, segmentation, reconstruction and object tracking use public (some gated) model weights. The **Hands** and **Final** stages, and hand calibration, additionally need a separately licensed WiLoR variant and FMHR checkpoints/templates that are **not distributed** here. See [THIRD_PARTY.md](THIRD_PARTY.md).
> - **Non-commercial use only.** Several components (EasyMocap, FoundationPose, Depth Anything 3 giant weights) carry non-commercial terms. See [License](#license).
> - Everything runs on **Linux with NVIDIA GPUs**. New-machine installation has not been validated on every GPU system; see [DEPENDENCIES.md](DEPENDENCIES.md) for exact versions.

## Processing configuration

| Setting | Value |
| --- | --- |
| Camera streams | 31 synchronized cameras, in the order fixed by `pipeline/rename_videos_31cams.py` ([camera names](docs/DATA_FORMAT.md)) |
| Processing rate | **10 FPS** (required by the Hands and Final stages; input videos may have any native rate) |
| Processing resolution | **640×360** by default |
| Final export | resampled to 30 FPS in `standardized_hoi_30fps/`; the first 500 output frames by default |

## Stages

The WebUI runs these stages in order; each can also be run on its own.

| Key | Stage | Main entrypoints (under `pipeline/` unless noted) |
| --- | --- | --- |
| `preprocess` | Rename camera videos, extract and synchronize frames, copy calibration, first-frame and video masks | `rename_videos_31cams.py`, `rename_videos_extract_frames_ffmpeg_pts_compat.py`, `prepare_bowl_0820_ck_ffmpeg_fast.py`, `webui/sync_calibration.py`, `sam3/sam3text_object.py`, `sam3/sam3text_multi_gpu.py` |
| `masks` | Re-run object segmentation only | `sam3/sam3text_object.py`, `webui/mask_views.py`, `sam3/sam3text_multi_gpu.py` |
| `recon` | Multi-view metric depth and object mesh | `Depth-Anything-3/da3_multiview_metric_align_uint16_v4_8gpu.py`, `MV-SAM3D/run_mv_sam3d_metric_scale.py` |
| `meshscale` | Optional silhouette-based mesh rescaling | `webui/mesh_scale_from_silhouettes.py` |
| `tracking` | 6D object tracking (FoundationPose or WAPR) and geometric post-optimization | `run_resize_camk_foundationpose_args_local.sh`, `webui/wapr_tracking.py`, `FoundationPose/postopt_geometry_global_pose.py` |
| `hands` | 2D hand detection, multi-view triangulation and MANO fitting | `run_extract_sapiens_easymocap_10fps_kalman.sh` → `WiLoR/demo_fast.py`, `apps/demo/mv1p_fast.py` |
| `final` | Joint hand–object refinement, standardized export and preview grids | `run_joint_pose_solver_only_and_export_bimanual_10fps_8gpu_hocap_meshsdf_31cam.sh`, `FoundationPose/export_standardized_mano_object_table_4rings_first500.py`, `run_render_outputmano_grid_8gpu.sh` |

Inputs, outputs, environments and prerequisites of each stage are listed in [`provenance/stages.json`](provenance/stages.json). Object tracking details are in [docs/POSE_TRACKING_V2.md](docs/POSE_TRACKING_V2.md).

## Setup

All commands below are run from this `bihoi_pipeline/` folder. Paths such as `.environments/` and `.models/` are the defaults; they move if you set `HOI_ENV_ROOT` or `HOI_MODEL_ROOT` in `.env`.

### 1. Configure

```bash
git clone https://github.com/Ropedia/DONIC.git
cd DONIC/bihoi_pipeline
cp .env.example .env
```

Edit `.env` for your machine, for example:

```dotenv
HOI_DATA_ROOT=/absolute/path/to/data
HOI_CALIBRATION_DIR=/absolute/path/to/calibration
WEBUI_ALLOWED_GPUS=0,1
WEBUI_PASSWORD=choose-a-password
```

`.env` is sourced as shell code by the launcher, so only put trusted values in it.

### 2. Install environments

Requirements: Linux, NVIDIA GPU(s) and driver, Conda/Miniforge, Git, FFmpeg/ffprobe and CUDA build tools. Each stage uses its own Conda environment (nine in total):

```bash
bash scripts/install.sh --component all --skip-native
```

Native GPU components need CUDA toolkits matching their PyTorch builds:

```bash
conda install -p .environments/sam3d-objects -c nvidia cuda-toolkit=12.1
conda install -p .environments/foundationpose -c nvidia cuda-toolkit=12.4
conda install -p .environments/wapr -c nvidia cuda-toolkit=12.8
conda install -p .environments/hocap-annotation -c nvidia cuda-toolkit=11.8

bash scripts/build_native.sh sam3d-objects
bash scripts/build_native.sh foundationpose
bash scripts/build_native.sh wapr
bash scripts/build_native.sh hocap-annotation
bash scripts/build_native.sh depthanything3
```

### 3. Download model weights

Accept the upstream licenses and request Hugging Face access to the gated **SAM3** and **SAM3D** models, then authenticate (for example `huggingface-cli login`, or set `HF_TOKEN`) and run:

```bash
for model in sam3 sam2 da3 mv-sam3d moge wilor; do
    bash scripts/download_models.sh --component "$model" --yes
done
```

Two model families are downloaded manually:

- **FoundationPose:** follow the [official instructions](https://github.com/NVlabs/FoundationPose#download-weights) and put `model_best.pth` in both `.models/foundationpose/2023-10-28-18-33-37/` and `.models/foundationpose/2024-01-11-20-02-45/`.
- **MANO:** register on the [MANO website](https://mano.is.tue.mpg.de/), put `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` in `.models/mano/`, then run `bash scripts/prepare_mano_regressors.sh`.

The **Hands** stage and hand calibration need the separately licensed WiLoR variant, which is not included. With permission and a copy of that variant:

```bash
.environments/handobj_webui/bin/python scripts/import_wilor.py --source /path/to/licensed/WiLoR
.environments/handobj_webui/bin/python scripts/setup_assets.py
bash scripts/check_env.sh
```

### 4. Hand calibration (once per data collector)

Each collector's MANO shape (beta) parameters are estimated once and reused for all of their captures. Calibration needs the WiLoR variant, MANO assets and the FMHR checkpoints/templates (`HOI_FMHR_MODEL_DIR` in `.env`).

```bash
# After reviewing and accepting the psbody-mesh license:
HOI_ACCEPT_PSBODY_MESH_LICENSE=1 bash scripts/install_hand_calibration_deps.sh

bash run_hand_calibration_args.sh \
    --data_name hands_collector01 \
    --start_sec 0 \
    --end_sec 0.1 \
    --hand_side right
```

Input: `data/hand_calibration/hands_collector01/` with `videos/`, `intri.yml` and `extri.yml` (calibration matching 640×360 frames). Output: `data/rigid/<person_name>/calib/restored_flat_real_scale.pt`. Map calibration captures to collectors in `configs/hand_calibration.json`, or pass `--person_name`. Right-hand calibration has been run end to end; left-hand calibration is not yet verified. Details: [docs/HAND_CALIBRATION.md](docs/HAND_CALIBRATION.md).

## Data layout

```text
data/
└── rigid/
    └── <person_name>/                       # one folder per data collector
        ├── calib/
        │   └── restored_flat_real_scale.pt  # from hand calibration
        └── <sequence>/
            └── videos/
                ├── take01/
                │   ├── node13_cam0.mp4
                │   ├── node13_cam1.mp4
                │   └── ...                  # 31 synchronized camera videos
                └── take02/                  # optional additional take
```

Camera calibration lives in a separate folder:

```text
calibration/
├── intri.yml
├── extri.yml
└── corrected_gsformat.pkl                   # currently required by preprocessing
```

The camera files must match the calibrated camera IDs and order; the pipeline does not calibrate cameras. See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for exact camera names and optional inputs.

## Run the WebUI

```bash
bash scripts/run_webui.sh
```

Open **http://127.0.0.1:7860** and sign in with the credentials from `.env` (default user `operator`; if `WEBUI_PASSWORD` is unset, a one-time password is printed at launch). On a remote GPU server, forward port 7860 over SSH. Do not expose the WebUI publicly.

In the WebUI:

1. Select the **data root**, a dataset (e.g. `rigid/collector01/my_sequence`) and a take.
2. Set the **object text prompt**, **10 FPS** and **640×360**.
3. Run all stages or pick individual ones, then inspect masks, depth, mesh, tracking, hands and previews in the interface.

Outputs are written under the selected dataset, e.g. `images/`, `masks/`, `depth/`, `meshes/compressed.ply` and `standardized_hoi_30fps/`.

## Tests

```bash
bash scripts/validate.sh     # unit and packaging tests (no GPU inference)
bash scripts/check_env.sh    # checks installed environments and assets
```

The heavier end-to-end procedure is described in [docs/INTEGRATION.md](docs/INTEGRATION.md).

## Optional demos

[`web_demos/`](web_demos/README.md) contains standalone WAPR and MoSE3 demo servers that are independent of the main pipeline. The patched WAPR core under `web_demos/third_party/WAPR/` is also used by the WebUI's WAPR tracking mode.

## License

This folder is an aggregation; there is no single license for all of it.

- Code written for this pipeline — `webui/`, `scripts/`, `tests/`, `configs/` (except `configs/models/`), `web_demos/` (except `third_party/`), and the scripts we added under `pipeline/` — is licensed under [CC BY-NC 4.0](LICENSE).
- Third-party code keeps its own license, including non-commercial terms for EasyMocap and FoundationPose and LGPL-2.1 for WAPR. Everything under `pipeline/FoundationPose/` is also subject to NVIDIA's FoundationPose license. See [THIRD_PARTY.md](THIRD_PARTY.md) and `licenses/`.
- Model weights, MANO assets, the WiLoR variant and datasets are not distributed; their own licenses apply.
