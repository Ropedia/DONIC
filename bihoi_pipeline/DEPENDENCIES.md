# Execution and dependency report

The active launcher now starts `webui/interaction_capture_webui.py`, whose `_steps_for` and `_plan_steps` functions define seven selectable stages. The unused alternate WebUI implementation is omitted. `provenance/stages.json` records inputs, outputs, prerequisites, entrypoints and resource modes. Dynamic Hydra targets in the MV-SAM3D config templates are part of the dependency graph, even though they are not normal imports.

## Stage entrypoints and execution graph

### 1 Prepare videos/images/calibration and masks (`preprocess`)

Environments: `easymocap`, `handobj_webui`, `sam3`. Resources: CPU helpers; single GPU first-frame SAM3; multi-GPU SAM3/SAM2 video masks.

Entrypoints:

- `rename_videos_31cams.py`
- `rename_videos_extract_frames_ffmpeg_pts_compat.py`
- `prepare_bowl_0820_ck_ffmpeg_fast.py`
- `webui/extract_reference_frames.py`
- `webui/shortest_video_end.py`
- `webui/equalize_camera_frames.py`
- `webui/sync_calibration.py`
- `webui/mask_views.py`
- `sam3/sam3text_object.py`
- `sam3/sam3text_multi_gpu.py`

Inputs: `videos/takeXX/*.mp4`, `operator rig calibration`, `text prompt`.

Outputs: `videos/*.mp4`, `images1/*.jpg`, `images/<cam>/*.jpg`, `mask/*.png`, `masks/<cam>/*.png`, `camera files`.

Models: sam3, sam2. Upstream: raw operator inputs.

### 1b Re-run segmentation (`masks`)

Environments: `sam3`, `handobj_webui`. Resources: single and multi-GPU.

Entrypoints:

- `sam3/sam3text_object.py`
- `sam3/sam3text_multi_gpu.py`
- `webui/mask_views.py`
- `webui/mask_views_all_cams.py`

Inputs: `images1`, `images`, `text prompt`, `camera files`.

Outputs: `mask`, `masks`.

Models: sam3, sam2. Upstream: preprocess.

### 2 Metric depth and object mesh (`recon`)

Environments: `depthanything3`, `sam3d-objects`, `sam3`, `handobj_webui`, `easymocap`. Resources: multi-GPU DA3, single GPU MV-SAM3D.

Entrypoints:

- `Depth-Anything-3/da3_multiview_metric_align_uint16_v4_8gpu.py`
- `MV-SAM3D/run_mv_sam3d_metric_scale.py`
- `MV-SAM3D/notebook/inference.py`
- `MV-SAM3D/texture_export.py`
- `webui/mask_views.py`
- `webui/equalize_camera_frames.py`
- `webui/sync_calibration.py`
- `webui/link_shared_take_resources.py`

Inputs: `images`, `masks`, `images1`, `mask`, `camera files`.

Outputs: `depth/<cam>/*.png`, `depth debug/backup folders`, `meshes/compressed.ply`, `meshes/metric_scale_info.json`, `optional meshes/textured.glb and sidecar`.

Models: da3, mv-sam3d, moge. Upstream: preprocess.

Later takes: DA3 per take, then link take01 mesh; waits for primary reconstruction.

### 2b Silhouette metric mesh rescaling (`meshscale`)

Environments: `easymocap`. Resources: CPU.

Entrypoints:

- `webui/mesh_scale_from_silhouettes.py`

Inputs: `meshes/compressed.ply`, `mask`, `camera files`.

Outputs: `rescaled meshes/compressed.ply`, `scale diagnostics`.

Models: none. Upstream: recon.

Later takes: Skipped because mesh is shared.

### 3 FoundationPose / WAPR and geometry postoptimization (`tracking`)

Environments: `foundationpose`, `easymocap`, and `wapr` for WAPR mode. Resources: scheduler-selected GPU, persistent per-camera workers, CPU image prefetch and asynchronous preview.

Entrypoints:

- `webui/wapr_tracking.py` (WAPR mode), `webui/foundationpose_reinit.py`
- `webui/masked_depth.py`, `webui/wapr_parallel.py`, `webui/tracking_preview.py`
- `web_demos/third_party/WAPR/wapr/estimator.py`, `tracking.py`, `pose_groups.py`
- `run_resize_camk_foundationpose_args_local.sh`
- `generate_cam_k.py`
- `FoundationPose/run_demo_d_working_depth_optional_no_force_bbox_center_fixed_fastio.py`
- `FoundationPose/postopt_geometry_global_pose.py`
- `FoundationPose/render_allcams_mesh_grid_fast_green.py`

Inputs: `images`, `masks`, `depth mm`, `mesh`, `camera files`.

Outputs: `cam_K.txt`, `debug_depth/ob_in_cam`, `debug_depth/geometry_global_opt/ob_in_world`, `tracking grid/video`.

Models: FoundationPose refiner/scorer; WAPR mode additionally uses WAPR with/without mask, SAPR and WBPS. Upstream: recon. See [6D tracking setup](docs/POSE_TRACKING_V2.md).

### 4 WiLoR detection, multiview triangulation and MANO (`hands`)

Environments: `wilor`, `easymocap`. Resources: scheduler-assigned single GPU; wrapper dispatches fitting jobs within that assignment.

Entrypoints:

- `run_extract_sapiens_easymocap_10fps_kalman.sh`
- `WiLoR/demo_fast.py [external licensed variant]`
- `apps/demo/mv1p_fast.py`

Inputs: `images`, `camera files`, `person prior/calib/restored_flat_real_scale.pt`, `licensed MANO/WiLoR assets`.

Outputs: `annots_fallback/<cam>/*.json`, `output/manol/smpl/*.json`, `output/manor/smpl/*.json`, `triangulation and Kalman state/diagnostics`.

Models: wilor, mano. Upstream: tracking.

### 5 Exact HO-Cap joint refinement, exports and grids (`final`)

Environments: `hocap-annotation`, `easymocap`. Resources: multi-GPU solver/rendering; CPU export and FFmpeg.

Entrypoints:

- `run_joint_pose_solver_only_and_export_bimanual_10fps_8gpu_hocap_meshsdf_31cam.sh`
- `FoundationPose/render_object_both_hands_joint_pose_solver_only_10fps_8gpu_hocap_meshsdf_31cam.py`
- `FoundationPose/export_outputmano_to_do_as_i_do_raw_bimanual_8gpu.py`
- `FoundationPose/export_outputmano_to_do_as_i_do_raw_bimanual.py` (included fallback)
- `FoundationPose/export_standardized_mano_object_table_4rings_first500.py`
- `run_render_outputmano_grid_8gpu.sh`
- `FoundationPose/render_object_both_hands_joint_pose_solver_only_10fps_8gpu_hocap_meshsdf_31cam_fastgrid.py`

Inputs: `depth`, `masks`, `images`, `camera files`, `meshes/compressed.ply`, `debug_depth/geometry_global_opt/ob_in_world`, `output/manol and/or manor`, `MANO assets`.

Outputs: `outputmano/contact_mano`, `outputmano/smoothed_ob_in_world`, `outputmano/object_mesh_scaled.ply`, `Do-As-I-Do raw export`, `standardized_hoi_30fps`, `outputmano/grid/final_hand_object_interaction.mp4`.

Models: mano. Upstream: hands, tracking, recon.

## Environment and CUDA matrix

| Environment | Observed Python | Observed PyTorch | torchvision | CUDA runtime | Main dependencies |
| --- | --- | --- | --- | --- | --- |
| handobj_webui | 3.10.21 | none | none | CPU | Gradio 6.29.1, NumPy 2.2.6, Pillow; proposed trimesh fallback from observed EasyMocap |
| easymocap | 3.9.23 | 2.3.1+cu121 | 0.18.1+cu121 | 12.1 | OpenCV, SciPy, smplx, chumpy, YAML, yacs, trimesh, pyrender, PyOpenGL |
| sam3 | 3.12.14 | 2.10.0+cu128 | 0.25.0+cu128 | 12.8 | SAM3, SAM2, timm, ftfy, regex, iopath, Hydra, OpenCV, pycocotools |
| depthanything3 | 3.11.16 | 2.10.0+cu128 | 0.25.0+cu128 | 12.8 | DA3, OmegaConf, einops, HF hub, OpenCV, evo, e3nn, pycolmap and API eager imports |
| sam3d-objects | 3.11.0 | 2.5.1+cu121 | 0.20.1+cu121 | 12.1 | sam3d_objects, MoGe, utils3d, spconv-cu121, PyTorch3D, Kaolin, nvdiffrast, gsplat, xatlas |
| foundationpose | 3.11.16 | 2.6.0+cu124 | 0.21.0+cu124 | 12.4 | PyTorch3D, nvdiffrast, OpenCV, Open3D, Kornia, ruamel.yaml, trimesh, native mycpp |
| wilor | 3.10.21 | 2.6.0+cu126 | 0.21.0+cu126 | 12.6 | licensed variant, ultralytics 8.1.34, smplx, chumpy, Lightning, timm, Hydra |
| hocap-annotation | 3.10.21 | 2.5.1+cu118 | 0.20.1+cu118 | 11.8 | HO-Cap loss + meshsdf_loss_cuda, smplx, chumpy, pyrender/PyOpenGL |
| wapr | 3.11.16 | 2.10.0+cu128 | 0.25.0+cu128 | 12.8 | patched WAPR core (bundled), nvdiffrast, Kornia, trimesh, OpenCV, ONNX, HF hub |

Every selected observed direct package pin and VCS commit is recorded in `environments/observed.json` and the component requirements files. Separate Conda YAMLs pin Python and describe system-independent build dependencies. These are not full transitive lockfiles. Optional packages not observed in a particular environment are explicitly recorded rather than assigned invented versions. In particular, fast-simplification is optional: the actual mesh compressor has existing fallback methods; the original environment did not have its distribution metadata.

PyTorch3D commits differ between FoundationPose and SAM3D; sharing these environments would change ABI compatibility. Kaolin is the observed 0.17.0 with Torch 2.5.1/CUDA 12.1. GSplat is pinned to `0b4dddf04cb687367602c01196913cde6a743d70`. MoGe is pinned to `a8c37341bc0325ca99b9d57981cc3bb2bd3e255b`. utils3d is pinned to `3913c65d81e05e47b9f367250cf8c0f7462a0900`. chumpy is pinned to `580566eafc9ac68b2614b64d6f7aaa84eebb70da`. Their full records remain reviewable in the environment metadata.

## Native extensions and external code

`mycpp` is C++/OpenMP, not CUDA. Its CMakeLists requires Boost system/program_options, Eigen and pybind11. It was compiled in the packaged tree using the existing FoundationPose environment. `Utils.py` imports it early from `FoundationPose/mycpp/build`; this path must exist after a fresh build.

PyTorch3D/nvdiffrast/gsplat and spconv have CUDA-specific binaries or build requirements. Optional texture generation additionally exercises Kaolin, xatlas and nvdiffrast. HO-Cap's exact MeshSDF loss requires `meshsdf_loss_cuda` built with CUDA 11.8 for the observed Torch version. Imported existing extensions passed; new compilation of all these extensions has not been tested. `build_native.sh` checks toolkit/runtime equality and requires a visible GPU before attempting the GPU builds. Matching toolkit install instructions are in README.

`WiLoR/demo_fast.py` is a separately imported adapted variant, not source distributed by this repository. The manifest retains the exact hashes required by the original hand pipeline. See THIRD_PARTY.md for the public reproducibility blocker. HO-Cap is fetched as an external pinned source dependency; no original modified installer is run, and no `rm -rf` cleanup is used by the packaging installer.

## Models and camera requirements

`configs/models/manifest.json` lists exact official model repository revisions and required files. SAM3 and SAM3D access is gated. FoundationPose official weights have separate score/refine version folders. Licensed MANO left/right files and locally derived learned joint regressor tables are private assets. The CPU conversion in `scripts/prepare_mano_regressors.sh` reproduces the original joint order and fingertip selection, including the different middle fingertip vertex for each side. The two generated tables were compared with the prepared originals and matched exactly; no learned values are published. MANO left/right files are shared between EasyMocap fitting, WiLoR and final refinement through ignored local links. WiLoR's mean-parameter NPZ is an external asset, not source.

Raw naming is tied to the exact calibrated camera order in `rename_videos_31cams.py`. Calibration is supplied by the operator. Preprocessing copies `intri.yml`, `extri.yml`, `corrected_gsformat.pkl`; the calibrated principal points/focal lengths must refer to the processed image resolution. Synchronization removes unavailable cameras from active YAMLs and preserves `.full` backups. Mask selection requires four radial rings with four suitable views in each of the first three. Final MeshSDF observation preparation requires at least eight usable calibrated depth views.

The primary reconstruction shares `meshes/` with later takes. Per-take video/images/masks/depth/tracking/hands/outputmano remain separate. Scheduler `after` records force secondary recon links to wait for primary mesh reconstruction. The media panel can regenerate optional texture using recorded reconstruction arguments while preserving the tracking mesh backup and verifying texture sidecar hashes.

## Additional WebUI dependencies and features

The main UI dynamically imports `scripts/hoi_qualitative/web_panel.py`. Its chain is `web_panel → media_config → run_single.sh → single_sequence → render_comparison`; the independent `render_hoi.py` batch CLI uses the same validator and runner. Source tests from the original media implementation are preserved. Rendering runs in EasyMocap with EGL and requires licensed MANO assets when hands are included. Figures, snapshots, real/orbit/world cameras, roll/crop controls, batch takes, previews, completed-result checks, and texture generation are retained.

Scheduler helper commands use the WebUI interpreter for Pillow/NumPy helpers, EasyMocap for OpenCV calibration synchronization, and explicit environment paths for native tools. FFmpeg/ffprobe are actual system prerequisites. tmux is optional; detached subprocess fallback remains available. ZIP/copy export, contact sheets, dataset screening, file deletion confirmation and stage resume remain part of the source.

## Excluded inactive projects

Standalone 2D Gaussian Splatting viewers, Sapiens repositories, FMHR, ReconViaGen, SVOR, tapnet, unrelated EasyMocap applications, original deployment bundles, real camera calibration, model checkpoints, dataset comparison JSONs, and copied OS rendering libraries are outside the active execution graph. The complete core source packages are retained where dynamic imports or package initialization require them; optional training/demo imports inside these packages are not evidence of an enabled pipeline stage.

## Separate per-person Hand Calibration setup

`run_hand_calibration_args.sh` calls `scripts/hand_calibration.py`, which uses the original frame extractor, privately imported `WiLoR/demo.py`, `apps/demo/mv1p713.py`, `FMHR/multihands_mano.py` → `FMHR/train_gcn.py` → `models/gcn.py`/`mesh_sampling.py`/`smplx.py`/`utils.py`, then `restore_gcn_mano_flat_real_scale_v3_all_bones.py`. `scripts/publish_hand_calibration.py` validates and publishes the original ten-beta contract. EasyMocap's existing package supplies the triangulation/config/camera utilities; no Sapiens stage is called by the original script despite its unused Sapiens variables.

Environments remain `easymocap` (CPU extraction, triangulation, FMHR CUDA inference, NumPy MANO restoration) and `wilor` (GPU detections). Extra requirements are `plyfile==1.1.3`, `pyzmq==27.2.0`, and `psbody-mesh` (observed installed version 0.4). The optional external source install is pinned to MPI-IS/mesh commit `49e70425cf373ec5269917012bda2944215c5ccd`; a fresh build at that pin is not yet verified. It needs a C++ compiler, Boost headers and OpenGL/GLU/GLUT libraries; its installer builds bundled CGAL mesh helpers. No psbody source or binaries are vendored. FMHR imports its unused MeshViewer under GLX to avoid the EGL GLUT-font import conflict; other stages retain EGL.

Required private assets: both MANO PKLs under `HOI_MANO_DIR`; both FMHR GCN checkpoints and both template OBJs under `HOI_FMHR_MODEL_DIR`; WiLoR checkpoint/detector/mean-parameter assets plus the exact licensed `demo.py` added to `provenance/restricted-wilor.json`. `--check` verifies paths and all required WiLoR source hashes. Model assets remain manual operator inputs.

The calibration graph is separate from the normal processing stages. Time/FPS select extraction once; downstream stages read these exact extracted frames and triangulated frame zero. Side selects `handr/manor` or `handl/manol`, then propagates to GCN/restoration and publication metadata. The person is an explicit CLI/WebUI selection or a verified mapping.
