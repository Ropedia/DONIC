# Hand Calibration

Hand calibration is a one-time, per-collector setup step. It estimates the collector's MANO shape (beta) parameters, which the **Hands** and **Final** stages then reuse for every capture of that collector. It runs at **10 FPS** and is separate from the normal processing stages; it is never added to their automatic job graph.

## Execution chain

`run_hand_calibration_args.sh` (or the WebUI calibration panel) runs:

1. `rename_videos_31cams.py --strict-calibration`: copy the calibration videos into an isolated workspace as `1.mp4`, `2.mp4`, ..., following `CALIB_CAMERA_ORDER` exactly. The source videos are never modified. Ambiguous, extra or missing cameras are rejected.
2. `rename_videos_extract_frames_ffmpeg_pts_compat_resize.py --skip-rename`: extract 640×360 images at 10 FPS from the selected half-open time interval.
3. `WiLoR/demo.py`: 2D hand detections in `annots_fallback/<camera>/000000.json`. This requires the separately licensed WiLoR variant (see [THIRD_PARTY.md](../THIRD_PARTY.md)).
4. `apps/demo/mv1p713.py`: multi-view triangulation with fallback 2D recovery, using `handr/manor` for the right hand or `handl/manol` for the left hand.
5. `FMHR/multihands_mano.py` → `train_gcn.run_infer_json` → `models/gcn.py`: regress MANO parameters from the frame-zero joints.
6. `FMHR/restore_gcn_mano_flat_real_scale_v3_all_bones.py`: rebuild the posed MANO geometry, estimate scale from all 20 skeleton bone segments, generate a flat hand and solve a regularized least-squares real-scale beta fit.
7. `scripts/publish_hand_calibration.py`: validate the result and save the per-collector prior.

## Calibration camera naming

The input folder is `<Hand Calibration Root>/<capture>/videos/`. Supply **one `.mp4` video per calibrated camera**, named after that camera; every camera in the list must be present. The conversion is 1-based and follows the list order: `node13_cam0.mp4` → `1.mp4`, `node13_cam1.mp4` → `2.mp4`, `node25_cam1.mp4` → `3.mp4`, and so on. The full list is `CALIB_CAMERA_ORDER` in `pipeline/rename_videos_31cams.py`.

A complete numbered capture (`1.mp4`, `2.mp4`, ...) or a complete `rpizero000_test*.mp4`, `rpizero001_test*.mp4`, ... capture is also accepted. In strict calibration mode, filename guessing, unknown cameras and missing cameras are errors. `intri.yml` and `extri.yml` must use the same numbered camera order.

## Collector mapping and output

A calibration capture (for example `hands_collector01`) is mapped to a collector name (for example `collector01`) through `configs/hand_calibration.json`. Add one entry per collector, or pass `--person_name` explicitly; names are never guessed from suffixes.

Output: `$HOI_DATA_ROOT/rigid/<person_name>/calib/restored_flat_real_scale.pt`. Its `real_scale_betas10_lstsq` and `original_betas10` fields are finite float tensors of shape `(10,)`, and `real_scale` is finite and positive. Publication adds capture, collector, side, interval and FPS metadata and is atomic. An existing prior is refused unless `--replace-prior` is given, in which case the old file is kept as a backup.

EasyMocap's `load_betas_from_restored_pt` reads `real_scale_betas10_lstsq`, and `set_fixed_betas_to_params` applies it to all frames. The Final stage and its standardized export use the same ten beta values.

## Tests

`tests/test_hand_calibration.py` covers argument, side and interval propagation, collector mapping, rejection of unmapped names and path escapes, the fixed 10 FPS rate, FFmpeg extraction from a generated video, beta validation, atomic publication with backups, and the downstream loaders. These tests use synthetic inputs; they do not run WiLoR, FMHR or GPU inference.

## Limits

- End-to-end calibration has been run on a right-hand capture. Left-hand calibration is implemented but has not been verified on real data.
- The least-squares real-scale beta vector is an approximation; it is not a separate global-scale parameter.
- Running calibration requires the separately licensed WiLoR variant, MANO assets, FMHR templates and checkpoints, and the external research-use `psbody-mesh` dependency. None of these are distributed here.
- `psbody-mesh` fails to import under EGL because its unused viewer loads GLUT fonts, so the launcher uses GLX for the FMHR step and EGL elsewhere.
