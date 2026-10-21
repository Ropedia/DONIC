# Operator data contract

Data discovery is `<HOI_DATA_ROOT>/rigid/<person_prior>/<sequence>`. take01 lives at the sequence root; later processed takes live under `takes/takeXX`. Raw videos are in `videos/takeXX`. The current UI accepts multiple datasets, multiple takes, per-dataset object text prompts and hand-side selection. Person prior `calib/restored_flat_real_scale.pt` contains at least ten MANO shape coefficients: a tensor/array or a dictionary under `real_scale_betas10_lstsq`, `real_scale_betas10`, `betas`, `beta`, `shapes`, or `shape`. This is operator-provided calibration, not a published model template.

The exact camera order, mapping raw stem to numbered camera ID, is:

1. `node13_cam0` → `1`
2. `node13_cam1` → `2`
3. `node25_cam1` → `3`
4. `node26_cam0` → `4`
5. `node26_cam1` → `5`
6. `node27_cam0` → `6`
7. `node27_cam1` → `7`
8. `node28_cam0` → `8`
9. `node28_cam1` → `9`
10. `node33_cam0` → `10`
11. `node33_cam1` → `11`
12. `node34_cam0` → `12`
13. `node34_cam1` → `13`
14. `node35_cam0` → `14`
15. `node35_cam1` → `15`
16. `node36_cam0` → `16`
17. `node36_cam1` → `17`
18. `node37_cam0` → `18`
19. `node37_cam1` → `19`
20. `node38_cam0` → `20`
21. `node38_cam1` → `21`
22. `node39_cam0` → `22`
23. `node39_cam1` → `23`
24. `node40_cam0` → `24`
25. `node40_cam1` → `25`
26. `node41_cam0` → `26`
27. `node41_cam1` → `27`
28. `node42_cam0` → `28`
29. `node42_cam1` → `29`
30. `node47_cam0` → `30`
31. `node47_cam1` → `31`

If source names match partially, `--allow-missing` preserves their calibration indices. If none match, the original script falls back to natural name order; the operator must ensure this corresponds to the calibration. No automatic camera recalibration is performed.

Calibration uses EasyMocap/OpenCV YAML: `names`, `K_<id>`, `dist_<id>` in intrinsics, `R_<id>`/`Rot_<id>` and `T_<id>` in extrinsics. `Rot`/`T` transform world to camera; centers are `-Rot.T @ T`. Metric units must be consistent. The original preparation also requires `corrected_gsformat.pkl`, even though later active stages primarily use YAML. Intrinsics must match resized frames (default 640×360).

Frames and per-frame masks use camera subdirectories with identical zero-padded frame stems. Reference images1 and mask use one numbered image directly in each directory. DA3 writes uint16 metric PNGs in `depth/<cam>/`; depth scale is 1000 units per meter in the tracking/final preset. Input images, masks, and depths must refer to the same sampling grid. The original reference frame is t=0 even when extraction starts later; this behavior is preserved.

The raw take rate need not be 10 FPS, but preprocessing samples at 10 FPS for the hand/final presets. Standardized export resamples to 30 FPS and limits to first 500 output frames by default. It includes camera/canonical coordinate metadata, object mesh/pose, MANO streams and source-frame coordinates. The qualitative renderer reads `standardized_hoi_30fps/metadata.json`, `source_frame_ids.npy`, object/MANO NPZ arrays and the object mesh.

No example capture, measured calibration, image, mesh, array, checkpoint or actual hand prior is versioned. Synthetic test fixtures are generated under temporary directories during tests and removed by the test harness.
