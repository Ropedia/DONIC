# Optional heavyweight integration procedure

No training is required. Run these steps only on a small operator-owned test take after environment/weight checks pass. They execute actual model inference and may take significant GPU time.

1. Put all raw videos and valid rig/person calibration under configured ignored roots. Run `scripts/check_env.sh` with models enabled.
2. Launch the UI, select one dataset/take and a concrete text prompt. Choose FPS=10, 640×360 and a short interval. Keep a separate test capture so existing frame/output replacement is intentional.
3. Run preprocess. Check equal camera frame counts, valid calibration indices, foreground masks and 12 selected mesh views. Re-run the separate masks stage if prompts need adjustment.
4. Run recon. Inspect uint16 millimeter depth and `meshes/compressed.ply` against calibrated geometry. Optionally run meshscale and texture generation; verify texture sidecar hash agreement.
5. Run tracking. Inspect pose reprojections, geometry postoptimization and grid/video outputs.
6. Import the licensed WiLoR variant, supply its detector/MANO/model assets and the person's beta prior, then run hands. Check annots_fallback, both MANO streams, RANSAC/Kalman diagnostics and hand-side selection.
7. Run final only after exact HO-Cap CUDA loss is available. Inspect loss CSV, refined object and hands, Do-As-I-Do export, first-500 30 FPS standardized export and final grid video. Compare against a reference run before claiming numerical reproduction.
8. Add take02 and verify its mesh link waits for take01 while depth/tracking/hands/final are take-specific. Test qualitative figure/video/snapshot generation on a completed standardized export.

These integrations were not executed during packaging; model checkpoints, real camera calibration and reconstruction fixtures were intentionally not transferred. The CPU smoke tests are not evidence that model-driven stages completed or that newly built CUDA extensions are numerically equivalent.
