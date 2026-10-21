# Local changes for the BiHOI pipeline (Ropedia/DONIC)

Upstream: https://github.com/WangYuLin-SEU/WAPR
Revision: 15eecbe77090a80ec454576dcf640f15ce64edf6
Modified source snapshot: 2026-10-15, integrated into the BiHOI pipeline (bihoi_pipeline/ in https://github.com/Ropedia/DONIC).

- `wapr/estimator.py`: optional explicit mesh-center XYZ for registration hypotheses;
  convert to original-mesh translation as `t = center - R @ mesh_center`.
- `wapr/tracking.py`: center-only three-candidate recipe, shared WAPR computation and
  batched SAPR refinement, retaining the complete six-candidate recipe for initialization.
- `wapr/pose_groups.py`: permit the 96-hypothesis WBPS scoring group for recovery.

The `wapr/` package, its runtime configuration, native source, fonts and license notices
are included. Upstream examples, demo videos, release workflows and wheel packaging
are omitted. This is source vendoring, not an unmodified upstream distribution.
Model weights and compiled native libraries are excluded; original licenses still apply.
