# Modified 2026-10-15 for the BiHOI pipeline (Ropedia/DONIC, bihoi_pipeline/).
# See ../LOCAL_CHANGES.md for the changes; original notices and license follow.
# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China
# Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
# SPDX-License-Identifier: LGPL-2.1-only
# 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
# Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

"""Supported hypothesis counts per instance in a public batched forward.

每物体支持的 WBPS 候选数量；多个物体的独立候选组批量处理。
"""

# Every integer from 2 through 25, plus regular-polyhedron views times
# 3, 4, 5, or 6 in-plane samples. These are pose counts, not recipe steps.
# 2 到 25 的每个整数，以及正多面体视角数乘 3、4、5、6 个面内采样。
# 这些是一次前向的位姿数，不是 recipe 的迭代次数。
wbps_polyhedron_views = (4, 6, 8, 12, 20)
wbps_inplane_samples = (3, 4, 5, 6)
wbps_group_sizes = frozenset(range(2, 26)) | frozenset(
    int(view) * int(sample)
    for view in wbps_polyhedron_views
    for sample in wbps_inplane_samples
)
# Capture-pipeline recovery: 12 icosahedron directions x 8 in-plane rotations.
# Keep all 96 candidates in one WBPS attention group, not separate subgroups.
wbps_group_sizes |= frozenset({96})
assert len(wbps_group_sizes) == 35 and max(wbps_group_sizes) == 120
