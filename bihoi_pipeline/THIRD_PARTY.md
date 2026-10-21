# Third-party source and model terms

Licenses were read from the local repositories and checked against official upstream documentation. Full license texts are in `licenses/`; original notices remain in source files. This aggregation does not establish compatibility between every license or grant commercial rights.

| Component | Source handling | Observed upstream/snapshot | License evidence |
| --- | --- | --- | --- |
| EasyMocap | Local package and MANO fitting modifications vendored | Original root has invalid/empty `.git`; per-file original hashes in provenance | `licenses/EasyMocap.txt`: noncommercial educational/research/nonprofit use; modified code must remain available |
| SAM3 | Package and the two active local entrypoints vendored | `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7` | `licenses/SAM3.txt`; custom SAM license. The upstream metadata's MIT classifier does not override this text |
| SAM2 | SAM2 package/configs vendored; CUDA postprocessing disabled by active stage | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | `licenses/SAM2.txt`, Apache-2.0 |
| Depth Anything 3 | Inference package and multiview script vendored | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | `licenses/Depth-Anything-3.txt`, Apache-2.0 code; selected giant model has CC-BY-NC-4.0 weights |
| MV-SAM3D | Actual local multiview implementation, inference helper and package vendored | `abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd`; nested SAM3D snapshot `f91db411c50efee93d8db7aeb323885650f6f722` | `licenses/MV-SAM3D.txt`, custom SAM3D terms |
| FoundationPose | Active entrypoints, transitive import closure, C++ helper and config templates vendored | Top-level customized source has no valid Git repository; do not equate it to the separate `data/FoundationPose` checkout | `licenses/FoundationPose.txt`, NVIDIA noncommercial terms |
| WiLoR | Adapted source excluded; hash-checked private local import only | Official upstream reference `fcb911312a38fa8badd30d9656a167485d61b8f9`; actual local variant identified by 27 source/config hashes | [Official license](https://github.com/rolpotamias/WiLoR/blob/main/license.txt), CC-BY-NC-ND-4.0 |
| HO-Cap Annotation | Pinned external installation, not committed | `f554ecdfdd1230f277a4dae443b104914caeb5a7` | `licenses/HO-Cap-Annotation.txt`: AGPL-3.0 text; package metadata incorrectly says GPL-3.0 |
| MoGe, PyTorch3D, nvdiffrast, gsplat, utils3d, chumpy | Pinned external package installs | Git commits in `environments/observed.json` and `scripts/build_native.sh` | Follow the license in each official upstream repository |
| Kaolin/spconv/xformers | External CUDA-specific packages | Observed version pins in environment records | Their upstream licenses and GPU/PyTorch compatibility requirements |
| MANO | Operator obtains and accepts terms directly | No weights, mean-parameter arrays or learned joint regressor tables committed | [Official MANO registration](https://mano.is.tue.mpg.de/) |

The local `WiLoR/demo_fast.py` is required by the working hand stage but is an adapted variant. Its no-derivatives license creates an unresolved public redistribution issue. `scripts/import_wilor.py` validates and copies only this variant's required source/config files into ignored `.external/WiLoR`, preserving its bytes. It does not distribute that code to new users or substitute an upstream demo with different outputs. A completely independent clone cannot run the hand stage until the operator supplies this exact licensed variant or the rights holder permits publishing it.

The HO-Cap install mechanism uses its actual upstream loss; it has no approximate replacement. External installation does not remove applicable AGPL obligations. Review combined distribution/service use before publishing a public service.

Model download revisions were read from official Hugging Face metadata without downloading checkpoints. Their separate source/model terms must be accepted by the operator. Source snapshot hashes describe local modifications; an upstream Git revision alone does not reproduce the customized source.

## Hand Calibration additions

FMHR's calibration source/import closure retains its Apache-2.0 license in `pipeline/FMHR/LICENSE` and `licenses/FMHR.txt`; the work is Fine-Grained Multi-View Hand Reconstruction Using Inverse Rendering by Qijun Gan, Wentong Li, Jinwei Ren and Jianke Zhu ([official repository](https://github.com/agnJason/FMHR)). Modified files identify packaging changes; the estimator's numerical routines are preserved. `models/mesh_sampling.py` attributes its CoMA-derived code; the complete [CoMA license](https://github.com/anuragranj/coma/blob/master/LICENSE) is preserved in `licenses/CoMA.txt`. EasyMocap calibration code retains its existing noncommercial terms.

The adapted `WiLoR/demo.py` is not distributed under its no-derivatives terms. Only its hash is published; the operator's licensed local import now verifies 27 required source/config files. `psbody-mesh` is an external dependency, with its [research-use terms](https://github.com/MPI-IS/mesh/blob/49e70425cf373ec5269917012bda2944215c5ccd/LICENSE.txt) in `licenses/psbody-mesh.txt`; its source/binaries are not copied into this repository. FMHR GCN checkpoints, template meshes, MANO files and per-person beta parameters are operator-supplied private assets, never published.

## V2 WAPR and standalone demos

WAPR core comes from [WangYuLin-SEU/WAPR](https://github.com/WangYuLin-SEU/WAPR)
at `15eecbe77090a80ec454576dcf640f15ce64edf6`. Source retains its
[LGPL-2.1-only license](web_demos/third_party/WAPR/LICENSE), author attribution,
third-party notices and separate weight license. Changed files and packaging scope
are listed in [LOCAL_CHANGES.md](web_demos/third_party/WAPR/LOCAL_CHANGES.md).
No WAPR weights are distributed. The four upstream weight files have CC-BY-ND-4.0 terms.

Bundled WAPR fonts retain DejaVu and WenQuanYi notices under `wapr/fonts/`.
The standalone demo's Noto Sans SC subset retains `web_demos/static/fonts/OFL.txt`.
These display assets are explicitly hashed in `configs/source-assets.json`.

MoSE3 source and weights are external, not vendored. Its optional demo uses the
revision documented in `web_demos/README.md`; consult upstream source/weight terms.
