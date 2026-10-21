<h1 align="center">DONIC: Generalist Dexterous Bimanual Tracking Control from Human Demonstrations</h1>

<p align="center">
  Mingze Sun,
  Ziang Cao,
  Yuan Fei,
  Qiankuan Tao,
  Zihan Wang,
  Tianqi Liu,
  Junhao Chen,
  Haiwen Diao,
  Fangzhou Hong,
  Zhaoxi Chen,
  Ziwei Liu<sup>†</sup>
</p>

<p align="center">
  Ropedia
</p>

<p align="center">
  <sup>†</sup>Corresponding author
</p>

<p align="center">
  <a href="https://ropedia.github.io/DONIC/"><img src="https://img.shields.io/badge/Project_Page-online-2ea44f" alt="Project Page"></a>
  <img src="https://img.shields.io/badge/arXiv-coming_soon-b31b1b?logo=arxiv&amp;logoColor=white" alt="DONIC paper on arXiv (coming soon)">
  <a href="./demo.mp4"><img src="https://img.shields.io/badge/Video-MP4-e53935?logo=youtube&amp;logoColor=white" alt="Demo video"></a>
  <a href="https://github.com/Ropedia/DONIC"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&amp;logoColor=white" alt="GitHub code"></a>
  <img src="https://img.shields.io/badge/🤗_BiHOI-coming_soon-f0c000" alt="BiHOI dataset coming soon">
</p>

https://github.com/user-attachments/assets/13836832-63da-4767-8acb-2be188fbb6aa

## Repository structure

| Folder | Contents | Status |
| --- | --- | --- |
| [`bihoi_pipeline/`](bihoi_pipeline/README.md) | Multi-view hand–object interaction processing pipeline used to build BiHOI (masks, metric depth, object mesh, 6D object tracking, bimanual MANO hands, standardized export, WebUI) | available |
| [`donic/`](donic/README.md) | DONIC training and evaluation code | coming soon |

## Release

| Resource | Status |
| --- | --- |
| BiHOI processing pipeline | [available](bihoi_pipeline/) |
| DONIC training and evaluation code | coming soon |
| 🤗 BiHOI dataset | coming soon |
| DONIC policy weights | coming soon |

The BiHOI pipeline is for non-commercial use and contains third-party code under its own licenses; see [`bihoi_pipeline/LICENSE`](bihoi_pipeline/LICENSE).

## Citation

```bibtex
@article{sun2026donic,
  title   = {DONIC: Generalist Dexterous Bimanual Tracking Control from Human Demonstrations},
  author  = {Sun, Mingze and Cao, Ziang and Fei, Yuan and Tao, Qiankuan and Wang, Zihan and Liu, Tianqi and Chen, Junhao and Diao, Haiwen and Hong, Fangzhou and Chen, Zhaoxi and Liu, Ziwei},
  journal = {Ropedia Technical Report},
  year    = {2026}
}
```
