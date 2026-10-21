# WAPR / MoSE3 本地 Web demo

两个独立的中文 Web demo，连接官方模型推理接口。网页服务仅依赖 Python 标准库；模型在各自的 Python 环境中执行，无需把新依赖装进 EasyMocap 环境。

## 启动网页

完成下方环境安装后，在本目录执行（启动脚本会读取仓库根目录的 `.env`）：

```bash
bash run_wapr.sh
# http://127.0.0.1:7863/wapr

bash run_mose3.sh
# http://127.0.0.1:7862/mose3
```

已有其他模型环境时，可显式指定：

```bash
bash run_wapr.sh --wapr-python /path/to/wapr-env/bin/python
# http://127.0.0.1:7863/wapr

bash run_mose3.sh --mose3-python /path/to/mose3-env/bin/python
# http://127.0.0.1:7862/mose3
```

也可以用一个服务打开两个页面，GPU 任务会在该服务内串行执行：

```bash
python3 server.py --port 7860 \
  --wapr-python /path/to/wapr-env/bin/python \
  --mose3-python /path/to/mose3-env/bin/python
# http://127.0.0.1:7860/wapr
# http://127.0.0.1:7860/mose3
```

只查看界面可直接运行 `python3 server.py`；未安装模型时，点击运行会在日志中说明缺少的依赖。默认仅监听 `127.0.0.1`。在远程服务器上，优先使用 SSH 端口转发，例如 `ssh -L 7860:127.0.0.1:7860 user@server`。

模型路径可通过 `--wapr-repo` / `--mose3-repo` 覆盖，或设置 `WAPR_REPO` / `MOSE3_REPO` 环境变量。`WAPR_PYTHON` / `MOSE3_PYTHON` 可代替对应命令行参数。MoSE3 默认读取 `mose3-tracker/MoSE3` Hugging Face 权重；使用本地权重时设置 `MOSE3_CHECKPOINT=/absolute/path/to/ckpt`。

## 安装模型

需要可用的 NVIDIA GPU、CUDA 版 PyTorch 和匹配的编译器。WAPR 默认使用仓库根目录
`.environments/wapr/bin/python`，MoSE3 使用 `.environments/mose3/bin/python`。
可通过 `.env` 的 `HOI_ENV_WAPR` / `HOI_ENV_MOSE3` 指向已有环境；也可用上述命令行参数。

### WAPR

V2 已包含修改后的 WAPR 源码，请保留该版本。安装步骤在仓库根目录执行：

```bash
bash scripts/install.sh --component wapr --skip-native
conda install -p .environments/wapr -c nvidia cuda-toolkit=12.8
bash scripts/build_native.sh wapr
bash scripts/check_env.sh --component wapr --skip-models
```

主 pipeline 默认使用 CUDA/nvdiffrast 渲染；独立 WAPR demo 使用上游 OGL 渲染，
还需要 OpenGL/EGL 开发头文件和 NVIDIA EGL 驱动。新建 WAPR 环境包含相关开发包；
已有环境需自行提供。首次 OGL 推理会编译扩展。`WAPR_BACKEND` 默认由网页设为 `torch`，
无需 TensorRT。官方估计器首次运行会下载缺失权重到 `WAPR_CACHE_DIR`，默认为
仓库的 `.models/wapr`。源码和权重许可证分别见 `third_party/WAPR/`。
本 demo 直接接收 Mask 或框，无需安装自动检测器或 SAM2。

### MoSE3

遵循 [MoSE3 官方安装说明](https://github.com/mose3-tracker/MoSE3)。官方测试组合为 PyTorch 2.6.0 + CUDA 12.6：

```bash
conda create -n mose3-web python=3.10 -y
conda activate mose3-web
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r third_party/MoSE3/requirements.txt
python -m pip install threadpoolctl
```

此 Web demo 使用官方 `inference.py`、`visualize.py`，不需要 Viser 的额外依赖。首次推理会通过 Hugging Face 下载权重，也可以预先下载：

```bash
hf download mose3-tracker/MoSE3 --local-dir /absolute/path/to/ckpt
export MOSE3_CHECKPOINT=/absolute/path/to/ckpt
```

### 获取可选 MoSE3 源码

WAPR 固定于 `15eecbe77090a80ec454576dcf640f15ce64edf6` 并包含 v2 补丁，
不需要重新 clone。MoSE3 外部源码固定于以下版本，在当前 `web_demos/` 目录执行：

```bash
git clone https://github.com/mose3-tracker/MoSE3.git third_party/MoSE3
git -C third_party/MoSE3 checkout c39ef00f7198515d709f40c668688d4f869255ae
```

先获取 MoSE3 源码，再安装上面的 MoSE3 环境。若使用命名环境 `mose3-web`，
在根目录 `.env` 设置 `HOI_ENV_MOSE3=mose3-web`。可使用 `MOSE3_REPO`
指向已有 checkout。外部源码、权重和运行数据均不提交到 v2 仓库。

## WAPR 使用

如果已通过上游工具下载 WAPR 的 `pose_lmo` 样例到模型缓存，可点击“载入样例”。MoSE3 同名按钮使用外部 checkout 中的官方 spin 视频。也可以直接上传自己的输入。

1. 上传 RGB 图像（PNG/JPEG）、深度（单通道 PNG/TIFF 或 H×W NPY）、三角网格（PLY/OBJ/STL/GLB）。
2. 在 RGB 预览上拖动框选目标，或切换为 Mask 模式并上传同尺寸的非零前景 Mask。框坐标使用原图像素，不受浏览器显示缩放影响。
3. 填写对应原始 RGB 图像的 `fx, fy, cx, cy`。不会自动缩放或猜测内参。
4. 填写深度和网格转换到米的乘数。例如毫米数据填写 `0.001`，米制数据填写 `1`。物体直径始终使用米，需填写真实值。
5. 运行后预览预测网格包围盒与 XYZ 坐标轴叠加，下载 `pose.npy`、`result.json`、`overlay.png`、`mesh_camera.ply` 或完整 ZIP。

位姿矩阵是 **object-to-camera**，即 `p_camera = R @ p_object + t`，OpenCV 相机坐标，+Z 向前，平移单位为米。`score_6d` 是官方排序分数，并非成功概率。叠加图绘制网格包围盒和坐标轴，不是渲染的物体分割轮廓。OBJ 如果需要外部 MTL/纹理，建议先导出成单文件 PLY 或 GLB。

## MoSE3 使用

1. 上传 RGB 视频（MP4/MOV/AVI/WebM/MKV）。浏览器预览推荐 H.264 MP4；实际读取由官方视频加载器完成。
2. 默认按 **10 FPS 采样、10 FPS 播放**：从 0 秒开始，每 0.1 秒选择时间上最近的原视频帧，覆盖完整视频。不会把 25/29.97 FPS 等视频简单按整数帧间隔跳帧。生成预览后显示采样帧数；查询帧索引对应这个 10 FPS 序列，界面同时显示时间和原视频帧号。
3. 需要处理布料等局部形变时，可启用“局部距离权重”。参数对应官方 `se3_dist_weight` 和 `se3_sigma_px`。
4. 查看官方轨迹、SE(3)、刚性分组视频和查询点。下载 `predictions.npz`、`config.json`、`result.json`、视频或完整 ZIP。

默认支持手动查询点：上传视频后点击“生成帧预览并选点”，拖动帧滑块或使用左右按钮选择采样帧，再直接点击图像添加查询点（最多 128 个）。支持撤销、清空和单独删除。帧号同时显示采样索引与原视频帧号；预览采用与模型相同的采样、中心裁剪和缩放，因此点击位置对应真实推理像素。

同一次推理的查询点位于同一帧。切换查询帧会清空已有点；改变视频、采样参数或中心裁剪后需要重新生成预览。运行后，轨迹和 SE(3) 可视化只展示所选点，并额外提供 `query_points.json` 和 `selected_tracks.npz`，后者包含这些点的逐帧轨迹、可见性和 SE(3) 数据。也可以切换到“自动网格”使用原有行为。

四个输出视频均固定为 10 FPS，页面播放和下载使用相同文件。采样帧数按 `原时长 × 10` 四舍五入，因此非 0.1 秒整数倍的视频，输出时长与原时长最多相差约 0.05 秒。低于 10 FPS 的输入会重复最近帧。结果 JSON 的 `sampling_fps`、`source_duration_seconds`、`output_duration_seconds`、`playback_fps` 记录采样与播放信息。旧结果仍保留原采样方式；使用新的 10 FPS 采样需重新生成预览并推理。

预测数据使用官方格式，世界坐标系为第一帧相机坐标，四元数顺序为 `wxyz`。单目预测不保证米制尺度。结果视频并排显示原始帧与预测可视化。

导出时，完整 `predictions.npz` 默认使用标准 NPZ 的无压缩存储，与视频绘制同时进行；数组值、精度、字段和 `np.load` 读取方式保持一致，文件体积会略增。需要节省磁盘时可设置 `DEMO_NPZ_COMPRESSION=1` 启用快速无损压缩。可视化直接读取内存中的预测，避免再次解压大型 NPZ。视频直接按目标帧率编码，保留全部采样帧。`result.json` 分别记录 `prediction_save_seconds`、`visualization_seconds`、`export_seconds`；推理耗时不再包含文件保存。安装 `threadpoolctl` 可限制 PCA 的 BLAS 线程数，避免多核服务器上的线程调度开销。

页面完成后即可预览和单独下载。完整 ZIP 在首次点击下载时生成，不再阻塞任务完成；已经压缩的 NPZ 和视频直接装入 ZIP，后续下载复用已有文件。

本机对同一份 97 帧、280×490、10 个手动查询点的预测做后处理对比：原流程保存约 111 秒、可视化约 33 秒、打包约 49 秒；新流程保存约 4.1 秒、可视化约 8.8 秒，并行后约 11.2 秒即可预览。按需打包约 4.8 秒。NPZ 体积增加约 16%，所有预测数组逐值一致，四个视频均为 97 帧、10 FPS、9.7 秒；这些耗时不含模型推理，实际速度随数据规模和硬件变化。

## 任务与文件

- 支持上传、运行日志、排队、取消、结果预览与下载；同一网页服务一次执行一个 GPU 任务。
- 分别启动两个独立服务时，任务队列也相互独立；需要限制显存占用时使用合并服务，或通过设备参数分配不同 GPU。
- 文件保存在 `runtime/uploads/`、`runtime/jobs/<任务 ID>/`；两个启动脚本分别使用 `runtime/wapr/` 和 `runtime/mose3/`。
- 服务不会自动删除用户上传或推理产物。服务重启后会恢复已有完整结果的成功任务；中断的任务不会自动重新推理，磁盘文件保留。
- 单文件上限 2 GiB。推理失败会保留日志以及已经生成的产物；不会生成模拟预测。取消会终止模型进程及其子进程。
- 这是可信本机使用的工具，没有用户账户或远程访问认证。若修改监听地址，应自行配置受控访问。

## 检查

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile server.py worker.py
# 使用已安装 numpy/opencv-python/trimesh 的模型环境测试输入与几何：
/path/to/model-env/bin/python -m unittest discover -s tests -p 'test_worker.py' -v
# 下载过官方样例后，通过实际 HTTP API 运行模型：
python3 tests/run_official_sample.py wapr --url http://127.0.0.1:7863
python3 tests/run_official_sample.py mose3 --url http://127.0.0.1:7862
```

WAPR 需要 OpenGL/EGL 开发头文件与 CUDA 编译器。WAPR 模型缓存默认为 `.models/wapr`，其余演示缓存位于 `runtime/cache/`；MoSE3 若发现 `runtime/checkpoints/MoSE3/model.safetensors`，优先使用已下载的本地权重。第一次 WAPR 推理会编译并缓存 OGL 扩展。

页面打包了 Noto Sans SC 字体子集，可离线显示中文；字体授权见 `static/fonts/OFL.txt`。

如需调整 CPU 线程数，可设置 `DEMO_TORCH_THREADS`，默认 8。不同机器上模型初始化耗时不同；结果中的耗时包含模型初始化，不表示纯 GPU 推理速度。

本工作区已在 NVIDIA H200 / PyTorch 2.10.0 + CUDA 12.8 上验证：WAPR 官方 `pose_lmo` 样例，以及 MoSE3 官方 `spin.mp4` 的 8 帧、最大边长 224 测试。已通过上传、参数与几何校验、浏览器交互、结果预览和下载检查。浏览器检查脚本为 `tests/browser_smoke.py`，需要单独安装 Playwright；网页服务本身无需这些测试依赖。完整样例记录保存在 `runtime/wapr_sample_job.json` 和 `runtime/mose3_sample_job.json`。

接口适配参考上述两个官方仓库。上游代码、第三方依赖及模型权重分别遵循其许可证；特别是 MoSE3 权重使用 CC BY-NC 4.0。
