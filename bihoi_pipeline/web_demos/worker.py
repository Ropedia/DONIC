"""Model adapters. Executed by server.py in the selected model environment."""
from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import time
import traceback


def require_cuda(device):
    import torch
    # Many-core hosts spend excessive time scheduling small checkpoint copies.
    torch.set_num_threads(max(1, int(os.environ.get("DEMO_TORCH_THREADS", "8"))))
    if not torch.cuda.is_available():
        raise RuntimeError("当前模型 Python 无法使用 CUDA。请检查 NVIDIA 驱动和 CUDA 版 PyTorch，或指定正确的模型 Python。")
    torch.cuda.set_device(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    return torch


def load_module(path, name):
    if not path.is_file():
        raise FileNotFoundError(f"缺少官方源码 {path}；请先按 README 安装，或指定 --mose3-repo。")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_wapr_inputs(config):
    import cv2
    import numpy as np
    import trimesh
    rgb_bgr = cv2.imread(config["rgb"], cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise ValueError("无法读取 RGB 图像")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    path = Path(config["depth"])
    depth = np.load(path, allow_pickle=False) if path.suffix == ".npy" else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2 or depth.shape != rgb.shape[:2]:
        raise ValueError("深度必须是单通道图像/数组，且尺寸必须与 RGB 一致；不自动缩放标定数据。")
    depth = depth.astype(np.float32) * config["depth_scale"]
    depth[~np.isfinite(depth) | (depth < 0)] = 0
    mask = None
    bbox = config.get("bbox")
    if bbox is None:
        mask = cv2.imread(config["mask"], cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != depth.shape:
            raise ValueError("Mask 必须与 RGB 尺寸一致")
        mask = (mask > 0).astype(np.uint8) * 255
        region = mask > 0
        if not region.any():
            raise ValueError("Mask 为空；请提供非零前景")
    else:
        x, y, w, h = bbox
        if x < 0 or y < 0 or w < 1 or h < 1 or x + w > rgb.shape[1] or y + h > rgb.shape[0]:
            raise ValueError("框选区域超出了原图范围")
        region = np.zeros(depth.shape, bool)
        region[y:y+h, x:x+w] = True
    if not (depth[region] > 0).any():
        raise ValueError("目标区域内没有有效深度；请检查深度单位和 Mask/框选位置")
    mesh = trimesh.load(config["mesh"], force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("需要包含三角面片的有效物体网格")
    mesh.apply_scale(config["mesh_scale"])
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("网格包含无效坐标")
    mesh.metadata["name"] = "object"
    K = np.array([[config["fx"], 0, config["cx"]],
                  [0, config["fy"], config["cy"]], [0, 0, 1]], np.float32)
    return rgb, depth, mesh, mask, bbox, K


def pose_overlay(rgb, mesh, pose, K, diameter):
    """Project the model's bounding box and object-coordinate axes onto RGB."""
    import cv2
    import numpy as np
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    corners = np.array(list(itertools.product(*zip(*mesh.bounds))), np.float32)
    axes = np.array([[0, 0, 0], [diameter * .35, 0, 0],
                     [0, diameter * .35, 0], [0, 0, diameter * .35]], np.float32)

    def project(points):
        cam = points @ pose[:3, :3].T + pose[:3, 3]
        p = cam @ K.T
        uv = p[:, :2] / np.maximum(p[:, 2:], 1e-8)
        valid = (cam[:, 2] > 1e-5) & np.isfinite(uv).all(axis=1)
        return np.clip(uv, -100000, 100000).astype(int), valid

    uv, valid = project(corners)
    for a, b in itertools.combinations(range(8), 2):
        if (corners[a] != corners[b]).sum() == 1 and valid[a] and valid[b]:
            cv2.line(image, tuple(uv[a]), tuple(uv[b]), (75, 235, 175), 2, cv2.LINE_AA)
    uv, valid = project(axes)
    for i, color in enumerate(((60, 60, 250), (70, 225, 90), (250, 160, 30)), 1):
        if valid[0] and valid[i]:
            cv2.arrowedLine(image, tuple(uv[0]), tuple(uv[i]), color, 3, cv2.LINE_AA, tipLength=.15)
            cv2.putText(image, "XYZ"[i-1], tuple(uv[i]), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    return image


def run_wapr(config):
    import cv2
    import numpy as np
    print("[1/4] 校验 RGB-D、目标区域和物体网格", flush=True)
    rgb, depth, mesh, mask, bbox, K = read_wapr_inputs(config)
    torch = require_cuda(config["device"])
    repo = Path(config["repo"])
    if (repo / "wapr").is_dir():
        sys.path.insert(0, str(repo))
    # EGL is suitable for a GPU server without an X11 display.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # Optional user-local OpenGL development headers, without changing system packages.
    native_prefix = Path(__file__).resolve().parent / ".native" / "usr"
    prefixes = []
    if native_prefix.is_dir():
        prefixes.append(str(native_prefix))
    try:
        import pybind11
        # The pybind11 CLI shell-quotes paths with spaces; CMake needs the raw path.
        prefixes.append(pybind11.get_cmake_dir())
    except ImportError:
        pass
    if prefixes:
        previous = os.environ.get("CMAKE_PREFIX_PATH", "")
        os.environ["CMAKE_PREFIX_PATH"] = os.pathsep.join(prefixes + ([previous] if previous else []))
    os.environ["WAPR_BACKEND"] = config.get("backend", "torch")
    from wapr import WAPREstimator, recipe
    recipe.visualize = False
    print("[2/4] 使用官方权重缓存；缺失时 WAPR 将自动下载", flush=True)
    print("[3/4] 加载模型，准备网格并估计 6D 位姿", flush=True)
    started = time.perf_counter()
    est = WAPREstimator(device=config["device"])
    pose_mesh = est.prepare_meshes([mesh])[0]
    out = est.estimate_one_category_one_instance(rgb, depth, K, pose_mesh, config["diameter_m"],
                                               bbox_xywh=bbox, mask=mask)
    torch.cuda.synchronize(config["device"])
    elapsed = time.perf_counter() - started
    pose = np.asarray(out["pose_4x4"], dtype=np.float64)
    score = float(out["score_6d"])
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.isfinite(score):
        raise RuntimeError("模型没有返回有效的位姿/分数")
    output = Path(config["output"])
    print("[4/4] 保存叠加图、位姿和相机坐标系网格", flush=True)
    if not cv2.imwrite(str(output / "overlay.png"), pose_overlay(rgb, mesh, pose, K, config["diameter_m"])):
        raise RuntimeError("无法保存叠加图")
    np.save(output / "pose.npy", pose)
    transformed = mesh.copy()
    transformed.apply_transform(pose)
    transformed.export(output / "mesh_camera.ply")
    result = {"model": "WAPR", "pose_4x4": pose.tolist(), "score_6d": score,
              "score_note": "排序分数，不是成功概率", "coordinate_frame": "object_to_camera (OpenCV +Z forward)",
              "translation_unit": "metre", "intrinsics": K.tolist(), "diameter_m": config["diameter_m"],
              "elapsed_seconds_including_setup": round(elapsed, 3)}
    return result


def run_mose3(config):
    import numpy as np
    repo = Path(config["repo"])
    sys.path.insert(0, str(repo))
    inference = load_module(repo / "inference.py", "mose3_demo_inference")
    torch = require_cuda(config["device"])
    device = torch.device(config["device"])
    all_frames = None
    if config.get("sample_all") or config.get("sampling_fps"):
        from mose3_points import load_video_frames
        all_frames, all_indices, original_size = load_video_frames(config)
        if len(all_frames) < 2 or config["query_idx"] >= len(all_frames):
            raise ValueError("查询帧超出视频范围，或视频不足两帧")
        if config.get("point_mode") == "manual" and all_indices != config["frame_indices"]:
            raise ValueError("完整视频帧与选点预览不一致，请重新生成预览")
        print(f"按 {config['sampling_fps']:g} FPS 采样，共 {len(all_frames)} 帧。" if config.get("sampling_fps") else f"完整读取 {len(all_frames)} 帧；不跳帧、不截断。",flush=True)
    print("[1/4] 加载 MoSE3 权重；首次使用 Hugging Face 模型 ID 时会下载", flush=True)
    started = time.perf_counter()
    model = inference.MoSE3.from_pretrained(config["checkpoint"], strict=True).to(device).eval()
    clip = inference.resolve_clip_config({
        "video_path": config["video"], "output_dir": config["output"],
        **{key: config[key] for key in ("num_frames", "start_frame", "frame_interval", "query_idx",
                                       "num_ref_pts", "se3_dist_weight", "se3_sigma_px", "central_crop", "fps")},
        "max_w": config["resolution"], "max_h": config["resolution"],
    })
    if config.get("point_mode") == "manual":
        clip["frame_indices"] = config["frame_indices"]
        # The upstream explicit-index loader requires num_frames to be the actual count.
        clip["num_frames"] = len(config["frame_indices"])
    print("[2/4] 视频采样、稠密 3D 轨迹预测和逐像素 SE(3) 拟合", flush=True)
    original_loader = inference.load_video_clip
    if all_frames is not None:
        clip.update(num_frames=len(all_frames), start_frame=0, frame_interval=1, frame_indices=all_indices)
        inference.load_video_clip = lambda *args, **kwargs: (all_frames, all_indices, original_size)
    try:
        payload = inference.run_clip(model, clip, device)
    finally:
        inference.load_video_clip = original_loader
    del all_frames
    inference_seconds = time.perf_counter() - started
    frames, height, width = payload["images"].shape[:3]
    result = {"model": "MoSE3", "frames": frames, "width": width, "height": height,
              "query_idx": int(payload["query_idx"]), "frame_indices": clip.get("frame_indices", []),
              "sample_all": bool(config.get("sample_all")),
              "sampling_fps": config.get("sampling_fps"),
              "coordinate_frame": "first camera frame", "quaternion_order": "wxyz",
              "scale_note": "单目预测尺度；不保证米制尺度",
              "visible_fraction": float(np.mean(payload["visibility"] > 0)),
              "valid_se3_fraction": float(np.mean(payload["se3_valid"])),
              "inference_seconds_including_setup": round(inference_seconds, 3)}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    result.update(export_mose3(config, clip, payload))
    return result


def export_mose3(config, clip, payload):
    """Export once from CPU memory, overlapping lossless compression and rendering."""
    from mose3_export import save_predictions, save_video
    from mose3_points import render_selected_points
    output = Path(config["output"])
    started = time.perf_counter()
    (output / "config.json").write_text(json.dumps(clip, indent=2))
    print("[3/4] 无损保存完整 predictions.npz，同时准备视频", flush=True)
    result = {"point_mode": config.get("point_mode", "grid")}
    with ThreadPoolExecutor(max_workers=1) as pool:
        saved = pool.submit(save_predictions, output / "predictions.npz", payload)
        save_video(output / "video.mp4", payload["images"], clip["fps"], quality=8)
        print("[4/4] 复用内存预测，使用官方绘制函数生成结果视频", flush=True)
        render_started = time.perf_counter()
        selection = render_selected_points({**config, "fps":clip["fps"]}, payload, clip.get("frame_indices"))
        result["visualization_seconds"] = round(time.perf_counter() - render_started, 3)
        if selection is not None:
            result.update(query_points_xy=selection["pixel_xy"], source_query_frame=selection["source_query_frame"],
                          selected_point_count=len(selection["pixel_xy"]))
        if not saved.done():
            print("  视频已生成，正在完成完整预测数据的无损保存。", flush=True)
        result["prediction_save_seconds"] = round(saved.result(), 3)
    for name in ("tracks.mp4", "se3.mp4", "rigidity.mp4", "query_points.png"):
        if not (output / name).is_file():
            raise RuntimeError(f"官方可视化未生成 {name}")
    from video_timing import align_result_videos
    result.update(align_result_videos(config, len(payload["images"])))
    result["export_seconds"] = round(time.perf_counter() - started, 3)
    print(f"  结果导出完成（{result['export_seconds']:.1f} 秒），视频时长已校验。", flush=True)
    return result


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        from mose3_points import prepare_preview
        prepare_preview(json.loads(Path(sys.argv[2]).read_text()))
        return
    config = json.loads(Path(sys.argv[1]).read_text())
    output = Path(config["output"])
    result = (run_wapr if config["app"] == "wapr" else run_mose3)(config)
    (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    if config["app"] == "wapr":
        from result_archive import build_result_archive
        build_result_archive(output)
    print("完成：全部结果已保存，可在页面预览和下载。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        print(f"缺少模型依赖 {exc.name}。请按 README 安装到对应的模型环境，并检查服务的 Python 路径。", flush=True)
        traceback.print_exc()
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
