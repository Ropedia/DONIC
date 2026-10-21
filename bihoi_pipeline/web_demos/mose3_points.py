"""Exact sampled-frame previews and manually selected MoSE3 trajectories."""
import importlib.util
import json
from pathlib import Path


SAMPLE_KEYS = ("num_frames", "start_frame", "frame_interval", "resolution", "central_crop", "sample_all", "sampling_fps")


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法读取官方模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_video_frames(config):
    import cv2
    import numpy as np
    io = load_source(Path(config["repo"]) / "mose3/utils/video_io.py", "mose3_preview_io")
    if config.get("sampling_fps"):
        return sample_video_at_fps(config, io)
    if not config.get("sample_all"):
        return io.load_video_clip(
            config["video"], num_frames=config["num_frames"], start_frame=config["start_frame"],
            frame_interval=config["frame_interval"], max_w=config["resolution"],
            max_h=config["resolution"], central_crop=config["central_crop"])
    # Decode until EOF instead of trusting codec frame-count metadata. Use exactly
    # the upstream crop and resize geometry, with no skipping or frame-count cap.
    capture = cv2.VideoCapture(config["video"])
    if not capture.isOpened():
        raise ValueError("无法读取视频")
    frames = []
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if not frames:
                original_size = (image.shape[1], image.shape[0])
                crop = io._central_square_crop_bounds(*original_size) if config["central_crop"] else None
                width, height = (crop[2]-crop[0],crop[3]-crop[1]) if crop else original_size
                target = io._target_size(width,height,config["resolution"],config["resolution"])
            if (image.shape[1],image.shape[0]) != original_size:
                raise ValueError("视频帧分辨率发生变化")
            if crop:
                x0,y0,x1,y1=crop
                image=image[y0:y1,x0:x1]
            if (image.shape[1],image.shape[0]) != target:
                image=cv2.resize(image,target,interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError("视频没有可解码帧")
    return np.stack(frames),list(range(len(frames))),original_size


def sample_video_at_fps(config, io):
    """Select nearest decoded frames on a 0.1-second clock, not a rounded stride."""
    import cv2
    import numpy as np
    from video_timing import probe_video
    rate = float(config["sampling_fps"])
    duration, _ = probe_video(config["video"])
    count = max(1, int(np.floor(duration * rate + 0.5)))
    capture = cv2.VideoCapture(config["video"])
    if not capture.isOpened():
        raise ValueError("无法读取视频")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frames, indices = [], []
    previous = None
    index, last_time = 0, -1.0

    def resized(frame):
        if crop:
            x0,y0,x1,y1=crop
            frame=frame[y0:y1,x0:x1]
        if (frame.shape[1],frame.shape[0]) != target:
            frame=cv2.resize(frame,target,interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index == 0:
                original_size = (frame.shape[1],frame.shape[0])
                crop = io._central_square_crop_bounds(*original_size) if config["central_crop"] else None
                width,height=(crop[2]-crop[0],crop[3]-crop[1]) if crop else original_size
                target=io._target_size(width,height,config["resolution"],config["resolution"])
                origin = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 - origin
            if not np.isfinite(timestamp) or (index and timestamp <= last_time):
                if not np.isfinite(source_fps) or source_fps <= 0:
                    raise ValueError("无法读取视频时间戳或帧率")
                timestamp=index/source_fps
            while len(frames) < count and len(frames)/rate <= timestamp + 1e-8:
                moment=len(frames)/rate
                if previous is not None and moment-last_time <= timestamp-moment:
                    chosen, chosen_index = previous, index-1
                else:
                    chosen, chosen_index = frame, index
                frames.append(resized(chosen))
                indices.append(chosen_index)
            previous, last_time = frame, timestamp
            index+=1
        if previous is None:
            raise ValueError("视频没有可解码帧")
        while len(frames) < count:
            frames.append(resized(previous))
            indices.append(index-1)
    finally:
        capture.release()
    return np.stack(frames), indices, original_size


def prepare_preview(config):
    import cv2
    frames, indices, original_size = load_video_frames(config)
    if len(frames) < 2:
        raise ValueError("实际采样少于两帧，请调整起始帧或采样间隔")
    output = Path(config["output"])
    for i, frame in enumerate(frames):
        if not cv2.imwrite(str(output / f"frame_{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            raise ValueError("无法写入预览帧")
    manifest = {key: config.get(key,None if key == "sampling_fps" else False) for key in SAMPLE_KEYS}
    manifest.update(video=config["video"], width=int(frames.shape[2]), height=int(frames.shape[1]),
                    frame_indices=indices, original_size=list(original_size))
    (output / "preview.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def pixel_points(points, width, height):
    """Normalized canvas coordinates -> exact processed-frame (x, y) pixels."""
    import numpy as np
    normalized = np.asarray(points, dtype=np.float64)
    if normalized.ndim != 2 or normalized.shape[1] != 2 or not len(normalized):
        raise ValueError("请至少选择一个查询点")
    if not np.isfinite(normalized).all() or (normalized < 0).any() or (normalized > 1).any():
        raise ValueError("查询点坐标超出图像范围")
    return np.minimum(np.floor(normalized * [width, height]).astype(np.int32), [width-1, height-1])


def render_selected_points(config, payload=None, frame_indices=None):
    """Use the official renderers on in-memory predictions; also support offline export."""
    import imageio.v2 as imageio
    import numpy as np
    import time
    from contextlib import nullcontext
    from mose3_export import save_video
    try:
        from threadpoolctl import threadpool_limits
        limits = threadpool_limits(limits=4, user_api="blas")
    except ImportError:
        limits = nullcontext()
    output = Path(config["output"])
    viz = load_source(Path(config["repo"]) / "visualize.py", "mose3_selected_visualize")
    if payload is None:
        # Load each member once for offline regeneration, too.
        with np.load(output / "predictions.npz", allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
    with limits:
        z = payload
        images, tracks, poses = z["images"], z["tracks"], z["camera_poses"]
        K, visibility, q = z["intrinsics"], z["visibility"], int(z["query_idx"])
        _, height, width, _ = images.shape
        selection = None
        edge = None
        if config.get("point_mode") == "manual":
            xy = pixel_points(config["query_points"], width, height)
            pts = xy[:, ::-1]  # Official drawing helpers use (row, column).
            if frame_indices is None:
                frame_indices = json.loads((output / "config.json").read_text())["frame_indices"]
            if frame_indices != config["frame_indices"]:
                raise ValueError("推理帧与选点预览不一致，请重新生成预览")
            selection = {"query_idx":q, "source_query_frame":frame_indices[q],
                         "normalized_xy":config["query_points"], "pixel_xy":xy.tolist(),
                         "width":width, "height":height, "source_frame_indices":frame_indices}
            (output / "query_points.json").write_text(json.dumps(selection, indent=2))
            rows, cols = pts[:, 0], pts[:, 1]
            np.savez_compressed(output / "selected_tracks.npz", query_points_xy=xy, query_idx=q,
                                source_frame_indices=frame_indices, tracks=tracks[:, rows, cols],
                                visibility=visibility[:, rows, cols], se3_quat=z["se3_quat"][:, rows, cols],
                                se3_trans=z["se3_trans"][:, rows, cols], se3_valid=z["se3_valid"][:, rows, cols],
                                camera_poses=poses, intrinsics=K)
        else:
            pts = viz.grid_points(height, width, config["stride"])
            edge = viz.depth_edges(z, .05)
            keep = ~viz._grow(edge, 4)
            pts = pts[keep[pts[:, 0], pts[:, 1]]]
        colors = viz.point_colors(pts, height, width)
        imageio.imwrite(output / "query_points.png", viz.draw_points(images[q], pts, colors))
        # Keep manual clicks, including depth edges. Grid mode retains the official edge filter.
        renderers = (
            ("tracks.mp4", lambda: viz.render_tracks(images, tracks, poses, K, visibility, pts, colors)),
            ("se3.mp4", lambda: viz.render_se3_axes(images, tracks, z["se3_quat"], z["se3_trans"],
                           z["se3_valid"], visibility, poses, K, q, pts, colors, edge=edge)),
            ("rigidity.mp4", lambda: viz.pca_rgb(z["rigidity_emb"])),
        )
        for name, render in renderers:
            started = time.perf_counter()
            save_video(output / name, images, config["fps"], rendered=render())
            print(f"  {name} 已生成（{time.perf_counter() - started:.1f} 秒）", flush=True)
        return selection
