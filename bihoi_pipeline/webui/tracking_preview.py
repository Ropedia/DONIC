#!/usr/bin/env python3
"""Latest-frame mesh and fused multi-view masked depth preview, outside pose inference.

Requests overwrite one small JSON file. A slow renderer skips stale preview
requests, never inference frames. Images and their metadata are published
atomically so a browser never reads a half-written JPEG.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


try:
    from .settings import PIPELINE_ROOT, PROJECT_ROOT, MODEL_ROOT, python_for
except ImportError:
    from settings import PIPELINE_ROOT, PROJECT_ROOT, MODEL_ROOT, python_for

ROOT = PIPELINE_ROOT
PREVIEW_PYTHON = python_for("foundationpose")


def overlay_depth_points(rgb, points, K, spacing=7, max_points=450, opacity=.35):
    """Project sparse red depth markers with a light, color-preserving tint."""
    import cv2
    import numpy as np

    points = np.asarray(points)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
    projected = points @ np.asarray(K).T
    valid = np.isfinite(projected).all(axis=1) & (projected[:, 2] > 0)
    projected = projected[valid]
    uv = projected[:, :2] / projected[:, 2:]
    height, width = rgb.shape[:2]
    # Filter before converting to integers, including extremely distant projections.
    uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) &
            (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)]
    pixels = np.rint(uv).astype(np.int32)
    # One point per screen cell keeps the green mesh visible even for dense clouds.
    _, ids = np.unique(pixels // spacing, axis=0, return_index=True)
    if len(ids) > max_points:
        ids = ids[np.linspace(0, len(ids) - 1, max_points, dtype=int)]
    coverage = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for x, y in pixels[ids]:
        cv2.circle(coverage, (int(x), int(y)), 1, 255, -1, cv2.LINE_8)
    selected = coverage > 0
    # Neutralize only marker pixels before tinting: ordinary alpha blending
    # over the green mesh turns a faint red glyph brown/green. Preserve local
    # brightness while retaining the cloud's red hue even inside the mesh.
    neutral = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[selected].astype(np.float32)
    rgb[selected] = np.rint(neutral[:, None] * (1 - opacity) +
                            np.array([255., 35., 35.]) * opacity).astype(np.uint8)
    return rgb


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data) + "\n")
    os.replace(temp, path)


class PreviewPublisher:
    def __init__(self, args):
        self.scene = args.scene_dir.resolve()
        self.output = (args.output_dir or self.scene / "debug_depth") / "live_tracking"
        self.mesh = (args.mesh_file or self.scene / "meshes/compressed.ply").resolve()
        self.process = None
        self.log = None
        self.request = dict(session=uuid.uuid4().hex, parent_pid=os.getpid(),
                            scene=str(self.scene), mesh=str(self.mesh),
                            fps=args.fps, depth_scale=args.depth_scale, width=args.width, height=args.height,
                            min_mask_area=args.min_mask_area, io_workers=args.io_workers,
                            cloud_outlier_neighbors=args.cloud_outlier_neighbors,
                            cloud_outlier_std_ratio=args.cloud_outlier_std_ratio,
                            no_cloud_outlier_removal=args.no_cloud_outlier_removal,
                            started=time.time(), finished=False)

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        for name in ("latest.json", "error.json"):
            (self.output / name).unlink(missing_ok=True)
        write_json(self.output / "request.json", self.request)
        try:
            self.log = (self.output / "renderer.log").open("w")
            env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            self.process = subprocess.Popen(
                [str(PREVIEW_PYTHON), "-u", str(Path(__file__).resolve()),
                 "--request", str((self.output / "request.json").resolve()),
                 "--session", self.request["session"]],
                env=env, stdout=self.log, stderr=subprocess.STDOUT)
        except OSError as exc:
            logging.warning("Live preview unavailable: %s", exc)
        return self

    def publish(self, frame, pose, cameras, completed, total, mode):
        self.request.update(frame=frame, pose=pose.tolist(), cameras=cameras[:6],
                            completed=completed, total=total, mode=mode, updated=time.time())
        try:
            write_json(self.output / "request.json", self.request)
        except OSError as exc:
            logging.warning("Could not publish live preview: %s", exc)

    def __exit__(self, exc_type, exc, tb):
        self.request.update(finished=True, failed=exc_type is not None)
        try:
            try:
                write_json(self.output / "request.json", self.request)
            except OSError as error:
                logging.warning("Could not finalize live preview: %s", error)
            if self.process is not None:
                try:
                    self.process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
        finally:
            if self.log is not None:
                self.log.close()


class Renderer:
    def __init__(self, request):
        import cv2
        import numpy as np
        import torch
        import nvdiffrast.torch as dr

        sys.path.insert(0, str(ROOT / "FoundationPose"))
        from render_allcams_mesh_grid_fast_green import load_mesh
        from Utils import make_mesh_tensors
        try:
            from .wapr_tracking import Capture, build_parser
        except ImportError:
            from wapr_tracking import Capture, build_parser
        cv2.setNumThreads(1)
        torch.set_num_threads(2)
        self.scene = Path(request["scene"])
        args = build_parser().parse_args(["--scene_dir", str(self.scene)])
        for key in ("width", "height", "depth_scale", "min_mask_area", "io_workers",
                    "cloud_outlier_neighbors", "cloud_outlier_std_ratio", "no_cloud_outlier_removal"):
            if key in request:
                setattr(args, key, request[key])
        self.capture = Capture(args)
        self.extrinsics = self.capture.extrinsics
        self.Ks = self.capture.Ks
        self.mesh = make_mesh_tensors(load_mesh(request["mesh"]), device="cuda")
        self.context = dr.RasterizeCudaContext()

    def render(self, request):
        import cv2
        import numpy as np
        import torch
        from Utils import nvdiffrast_render
        from render_allcams_mesh_grid_fast_green import find_frame_image, scale_K, make_grid, draw_label_rgb

        views = []
        pose = np.asarray(request["pose"], np.float32)
        points_world, cloud_cameras, self.cloud_stats = self.capture.fused_depth_cloud(request["frame"])
        self.cloud_stats["cameras"] = cloud_cameras
        # Bound projection cost independently of the full-density init centroid.
        # The same world-space subset is projected into every preview camera.
        if len(points_world) > 24000:
            points_world = points_world[np.linspace(0, len(points_world) - 1, 24000, dtype=int)]
        for cam in request["cameras"] or list(self.extrinsics)[:6]:
            path = find_frame_image(str(self.scene / "images"), cam, request["frame"])
            bgr = cv2.imread(path) if path else None
            if bgr is None:
                continue
            height, width = bgr.shape[:2]
            rgb = cv2.cvtColor(cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            K = scale_K(self.Ks[cam], 320 / width, 180 / height)
            with torch.inference_mode():
                color, depth, _ = nvdiffrast_render(
                    K=K, H=180, W=320,
                    ob_in_cams=torch.as_tensor((self.extrinsics[cam] @ pose)[None], dtype=torch.float32, device="cuda"),
                    glctx=self.context, context="cuda", mesh_tensors=self.mesh,
                    output_size=np.array([180, 320]), use_light=False)
            foreground = color[0].clamp(0, 1).cpu().numpy() * 255
            valid = depth[0].cpu().numpy() > 1e-6
            rgb[valid] = (rgb[valid] * .35 + foreground[valid] * .65).astype(np.uint8)
            ext = self.extrinsics[cam]
            points = points_world @ ext[:3, :3].T + ext[:3, 3]
            overlay_depth_points(rgb, points, K)
            # Draw the mesh silhouette last so even overlapping depth points
            # cannot obscure the tracked pose. Include holes such as handles.
            contours, _ = cv2.findContours(valid.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, (12, 40, 18), 4, cv2.LINE_AA)
            cv2.drawContours(rgb, contours, -1, (70, 255, 105), 2, cv2.LINE_AA)
            label = f"cam {cam} | {request['frame']} | {request['mode']}"
            if not len(points):
                label += " | no depth"
            views.append(draw_label_rgb(rgb, label))
        if not views:
            raise ValueError("No camera images available for the latest tracking frame")
        grid = make_grid(views, cols=3)
        legend = np.full((28, grid.shape[1], 3), 24, dtype=np.uint8)
        cv2.putText(legend, "Green: tracked mesh", (10, 19), cv2.FONT_HERSHEY_SIMPLEX, .48, (70, 255, 70), 1, cv2.LINE_AA)
        filter_label = "SOR" if self.cloud_stats["status"] == "filtered" else self.cloud_stats["status"]
        label = (f"Red: fused cloud ({len(cloud_cameras)} views) | {filter_label}: "
                 f"{self.cloud_stats['kept_points']}/{self.cloud_stats['raw_points']} pts")
        cv2.putText(legend, label, (205, 19), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 70, 70), 1, cv2.LINE_AA)
        return np.concatenate((legend, grid), axis=0)


def serve(request_path, session):
    import cv2
    renderer, last_frame = None, None
    output = request_path.parent
    while True:
        request = json.loads(request_path.read_text())
        if request["session"] != session:
            return
        if renderer is None:
            renderer = Renderer(request)
        frame = request.get("frame")
        if frame is not None and frame != last_frame:
            try:
                rgb = renderer.render(request)
                filename = f"{session}-{frame}.jpg"
                ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    raise ValueError("JPEG encoding failed")
                temp = output / (filename + ".tmp")
                temp.write_bytes(encoded.tobytes())
                os.replace(temp, output / filename)
                write_json(output / "latest.json", dict(request, image=filename, rendered=time.time(), depth_cloud=renderer.cloud_stats,
                                                       visualization_style="light_red_cloud_emphasized_pose"))
                last_frame = frame
                # Retain a few recent images for concurrent browser reads.
                old = sorted(output.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
                for path in old[4:]:
                    path.unlink(missing_ok=True)
                print(f"Preview {frame}: {request['completed']}/{request['total']}", flush=True)
            except Exception as exc:
                logging.exception("Preview rendering failed")
                write_json(output / "error.json", dict(session=session, error=str(exc)))
                if request.get("finished"):
                    return
        # Reread after a slow render: the producer may have advanced or finished.
        latest = json.loads(request_path.read_text())
        if latest["session"] != session:
            return
        if latest.get("finished") and latest.get("frame") == last_frame:
            return
        # A replacement preview worker can outlive the producer's original
        # subprocess handle. Always consume a published final frame before exit.
        if latest.get("finished"):
            continue
        try:
            os.kill(request["parent_pid"], 0)
        except ProcessLookupError:
            return
        time.sleep(.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    try:
        serve(args.request, args.session)
    except Exception as exc:
        write_json(args.request.parent / "error.json", dict(session=args.session, error=str(exc)))
        raise
