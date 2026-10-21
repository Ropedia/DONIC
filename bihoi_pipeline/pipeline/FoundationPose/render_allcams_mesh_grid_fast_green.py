#!/usr/bin/env python3
"""
Faster all-camera FoundationPose mesh-overlay grid renderer.

Main speed improvements over the previous version:
1) Render each camera directly at a SMALL overlay resolution (default 256x144),
   instead of rendering 640x360 and then shrinking again for the grid.
2) By default, ONLY save the MP4 (no per-frame PNGs).
3) Use uniform GREEN mesh color by default.
4) Optional max_frames / frame range for quick testing.

Pose convention:
    pose_world = ORIGINAL_MESH -> WORLD
    extri.yml  = WORLD -> CAMERA
    pose_cam   = T_cam_from_world @ pose_world
"""

import argparse
import glob
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

try:
    import nvdiffrast.torch as dr
except Exception as e:
    raise ImportError("Please run inside the FoundationPose environment with nvdiffrast.") from e

try:
    from Utils import make_mesh_tensors, nvdiffrast_render
except Exception as e:
    raise ImportError(
        "Could not import FoundationPose Utils.py. Run from the FoundationPose repo root."
    ) from e


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def natural_key(s):
    s = str(s)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def find_frame_image(images_root, cam, frame_id):
    folder = os.path.join(images_root, str(cam))
    for ext in IMAGE_EXTS:
        p = os.path.join(folder, f"{frame_id}{ext}")
        if os.path.exists(p):
            return p
    cands = []
    for p in glob.glob(os.path.join(folder, f"{frame_id}.*")):
        if os.path.splitext(p)[1].lower() in IMAGE_EXTS:
            cands.append(p)
    return sorted(cands)[0] if cands else None


def read_extrinsics(extri_path):
    if not os.path.isfile(extri_path):
        raise FileNotFoundError(f"Cannot find extri.yml: {extri_path}")

    fs = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open {extri_path}")

    names_node = fs.getNode("names")
    if names_node.empty():
        fs.release()
        raise ValueError("No 'names' node in extri.yml")

    names = [names_node.at(i).string() for i in range(names_node.size())]
    ext_by_cam = {}

    for cam in names:
        rot = fs.getNode(f"Rot_{cam}").mat()
        t = fs.getNode(f"T_{cam}").mat()
        if rot is None or t is None:
            print(f"[WARN] skip cam {cam}: missing Rot/T")
            continue
        ext = np.eye(4, dtype=np.float64)
        ext[:3, :3] = np.asarray(rot, dtype=np.float64).reshape(3, 3)
        ext[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
        ext_by_cam[str(cam)] = ext

    fs.release()
    cams = sorted([str(x) for x in names if str(x) in ext_by_cam], key=natural_key)
    return cams, ext_by_cam


def read_cam_k_txt(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find {path}")
    vals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals.extend(float(x) for x in line.replace(",", " ").split())
    if len(vals) != 9:
        raise ValueError(f"Expected 9 numbers in {path}, got {len(vals)}")
    return np.asarray(vals, dtype=np.float64).reshape(3, 3)


def auto_find_pose_dir(data_dir):
    candidates = []
    for d in glob.glob(os.path.join(data_dir, "**", "geometry_global_opt/ob_in_world"), recursive=True):
        txts = glob.glob(os.path.join(d, "*.txt"))
        if txts:
            newest = max(os.path.getmtime(p) for p in txts)
            candidates.append((newest, len(txts), os.path.abspath(d)))
    if not candidates:
        raise FileNotFoundError(f"No ob_in_world directory found under {data_dir}")
    candidates.sort(reverse=True)
    chosen = candidates[0][2]
    print(f"[pose] auto-selected: {chosen}")
    return chosen


def auto_find_mesh(data_dir):
    preferred = [
        os.path.join(data_dir, "meshes", "compressed.ply"),
        os.path.join(data_dir, "meshes", "fuse_post.ply"),
    ]
    for p in preferred:
        if os.path.isfile(p):
            return p
    cands = sorted(glob.glob(os.path.join(data_dir, "meshes", "*.ply")), key=natural_key)
    if cands:
        return cands[0]
    raise FileNotFoundError(f"No .ply mesh found in {os.path.join(data_dir, 'meshes')}")


def load_mesh(mesh_file, uniform_color_rgb=(0, 255, 0)):
    mesh = trimesh.load(mesh_file, force="mesh", process=False, maintain_order=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(mesh)}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Empty mesh: {mesh_file}")

    if uniform_color_rgb is not None:
        rgb = np.asarray(uniform_color_rgb, dtype=np.uint8).reshape(1, 3)
        rgba = np.concatenate(
            [np.tile(rgb, (len(mesh.vertices), 1)),
             np.full((len(mesh.vertices), 1), 255, dtype=np.uint8)],
            axis=1,
        )
        mesh.visual.vertex_colors = rgba
    return mesh


def resize_rgb(img_rgb, size_wh):
    w, h = size_wh
    if img_rgb.shape[1] == w and img_rgb.shape[0] == h:
        return img_rgb
    return cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)


def scale_K(K, sx, sy):
    K2 = np.asarray(K, dtype=np.float64).copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2


def draw_label_rgb(img, text, x=6, y=18):
    out = img.copy()
    cv2.putText(out, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_grid(images, cols=6, pad=2):
    if not images:
        return None
    cell_h = max(im.shape[0] for im in images)
    cell_w = max(im.shape[1] for im in images)
    cols = max(1, int(cols))
    rows = int(math.ceil(len(images) / cols))
    grid_h = rows * cell_h + (rows + 1) * pad
    grid_w = cols * cell_w + (cols + 1) * pad
    if grid_h % 2:
        grid_h += 1
    if grid_w % 2:
        grid_w += 1
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for i, im in enumerate(images):
        r = i // cols
        c = i % cols
        y0 = pad + r * (cell_h + pad)
        x0 = pad + c * (cell_w + pad)
        h, w = im.shape[:2]
        grid[y0:y0+h, x0:x0+w] = im
    return grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--images_dir", default=None)
    p.add_argument("--mesh_file", default=None)
    p.add_argument("--pose_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--output_video", default=None)

    p.add_argument("--render_w", type=int, default=256, help="Per-camera render width. Smaller = faster.")
    p.add_argument("--render_h", type=int, default=144, help="Per-camera render height. Smaller = faster.")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--alpha", type=float, default=0.40)
    p.add_argument("--grid_cols", type=int, default=6)
    p.add_argument("--pad", type=int, default=2)
    p.add_argument("--save_frames", type=int, choices=[0, 1], default=0, help="Default 0 for speed.")
    p.add_argument("--use_light", type=int, choices=[0, 1], default=0, help="Default 0 for speed.")
    p.add_argument("--outline", type=int, choices=[0, 1], default=1)
    p.add_argument("--outline_color", nargs=3, type=int, default=[0, 255, 0], metavar=("R", "G", "B"))
    p.add_argument("--green", type=int, choices=[0, 1], default=1, help="Default 1: render mesh as green.")
    p.add_argument("--uniform_color", nargs=3, type=int, default=[0, 255, 0], metavar=("R", "G", "B"))
    p.add_argument("--start", type=int, default=0, help="Start index in sorted pose files.")
    p.add_argument("--end", type=int, default=-1, help="Exclusive end index. -1 means all.")
    p.add_argument("--max_frames", type=int, default=-1, help="Optional cap after start/end filtering.")
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    cv2.setNumThreads(1)

    data_dir = os.path.abspath(args.data_dir)
    images_dir = os.path.abspath(args.images_dir) if args.images_dir else os.path.join(data_dir, "images")
    mesh_file = os.path.abspath(args.mesh_file) if args.mesh_file else auto_find_mesh(data_dir)
    pose_dir = os.path.abspath(args.pose_dir) if args.pose_dir else auto_find_pose_dir(data_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(data_dir, "mesh_overlay_allcams_fast_green")
    output_video = os.path.abspath(args.output_video) if args.output_video else os.path.join(output_dir, "mesh_overlay_allcams_fast_green.mp4")
    frames_dir = os.path.join(output_dir, "grid_frames")
    os.makedirs(output_dir, exist_ok=True)
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    cams, ext_by_cam = read_extrinsics(os.path.join(data_dir, "extri.yml"))
    K_input = read_cam_k_txt(os.path.join(data_dir, "cam_K.txt"))

    color = tuple(args.uniform_color) if args.green else None
    mesh = load_mesh(mesh_file, uniform_color_rgb=color)
    mesh_tensors = make_mesh_tensors(mesh, device="cuda")
    glctx = dr.RasterizeCudaContext()

    pose_files = sorted(glob.glob(os.path.join(pose_dir, "*.txt")), key=lambda p: natural_key(Path(p).stem))
    if not pose_files:
        raise FileNotFoundError(f"No pose txt files found in {pose_dir}")

    end_idx = len(pose_files) if args.end < 0 else min(len(pose_files), args.end)
    pose_files = pose_files[max(0, args.start):end_idx]
    if args.max_frames > 0:
        pose_files = pose_files[:args.max_frames]

    if not pose_files:
        raise ValueError("No frames selected after start/end/max_frames filtering.")

    render_w = int(args.render_w)
    render_h = int(args.render_h)
    alpha = float(np.clip(args.alpha, 0.0, 1.0))

    # Probe the first available image to scale K correctly to render size.
    probe_path = None
    for pose_path in pose_files:
        frame_id = Path(pose_path).stem
        for cam in cams:
            probe_path = find_frame_image(images_dir, cam, frame_id)
            if probe_path is not None:
                break
        if probe_path is not None:
            break
    if probe_path is None:
        raise FileNotFoundError(f"No input images found under {images_dir}")

    probe_bgr = cv2.imread(probe_path, cv2.IMREAD_COLOR)
    if probe_bgr is None:
        raise RuntimeError(f"Failed to read probe image {probe_path}")
    input_h, input_w = probe_bgr.shape[:2]
    sx = render_w / float(input_w)
    sy = render_h / float(input_h)
    K_render = scale_K(K_input, sx, sy)

    print(f"[data]       {data_dir}")
    print(f"[images]     {images_dir}")
    print(f"[mesh]       {mesh_file}")
    print(f"[pose_dir]   {pose_dir}")
    print(f"[frames]     {len(pose_files)}")
    print(f"[input]      {input_w}x{input_h}")
    print(f"[render]     {render_w}x{render_h} per cam")
    print(f"[grid cols]  {args.grid_cols}")
    print(f"[alpha]      {alpha}")
    print(f"[green mesh] {bool(args.green)}")
    print(f"[save pngs]  {bool(args.save_frames)}")
    print(f"[output]     {output_video}")

    writer = None
    rendered = 0
    skipped = 0

    for idx, pose_path in enumerate(pose_files):
        frame_id = Path(pose_path).stem
        try:
            pose_world = np.loadtxt(pose_path, dtype=np.float64).reshape(4, 4)
        except Exception as e:
            print(f"[WARN] frame={frame_id}: bad pose: {e}")
            skipped += 1
            continue

        backgrounds = []
        poses_cam = []
        used_cams = []

        for cam in cams:
            img_path = find_frame_image(images_dir, cam, frame_id)
            if img_path is None:
                continue
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = resize_rgb(rgb, (render_w, render_h))

            pose_cam = ext_by_cam[cam] @ pose_world
            backgrounds.append(rgb)
            poses_cam.append(pose_cam.astype(np.float32))
            used_cams.append(cam)

        if not poses_cam:
            print(f"[WARN] frame={frame_id}: no views found")
            skipped += 1
            continue

        poses_t = torch.as_tensor(np.stack(poses_cam, axis=0), dtype=torch.float32, device="cuda")

        with torch.inference_mode():
            color_t, depth_t, _ = nvdiffrast_render(
                K=K_render,
                H=render_h,
                W=render_w,
                ob_in_cams=poses_t,
                glctx=glctx,
                context="cuda",
                get_normal=False,
                mesh_tensors=mesh_tensors,
                mesh=None,
                output_size=np.asarray([render_h, render_w]),
                use_light=bool(args.use_light),
                light_dir=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
                w_ambient=0.8,
                w_diffuse=0.5,
            )

        render_rgb = (color_t.clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
        depth = depth_t.detach().cpu().numpy()

        views = []
        outline_color = tuple(int(x) for x in args.outline_color)
        for j, cam in enumerate(used_cams):
            bg = backgrounds[j].astype(np.float32)
            fg = render_rgb[j].astype(np.float32)
            valid = np.isfinite(depth[j]) & (depth[j] > 1e-6)
            a = (valid.astype(np.float32) * alpha)[..., None]
            blended = np.clip(bg * (1.0 - a) + fg * a, 0, 255).astype(np.uint8)

            if args.outline:
                mask_u8 = (valid.astype(np.uint8) * 255)
                cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(blended, cnts, -1, outline_color, 1, cv2.LINE_AA)

            blended = draw_label_rgb(blended, f"cam {cam} | {frame_id}")
            views.append(blended)

        grid = make_grid(views, cols=args.grid_cols, pad=args.pad)
        if grid is None:
            skipped += 1
            continue

        if args.save_frames:
            cv2.imwrite(os.path.join(frames_dir, f"{frame_id}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

        if writer is None:
            gh, gw = grid.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video, fourcc, float(args.fps), (gw, gh))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {output_video}")
            print(f"[video] grid size = {gw}x{gh}")

        writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        rendered += 1

        if (idx + 1) % 20 == 0 or idx == 0 or idx + 1 == len(pose_files):
            print(f"[progress] {idx+1}/{len(pose_files)} frame={frame_id} views={len(used_cams)}")

    if writer is not None:
        writer.release()

    print("\n[DONE]")
    print(f"  rendered frames : {rendered}")
    print(f"  skipped frames  : {skipped}")
    print(f"  video           : {output_video}")
    if args.save_frames:
        print(f"  grid PNGs       : {frames_dir}")


if __name__ == "__main__":
    main()
