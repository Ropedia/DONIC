#!/usr/bin/env python3
"""Track the selected pose backend and preserve previous Stage 3 outputs."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid


METHODS = ("foundationpose", "wapr")
MARKER = "pose_tracking.json"
OUTPUTS = ("debug_depth", "mesh_overlay_allcams_fast_green")


def read_state(scene):
    path = Path(scene) / MARKER
    if not path.exists():
        return {"method": "foundationpose", "status": "legacy"}
    try:
        state = json.loads(path.read_text())
        if isinstance(state, dict):
            return state
    except (ValueError, OSError):
        pass
    return {"method": None, "status": "invalid"}


def write_state(scene, state):
    path = Path(scene) / MARKER
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(temp, path)


def prepare(scene, method):
    scene = Path(scene)
    if method not in METHODS:
        raise ValueError(f"Unknown tracking method: {method}")
    previous = read_state(scene)
    archive = scene / "tracking_history" / (
        time.strftime("%Y%m%d-%H%M%S") + "-" + str(previous.get("method")) + "-" + uuid.uuid4().hex[:8])
    existing = [scene / name for name in (*OUTPUTS, MARKER) if (scene / name).exists()]
    if existing:
        archive.mkdir(parents=True)
        for path in existing:
            path.rename(archive / path.name)
        print(f"Previous tracking outputs preserved: {archive}", flush=True)
    (scene / "debug_depth").mkdir(parents=True, exist_ok=True)
    write_state(scene, {"method": method, "status": "running", "started": time.time(),
                        "previous_outputs": str(archive) if existing else None})


def complete(scene, method):
    scene = Path(scene)
    state = read_state(scene)
    if state.get("method") != method or state.get("status") != "running":
        raise ValueError("Tracking run does not match the prepared backend")
    poses = list((scene / "debug_depth/geometry_global_opt/ob_in_world").glob("*.txt"))
    video = scene / "mesh_overlay_allcams_fast_green/mesh_overlay_allcams_fast_green.mp4"
    if not poses or not video.is_file() or video.stat().st_size == 0:
        raise ValueError("Tracking post-optimization or preview video is missing")
    state.update(status="complete", completed=time.time(), pose_frames=len(poses))
    write_state(scene, state)
    print(f"Completed {method} tracking: {len(poses)} frames", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "complete"))
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "complete": complete}[args.action](args.scene, args.method)
