"""FoundationPose registration worker for WAPR first-frame initialization.

Run in the existing FoundationPose environment: its CUDA extensions and Torch
version are independent of WAPR's. Only RGB-D inputs and original-mesh poses
cross the process boundary. The worker owns the registration networks and mesh.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np


try:
    from .settings import PIPELINE_ROOT, PROJECT_ROOT, MODEL_ROOT, python_for
except ImportError:
    from settings import PIPELINE_ROOT, PROJECT_ROOT, MODEL_ROOT, python_for

ROOT = PIPELINE_ROOT
HYPOTHESES = 240
REFINE_ITERATIONS = 5


def rotation_grid_240(camera_views):
    """FoundationPose view/in-plane construction with exactly 40 x 6 rotations.

    Its minimum-40 icosphere actually has 42 vertices. Select 40 deterministically
    by farthest-point sampling, then apply all six 60-degree in-plane rotations.
    No symmetry pruning: registration must refine all 240 hypotheses.
    """
    views = np.asarray(camera_views, dtype=np.float64)
    if views.ndim != 3 or views.shape[1:] != (4, 4) or len(views) < 40 or not np.isfinite(views).all():
        raise ValueError("Expected at least 40 finite FoundationPose camera views")
    directions = views[:, :3, 3]
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms == 0):
        raise ValueError("Camera directions must be nonzero")
    directions = directions / norms[:, None]
    ids, distances = [], np.full(len(views), np.inf)
    current = 0
    for _ in range(40):
        ids.append(current)
        distances = np.minimum(distances, np.sum((directions - directions[current]) ** 2, axis=1))
        distances[ids] = -np.inf
        current = int(np.argmax(distances))
    rotations = []
    for view in views[ids]:
        for angle in np.deg2rad(np.arange(0, 360, 60)):
            spin = np.eye(4)
            c, s = np.cos(angle), np.sin(angle)
            spin[:2, :2] = [[c, -s], [s, c]]
            pose = np.linalg.inv(view @ spin)
            pose[:3, 3] = 0  # register() assigns the fused depth center.
            rotations.append(pose)
    return np.asarray(rotations, dtype=np.float32)


class FoundationPoseClient:
    def __init__(self, args):
        self.process = None
        self.directory = tempfile.TemporaryDirectory(prefix="wapr-fp-reinit-")
        output = args.output_dir or args.scene_dir / "debug_depth"
        output.mkdir(parents=True, exist_ok=True)
        self.log_path = output / "foundationpose_reinit.log"
        self.log = self.log_path.open("a")
        self.sequence = 0
        try:
            python = args.foundationpose_python
            if not python.is_file():
                raise FileNotFoundError(f"FoundationPose Python not found: {python}")
            env = dict(os.environ, OMP_NUM_THREADS=str(args.cpu_threads), OPENBLAS_NUM_THREADS="2")
            env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
            self.process = subprocess.Popen(
                [str(python), "-u", str(Path(__file__).resolve()), "--worker",
                 "--mesh", str((args.mesh_file or args.scene_dir / "meshes/compressed.ply").resolve()),
                 "--device", args.device, "--threads", str(args.cpu_threads),
                 "--work-dir", self.directory.name],
                cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=self.log,
                stderr=subprocess.STDOUT, text=True)
        except BaseException:
            self.close()
            raise

    def register(self, obs, center_camera):
        self.sequence += 1
        request = Path(self.directory.name) / f"{self.sequence}.npz"
        response = request.with_suffix(".json")
        np.savez(request, rgb=obs["rgb"], depth=obs["depth"], mask=obs["mask"],
                 K=obs["K"], center_camera=center_camera)
        try:
            self.process.stdin.write(json.dumps(dict(request=str(request), response=str(response))) + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + 600
            while not response.exists():
                if self.process.poll() is not None:
                    raise RuntimeError(f"FoundationPose worker exited ({self.process.returncode}); see {self.log_path}")
                if time.monotonic() > deadline:
                    self.close()
                    raise TimeoutError(f"FoundationPose registration timed out; see {self.log_path}")
                time.sleep(.1)
            result = json.loads(response.read_text())
            if "error" in result:
                raise RuntimeError(f"FoundationPose registration failed: {result['error']}; see {self.log_path}")
            pose = np.asarray(result["pose_4x4"], dtype=np.float64)
            if (pose.shape != (4, 4) or not np.isfinite(pose).all() or pose[2, 3] <= 0
                    or not np.allclose(pose[3], [0, 0, 0, 1]) or not np.isfinite(result["score"])):
                raise RuntimeError("FoundationPose returned an invalid pose/score")
            if result["hypotheses"] != HYPOTHESES or result["refine_iterations"] != REFINE_ITERATIONS:
                raise RuntimeError("FoundationPose registration recipe differs from 240 hypotheses / 5 iterations")
            result["pose_4x4"] = pose
            return result
        finally:
            request.unlink(missing_ok=True)
            response.unlink(missing_ok=True)

    def close(self):
        if self.process is not None:
            try:
                self.process.stdin.close()
            except (OSError, BrokenPipeError):
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        self.log.close()
        self.directory.cleanup()


def serve(args):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "FoundationPose"))
    import torch
    import trimesh
    from estimater import FoundationPose
    from Utils import sample_views_icosphere

    torch.cuda.set_device(args.device)
    torch.set_num_threads(args.threads)

    class RegistrationEstimator(FoundationPose):
        def make_rotation_grid(self, min_n_views=40, inplane_step=60):
            self.rot_grid = torch.as_tensor(rotation_grid_240(sample_views_icosphere(n_views=40)),
                                            device=args.device)

    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    estimator = RegistrationEstimator(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                                  mesh=mesh, debug=0, debug_dir=args.work_dir)
    print(f"[FoundationPose registration] ready: {len(estimator.rot_grid)} hypotheses, {REFINE_ITERATIONS} iterations", flush=True)
    try:
        for line in sys.stdin:
            task = json.loads(line)
            response = Path(task["response"])
            try:
                started = time.perf_counter()
                with np.load(task["request"], allow_pickle=False) as data:
                    pose = estimator.register(
                        K=data["K"], rgb=data["rgb"], depth=data["depth"], ob_mask=data["mask"],
                        init_xyz=data["center_camera"], iteration=REFINE_ITERATIONS,
                        lock_translation_to_init=False)
                if len(estimator.poses) != HYPOTHESES:
                    raise RuntimeError("FoundationPose did not refine all 240 hypotheses")
                result = dict(pose_4x4=pose.tolist(), score=float(estimator.scores[0].item()),
                              best_id=int(estimator.best_id.item()), hypotheses=len(estimator.poses),
                              refine_iterations=REFINE_ITERATIONS, seconds=time.perf_counter() - started)
                print(f"[FoundationPose registration] best={result['best_id']} score={result['score']:.6f} "
                      f"{result['seconds']:.3f}s", flush=True)
            except Exception as exc:
                traceback.print_exc()
                result = dict(error=str(exc))
            pending = response.with_suffix(".tmp")
            pending.write_text(json.dumps(result) + "\n")
            os.replace(pending, response)
    finally:
        if estimator.mesh_path:
            Path(estimator.mesh_path).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--work-dir", required=True)
    serve(parser.parse_args())
