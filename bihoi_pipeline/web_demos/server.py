"""Dependency-free local web front end for WAPR and MoSE3.

Model inference runs in a separate, user-selected Python environment.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit
import uuid

from mose3_points import SAMPLE_KEYS
from result_archive import build_result_archive, result_files

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "webui"))
from settings import MODEL_ROOT, python_for
os.environ.setdefault("WAPR_CACHE_DIR", str(MODEL_ROOT / "wapr"))
APPS = {"wapr", "mose3"}
MAX_UPLOAD = 2 * 1024**3


def default_python(app):
    return os.environ.get(f"{app.upper()}_PYTHON", str(python_for(app)))


def numeric(data, key, low, high, integer=False):
    value = data.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} 必须是数字")
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{key} 必须是数字") from None
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{key} 必须在 {low} 到 {high} 之间")
    if integer and result != int(result):
        raise ValueError(f"{key} 必须是整数")
    return int(result) if integer else result


class DemoState:
    def __init__(self, data_dir, app="all", interpreters=None, repos=None):
        self.data = Path(data_dir).resolve()
        self.uploads = self.data / "uploads"
        self.jobs_dir = self.data / "jobs"
        self.previews_dir = self.data / "previews"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir.mkdir(parents=True, exist_ok=True)
        self.app = app
        self.python = interpreters or {a: default_python(a) for a in APPS}
        self.repos = repos or {a: str(ROOT / "third_party" / ("WAPR" if a == "wapr" else "MoSE3")) for a in APPS}
        self.jobs = {}
        self.lock = threading.RLock()
        self.archive_locks = {}
        # Serialize GPU work, including work submitted from the other demo page.
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Finished files are durable; retain their result pages across service updates.
        for directory in self.jobs_dir.iterdir():
            if not re.fullmatch(r"[0-9a-f]{32}", directory.name) or not (directory / "result.json").is_file():
                continue
            try:
                result = json.loads((directory / "result.json").read_text())
                model = result.get("model", "").lower()
                if model not in APPS or self.app not in ("all", model):
                    continue
                log_path = directory / "run.log"
                self.jobs[directory.name] = {"id":directory.name, "app":model, "status":"succeeded",
                    "created":directory.stat().st_mtime, "log":log_path.read_text(errors="replace")[-50000:] if log_path.is_file() else "",
                    "artifacts":self.artifact_names(directory, complete=True),
                    "result":result, "process":None}
            except (OSError, ValueError, AttributeError):
                continue

    @staticmethod
    def artifact_names(directory, complete=False):
        names = [p.name for p in result_files(directory)]
        if complete or (directory / "results.zip").is_file():
            names.append("results.zip")
        return sorted(names)

    def archive(self, job_id):
        with self.lock:
            if job_id not in self.jobs or self.jobs[job_id]["status"] != "succeeded":
                raise KeyError("任务尚未完成，无法打包下载")
            lock = self.archive_locks.setdefault(job_id, threading.Lock())
        # Concurrent downloads share one atomic archive build; inference is not blocked.
        with lock:
            return build_result_archive(self.jobs_dir / job_id)

    def example(self, app):
        if app not in APPS or self.app not in ("all", app):
            raise KeyError("此服务未启用该样例")
        if app == "wapr":
            base = Path(os.environ.get("WAPR_CACHE_DIR", str(ROOT / "runtime/cache/wapr"))) / "cache/samples/pose_lmo"
            paths = {key: base / name for key, name in {"rgb":"rgb.png", "depth":"depth.npy", "mesh":"model.ply", "mask":"mask.png"}.items()}
            if not (base / "meta.json").is_file():
                raise KeyError("未下载 WAPR 官方 pose_lmo 样例")
            meta = json.loads((base / "meta.json").read_text())
            values = {key:meta[key] for key in ("fx", "fy", "cx", "cy", "diameter_m")}
            values.update(region="mask", depth_scale=1, mesh_scale=1, backend="torch")
        else:
            paths = {"video":Path(self.repos[app]) / "examples/spin.mp4"}
            values = dict(num_frames=8, resolution=224, num_ref_pts=1024, query_idx=0, start_frame=0, frame_interval=0)
        if not all(p.is_file() for p in paths.values()):
            raise KeyError("官方样例文件不完整，请先下载样例")
        return paths, values

    def upload_path(self, token):
        if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{32}\.[a-z0-9]{1,10}", token):
            raise ValueError("无效的上传文件")
        path = self.uploads / token
        if not path.is_file():
            raise ValueError("上传文件不存在，请重新上传")
        return str(path)

    def validate(self, raw):
        app = raw.get("app")
        if app not in APPS or self.app not in ("all", app):
            raise ValueError("此服务未启用该 demo")
        out = {"app": app, "repo": self.repos[app]}
        device = raw.get("device", "cuda:0")
        if not re.fullmatch(r"cuda(?::\d+)?", str(device)):
            raise ValueError("请选择 CUDA GPU，例如 cuda:0")
        out["device"] = device
        if app == "wapr":
            for key in ("rgb", "depth", "mesh"):
                out[key] = self.upload_path(raw.get(key))
            if Path(out["rgb"]).suffix not in (".png", ".jpg", ".jpeg"):
                raise ValueError("RGB 需要 PNG 或 JPEG")
            if Path(out["depth"]).suffix not in (".png", ".npy", ".tif", ".tiff"):
                raise ValueError("深度需要 PNG、TIFF 或 NPY")
            if Path(out["mesh"]).suffix not in (".ply", ".obj", ".stl", ".glb"):
                raise ValueError("网格需要 PLY、OBJ、STL 或 GLB")
            if raw.get("region") == "mask":
                out["mask"] = self.upload_path(raw.get("mask"))
                if Path(out["mask"]).suffix not in (".png", ".jpg", ".jpeg"):
                    raise ValueError("Mask 需要 PNG 或 JPEG")
                out["bbox"] = None
            elif raw.get("region") == "box":
                out["mask"] = None
                out["bbox"] = [numeric(raw, k, 0 if k in ("x", "y") else 1, 100000, True) for k in ("x", "y", "w", "h")]
            else:
                raise ValueError("请选择 Mask 或框选区域")
            for key in ("fx", "fy"):
                out[key] = numeric(raw, key, 0.000001, 1e7)
            for key in ("cx", "cy"):
                out[key] = numeric(raw, key, -1e7, 1e7)
            for key in ("depth_scale", "mesh_scale", "diameter_m"):
                out[key] = numeric(raw, key, 1e-10, 1e6)
            if raw.get("backend") not in ("torch", "trt"):
                raise ValueError("WAPR 后端必须为 torch 或 trt")
            out["backend"] = raw["backend"]
        else:
            out["sampling_fps"] = numeric(raw,"sampling_fps",10,10) if raw.get("sampling_fps") is not None else None
            out["sample_all"] = raw.get("sample_all") is True and not out["sampling_fps"]
            automatic_count = bool(out["sampling_fps"] or out["sample_all"])
            out["video"] = self.upload_path(raw.get("video"))
            if Path(out["video"]).suffix not in (".mp4", ".mov", ".avi", ".webm", ".mkv"):
                raise ValueError("请选择 MP4、MOV、AVI、WebM 或 MKV 视频")
            for key, low, high in (("num_frames", 2, 10000000 if automatic_count else 300), ("start_frame", 0, 10000000),
                                  ("frame_interval", 0, 100000), ("query_idx", 0, 10000000 if automatic_count else 299),
                                  ("resolution", 112, 1036), ("num_ref_pts", 64, 50000),
                                  ("stride", 8, 256), ("fps", 1, 60)):
                out[key] = numeric(raw, key, low, high, True)
            if not automatic_count and out["query_idx"] >= out["num_frames"]:
                raise ValueError("查询帧索引必须小于采样帧数")
            out["se3_sigma_px"] = numeric(raw, "se3_sigma_px", 0.1, 1000)
            out["se3_dist_weight"] = raw.get("se3_dist_weight") is True
            out["central_crop"] = raw.get("central_crop") is True
            mode = raw.get("point_mode", "grid")
            if mode not in ("grid", "manual"):
                raise ValueError("查询点模式必须是 manual 或 grid")
            out["point_mode"] = mode
            if mode == "manual":
                preview_id = raw.get("point_preview_id", "")
                if not isinstance(preview_id, str) or not re.fullmatch(r"[0-9a-f]{32}", preview_id):
                    raise ValueError("请先生成选点预览")
                manifest_path = self.previews_dir / preview_id / "preview.json"
                if not manifest_path.is_file():
                    raise ValueError("选点预览不存在，请重新生成")
                manifest = json.loads(manifest_path.read_text())
                keys = ("video", "resolution", "central_crop", "sampling_fps") if out["sampling_fps"] else (("video", "resolution", "central_crop", "sample_all", "sampling_fps") if out["sample_all"] else ("video", *SAMPLE_KEYS))
                if any(out[key] != manifest.get(key,False if key == "sample_all" else None) for key in keys):
                    raise ValueError("视频或采样参数已改变，请重新生成选点预览")
                if out["query_idx"] >= len(manifest["frame_indices"]):
                    raise ValueError("查询帧超出实际采样帧数")
                points = raw.get("query_points")
                if not isinstance(points, list) or not 1 <= len(points) <= 128:
                    raise ValueError("请在同一查询帧上点击选择 1 到 128 个点")
                for point in points:
                    if not isinstance(point, list) or len(point) != 2:
                        raise ValueError("查询点必须为 [x, y] 坐标")
                    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in point):
                        raise ValueError("查询点坐标必须在 0 到 1 之间")
                out["query_points"] = points
                out["frame_indices"] = manifest["frame_indices"]
            # Checkpoint is server configuration, never an arbitrary uploaded executable.
            local_ckpt = ROOT / "runtime" / "checkpoints" / "MoSE3"
            out["checkpoint"] = os.environ.get("MOSE3_CHECKPOINT", str(local_ckpt) if (local_ckpt / "model.safetensors").is_file() else "mose3-tracker/MoSE3")
        return out

    def prepare_preview(self, raw):
        # Preview runs only the upstream CPU decoder, never the model or GPU queue.
        config = self.validate({**raw, "app":"mose3", "point_mode":"grid", "query_idx":0})
        preview_id = uuid.uuid4().hex
        directory = self.previews_dir / preview_id
        directory.mkdir()
        config["output"] = str(directory)
        request = directory / "input.json"
        request.write_text(json.dumps(config))
        interpreter = shutil.which(self.python["mose3"])
        if not interpreter:
            raise ValueError("MoSE3 Python 不存在，请检查 --mose3-python")
        try:
            result = subprocess.run([interpreter, "-u", str(ROOT / "worker.py"), "--preview", str(request)],
                                    capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            raise ValueError("视频预览超时，请缩短视频或减少采样帧数") from None
        if result.returncode:
            raise ValueError("无法准备选点预览：" + (result.stderr or result.stdout)[-1800:])
        manifest = json.loads((directory / "preview.json").read_text())
        return {"id":preview_id, "width":manifest["width"], "height":manifest["height"],
                "frame_indices":manifest["frame_indices"],
                "frames":[f"/previews/{preview_id}/frame_{i:04d}.png" for i in range(len(manifest["frame_indices"]))]}

    def submit(self, raw):
        config = self.validate(raw)
        job_id = uuid.uuid4().hex
        directory = self.jobs_dir / job_id
        directory.mkdir()
        config["output"] = str(directory)
        (directory / "input.json").write_text(json.dumps(config), encoding="utf-8")
        job = {"id": job_id, "app": config["app"], "status": "queued", "created": time.time(),
               "log": "任务已排队。\n", "artifacts": [], "result": None, "process": None}
        with self.lock:
            self.jobs[job_id] = job
            self.pool.submit(self.run, job_id, config)
        return self.snapshot(job_id)

    def snapshot(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError("任务不存在；服务重启后请重新提交")
            if self.jobs[job_id]["status"] == "succeeded":
                result_path = self.jobs_dir / job_id / "result.json"
                if result_path.is_file():
                    self.jobs[job_id]["result"] = json.loads(result_path.read_text())
            return {k: v for k, v in self.jobs[job_id].items() if k != "process"}

    def run(self, job_id, config):
        directory = Path(config["output"])
        job = self.jobs[job_id]
        try:
            interpreter = shutil.which(self.python[config["app"]])
            if not interpreter:
                raise ValueError("模型 Python 不存在，请通过 --wapr-python / --mose3-python 配置")
            command = [interpreter, "-u", str(ROOT / "worker.py"), str(directory / "input.json")]
            with self.lock:
                if job["status"] == "cancelled":
                    return
                job["status"] = "running"
                env = os.environ.copy()
                env.setdefault("HF_HOME", str(ROOT / "runtime" / "cache" / "huggingface"))
                env.setdefault("WAPR_CACHE_DIR", str(ROOT / "runtime" / "cache" / "wapr"))
                job["process"] = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                   text=True, encoding="utf-8", errors="replace",
                                                   start_new_session=True, env=env)
                process = job["process"]
            with (directory / "run.log").open("w", encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    with self.lock:
                        job["log"] = (job["log"] + line)[-50000:]
            code = process.wait()
            with self.lock:
                if job["status"] != "cancelled":
                    job["status"] = "succeeded" if code == 0 else "failed"
                result_path = directory / "result.json"
                if result_path.exists():
                    job["result"] = json.loads(result_path.read_text())
                job["artifacts"] = self.artifact_names(directory, complete=job["status"] == "succeeded")
        except Exception as exc:
            with self.lock:
                if job["status"] != "cancelled":
                    job["status"] = "failed"
                job["log"] += f"\n{type(exc).__name__}: {exc}\n"
        finally:
            with self.lock:
                job["process"] = None

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError("任务不存在")
            if job["status"] in ("queued", "running"):
                job["status"] = "cancelled"
                job["log"] += "\n已取消任务。\n"
                process = job["process"]
                if process and process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    threading.Thread(target=self._kill_later, args=(process,), daemon=True).start()
        return self.snapshot(job_id)

    @staticmethod
    def _kill_later(process):
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def close(self):
        for job_id in list(self.jobs):
            self.cancel(job_id)
        self.pool.shutdown(wait=True, cancel_futures=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "PoseDemos/1.0"

    @property
    def state(self):
        return self.server.state

    def reply(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, path):
        if not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        value = self.headers.get("Range")
        if value:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", value)
            if not match:
                self.send_error(416)
                return
            start = int(match[1])
            end = min(int(match[2]), end) if match[2] else end
            if start > end:
                self.send_error(416)
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if path.suffix in (".npz", ".npy", ".zip", ".ply"):
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with path.open("rb") as source:
            source.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        try:
            if path == "/api/config":
                self.reply(200, {"app": self.state.app, "python": self.state.python,
                                 "repos": self.state.repos, "max_upload": MAX_UPLOAD})
            elif re.fullmatch(r"/api/example/(wapr|mose3)", path):
                app = path.rsplit("/", 1)[-1]
                paths, values = self.state.example(app)
                self.reply(200, {"values":values, "files":{key:{"name":p.name,"url":f"/examples/{app}/{key}"} for key,p in paths.items()}})
            elif re.fullmatch(r"/examples/(wapr|mose3)/(rgb|depth|mesh|mask|video)", path):
                _, _, app, key = path.split("/")
                paths, _ = self.state.example(app)
                self.serve_file(paths[key])
            elif re.fullmatch(r"/api/jobs/[0-9a-f]{32}", path):
                self.reply(200, self.state.snapshot(path.rsplit("/", 1)[-1]))
            elif re.fullmatch(r"/previews/[0-9a-f]{32}/frame_[0-9]{4}\.png", path):
                _, _, preview_id, name = path.split("/")
                self.serve_file(self.state.previews_dir / preview_id / name)
            elif re.fullmatch(r"/files/[0-9a-f]{32}/[\w.-]+", path):
                _, _, job_id, name = path.split("/")
                if name in ("input.json", ".", ".."):
                    self.send_error(404)
                elif name == "results.zip":
                    self.serve_file(self.state.archive(job_id))
                else:
                    self.serve_file(self.state.jobs_dir / job_id / name)
            elif path in ("/", "/wapr", "/mose3"):
                self.serve_file(ROOT / "static" / "index.html")
            elif path in ("/app.js", "/style.css"):
                self.serve_file(ROOT / "static" / path[1:])
            elif path in ("/fonts/noto-sans-sc.woff2", "/fonts/OFL.txt"):
                self.serve_file(ROOT / "static" / path[1:])
            else:
                self.send_error(404)
        except KeyError as exc:
            self.reply(404, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        # Local tools can start model subprocesses: reject cross-origin requests.
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + self.headers.get("Host", ""):
            self.reply(403, {"error": "不允许跨站请求"})
            return
        path = urlsplit(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("请求体为空")
            if path == "/api/upload":
                if length > MAX_UPLOAD:
                    self.reply(413, {"error": "单个文件不能超过 2 GiB"})
                    return
                name = unquote(self.headers.get("X-Filename", ""))
                ext = Path(name).suffix.lower()
                if ext not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".ply", ".obj",
                               ".stl", ".glb", ".mp4", ".mov", ".avi", ".webm", ".mkv"):
                    raise ValueError("不支持该文件格式")
                token = uuid.uuid4().hex + ext
                target = self.state.uploads / token
                try:
                    with target.open("wb") as output:
                        remaining = length
                        while remaining:
                            chunk = self.rfile.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError("上传中断")
                            output.write(chunk)
                            remaining -= len(chunk)
                except Exception:
                    target.unlink(missing_ok=True)
                    raise
                self.reply(201, {"token": token, "name": name, "size": length})
            else:
                if length > 65536:
                    self.reply(413, {"error": "参数过大"})
                    return
                raw = json.loads(self.rfile.read(length))
                if not isinstance(raw, dict):
                    raise ValueError("参数必须为 JSON 对象")
                if path == "/api/jobs":
                    self.reply(202, self.state.submit(raw))
                elif path == "/api/mose3/preview":
                    self.reply(201, self.state.prepare_preview(raw))
                elif re.fullmatch(r"/api/jobs/[0-9a-f]{32}/cancel", path):
                    self.reply(200, self.state.cancel(path.split("/")[3]))
                else:
                    self.send_error(404)
        except (ValueError, TypeError) as exc:
            self.reply(400, {"error": str(exc)})
        except KeyError as exc:
            self.reply(404, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", choices=("all", "wapr", "mose3"), default="all")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "runtime")
    for app, repo in (("wapr", "WAPR"), ("mose3", "MoSE3")):
        parser.add_argument(f"--{app}-python", default=default_python(app))
        parser.add_argument(f"--{app}-repo", default=os.environ.get(f"{app.upper()}_REPO", str(ROOT / "third_party" / repo)))
    args = parser.parse_args()
    state = DemoState(args.data_dir, args.app, {a: getattr(args, a + "_python") for a in APPS},
                      {a: str(Path(getattr(args, a + "_repo")).expanduser().resolve()) for a in APPS})
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.state = state
    print(f"Demo 已启动: http://{args.host}:{server.server_port}/" + (args.app if args.app != "all" else ""), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.close()


if __name__ == "__main__":
    main()
