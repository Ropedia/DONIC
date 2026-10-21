import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import DemoState, Handler, ThreadingHTTPServer


class StateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = DemoState(self.temp.name, interpreters={app: sys.executable for app in ("wapr", "mose3")})
        self.token = "a" * 32 + ".mp4"
        (self.state.uploads / self.token).write_bytes(b"video")
        self.raw = dict(app="mose3", video=self.token, num_frames=24, start_frame=0,
                        frame_interval=0, query_idx=0, resolution=280,
                        num_ref_pts=12000, stride=48, fps=8, se3_sigma_px=8)

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def test_sampling_validation(self):
        for field, bad in (("num_frames", 1), ("query_idx", 24), ("resolution", "nan"),
                           ("num_ref_pts", 1.5), ("fps", True), ("device", "cpu")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.state.validate({**self.raw, field: bad})
        self.assertEqual(self.state.validate(self.raw)["query_idx"], 0)

    def test_reject_path_traversal_and_wrong_file_types(self):
        for token in ("../../server.py", "/etc/passwd", "b" * 32 + ".mp4", None):
            with self.subTest(token=token), self.assertRaises(ValueError):
                self.state.upload_path(token)
        token = "c" * 32 + ".png"
        (self.state.uploads / token).touch()
        with self.assertRaisesRegex(ValueError, "视频"):
            self.state.validate({**self.raw, "video": token})

    def test_client_cannot_override_interpreter_or_checkpoint(self):
        result = self.state.validate({**self.raw, "python": "/tmp/untrusted",
                                      "checkpoint": "/tmp/untrusted", "repo": "/tmp/untrusted"})
        self.assertNotIn("python", result)
        self.assertEqual(result["repo"], self.state.repos["mose3"])
        self.assertNotEqual(result["checkpoint"], "/tmp/untrusted")

    def test_disabled_app(self):
        self.state.app = "wapr"
        with self.assertRaises(ValueError):
            self.state.validate(self.raw)

    def manual_request(self):
        preview_id = "d" * 32
        folder = self.state.previews_dir / preview_id
        folder.mkdir(exist_ok=True)
        cfg = self.state.validate(self.raw)
        manifest = {key:cfg[key] for key in ("video", "num_frames", "start_frame", "frame_interval", "resolution", "central_crop")}
        manifest["frame_indices"] = [0, 5, 10, 15]
        (folder / "preview.json").write_text(json.dumps(manifest))
        return {**self.raw, "point_mode":"manual", "point_preview_id":preview_id,
                "query_idx":2, "query_points":[[.25,.5],[1,0]]}

    def test_manual_points_keep_selected_frame(self):
        result = self.state.validate(self.manual_request())
        self.assertEqual(result["query_idx"], 2)
        self.assertEqual(result["frame_indices"], [0,5,10,15])
        self.assertEqual(result["query_points"], [[.25,.5],[1,0]])

    def test_manual_points_require_matching_preview_and_valid_coordinates(self):
        raw = self.manual_request()
        for changes in ({"query_points":[]}, {"query_points":[[1.1,.5]]},
                        {"query_points":[[True,.5]]}, {"query_idx":4},
                        {"resolution":518}, {"central_crop":True}, {"point_preview_id":"../../escape"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.state.validate({**raw, **changes})

    def test_all_frames_accepts_query_above_old_sampling_limit(self):
        result=self.state.validate({**self.raw,"sample_all":True,"query_idx":400,"num_frames":900})
        self.assertTrue(result["sample_all"])
        self.assertEqual(result["query_idx"],400)

    def test_worker_success_and_artifact_collection(self):
        class Process:
            def __init__(self, command, **kwargs):
                config = json.loads(Path(command[-1]).read_text())
                output = Path(config["output"])
                (output / "result.json").write_text('{"model":"MoSE3"}')
                (output / "tracks.mp4").write_bytes(b"test artifact")
                self.stdout = io.StringIO("model loaded\ninference finished\n")

            def wait(self):
                return 0

        with patch("server.subprocess.Popen", Process):
            submitted = self.state.submit(self.raw)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                job = self.state.snapshot(submitted["id"])
                if job["status"] in ("succeeded", "failed"):
                    break
                time.sleep(.01)
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["result"]["model"], "MoSE3")
        self.assertIn("tracks.mp4", job["artifacts"])
        self.assertIn("results.zip", job["artifacts"])
        self.assertFalse((self.state.jobs_dir / job["id"] / "results.zip").exists())
        self.assertNotIn("input.json", job["artifacts"])
        self.assertNotIn("process", job)
        self.assertIn("inference finished", job["log"])

    def test_invalid_python_fails_with_actionable_message(self):
        self.state.python["mose3"] = "/does/not/exist/python"
        submitted = self.state.submit(self.raw)
        for _ in range(100):
            job = self.state.snapshot(submitted["id"])
            if job["status"] == "failed":
                break
            time.sleep(.01)
        self.assertEqual(job["status"], "failed")
        self.assertIn("--mose3-python", job["log"])

    def test_cancel_queued_job_does_not_start_worker(self):
        self.state.jobs["queued"] = dict(id="queued", status="queued", process=None, log="")
        self.assertEqual(self.state.cancel("queued")["status"], "cancelled")
        with patch("server.subprocess.Popen") as popen:
            self.state.run("queued", {"output": self.temp.name, "app": "mose3"})
        popen.assert_not_called()

    def test_restart_restores_completed_job_without_eager_zip(self):
        directory = self.state.jobs_dir / ("b" * 32)
        directory.mkdir()
        (directory / "result.json").write_text('{"model":"MoSE3"}')
        (directory / "tracks.mp4").write_bytes(b"video")
        restored = DemoState(self.temp.name)
        try:
            job = restored.snapshot(directory.name)
            self.assertEqual(job["status"], "succeeded")
            self.assertIn("results.zip", job["artifacts"])
            self.assertFalse((directory / "results.zip").exists())
        finally:
            restored.close()


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.state = DemoState(cls.temp.name)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.state = cls.state
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()
        cls.state.close()
        cls.temp.cleanup()

    def fetch(self, path, data=None, headers=None):
        return urlopen(Request(self.url + path, data=data, headers=headers or {}), timeout=5)

    def test_both_demo_pages_and_assets(self):
        for path, content in (("/wapr", b"POSE LAB"), ("/mose3", b"POSE LAB"),
                              ("/app.js", b"mose3Fields"), ("/style.css", b"@media")):
            with self.subTest(path=path), self.fetch(path) as response:
                self.assertEqual(response.status, 200)
                self.assertIn(content, response.read())
        with self.fetch("/api/config") as response:
            self.assertEqual(json.load(response)["app"], "all")

    def test_binary_upload(self):
        payload = bytes(range(256))
        with self.fetch("/api/upload", payload, {"X-Filename": "input.png", "Content-Type": "application/octet-stream"}) as response:
            result = json.load(response)
        self.assertEqual((self.state.uploads / result["token"]).read_bytes(), payload)
        self.assertEqual(result["size"], 256)

    def test_cross_origin_and_invalid_upload_are_rejected(self):
        for headers, expected in (({"X-Filename":"bad.py"}, 400),
                                  ({"X-Filename":"file.png", "Origin":"https://untrusted.example"}, 403)):
            with self.subTest(headers=headers), self.assertRaises(HTTPError) as raised:
                self.fetch("/api/upload", b"data", headers)
            self.assertEqual(raised.exception.code, expected)
            raised.exception.close()

    def test_result_video_range_and_private_config(self):
        folder = self.state.jobs_dir / ("f" * 32)
        folder.mkdir(exist_ok=True)
        (folder / "tracks.mp4").write_bytes(b"0123456789")
        (folder / "input.json").write_text("private")
        with self.fetch(f"/files/{folder.name}/tracks.mp4", headers={"Range":"bytes=2-5"}) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b"2345")
            self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
        with self.assertRaises(HTTPError) as raised:
            self.fetch(f"/files/{folder.name}/input.json")
        self.assertEqual(raised.exception.code, 404)
        raised.exception.close()

    def test_zip_is_built_on_download_and_cached_without_recompressing_binary(self):
        folder = self.state.jobs_dir / ("e" * 32)
        folder.mkdir()
        (folder / "tracks.mp4").write_bytes(b"encoded video")
        (folder / "predictions.npz").write_bytes(b"compressed predictions")
        (folder / "result.json").write_text('{"model":"MoSE3"}')
        (folder / "input.json").write_text("private")
        self.state.jobs[folder.name] = dict(id=folder.name, status="succeeded", process=None,
                                          log="", result={})
        self.assertFalse((folder / "results.zip").exists())
        with self.fetch(f"/files/{folder.name}/results.zip") as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                self.assertNotIn("input.json", archive.namelist())
                self.assertEqual(archive.read("tracks.mp4"), b"encoded video")
                self.assertEqual(archive.getinfo("predictions.npz").compress_type, zipfile.ZIP_STORED)
        previous = (folder / "results.zip").stat().st_mtime_ns
        self.state.archive(folder.name)
        self.assertEqual((folder / "results.zip").stat().st_mtime_ns, previous)
        (folder / "result.json").write_text('{"model":"MoSE3", "updated":true}')
        self.state.archive(folder.name)
        with zipfile.ZipFile(folder / "results.zip") as archive:
            self.assertTrue(json.loads(archive.read("result.json"))["updated"])


if __name__ == "__main__":
    unittest.main()
