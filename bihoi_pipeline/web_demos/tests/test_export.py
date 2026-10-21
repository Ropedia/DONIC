"""Round-trip the full prediction format and exercise real CPU export, without a GPU."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import numpy as np
    import imageio.v2 as imageio
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

from mose3_export import save_predictions, save_video
from video_timing import probe_video
from worker import export_mose3


@unittest.skipUnless(AVAILABLE, "requires model environment")
class ExportTests(unittest.TestCase):
    def test_npz_roundtrip_preserves_values_shapes_and_dtypes(self):
        payload = {"half":np.arange(24, dtype=np.float16).reshape(2,3,4),
                   "fortran":np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3,4)),
                   "valid":np.array([True,False]), "query_idx":np.asarray(3)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.npz"
            for level in ("0", "1"):
                with self.subTest(compression=level), patch.dict(os.environ, DEMO_NPZ_COMPRESSION=level):
                    save_predictions(path, payload)
                    with np.load(path, allow_pickle=False) as saved:
                        self.assertEqual(set(saved.files), set(payload))
                        for key, expected in payload.items():
                            self.assertEqual(saved[key].dtype, expected.dtype)
                            np.testing.assert_array_equal(saved[key], expected)

    def test_real_manual_and_grid_export_reuses_memory_and_keeps_ten_fps(self):
        repo = Path(__file__).resolve().parents[1] / "third_party/MoSE3"
        if not (repo / "visualize.py").is_file():
            self.skipTest("official checkout unavailable")
        frames, height, width = 3, 28, 28
        y, x = np.mgrid[:height,:width]
        tracks = np.stack([(x-14)/30, (y-14)/30, np.ones_like(x)], -1).astype(np.float32)
        rng = np.random.default_rng(0)
        quat = np.zeros((frames,height,width,4),np.float32); quat[...,0] = 1
        payload = dict(images=rng.integers(0,255,(frames,height,width,3),dtype=np.uint8),
                       tracks=np.repeat(tracks[None],frames,axis=0),
                       camera_poses=np.repeat(np.eye(4,dtype=np.float32)[None],frames,axis=0),
                       intrinsics=np.array([[30,0,14],[0,30,14],[0,0,1]],np.float32),
                       query_idx=np.array(1), visibility=np.ones((frames,height,width),np.float32),
                       se3_quat=quat, se3_trans=np.zeros((frames,height,width,3),np.float32),
                       se3_valid=np.ones((frames,height,width),bool),
                       rigidity_emb=rng.normal(size=(frames,height,width,4)).astype(np.float32))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            save_video(source, payload["images"], 10)
            for mode in ("manual", "grid"):
                with self.subTest(mode=mode):
                    out = Path(directory) / mode; out.mkdir()
                    config = dict(output=str(out), repo=str(repo), video=str(source), point_mode=mode,
                                  fps=10, sampling_fps=10, query_points=[[.5,.5]], frame_indices=[0,1,2], stride=14)
                    with patch("numpy.load", side_effect=AssertionError("must reuse in-memory predictions")):
                        result = export_mose3(config, dict(fps=10,frame_indices=[0,1,2]), payload)
                    self.assertEqual(result["playback_fps"],10)
                    for name in ("video.mp4", "tracks.mp4", "se3.mp4", "rigidity.mp4"):
                        duration, count = probe_video(out/name)
                        self.assertEqual(count,3)
                        self.assertAlmostEqual(duration,.3,places=3)
                    with imageio.get_reader(out/"tracks.mp4") as reader:
                        self.assertEqual(reader.get_data(0).shape, (height,width*2,3))
                    self.assertFalse((out/"results.zip").exists())
                    if mode == "manual":
                        with np.load(out/"selected_tracks.npz") as selected:
                            np.testing.assert_array_equal(selected["tracks"],payload["tracks"][:,[14],[14]])
                        self.assertEqual(result["query_points_xy"],[[14,14]])
                        self.assertEqual(json.loads((out/"query_points.json").read_text())["source_query_frame"],1)


if __name__ == "__main__":
    unittest.main()
