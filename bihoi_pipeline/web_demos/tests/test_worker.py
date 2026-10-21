from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import cv2
    import numpy as np
    import trimesh
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

from worker import pose_overlay, read_wapr_inputs
from mose3_points import pixel_points, prepare_preview, load_video_frames


@unittest.skipUnless(AVAILABLE, "requires numpy, opencv-python and trimesh in a model environment")
class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        p = Path(self.temp.name)
        cv2.imwrite(str(p / "rgb.png"), np.zeros((80, 100, 3), np.uint8))
        np.save(p / "depth.npy", np.full((80, 100), 1000, np.uint16))
        mask = np.zeros((80, 100), np.uint8)
        mask[20:60, 25:75] = 255
        cv2.imwrite(str(p / "mask.png"), mask)
        trimesh.creation.box(extents=(100, 100, 100)).export(p / "mesh.ply")
        self.config = {"rgb":str(p / "rgb.png"), "depth":str(p / "depth.npy"),
                       "mask":str(p / "mask.png"), "mesh":str(p / "mesh.ply"),
                       "bbox":None, "depth_scale":.001, "mesh_scale":.001,
                       "fx":100, "fy":100, "cx":50, "cy":40}

    def tearDown(self):
        self.temp.cleanup()

    def test_metric_conversion_preserves_original_resolution(self):
        rgb, depth, mesh, mask, box, K = read_wapr_inputs(self.config)
        self.assertEqual(rgb.shape, (80, 100, 3))
        np.testing.assert_allclose(depth, 1)
        np.testing.assert_allclose(mesh.extents, .1)
        self.assertEqual(np.count_nonzero(mask), 2000)
        self.assertIsNone(box)
        np.testing.assert_allclose(K, [[100,0,50],[0,100,40],[0,0,1]])

    def test_depth_size_mismatch_is_rejected(self):
        np.save(self.config["depth"], np.ones((40, 50)))
        with self.assertRaisesRegex(ValueError, "尺寸"):
            read_wapr_inputs(self.config)

    def test_empty_mask_and_invalid_depth_are_rejected(self):
        cv2.imwrite(self.config["mask"], np.zeros((80,100),np.uint8))
        with self.assertRaisesRegex(ValueError, "Mask 为空"):
            read_wapr_inputs(self.config)
        np.save(self.config["depth"], np.full((80,100),np.nan))
        with self.assertRaisesRegex(ValueError, "有效深度"):
            read_wapr_inputs({**self.config,"bbox":[10,10,20,20]})

    def test_box_outside_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "超出"):
            read_wapr_inputs({**self.config, "bbox":[90,70,20,20]})

    def test_overlay_projects_object_in_camera_coordinates(self):
        rgb, _, mesh, _, _, K = read_wapr_inputs(self.config)
        pose = np.eye(4)
        pose[2,3] = 1
        overlay = pose_overlay(rgb, mesh, pose, K, .1)
        self.assertEqual(overlay.shape, rgb.shape)
        self.assertGreater(np.count_nonzero(overlay[30:51,40:70]), 0)
        self.assertEqual(np.count_nonzero(overlay[:10,:10]), 0)

    def test_click_coordinates_map_to_processed_pixels(self):
        np.testing.assert_array_equal(pixel_points([[0,0],[.25,.5],[1,1]],168,224),
                                      [[0,0],[42,112],[167,223]])
        for points in ([[1.1,0]], [[0,float("nan")]], []):
            with self.assertRaises(ValueError):
                pixel_points(points,168,224)

    def test_preview_uses_actual_sampling_and_cropping(self):
        repo = Path(__file__).resolve().parents[1] / "third_party/MoSE3"
        if not (repo / "mose3/utils/video_io.py").is_file():
            self.skipTest("official MoSE3 checkout unavailable")
        output = Path(self.temp.name)
        video = output / "sample.avi"
        writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"MJPG"),10,(100,80))
        self.assertTrue(writer.isOpened())
        for i in range(10):
            writer.write(np.full((80,100,3),i*20,np.uint8))
        writer.release()
        manifest=prepare_preview(dict(repo=str(repo),video=str(video),output=str(output),
                                      num_frames=4,start_frame=1,frame_interval=2,resolution=112,central_crop=True))
        self.assertEqual(manifest["frame_indices"],[1,3,5,7])
        self.assertEqual(manifest["width"],manifest["height"])
        frame=cv2.imread(str(output / "frame_0002.png"))
        self.assertAlmostEqual(float(frame.mean()),100,delta=3)
        config = dict(repo=str(repo),video=str(video),output=str(output),num_frames=2,
                      start_frame=5,frame_interval=3,resolution=112,central_crop=False,sample_all=True)
        frames, indices, _ = load_video_frames(config)
        self.assertEqual(indices,list(range(10)))
        self.assertEqual(len(frames),10)
        self.assertAlmostEqual(float(frames[0].mean()),0,delta=3)
        self.assertAlmostEqual(float(frames[-1].mean()),180,delta=3)
        frames,indices,_=load_video_frames({**config,"sampling_fps":10,"sample_all":False})
        self.assertEqual(indices,list(range(10)))
        self.assertEqual(len(frames),10)

    def test_time_sampling_does_not_use_integer_stride(self):
        repo=Path(__file__).resolve().parents[1]/"third_party/MoSE3"
        if not (repo/"mose3/utils/video_io.py").is_file():
            self.skipTest("official MoSE3 checkout unavailable")
        path=Path(self.temp.name)/"non_integer_rate.avi"
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),25,(100,80))
        for i in range(25): writer.write(np.full((80,100,3),i*8,np.uint8))
        writer.release()
        frames,indices,_=load_video_frames(dict(video=str(path),repo=str(repo),sampling_fps=10,
                                              central_crop=False,resolution=112))
        self.assertEqual(len(frames),10)
        self.assertEqual(indices[0],0)
        self.assertTrue({2,3}.issubset(set(np.diff(indices))))
        self.assertLessEqual(abs(indices[-1]/25-.9),.020001)


if __name__ == "__main__":
    unittest.main()
