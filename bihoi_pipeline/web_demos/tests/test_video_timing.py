from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_timing import probe_video, retime_video, align_result_videos, VIDEO_NAMES


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires ffmpeg and ffprobe")
class VideoTimingTests(unittest.TestCase):
    def setUp(self):
        try:
            import imageio.v2 as imageio
            import numpy as np
        except ImportError:
            self.skipTest("requires imageio and numpy")
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "tracks.mp4"
        frames = [np.full((32,48,3),i*25,np.uint8) for i in range(8)]
        imageio.mimwrite(self.path, frames, fps=8, macro_block_size=1)

    def tearDown(self):
        if hasattr(self, "temp"):
            self.temp.cleanup()

    def test_fractional_playback_rate_preserves_all_frames(self):
        self.assertEqual(probe_video(self.path), (1.0,8))
        rate = retime_video(self.path,9.7,8)
        duration, count = probe_video(self.path)
        self.assertAlmostEqual(duration,9.7,places=3)
        self.assertEqual(count,8)
        self.assertAlmostEqual(rate,8/9.7)
        self.assertEqual(probe_video(self.path.parent/"original_timing/tracks.mp4"),(1.0,8))

    def test_retiming_is_idempotent(self):
        retime_video(self.path,3.6,8)
        first = self.path.read_bytes()
        retime_video(self.path,3.6,8)
        self.assertEqual(self.path.read_bytes(),first)
        self.assertAlmostEqual(probe_video(self.path)[0],3.6,places=3)

    def test_mismatched_frame_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"帧数"):
            retime_video(self.path,3.6,9)
        self.assertEqual(probe_video(self.path),(1.0,8))

    def test_fixed_sampling_also_fixes_playback_to_ten_fps(self):
        for name in VIDEO_NAMES:
            path=self.path.parent/name
            if path!=self.path: shutil.copy2(self.path,path)
        timing=align_result_videos({"video":str(self.path),"output":str(self.path.parent),"sampling_fps":10},8)
        self.assertEqual(timing["playback_fps"],10)
        self.assertEqual(timing["output_duration_seconds"],.8)
        for name in VIDEO_NAMES: self.assertEqual(probe_video(self.path.parent/name),(.8,8))


if __name__ == "__main__":
    unittest.main()
