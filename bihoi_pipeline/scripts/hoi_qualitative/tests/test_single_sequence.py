import os
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE','0')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from single_sequence import camera_for_crop,crop_box,video_ids,project_bounds,ProjectionError,retreat_camera,custom_extrinsic
import trimesh

class SingleSequenceTests(unittest.TestCase):
    def test_retreat_preserves_orientation_and_moves_backward(self):
        e=np.eye(4);e[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
        adjusted,distance=retreat_camera(e,-.591181)
        self.assertAlmostEqual(distance,.691181)
        np.testing.assert_array_equal(adjusted[:3,:3],e[:3,:3])
        delta=np.linalg.inv(adjusted)[:3,3]-np.linalg.inv(e)[:3,3]
        np.testing.assert_allclose(delta,-e[2,:3]*distance)
        self.assertAlmostEqual(-.591181+distance,.1)
        unchanged,distance=retreat_camera(e,.02)
        np.testing.assert_array_equal(unchanged,e);self.assertEqual(distance,0.)
        with self.assertRaises(ValueError):retreat_camera(e,float('nan'))

    def test_crop_projection_is_same_as_pixel_crop_resize(self):
        k=np.array([[500.,0,320],[0,505,180],[0,0,1]])
        box=[110.,-20.,610.,280.];resolution=[1920,1080];aa=2
        _,new=camera_for_crop(k,box,resolution,aa)
        p=np.array([[.1,.2,1.],[-.3,.2,2.]])
        orig=p@k.T;orig=orig[:,:2]/orig[:,2:]
        expected=(orig-box[:2])*np.array(resolution)*aa/(np.array(box[2:])-box[:2])
        actual=p@new.T;actual=actual[:,:2]/actual[:,2:]
        np.testing.assert_allclose(actual,expected)

    def test_fixed_crop_contains_union_and_preserves_aspect(self):
        lo=np.array([-12.,24.]);hi=np.array([400.,360.])
        b=np.array(crop_box(lo,hi,16/9,1.22))
        self.assertTrue((b[:2]<lo).all() and (b[2:]>hi).all())
        self.assertAlmostEqual((b[2]-b[0])/(b[3]-b[1]),16/9)

    def test_near_plane_error_identifies_frame_camera_and_part(self):
        mesh=trimesh.Trimesh(vertices=[[0,0,-.02],[.01,0,.04],[0,.01,.06]],faces=[[0,1,2]],process=False)
        with self.assertRaises(ProjectionError) as caught:
            project_bounds([(mesh,'left')],np.eye(3),np.eye(4),5,'25')
        details=caught.exception.details
        self.assertEqual(details['source_frame'],5)
        self.assertEqual(details['camera'],'25')
        self.assertEqual(details['invalid_parts'][0]['part'],'left')
        self.assertEqual(details['invalid_parts'][0]['vertices_behind_near_plane'],1)
        self.assertIn('fixed cropping cannot fix',str(caught.exception))

    def test_missing_video_frames_are_not_skipped_or_replaced(self):
        with self.assertRaisesRegex(ValueError,'Missing video source frames'):
            video_ids(np.array([0,1,3]),0,4)
        self.assertEqual(video_ids(np.arange(10),2,5),[2,3,4])
        self.assertEqual(video_ids(np.arange(4),2,None),[2,3])

    def test_custom_camera_identity_orbit_is_the_real_camera(self):
        e=np.eye(4);e[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]];e[:3,3]=[.1,-.2,.9]
        spec={'type':'orbit','yaw_degrees':0,'pitch_degrees':0,'distance_scale':1,'offset_right_up_forward_m':[0,0,0]}
        np.testing.assert_allclose(custom_extrinsic(e,spec,[.3,.1,.2]),e,atol=1e-12)

    def test_custom_orbit_keeps_pivot_distance_and_view_of_pivot(self):
        e=np.eye(4);pivot=np.array([0.,0.,1.])  # camera at origin looking +z at the pivot
        new=custom_extrinsic(e,{'type':'orbit','yaw_degrees':90,'pitch_degrees':0,'distance_scale':.5},pivot)
        c2w=np.linalg.inv(new)
        self.assertAlmostEqual(np.linalg.norm(c2w[:3,3]-pivot),.5)
        np.testing.assert_allclose(core_project(new,pivot),[0,0],atol=1e-9)  # pivot stays on the optical axis
        self.assertAlmostEqual(abs(c2w[0,3]),.5)  # moved sideways, not up/down
        self.assertAlmostEqual(c2w[1,3],0)

    def test_custom_world_camera_looks_at_target_with_upright_horizon(self):
        e=np.eye(4);new=custom_extrinsic(e,{'type':'world','position':[1,0,0],'look_at':[0,0,1]},[9,9,9])
        np.testing.assert_allclose(core_project(new,[0,0,1]),[0,0],atol=1e-9)
        np.testing.assert_allclose(np.linalg.inv(new)[:3,3],[1,0,0])
        np.testing.assert_allclose(np.linalg.inv(new)[:3,1],[0,1,0],atol=1e-9)  # image down == reference down
        with self.assertRaises(ValueError):custom_extrinsic(e,{'type':'world','position':[0,0,1],'look_at':[0,0,1]},[0,0,0])
        with self.assertRaises(ValueError):custom_extrinsic(e,{'type':'world','position':[0,0,0],'look_at':[0,1,0]},[0,0,0])

    def test_roll_turns_picture_clockwise_about_optical_axis(self):
        e=np.eye(4);e[:3,3]=[.1,-.2,.3]
        new=custom_extrinsic(e,{'type':'real','roll_degrees':90},[0,0,0])
        np.testing.assert_allclose(np.linalg.inv(new)[:3,3],np.linalg.inv(e)[:3,3])  # position unchanged
        p=np.linalg.inv(e)[:3,:3]@np.array([.5,0,1.])+np.linalg.inv(e)[:3,3]    # right of center
        np.testing.assert_allclose(core_project(new,p),[0,.5],atol=1e-9)         # -> below center (y down)
        np.testing.assert_allclose(custom_extrinsic(e,{'type':'real','roll_degrees':0},[0,0,0]),e,atol=1e-12)
        world=custom_extrinsic(e,{'type':'world','position':[1,0,0],'look_at':[0,0,1],'roll_degrees':-30},[0,0,0])
        np.testing.assert_allclose(core_project(world,[0,0,1]),[0,0],atol=1e-9)  # still looks at the target


def core_project(e,p):
    v=e[:3,:3]@np.asarray(p,float)+e[:3,3];return v[:2]/v[2]

if __name__=='__main__':unittest.main()
