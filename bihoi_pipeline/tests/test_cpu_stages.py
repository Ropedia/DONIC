"""Synthetic fixtures only: FFmpeg extraction, numbering, masks, calibration synchronization."""
import importlib.util,json,math,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'webui'),str(ROOT/'pipeline')]
from rename_videos_31cams import CALIB_CAMERA_ORDER
class CPUStageTests(unittest.TestCase):
 def run_script(self,relative,*args):
  result=subprocess.run([sys.executable,str(ROOT/relative),*map(str,args)],cwd=ROOT,capture_output=True,text=True)
  self.assertEqual(result.returncode,0,result.stdout+result.stderr);return result
 def test_video_numbering_reference_and_pts_extraction(self):
  with tempfile.TemporaryDirectory(prefix='hoi CPU fixture ') as directory:
   root=Path(directory);raw=root/'raw';raw.mkdir();dataset=root/'dataset';videos=dataset/'videos'
   for name in CALIB_CAMERA_ORDER[:2]:
    result=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi','-i','color=c=black:s=64x36:r=10:d=1.2','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(raw/(name+'.mp4'))],capture_output=True,text=True)
    self.assertEqual(result.returncode,0,result.stderr)
   self.run_script('pipeline/rename_videos_31cams.py','--input_dir',raw,'--output_dir',videos,'--allow-missing')
   self.assertEqual(sorted(p.name for p in videos.iterdir()),['1.mp4','2.mp4'])
   self.assertEqual(len(list(raw.glob('*.mp4'))),2)
   result=self.run_script('webui/shortest_video_end.py','--video_dir',videos,'--fps',10,'--max-end-sec',.8)
   self.assertAlmostEqual(float(result.stdout.strip()),.8)
   self.run_script('webui/extract_reference_frames.py','--video_dir',videos,'--output_dir',dataset/'images1')
   self.assertEqual(len(list((dataset/'images1').glob('*.jpg'))),2)
   self.run_script('pipeline/rename_videos_extract_frames_ffmpeg_pts_compat.py','--data-name',dataset,'--start-sec',0,'--end-sec',.8,'--fps',10,'--workers',2,'--ffmpeg-threads',1,'--skip-rename','--execute','--verify-count')
   counts=[len(list(p.glob('*.jpg'))) for p in (dataset/'images').iterdir()]
   self.assertEqual(counts,[8,8])
 def test_equalization_dry_run_preserves_fixture_files(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory)
   for cam,count in [('1',3),('2',2)]:
    folder=root/'images'/cam;folder.mkdir(parents=True)
    for idx in range(count):Image.new('RGB',(16,16)).save(folder/f'{idx:06}.jpg')
   self.run_script('webui/equalize_camera_frames.py','--data_dir',root,'--dirs','images','--dry-run')
   self.assertEqual(len(list((root/'images/1').iterdir())),3)
   self.run_script('webui/equalize_camera_frames.py','--data_dir',root,'--dirs','images')
   self.assertEqual([len(list(p.iterdir())) for p in (root/'images').iterdir()],[2,2])
 def test_masks_four_ring_selection_and_calibration_sync(self):
  import cv2
  from easymocap.mytools.camera_utils import write_camera,read_camera
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'mask').mkdir();cams={}
   for idx in range(1,33):
    radius=1+(idx-1)//8;angle=(idx-1)%8*math.pi/4;center=np.array([radius*math.cos(angle),radius*math.sin(angle),0.])
    cams[str(idx)]={'K':np.array([[100.,0,16],[0,100,16],[0,0,1]]),'dist':np.zeros((1,5)),'R':np.eye(3),'Rot':np.eye(3),'T':-center.reshape(3,1)}
    Image.new('L',(32,32),255).save(root/'mask'/f'{idx}.png')
    if idx<=31:
     (root/'images'/str(idx)).mkdir(parents=True)
     Image.new('RGB',(32,32)).save(root/'images'/str(idx)/'000000.jpg')
   write_camera(cams,str(root))
   result=self.run_script('webui/mask_views.py','select','--data_dir',root)
   assignments=dict(part.strip().split('=',1) for part in result.stdout.strip().split(';'))
   self.assertEqual(len(assignments['MESH_CAMS'].strip('"\'').split(',')),12)
   self.run_script('webui/sync_calibration.py','--data_dir',root)
   final=read_camera(str(root/'intri.yml'),str(root/'extri.yml'));final.pop('basenames',None)
   self.assertEqual(len(final),31);self.assertTrue((root/'intri.yml.full').exists())
if __name__=='__main__':unittest.main()
