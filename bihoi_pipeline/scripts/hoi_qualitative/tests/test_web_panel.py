"""Run with the handobj_webui environment; validates browser-to-runner contracts."""
import os
os.environ['GRADIO_ANALYTICS_ENABLED']='False'
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from web_panel import all_sequences,batch_frame_range,build_config,dataset_path,Controller,discover,group_dir,output_root,texture_command,textured_status

class WebPanelTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        data=self.root/'rigid/test/sequence';export=data/'standardized_hoi_30fps';export.mkdir(parents=True)
        (export/'metadata.json').write_text(json.dumps({'resampled':False}))
        np.save(export/'source_frame_ids.npy',np.array([0,1,2,3]))
        for n in ['obj_pose.npz','object_mesh.ply','mano_left.npz','mano_right.npz']:(export/n).touch()
        for n in ['intri.yml','extri.yml']:(data/n).touch()
        self.v=[str(self.root),'test','sequence','3',True,'vertex',640,360,'0,2',True,50,15,0,'4','640x360','fixed_crop',False,2]
    def tearDown(self):self.tmp.cleanup()
    def test_video_ignores_novel_parameters(self):
        v=self.v.copy();v[8]='invalid unused frames';v[10]='invalid unused yaw'
        c=build_config(*v,'video');self.assertNotIn('figure',c);self.assertEqual(c['video']['end_frame_exclusive'],4)
    def test_figure_ignores_video_parameters(self):
        v=self.v.copy();v[12]='invalid unused start';v[14]='invalid unused resolution'
        c=build_config(*v,'figure');self.assertNotIn('video',c);self.assertEqual(c['figure']['frames'],[0,2])
    def test_missing_texture_not_silently_substituted(self):
        v=self.v.copy();v[5]='texture'
        with self.assertRaisesRegex(ValueError,'Missing'):build_config(*v,'figure')
    def test_video_camera_defaults_to_real(self):
        self.assertEqual(build_config(*self.v,'video')['video']['custom_camera'],{'type':'real'})
    def test_custom_camera_parameters_are_validated(self):
        c=build_config(*self.v,'video','orbit',30,10,.8,0,.05,0)['video']['custom_camera']
        self.assertEqual(c['type'],'orbit');self.assertEqual(c['offset_right_up_forward_m'],[0,.05,0])
        c=build_config(*self.v,'video','world',world_position='0.5，-0.3 1.2')['video']['custom_camera']
        self.assertEqual(c,{'type':'world','position':[.5,-.3,1.2],'look_at':None})
        with self.assertRaisesRegex(ValueError,'3 个数值'):build_config(*self.v,'video','world',world_position='1,2')
        with self.assertRaisesRegex(ValueError,'> 0'):build_config(*self.v,'video','orbit',orbit_scale=0)
        self.assertEqual(build_config(*self.v,'video','real',roll=15)['video']['custom_camera'],{'type':'real','roll_degrees':15.0})
        self.assertEqual(build_config(*self.v,'video','world',world_position='1,2,3',roll=-90)['video']['custom_camera']['roll_degrees'],-90.0)
        with self.assertRaisesRegex(ValueError,'roll'):build_config(*self.v,'video','real',roll='abc')
    def test_snapshot_frame_must_exist(self):
        self.assertEqual(build_config(*self.v,'snapshot',snapshot_frame=2)['snapshot'],{'frame':2})
        with self.assertRaisesRegex(ValueError,'Missing'):build_config(*self.v,'snapshot',snapshot_frame=9)
    def test_reject_path_escape(self):
        with self.assertRaises(ValueError):dataset_path(self.root,'test','../../')
    def test_reject_missing_video_frame(self):
        np.save(self.root/'rigid/test/sequence/standardized_hoi_30fps/source_frame_ids.npy',[0,1,3])
        with self.assertRaisesRegex(ValueError,'Missing'):build_config(*self.v,'video')
    def test_poll_never_publishes_partial_or_stale_outputs(self):
        output=self.root/'artifacts';output.mkdir();(output/'figure.png').write_text('partial')
        (output/'status.json').write_text('{"status":"complete"}')
        meta={'state':{'status':'running'},'media_export':{'output':str(output),'mode':'figure','config':str(self.root/'config.json')}}
        c=Controller({'_find_run':lambda _:meta,'_run_status_md':lambda _:'running','_run_log_view':lambda *_:'log'})
        result=c.poll('id','');self.assertIsNone(result[2]);self.assertIsNone(result[3])
        self.assertEqual(c.poll('id','id:running')[-1],'id:running')
    def backend(self,**extra):
        class Step:
            def __init__(s,title,cwd,command,conda_env,gpu=None):s.cwd,s.conda_env=cwd,conda_env
        return {'RUNS_DIR':self.root/'runs','Step':Step,'_active_runs':lambda:[],'_shell_for_step':lambda _:'',
                '_write_json_atomic':lambda p,d:Path(p).write_text(json.dumps(d)),'_launch_detached':lambda *_:'test',
                '_parse_gpu_list':lambda t:[] if t=='auto' else [int(x) for x in t.split(',')],**extra}
    def test_submit_writes_media_under_output_reconstruction(self):
        run_id=Controller(self.backend()).submit('video',False,self.v,'auto');run=self.root/'runs'/run_id
        expected=self.root.resolve()/'output_reconstruction/test/sequence'/run_id
        self.assertEqual(json.loads((run/'media_config.json').read_text())['output'],str(expected))
        self.assertEqual(json.loads((run/'meta.json').read_text())['media_export']['output'],str(expected/'final/video'))
        (self.root/'rigid/test/sequence/takes/take02').mkdir(parents=True,exist_ok=True)
        self.assertEqual(output_root(self.root,'test','sequence/takes/take02'),self.root.resolve()/'output_reconstruction/test/sequence/takes/take02')

class BatchTests(WebPanelTests):
    def setUp(self):
        super().setUp();import shutil
        seq=self.root/'rigid/test'
        shutil.copytree(seq/'sequence',seq/'late');np.save(seq/'late/standardized_hoi_30fps/source_frame_ids.npy',np.array([2,3]))
        shutil.copytree(seq/'sequence',seq/'broken');(seq/'broken/standardized_hoi_30fps/obj_pose.npz').unlink()
        shutil.copytree(seq/'sequence/standardized_hoi_30fps',seq/'sequence/takes/take02/standardized_hoi_30fps')
        (seq/'no_export').mkdir()
    def test_all_sequences_lists_exports_and_takes(self):
        self.assertEqual(all_sequences(self.root,'test'),['broken','late','sequence','sequence/takes/take02'])
        self.assertEqual(all_sequences(self.root,'../x'),[])
    def test_submit_batch_one_step_per_sequence(self):
        rid,_,skipped=Controller(self.backend()).submit_batch('video',False,self.v,['sequence','late','broken','late'],'0,1',8,'full')
        self.assertEqual([s['dataset'] for s in skipped],['broken']);self.assertIn('Missing',skipped[0]['error'])
        meta=json.loads((self.root/'runs'/rid/'meta.json').read_text())
        self.assertEqual([s['logical_data_name'] for s in meta['steps']],['rigid/test/sequence','rigid/test/late'])
        self.assertEqual(meta['scheduler']['parallel_datasets'],2);self.assertEqual(meta['scheduler']['allowed_gpus'],[0,1])
        self.assertNotIn('media_export',meta)
        items=meta['media_batch']['items']
        self.assertEqual(items[1]['output'],str(self.root.resolve()/'output_reconstruction/test/late'/rid/'final/video'))
        late=json.loads(Path(items[1]['config']).read_text())['video']
        self.assertEqual((late['start_frame'],late['end_frame_exclusive']),(2,None))
    def test_batch_inherits_start_end_and_clips_to_each_sequence(self):
        v=self.v.copy();v[12]=1;v[13]=''
        rid,items,skipped=Controller(self.backend()).submit_batch('video',False,v,['sequence','late'],'auto',4)
        self.assertEqual(skipped,[])
        ranges=[(c['start_frame'],c['end_frame_exclusive']) for c in (json.loads(Path(i['config']).read_text())['video'] for i in items)]
        self.assertEqual(ranges,[(1,None),(2,None)])
        self.assertEqual(items[0]['note'],'');self.assertIn('2–4',items[1]['note'])
        v[12]=0;v[13]='2'
        rid,items,skipped=Controller(self.backend()).submit_batch('video',False,v,['sequence','late'],'auto',4,'inherit')
        self.assertEqual(json.loads(Path(items[0]['config']).read_text())['video']['end_frame_exclusive'],2)
        self.assertEqual([s['dataset'] for s in skipped],['late']);self.assertIn('无交集',skipped[0]['error'])
    def test_batch_frame_range(self):
        ids=np.array([225,226,227])
        self.assertEqual(batch_frame_range(ids,70,'','inherit'),(225,'','帧段裁剪为 225–228（该序列已有 225–228）'))
        self.assertEqual(batch_frame_range(ids,226,'300','inherit')[:2],(226,228))
        self.assertEqual(batch_frame_range(ids,70,'100','full'),(225,'',''))
        self.assertEqual(batch_frame_range(np.arange(300),70,None,'inherit'),(70,'',''))
    def test_batch_rejects_when_nothing_valid(self):
        with self.assertRaisesRegex(ValueError,'broken'):Controller(self.backend()).submit_batch('video',False,self.v,['broken'],'auto',4,'full')
        with self.assertRaisesRegex(ValueError,'勾选'):Controller(self.backend()).submit_batch('video',False,self.v,[],'auto',4,'full')
        self.assertFalse((self.root/'runs').exists())
    def test_batch_snapshot_is_always_preview(self):
        rid,_,_=Controller(self.backend()).submit_batch('snapshot',False,self.v+['real',0,0,1,0,0,0,'','',0,2],['sequence'],'auto',4,'full')
        meta=json.loads((self.root/'runs'/rid/'meta.json').read_text())
        self.assertTrue(meta['media_batch']['preview']);self.assertIn('--preview',meta['steps'][0]['command'])
    def test_batch_poll_and_zip_only_completed_items(self):
        c=Controller(self.backend());rid,_,_=c.submit_batch('video',False,self.v,['sequence','late'],'auto',4,'full')
        meta=json.loads((self.root/'runs'/rid/'meta.json').read_text())
        meta.update(dir=str(self.root/'runs'/rid),state={'status':'running','steps':{'0':{'status':'done'},'1':{'status':'running'}}})
        out=Path(meta['media_batch']['items'][0]['output']);out.mkdir(parents=True)
        (out/'status.json').write_text('{"status":"complete"}');(out/'camera_white.mp4').write_text('v');(out/'first_frame.png').write_text('i')
        c.b.update({'_find_run':lambda _:meta,'_run_status_md':lambda _:'running','_list_runs':lambda:[meta]})
        status,items,signature=c.batch_poll(rid,None)
        self.assertIn('late',status);self.assertEqual(items.choices,[('sequence','sequence')])
        self.assertEqual(c.batch_poll(rid,signature)[2],signature)
        self.assertEqual(c.batch_item_media(rid,'sequence'),(str(out/'first_frame.png'),str(out/'camera_white.mp4')))
        self.assertEqual(c.batch_item_media(rid,'late'),(None,None))
        path,n=c.batch_zip(rid);self.assertEqual(n,1)
        import zipfile
        self.assertEqual(sorted(zipfile.ZipFile(path).namelist()),['sequence/camera_white.mp4'])
        self.assertEqual(len(c.batch_choices()),1)

class DatasetSelectionTests(WebPanelTests):
    def test_changing_dataset_changes_all_model_paths(self):
        import shutil
        shutil.copytree(self.root/'rigid/test/sequence',self.root/'rigid/test/other')
        v=self.v.copy();v[2]='other'
        c=build_config(*v,'figure')
        self.assertEqual(Path(c['dataset']).name,'other')
        self.assertEqual(Path(c['export']).parent.name,'other')
        self.assertNotIn('fwd_04',json.dumps(c))

    def test_visualization_choice_is_used_not_first_dataset(self):
        import shutil
        shutil.copytree(self.root/'rigid/test/sequence',self.root/'rigid/test/other')
        result=discover(str(self.root),'test',['sequence','other'],'sequence','other')
        self.assertEqual(result.value,'other')
        self.assertIsNone(discover(str(self.root),'test',['sequence','other']).value)

    def test_visualization_can_select_dataset_not_checked_for_reconstruction(self):
        import shutil
        shutil.copytree(self.root/'rigid/test/sequence',self.root/'rigid/test/other')
        result=discover(str(self.root),'test',['sequence'],'sequence','other')
        self.assertEqual(result.value,'other')
        self.assertIsNone(discover(str(self.root),'test',['sequence'],'sequence','missing').value)

    def test_history_is_filtered_to_current_dataset(self):
        a=self.root/'rigid/test/sequence';b=self.root/'rigid/test/other'
        metas=[{'run_id':'a','datasets':['sequence'],'steps':[{'dataset_path':str(a)}],'media_export':{'mode':'figure'},'state':{'status':'done'}},
               {'run_id':'b','datasets':['other'],'steps':[{'dataset_path':str(b)}],'media_export':{'mode':'figure'},'state':{'status':'done'}}]
        c=Controller({'_list_runs':lambda:metas})
        self.assertEqual([rid for _,rid in c.choices(b)],['b'])

    def test_old_completed_result_is_cleared_on_dataset_switch(self):
        meta={'steps':[{'dataset_path':str(self.root/'rigid/test/sequence')}],
              'state':{'status':'done'},'media_export':{'output':'old_output','mode':'figure'}}
        c=Controller({'_find_run':lambda _:meta})
        result=c.poll('old','old:done',self.root/'rigid/test/other')
        self.assertIsNone(result[2]);self.assertIsNone(result[3]);self.assertEqual(result[4],[])

class TexturedModelTests(WebPanelTests):
    def setUp(self):
        super().setUp()
        import hashlib
        self.group=self.root/'rigid/test/sequence';meshes=self.group/'meshes';meshes.mkdir()
        for n in ['images1','mask','depth']:(self.group/n).mkdir()
        (self.group/'images1/1.jpg').touch();(meshes/'compressed.ply').write_bytes(b'ply')
        (meshes/'metric_scale_info.json').write_text(json.dumps({'args':{'mesh_cam_ids':'1,5','depth_cam_ids':'1,2,3','depth_unit':'mm',
            'pixel_stride':2,'mask_erode':1,'max_points_per_cam':50000,'max_size_mb':9.8,'seed':42,'depth_scale':None,'save_debug_pointcloud':True,'generate_texture':False}}))
        self.code=self.root/'code';(self.code/'MV-SAM3D').mkdir(parents=True)
        for n in ['run_mv_sam3d_metric_scale.py','texture_export.py']:(self.code/'MV-SAM3D'/n).touch()
        self.sha=lambda b:hashlib.sha256(b).hexdigest()
    def test_takes_share_the_group_textured_model(self):
        self.assertEqual(group_dir(self.group/'takes/take02'),self.group);self.assertEqual(group_dir(self.group),self.group)
    def test_command_replays_recorded_reconstruction_with_texture(self):
        command,recorded=texture_command(self.group,self.code)
        self.assertEqual(recorded['mesh_cam_ids'],'1,5')
        for token in ['--generate_texture','--mesh_cam_ids 1,5','--depth_cam_ids 1,2,3','--depth_unit mm','--seed 42','--save_debug_pointcloud',
                      f'--setting_name {self.group.name}','before_texture','compressed.ply','metric_scale_info.json','CUDA_VISIBLE_DEVICES="$GPU_PICK"']:
            self.assertIn(token,command)
        self.assertNotIn('--depth_scale',command);self.assertNotIn('--generate_texture False',command);self.assertNotIn('MESH_CAMS',command)
    def test_verified_textured_model_is_generated_only_once(self):
        meshes=self.group/'meshes';(meshes/'textured.glb').write_bytes(b'glb')
        (meshes/'textured.json').write_text(json.dumps({'status':'complete','glb_sha256':self.sha(b'glb'),'tracking_ply_sha256':self.sha(b'ply')}))
        self.assertIsNotNone(textured_status(meshes))
        with self.assertRaisesRegex(ValueError,'只生成一次'):texture_command(self.group,self.code)
    def test_stale_textured_model_does_not_block_regeneration(self):
        meshes=self.group/'meshes';(meshes/'textured.glb').write_bytes(b'glb')
        (meshes/'textured.json').write_text(json.dumps({'status':'complete','glb_sha256':self.sha(b'glb'),'tracking_ply_sha256':self.sha(b'other')}))
        self.assertIsNone(textured_status(meshes));texture_command(self.group,self.code)
    def test_missing_reconstruction_inputs_are_rejected(self):
        (self.group/'meshes/compressed.ply').unlink()
        with self.assertRaisesRegex(ValueError,'Missing'):texture_command(self.group,self.code)
    def test_texture_task_is_listed_for_takes_of_its_group(self):
        meta={'run_id':'t','datasets':['sequence'],'steps':[{'dataset_path':str(self.group)}],'media_export':{'mode':'texture'},'state':{'status':'done'}}
        c=Controller({'_list_runs':lambda:[meta]})
        self.assertEqual([rid for _,rid in c.choices(self.group/'takes/take02')],['t'])
        meta['media_export']['mode']='figure';self.assertEqual(c.choices(self.group/'takes/take02'),[])
    def test_texture_poll_publishes_only_verified_model(self):
        meshes=self.group/'meshes';(meshes/'textured.glb').write_bytes(b'glb')
        (meshes/'textured.json').write_text(json.dumps({'status':'complete','texture_resolution':[1024,1024],'glb_sha256':self.sha(b'glb'),'tracking_ply_sha256':self.sha(b'ply')}))
        meta={'state':{'status':'done'},'steps':[{'dataset_path':str(self.group)}],'media_export':{'output':str(meshes),'mode':'texture','config':'cfg'}}
        c=Controller({'_find_run':lambda _:meta,'_run_status_md':lambda _:'done','_run_log_view':lambda *_:'log'})
        result=c.poll('t','',self.group);self.assertIn(str(meshes/'textured.glb'),result[4]);self.assertIn('通过校验',result[0])
        (meshes/'compressed.ply').write_bytes(b'changed')
        result=c.poll('t','',self.group);self.assertEqual(result[4],['cfg'])

if __name__=='__main__':unittest.main()
