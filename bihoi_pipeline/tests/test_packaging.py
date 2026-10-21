import ast,importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'webui'),str(ROOT/'scripts/hoi_qualitative')]
import interaction_capture_webui as app
class PackageTests(unittest.TestCase):
 def test_import_is_packaged(self):self.assertEqual(Path(app.__file__).resolve(),ROOT/'webui/interaction_capture_webui.py')
 def test_ui_and_visualization_panel_construct(self):self.assertGreater(len(app.make_ui(str(ROOT/'data')).blocks),250)
 def test_configuration_files_parse(self):
  import yaml
  for p in (ROOT/'configs').rglob('*'):
   if p.suffix=='.json':json.loads(p.read_text())
   if p.suffix in ['.yaml','.yml']:yaml.safe_load(p.read_text())
 def test_path_traversal_is_rejected(self):
  for value in ['.','..','../outside','a/b','x;echo hi']:
   with self.assertRaises(ValueError):app._validate_name(value,'dataset')
 def test_all_stage_commands_and_shared_take_dependency(self):
  with tempfile.TemporaryDirectory(prefix='hoi commands with spaces ') as directory:
   root=Path(directory);(root/'rigid/prior/sequence/videos/take01').mkdir(parents=True)
   (root/'rigid/prior/sequence/videos/take02').mkdir(parents=True)
   records=[]
   for take in ['take01','take02']:
    job=app.Job('sequence','prior','a metal object',take=take)
    target=app._dataset_root(root,job);target.mkdir(parents=True,exist_ok=True)
    alias=app._ensure_pipeline_data_name(root,job)
    self.assertEqual((app.EASYMOCAP_ROOT/alias).resolve(),target.resolve())
    for stage in app.STAGE_KEYS:
     steps=app._steps_for(stage,job,app.EASYMOCAP_ROOT,root,0.,1.,10.,640,360,1,'all','auto','auto','auto','mm',2,1,50000,9.8,False,1000.,500,True,True)
     if stage=='meshscale' and take=='take02':self.assertEqual(steps,[]);continue
     self.assertTrue(steps,stage)
     for step in steps:
      self.assertTrue(step.cwd.exists(),str(step.cwd))
      shell=app._shell_for_step(step)
      result=subprocess.run(['bash','-n'],input=shell,text=True,capture_output=True)
      self.assertEqual(result.returncode,0,step.title+result.stderr)
      self.assertNotIn('/mnt/local/',shell)
      records.append((stage,take,step))
   links=[s for stage,take,s in records if take=='take02' and s.after]
   self.assertTrue(any(s.after['title']==app.MESH_RECON_STEP_TITLE for s in links))
 def test_hand_gpu_selection_obeys_outer_assignment(self):
  wrapper=(ROOT/'pipeline/run_extract_sapiens_easymocap_10fps_kalman.sh').read_text()
  function=wrapper.split('refresh_eligible_gpus() {',1)[1].split('\n}\n',1)[0]
  script='nvidia-smi() { printf "0, 140000\\n1, 139000\\n3, 138000\\n"; }\n'
  script+='die() { echo "$*" >&2; exit 1; }\nGPU_MIN_FREE_VRAM_MB=32768\n'
  script+='refresh_eligible_gpus() {'+function+'\n}\nrefresh_eligible_gpus\nprintf "%s\\n" "${ELIGIBLE_GPUS[@]}"\n'
  for pick,visible,expected in [('3','0,1,3',['3']),('','1,3',['1','3'])]:
   env=os.environ.copy();env.update(GPU_PICK=pick,CUDA_VISIBLE_DEVICES=visible)
   result=subprocess.run(['bash','-c',script],env=env,text=True,capture_output=True)
   self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stdout.strip().splitlines(),expected)
 def test_non_10fps_hands_are_rejected(self):
  with tempfile.TemporaryDirectory() as directory:
   (Path(directory)/'rigid/prior/sequence').mkdir(parents=True)
   with self.assertRaisesRegex(ValueError,'10 FPS'):
    app._plan_steps(['hands'],'prior',['sequence'],[['sequence','object']],'bimanual',directory,0.,1.,30.,640,360,1,'all','auto','auto','auto','mm',2,1,50000,9.8,False,1000.,500,skip_done=False)
 def test_model_download_is_opt_in_and_official(self):
  manifest=json.loads((ROOT/'configs/models/manifest.json').read_text())
  for name,spec in manifest.items():
   if 'repo' in spec:self.assertEqual(len(spec['revision']),40)
  script=ROOT/'scripts/download_models.py'
  result=subprocess.run([sys.executable,str(script),'--component','sam3'],capture_output=True,text=True)
  self.assertNotEqual(result.returncode,0);self.assertIn('No files downloaded',result.stderr)
 def test_asset_linking_can_be_rerun_without_touching_sources(self):
  with tempfile.TemporaryDirectory() as directory:
   spec=importlib.util.spec_from_file_location('assets',ROOT/'scripts/setup_assets.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   root=Path(directory);target=root/'weights';target.mkdir();dest=root/'legacy';module.link(dest,target);module.link(dest,target)
   conflict=root/'existing';conflict.mkdir()
   with self.assertRaises(RuntimeError):module.link(conflict,target)
if __name__=='__main__':unittest.main()
