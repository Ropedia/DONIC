"""Validate package-local imports, environment prefixes, hardware, and user-provided assets."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODULES={
 'handobj_webui':['gradio','numpy','PIL','interaction_capture_webui','web_panel'],
 'easymocap':['cv2','scipy','torch','easymocap.smplmodel.body_param','easymocap.mytools.camera_utils','pyrender','single_sequence'],
 'sam3':['sam3','sam2.build_sam','timm','cv2'],
 'depthanything3':['depth_anything_3.api','cv2'],
 'sam3d-objects':['sam3d_objects','sam3d_objects.pipeline.inference_pipeline_pointmap','pytorch3d','kaolin','nvdiffrast.torch','spconv.pytorch','gsplat','moge.model.v1'],
 'foundationpose':['Utils','estimater','pytorch3d','nvdiffrast.torch','kornia'],
 'wapr':['wapr.estimator','wapr.tracking','nvdiffrast.torch','cv2','scipy'],
 'wilor':['wilor','demo_fast','ultralytics','smplx'],
 'hocap-annotation':['hocap_annotation.loss','meshsdf_loss_cuda','smplx','pyrender']}
def env_prefix(name):
 value=os.getenv('HOI_ENV_'+name.upper().replace('-','_'),str(Path(os.getenv('HOI_ENV_ROOT',ROOT/'.environments'))/name))
 if not Path(value).is_absolute():
  conda=os.getenv('CONDA_EXE') or shutil.which('conda')
  if not conda:raise RuntimeError('Named environment requires CONDA_EXE')
  value=str(Path(subprocess.check_output([conda,'info','--base'],text=True).strip())/'envs'/value)
 return Path(value)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--component',choices=['all',*MODULES],default='all');p.add_argument('--skip-models',action='store_true');p.add_argument('--skip-gpu',action='store_true');p.add_argument('--json-output',type=Path);a=p.parse_args()
 selected=MODULES if a.component=='all' else {a.component:MODULES[a.component]};results=[]
 def record(kind,name,ok,detail):
  rec={'kind':kind,'name':name,'ok':bool(ok),'detail':detail};results.append(rec);print(('PASS' if ok else 'FAIL'),kind,name,detail)
 for exe in ['ffmpeg','ffprobe','git']:record('executable',exe,shutil.which(exe) is not None,shutil.which(exe))
 if not a.skip_gpu and any(n!='handobj_webui' for n in selected):
  try:
   x=subprocess.run(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=30)
   record('driver','nvidia-smi',x.returncode==0,x.stdout.strip() or x.stderr.strip())
   if x.returncode==0:
    ids={int(line.split(',')[0]) for line in x.stdout.splitlines() if line.strip()}
    allowed=os.getenv('WEBUI_ALLOWED_GPUS','all')
    if allowed.lower() not in ['all','auto','']:
     try:requested={int(n.strip()) for n in allowed.split(',')};ok=bool(requested) and requested<=ids
     except ValueError:ok=False
     record('gpu configuration','WEBUI_ALLOWED_GPUS',ok,allowed)
    budget=int(os.getenv('WEBUI_VRAM_PER_STEP_MB','32768'));record('gpu configuration','VRAM budget',budget>0,budget)

  except Exception as e:record('driver','nvidia-smi',False,str(e))
 for name,modules in selected.items():
  try:python=env_prefix(name)/'bin/python'
  except Exception as e:record('environment',name,False,str(e));continue
  if not python.is_file():record('environment',name,False,'Missing interpreter');continue
  # Always put the NEW sources before the existing environment's editable upstream installs.
  paths=[ROOT/'web_demos/third_party/WAPR',ROOT/'webui',ROOT/'pipeline',ROOT/'scripts/hoi_qualitative',ROOT/'pipeline/FoundationPose',ROOT/'pipeline/sam3',ROOT/'pipeline/sam3/sam2',ROOT/'pipeline/Depth-Anything-3/src',ROOT/'pipeline/MV-SAM3D',ROOT/'pipeline/WiLoR',Path(os.getenv('HOI_EXTERNAL_ROOT',ROOT/'.external'))/'HO-Cap-Annotation']
  code="""
import sys,importlib,json
sys.path[:0]=json.loads(sys.argv[1])
out=[]
for name in json.loads(sys.argv[2]):
 try:
  m=importlib.import_module(name)
  local_roots={'wapr':'web_demos/third_party/WAPR/wapr','interaction_capture_webui':'webui','web_panel':'scripts/hoi_qualitative','easymocap':'pipeline/easymocap','sam3':'pipeline/sam3','sam2':'pipeline/sam3/sam2','depth_anything_3':'pipeline/Depth-Anything-3','sam3d_objects':'pipeline/MV-SAM3D','Utils':'pipeline/FoundationPose','estimater':'pipeline/FoundationPose','demo_fast':'pipeline/WiLoR','hocap_annotation':'.external/HO-Cap-Annotation'}
  part=local_roots.get(name.split('.')[0])
  if part and getattr(m,'__file__',None):
   from pathlib import Path
   root=Path(sys.argv[4]);expected=Path(__import__('os').environ.get('HOI_EXTERNAL_ROOT',root/'.external'))/'HO-Cap-Annotation' if name.startswith('hocap_annotation') else root/part
   if not Path(m.__file__).resolve().is_relative_to(expected.resolve()):raise ImportError('Module resolved outside packaged source: '+m.__file__)
  out.append({'module':name,'ok':True,'path':getattr(m,'__file__',None)})
 except Exception as e:out.append({'module':name,'ok':False,'error':type(e).__name__+': '+str(e)})
if sys.argv[3]=='1':
 try:
  import torch
  out.append({'module':'torch.cuda','ok':torch.cuda.is_available(),'torch':torch.__version__,'cuda':torch.version.cuda,'devices':torch.cuda.device_count()})
 except Exception as e:out.append({'module':'torch.cuda','ok':False,'error':str(e)})
print('HOI_CHECK_JSON='+json.dumps({'python':sys.version.split()[0],'imports':out}))
"""
  try:
   x=subprocess.run([str(python),'-c',code,json.dumps(list(map(str,paths))),json.dumps(modules),'1' if name!='handobj_webui' and not a.skip_gpu else '0',str(ROOT)],cwd=ROOT,capture_output=True,text=True,timeout=180)
   line=next((line for line in reversed(x.stdout.splitlines()) if line.startswith('HOI_CHECK_JSON=')),None)
   if line is None:record('environment',name,False,(x.stdout+x.stderr)[-4000:]);continue
   detail=json.loads(line.split('=',1)[1]);record('environment',name,all(i['ok'] for i in detail['imports']),detail)
  except Exception as e:record('environment',name,False,str(e))
 if not a.skip_models:
  model=Path(os.getenv('HOI_MODEL_ROOT',ROOT/'.models'));manifest=json.loads((ROOT/'configs/models/manifest.json').read_text())
  needed={'sam3':['sam3','sam2'],'depthanything3':['da3'],'sam3d-objects':['mv-sam3d','moge'],'foundationpose':['foundationpose'],'wapr':['foundationpose'],'wilor':['wilor','mano'],'easymocap':['mano'],'hocap-annotation':['mano']}
  for part in sorted({part for name in selected for part in needed.get(name,[])}):
   spec=manifest[part]
   files=spec['required_files'] if 'required_files' in spec else list(spec.get('files',{}).values()) if 'files' in spec else [spec['file']] if 'file' in spec else ['MANO_LEFT.pkl','MANO_RIGHT.pkl'] if part=='mano' else ['2023-10-28-18-33-37/model_best.pth','2024-01-11-20-02-45/model_best.pth']
   asset_root=Path(os.getenv('HOI_MANO_DIR',model/part)) if part=='mano' else model/part
   missing=[f for f in files if not (asset_root/f).is_file() or (asset_root/f).stat().st_size==0]
   record('models',part,not missing,{'missing':missing,'root':str(asset_root)})
  if a.component!='handobj_webui':
   cal=Path(os.getenv('HOI_CALIBRATION_DIR',ROOT/'.calibration'));missing=[f for f in ['intri.yml','extri.yml','corrected_gsformat.pkl'] if not (cal/f).is_file()]
   record('calibration','rig',not missing,{'missing':missing,'root':str(cal)})
 if a.json_output:a.json_output.parent.mkdir(parents=True,exist_ok=True);a.json_output.write_text(json.dumps(results,indent=2)+'\n')
 return 0 if all(r['ok'] for r in results) else 1
if __name__=='__main__':sys.exit(main())
