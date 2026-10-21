"""Audit either the exact Git index or source tree; do not print secret values."""
import argparse,ast,hashlib,json,os,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ASSETS=json.loads((ROOT/'configs/source-assets.json').read_text())
RUNTIME={'.git','.external','.models','.calibration','.environments','.webui_runs','data','outputs','logs','__pycache__','build'}
FORBIDDEN={'.pt','.pth','.ckpt','.safetensors','.onnx','.pkl','.pickle','.npy','.npz','.mp4','.avi','.mov','.mkv','.png','.jpg','.jpeg','.gif','.webp','.ply','.obj','.stl','.glb','.gltf','.so','.o','.a','.zip','.tar','.bin'}
PATTERNS={
 'AWS access key':r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
 'GitHub token':r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
 'Hugging Face token':r'\bhf_[A-Za-z0-9]{30,}\b',
 'OpenAI secret':r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b',
 'Slack token':r'\bxox[baprs]-[A-Za-z0-9-]{20,}\b',
 'Private key':r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
 'Credential in URL':r'https?://[^\s/"\']+:[^\s/@"\']+@',
}
def git(args):return subprocess.check_output(['git',*args],cwd=ROOT)
def source_files():
 if (ROOT/'.git').exists():
  names=git(['ls-files','--cached','--others','--exclude-standard','-z']).decode().split('\0')
  yield from sorted(set(name for name in names if name))
  return
 for base,dirs,files in os.walk(ROOT,followlinks=False):
  dirs[:]=sorted(d for d in dirs if not (
   (Path(base)==ROOT and d in RUNTIME)
   or (Path(base)/d).relative_to(ROOT).as_posix() in {'pipeline/data','pipeline/rigid','pipeline/.webui_data_roots'}
   or d in {'__pycache__','build','.git'}
   or (Path(base)/d).is_symlink()))
  for name in sorted(files):
   p=Path(base)/name
   if p.is_symlink():continue
   yield str(p.relative_to(ROOT))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--staged',action='store_true');p.add_argument('--syntax-only',action='store_true');p.add_argument('--json-output',type=Path);a=p.parse_args()
 files=list(source_files());index={};issues=[];reviews=[];total=0
 if a.staged:
  entries=git(['ls-files','--stage','-z']).split(b'\0');files=[]
  for entry in entries:
   if not entry:continue
   fields,path=entry.split(b'\t',1);mode,sha,stage=fields.decode().split();path=path.decode();files.append(path);index[path]=sha
   if mode=='160000':issues.append({'path':path,'type':'accidental Git submodule'})
   if mode=='120000':issues.append({'path':path,'type':'unexpected publishable source symlink'})
   if stage!='0':issues.append({'path':path,'type':'unresolved index stage'})
 for name in files:
  path=ROOT/name;data=git(['cat-file','blob',index[name]]) if a.staged else path.read_bytes();total+=len(data)
  allowed_asset=ASSETS.get(name)==hashlib.sha256(data).hexdigest()
  if len(data)>5*1024*1024 and not allowed_asset:issues.append({'path':name,'type':'source larger than 5 MiB','bytes':len(data)})
  if path.suffix.lower() in FORBIDDEN:issues.append({'path':name,'type':'forbidden runtime/data extension'})
  if path.name.startswith('J_regressor_mano_') and path.suffix.lower()=='.txt':issues.append({'path':name,'type':'learned MANO parameters stored as text'})
  if path.name in ['.env','intri.yml','extri.yml','corrected_gsformat.pkl'] or name.startswith(('.external/','.models/','.environments/','.calibration/')):issues.append({'path':name,'type':'private/runtime path'})
  if a.staged and subprocess.run(['git','check-ignore','--no-index','-q','--',name],cwd=ROOT).returncode==0:issues.append({'path':name,'type':'tracked despite .gitignore'})
  try:text=data.decode('utf8')
  except UnicodeDecodeError:
   if name!='pipeline/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz' and not allowed_asset:issues.append({'path':name,'type':'unexpected binary source'})
   continue
  if path.suffix=='.py':
   try:ast.parse(text,filename=name)
   except SyntaxError as e:issues.append({'path':name,'type':'Python syntax','line':e.lineno,'detail':e.msg})
  if path.suffix=='.sh':
   result=subprocess.run(['bash','-n'],input=text,text=True,capture_output=True)
   if result.returncode:issues.append({'path':name,'type':'Shell syntax','detail':result.stderr})
  if a.syntax_only:continue
  for label,pattern in PATTERNS.items():
   for m in re.finditer(pattern,text):issues.append({'path':name,'line':text[:m.start()].count('\n')+1,'type':label})
  # Flag literal credential assignments for human review, without exposing values.
  for m in re.finditer(r'(?im)\b(?:password|passwd|api_key|access_token|auth_token)\s*(?:=|:)\s*["\']([^"\'\n]{6,})["\']',text):
   value=m.group(1)
   if not any(x in value.lower() for x in ['example','replace','placeholder','your','${','getenv','token_urlsafe','password','default']):reviews.append({'path':name,'line':text[:m.start()].count('\n')+1,'type':'literal credential assignment review'})
 report={'scope':'exact Git index' if a.staged else 'source tree excluding runtime roots','files':len(files),'bytes':total,'issues':issues,'literal_assignment_reviews':reviews,'largest_files':sorted([{'path':f,'bytes':len(git(['cat-file','blob',index[f]])) if a.staged else (ROOT/f).stat().st_size} for f in files],key=lambda r:r['bytes'],reverse=True)[:10]}
 if a.json_output:a.json_output.parent.mkdir(parents=True,exist_ok=True);a.json_output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2));return 1 if issues or reviews else 0
if __name__=='__main__':sys.exit(main())
