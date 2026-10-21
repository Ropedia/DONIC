"""Explicit official checkpoint downloads; never called by basic install.sh."""
import argparse,json,os,shutil,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 manifest=json.loads((ROOT/'configs/models/manifest.json').read_text())
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--component',choices=manifest,required=True);p.add_argument('--yes',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 spec=manifest[a.component];model=Path(os.getenv('HOI_MODEL_ROOT',ROOT/'.models'))/a.component
 if 'manual' in spec:print(spec['manual']);return
 print('Official source:',spec.get('repo',spec.get('url')),'revision:',spec.get('revision','dated official release'),'destination:',model)
 if spec.get('gated'):print('Gated model: accept upstream terms and authenticate Hugging Face before downloading. Tokens stay in environment/cache.')
 if a.dry_run:return
 if not a.yes:raise SystemExit('No files downloaded. Rerun with --yes after reviewing model licenses and disk requirements.')
 model.mkdir(parents=True,exist_ok=True)
 if 'url' in spec:
  dest=model/spec['file']
  if dest.exists():raise SystemExit('Existing SAM2 checkpoint left untouched; validate it before rerunning.')
  temp=dest.with_suffix(dest.suffix+'.part')
  with urllib.request.urlopen(spec['url']) as response,temp.open('xb') as output:shutil.copyfileobj(response,output)
  temp.replace(dest)
 else:
  from huggingface_hub import hf_hub_download
  for remote,relative in spec['files'].items():
   target=model/relative
   if target.exists():print('Existing file left untouched:',relative);continue
   cached=hf_hub_download(repo_id=spec['repo'],repo_type=spec.get('repo_type','model'),revision=spec['revision'],filename=remote,token=os.environ.get('HF_TOKEN'))
   target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(cached,target)
 print('Download complete. Run scripts/setup_assets.py and scripts/check_env.sh.')
if __name__=='__main__':main()
