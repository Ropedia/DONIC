"""Fetch pinned upstream HO-Cap source into an ignored operator dependency directory."""
import argparse,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES={'hocap':('HO-Cap-Annotation','https://github.com/JWRoboticsVision/HO-Cap-Annotation.git','f554ecdfdd1230f277a4dae443b104914caeb5a7')}
def run(args,cwd):return subprocess.check_output(args,cwd=cwd,text=True).strip()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('component',choices=SOURCES);p.add_argument('--dry-run',action='store_true');p.add_argument('--source-checkout',type=Path,help='Optional existing source checkout: read Git objects locally rather than downloading');a=p.parse_args()
 name,url,commit=SOURCES[a.component];dest=Path(os.getenv('HOI_EXTERNAL_ROOT',ROOT/'.external'))/name
 if a.dry_run:print(url,commit,'->',dest);return
 if (dest/'.git').is_dir():
  if run(['git','rev-parse','HEAD'],dest)!=commit:raise SystemExit('Existing dependency has a different commit; no checkout/reset performed.')
  if run(['git','diff','--name-only','HEAD'],dest):raise SystemExit('Existing dependency has modified tracked files; no changes made.')
  print('Pinned dependency already present');return
 if dest.exists():raise SystemExit('Refusing to populate an existing non-repository dependency directory.')
 dest.mkdir(parents=True)
 run(['git','init'],dest);run(['git','remote','add','origin',url],dest)
 run(['git','fetch','--depth','1',str(a.source_checkout.resolve()) if a.source_checkout else 'origin',commit],dest);run(['git','checkout','--detach','FETCH_HEAD'],dest)
 if run(['git','rev-parse','HEAD'],dest)!=commit:raise SystemExit('Fetched commit mismatch')
 print('Fetched',name,commit)
if __name__=='__main__':main()
