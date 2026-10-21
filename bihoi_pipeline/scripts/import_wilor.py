"""Import an operator's licensed local WiLoR source without redistributing modifications."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
 manifest=json.loads((ROOT/'provenance/restricted-wilor.json').read_text());source=a.source.expanduser().resolve()
 dest=Path(os.environ.get('HOI_EXTERNAL_ROOT',ROOT/'.external'))/'WiLoR'
 # Validate everything before copying anything. Extra files/weights/.git are never copied.
 for rec in manifest['required_files']:
  file=source/rec['path']
  if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=rec['sha256']:
   raise SystemExit('Required pipeline variant missing or different: '+rec['path'])
 for rec in manifest['required_files']:
  target=dest/rec['path'];target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists() and target.read_bytes()!=(source/rec['path']).read_bytes():raise SystemExit('Refusing to overwrite different file: '+str(target))
  shutil.copy2(source/rec['path'],target)
 link=ROOT/'pipeline/WiLoR'
 if link.is_symlink() and link.resolve()==dest.resolve():pass
 elif link.exists() or link.is_symlink():raise SystemExit('Existing pipeline/WiLoR conflicts; no files removed.')
 else:link.symlink_to(dest.resolve(),target_is_directory=True)
 print('Imported',len(manifest['required_files']),'source/config files locally; weights were not copied. Directory is Git-ignored.')
if __name__=='__main__':main()
