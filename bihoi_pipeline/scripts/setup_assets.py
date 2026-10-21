"""Create non-destructive ignored links from legacy stage paths to configured assets."""
import argparse,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def link(dest,target):
 dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.is_symlink() and dest.resolve()==target.resolve():return
 if dest.exists() or dest.is_symlink():raise RuntimeError('Existing asset path conflicts (not overwritten): '+str(dest))
 dest.symlink_to(target.resolve(),target_is_directory=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 model=Path(os.getenv('HOI_MODEL_ROOT',ROOT/'.models')).expanduser().resolve()
 mano_models=Path(os.getenv('HOI_MANO_DIR',model/'mano')).expanduser().resolve()
 links={ROOT/'pipeline/sam3/checkpoints':model/'sam3',ROOT/'pipeline/sam3/sam2/checkpoints':model/'sam2',ROOT/'pipeline/MV-SAM3D/checkpoints/hf':model/'mv-sam3d',ROOT/'pipeline/FoundationPose/weights':model/'foundationpose',ROOT/'pipeline/FoundationPose/mycpp/mano':mano_models}
 for dest,target in links.items():
  if a.dry_run:print(dest.relative_to(ROOT),'->',target);continue
  target.mkdir(parents=True,exist_ok=True);link(dest,target)
 if a.dry_run:return
 # MANO models and learned joint tables are private operator assets, including text tables.
 smpl=ROOT/'pipeline/data/smplx/smplh';smpl.mkdir(parents=True,exist_ok=True)
 for side in ['LEFT','RIGHT']:
  link(smpl/f'MANO_{side}.pkl',mano_models/f'MANO_{side}.pkl')
  reg=mano_models/f'J_regressor_mano_{side}.txt';link(smpl.parent/reg.name,reg)
 for part in ['mv-sam3d','foundationpose']:
  for config in (ROOT/'configs/models'/part).rglob('*'):
   if config.is_file():
    target=model/part/config.relative_to(ROOT/'configs/models'/part);target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():shutil.copy2(config,target)
 # Local WiLoR code may exist after a separately licensed import.
 ext=Path(os.getenv('HOI_EXTERNAL_ROOT',ROOT/'.external'))/'WiLoR'
 if ext.is_dir():
  link(ROOT/'pipeline/WiLoR',ext)
  pre=ext/'pretrained_models';pre.mkdir(exist_ok=True)
  for name in ['wilor_final.ckpt','detector.pt']:
   target=(model/'mano/MANO_RIGHT.pkl') if name.startswith('MANO') else model/'wilor'/name
   link(pre/name,target)
  mano=ext/'mano_data';mano.mkdir(exist_ok=True)
  link(mano/'MANO_RIGHT.pkl',mano_models/'MANO_RIGHT.pkl')
  link(mano/'mano_mean_params.npz',model/'wilor/mano_mean_params.npz')
 print('Asset links prepared; no weights downloaded, no existing files overwritten.')
if __name__=='__main__':main()
