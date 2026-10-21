"""Derive EasyMocap joint tables locally from operator-supplied licensed MANO models."""
import argparse,os,pickle
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
# The original pipeline's 21-joint order: wrist, thumb, index, middle, ring, little.
JOINT_ORDER=[0,13,14,15,None,1,2,3,None,4,5,6,None,10,11,12,None,7,8,9,None]
TIP_VERTICES={'LEFT':{4:745,8:317,12:445,16:556,20:673},'RIGHT':{4:745,8:317,12:444,16:556,20:673}}
def build_table(model_file,side):
 # MANO pickle loading is restricted to the operator's supplied, trusted model file.
 with Path(model_file).open('rb') as f:model=pickle.load(f,encoding='latin1')
 base=model['J_regressor'];base=base.toarray() if hasattr(base,'toarray') else np.asarray(base)
 if base.shape!=(16,778):raise ValueError('Expected MANO J_regressor shape (16,778), got '+str(base.shape))
 table=np.zeros((21,778),dtype=base.dtype)
 for row,joint in enumerate(JOINT_ORDER):
  if joint is not None:table[row]=base[joint]
 for row,vertex in TIP_VERTICES[side].items():table[row,vertex]=1
 return table

def main():
 default=Path(os.getenv('HOI_MANO_DIR',Path(os.getenv('HOI_MODEL_ROOT',ROOT/'.models'))/'mano'))
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mano-dir',type=Path,default=default);p.add_argument('--output-dir',type=Path);a=p.parse_args()
 output=a.output_dir or a.mano_dir
 tables={side:build_table(a.mano_dir/f'MANO_{side}.pkl',side) for side in ['LEFT','RIGHT']}
 output.mkdir(parents=True,exist_ok=True)
 for side,table in tables.items():
  dest=output/f'J_regressor_mano_{side}.txt'
  if dest.exists():
   text=np.loadtxt(dest,comments='#');existing=np.zeros(table.shape);existing[text[:,0].astype(int),text[:,1].astype(int)]=text[:,2]
   if not np.allclose(existing,table,rtol=0,atol=1e-12):raise SystemExit('Existing parameter table differs; no overwrite: '+str(dest))
   print('Existing table verified:',dest);continue
  rows,cols=np.nonzero(table)
  data=np.column_stack([rows,cols,table[rows,cols]])
  with dest.open('x') as f:np.savetxt(f,data,fmt=['%d','%d','%.17g'],header='21 778')
  print('Generated local model parameters:',dest)
if __name__=='__main__':main()
