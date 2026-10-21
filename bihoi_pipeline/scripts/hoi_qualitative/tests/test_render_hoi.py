"""Command-line batch contract: discovery, per-sequence frame ranges, skipping, MANO lookup and the runner (rendering stubbed)."""
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import render_hoi

# Stands in for `python single_sequence.py ...`: marks the requested output complete (or failed for "broken_run").
STUB='''import json,sys
from pathlib import Path
a=sys.argv[2:];cfg=json.loads(Path(a[a.index('--config')+1]).read_text());mode=a[a.index('--mode')+1]
out=Path(cfg['output'])/('preview' if '--preview' in a else 'final')/mode;out.mkdir(parents=True,exist_ok=True)
fail='broken_run' in cfg['dataset']
(out/'status.json').write_text(json.dumps({'status':'error','error':'boom'} if fail else {'status':'complete'}))
print('rendered',cfg['dataset'],'egl',a[a.index('--egl-device')+1] if '--egl-device' in a else None)
sys.exit(1 if fail else 0)
'''


class RenderHoiTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        base=self.root/'data'
        self.make(base/'sequence',[0,1,2,3]);self.make(base/'late',[2,3]);self.make(base/'sequence/takes/take02',[0,1,2,3])
        self.make(base/'broken',[0,1,2,3]);(base/'broken/standardized_hoi_30fps/obj_pose.npz').unlink()
        (base/'no_export').mkdir()
        self.mano=self.root/'mano';self.mano.mkdir()
        for n in ['MANO_LEFT.pkl','MANO_RIGHT.pkl']:(self.mano/n).touch()
        self.base=base;self.out=self.root/'out'
    def tearDown(self):self.tmp.cleanup()
    @staticmethod
    def make(data,ids):
        export=data/'standardized_hoi_30fps';export.mkdir(parents=True)
        (export/'metadata.json').write_text(json.dumps({'resampled':False}));np.save(export/'source_frame_ids.npy',np.array(ids))
        for n in ['obj_pose.npz','object_mesh.ply','mano_left.npz','mano_right.npz']:(export/n).touch()
        for n in ['intri.yml','extri.yml']:(data/n).touch()
    def cli(self,*argv):
        buffer=io.StringIO()
        with redirect_stdout(buffer):code=render_hoi.main([*map(str,argv),'--output',str(self.out),'--mano-dir',str(self.mano)])
        return code,buffer.getvalue()
    def configs(self):
        return {str(p.parent.parent.relative_to(self.out)):json.loads(p.read_text()) for p in self.out.rglob('media_config.json')}

    def test_under_inherits_start_clips_and_skips(self):
        code,text=self.cli('--under',self.base,'--mode','video','--start',1,'--dry-run')
        self.assertEqual(code,1);self.assertIn('[跳过] broken',text);self.assertIn('帧段裁剪为 2–4',text)
        configs=self.configs()
        self.assertEqual(sorted(configs),['late','sequence','sequence/takes/take02'])
        self.assertEqual([configs[n]['video']['start_frame'] for n in ['sequence','late']],[1,2])
        settings=json.loads(Path(configs['late']['render_settings']).read_text())
        self.assertEqual(settings['mano_dir'],str(self.mano.resolve()))
    def test_full_range_and_custom_camera(self):
        code,_=self.cli(self.base/'late','--mode','video','--start',0,'--end',1,'--full-range','--view','orbit','--orbit-yaw',30,'--dry-run')
        video=self.configs()['late']['video']
        self.assertEqual(code,0);self.assertEqual((video['start_frame'],video['end_frame_exclusive']),(2,None))
        self.assertEqual(video['custom_camera']['yaw_degrees'],30.0)
    def test_explicit_take_is_named_by_dataset(self):
        self.assertEqual(render_hoi.sequence_name(self.base/'sequence/takes/take02'),'sequence/takes/take02')
    def test_same_name_from_different_folders_is_refused(self):
        self.make(self.root/'other/late',[0,1])
        with self.assertRaisesRegex(SystemExit,'同名'):self.cli(self.base/'late',self.root/'other/late','--mode','video','--dry-run')
    def test_missing_mano_needs_no_hands(self):
        shutil.rmtree(self.mano)
        with self.assertRaisesRegex(SystemExit,'MANO'):self.cli(self.base/'sequence','--mode','video','--dry-run')
        code,_=self.cli(self.base/'sequence','--mode','video','--no-hands','--dry-run');self.assertEqual(code,0)
    def test_snapshot_needs_frame_and_is_preview(self):
        with redirect_stdout(io.StringIO()),self.assertRaises(SystemExit),open(os.devnull,'w') as null:
            sys.stderr,err=null,sys.stderr
            try:render_hoi.main([str(self.base/'sequence'),'--mode','snapshot'])
            finally:sys.stderr=err
        self.cli(self.base/'sequence','--mode','snapshot','--snapshot-frame',2,'--dry-run')
        self.assertEqual(self.configs()['sequence']['snapshot'],{'frame':2})
    def test_runner_reports_each_sequence(self):
        stub=self.root/'stub.py';stub.write_text(STUB)
        python=self.root/'python';python.write_text(f'#!/bin/sh\nexec {sys.executable} {stub} "$@"\n');python.chmod(python.stat().st_mode|stat.S_IEXEC)
        self.make(self.base/'broken_run',[0,1])
        code,text=self.cli(self.base/'sequence',self.base/'broken_run','--mode','video','--gpus','4,5','--jobs',2,'--python',python)
        self.assertEqual(code,1);self.assertIn('egl 4',text);self.assertIn('egl 5',text)
        summary=json.loads(next(self.out.glob('batch_*.json')).read_text())
        status={j['sequence']:(j['status'],j['error']) for j in summary['jobs']}
        self.assertEqual(status,{'sequence':('complete',''),'broken_run':('error','boom')})
        self.assertTrue(Path(summary['jobs'][0]['log']).read_text().startswith('rendered'))


if __name__=='__main__':unittest.main()
