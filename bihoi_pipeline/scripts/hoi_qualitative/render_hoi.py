#!/usr/bin/env python3
"""Render white-background videos, figures or snapshot previews of HOI reconstructions from the command line.

A sequence folder holds intri.yml, extri.yml and standardized_hoi_30fps/ (plus images/<camera>/ for the RGB
input column, meshes/textured.glb for --material texture). Several sequences are one batch with the same
parameters; each runs single_sequence.py in its own process, so one failure does not stop the others.

Examples:
  python render_hoi.py /data/rigid/fwd_capture/0818_fwd_04 --mode video --preview
  python render_hoi.py --under /data/rigid/fwd_capture --mode video --start 70 --view orbit --orbit-yaw 30 --gpus 0,1 --jobs 2
"""
import argparse
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from media_config import EXPORT,RENDER_SETTINGS,batch_frame_range,find_sequences,sequence_config,write_render_settings

SCRIPTS=Path(__file__).resolve().parent
RESULT={'video':'camera_white.mp4','figure':'figure.png','snapshot':'snapshot.png'}


def parse_args(argv=None):
    ap=argparse.ArgumentParser(description=__doc__.split('\n\n')[0],formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('sequences',nargs='*',type=Path,help='序列目录（含 intri.yml、extri.yml、standardized_hoi_30fps/）')
    ap.add_argument('--under',type=Path,help='批量：渲染该目录下所有带 standardized_hoi_30fps/ 的序列及其 takes/*')
    ap.add_argument('--mode',choices=['video','figure','snapshot'],required=True,help='白底视频 / 制图 / 指定帧预览')
    ap.add_argument('--preview',action='store_true',help='预览：视频只渲染前 --preview-frames 帧，制图只渲染第一个帧号（snapshot 总是预览）')
    ap.add_argument('--output',type=Path,default=Path('hoi_render_output'),help='输出根目录')
    ap.add_argument('--mano-dir',type=Path,help='含 MANO_LEFT.pkl / MANO_RIGHT.pkl 的目录（默认取 render settings 的 mano_dir）')
    ap.add_argument('--render-settings',type=Path,default=RENDER_SETTINGS,help='渲染外观：手/物体颜色、光照、超采样、裁剪余量')
    g=ap.add_argument_group('通用')
    g.add_argument('--camera',default='3',help='真实相机号（intri.yml 中 K_<camera>）')
    g.add_argument('--no-hands',action='store_true',help='只渲染物体')
    g.add_argument('--material',choices=['vertex','texture','gray'],default='vertex',help='物体：PLY 顶点颜色 / meshes/textured.glb 贴图 / 灰模')
    g.add_argument('--calib-size',default='640x360',help='内参对应的原始图像分辨率（不是输出分辨率）')
    v=ap.add_argument_group('白底视频 / 指定帧预览')
    v.add_argument('--start',type=int,default=0,help='起始源帧号')
    v.add_argument('--end',default='',help='结束源帧号（不包含）；留空到末帧')
    v.add_argument('--full-range',action='store_true',help='每个序列用自己的完整帧段，忽略 --start/--end')
    v.add_argument('--resolution',default='1920x1080',help='输出分辨率，宽高为偶数')
    v.add_argument('--framing',choices=['fixed_crop','full_frame'],default='fixed_crop',help='整段固定裁剪到手物 / 完整传感器画幅')
    v.add_argument('--save-frames',action='store_true',help='另存逐帧 PNG')
    v.add_argument('--preview-frames',type=int,default=12)
    v.add_argument('--snapshot-frame',type=int,help='--mode snapshot 渲染的源帧号')
    c=ap.add_argument_group('自定义相机（白底视频 / 指定帧预览；内参沿用 --camera）')
    c.add_argument('--view',choices=['real','orbit','world'],default='real',help='真实相机 / 绕场景中心环绕 / 世界坐标指定')
    c.add_argument('--orbit-yaw',type=float,default=0,help='orbit：绕相机上方向旋转（度）')
    c.add_argument('--orbit-pitch',type=float,default=0,help='orbit：正值=相机升高向下俯视（度）')
    c.add_argument('--orbit-scale',type=float,default=1,help='orbit：到场景中心距离倍数（<1 拉近）')
    c.add_argument('--offset-right',type=float,default=0,help='orbit：相机系右移（米）')
    c.add_argument('--offset-up',type=float,default=0,help='orbit：相机系上移（米）')
    c.add_argument('--offset-forward',type=float,default=0,help='orbit：相机系前移（米）')
    c.add_argument('--position',default='',help='world：相机位置 x,y,z（世界系，米）')
    c.add_argument('--look-at',default='',help='world：注视点 x,y,z；留空=场景中心')
    c.add_argument('--roll',type=float,default=0,help='绕光轴旋转画面（度，正值顺时针；三种模式均生效）')
    f=ap.add_argument_group('制图（Input + Camera View + Novel View）')
    f.add_argument('--frames',default='60,140,220',help='源帧号，逗号分隔')
    f.add_argument('--no-input',action='store_true',help='不放原始 RGB 列（不需要 images/）')
    f.add_argument('--novel-yaw',type=float,default=50)
    f.add_argument('--novel-pitch',type=float,default=15)
    r=ap.add_argument_group('运行')
    r.add_argument('--gpus',default='',help='EGL 设备号列表，如 0,1；按序列轮流分配（默认不指定）')
    r.add_argument('--jobs',type=int,default=1,help='同时渲染的序列数')
    r.add_argument('--python',default=sys.executable,help='运行 single_sequence.py 的 Python')
    r.add_argument('--dry-run',action='store_true',help='只校验并写出各序列配置，不渲染')
    args=ap.parse_args(argv)
    if not args.sequences and not args.under:ap.error('请给出序列目录，或用 --under 指定批量目录')
    if args.mode=='snapshot' and args.snapshot_frame is None:ap.error('--mode snapshot 需要 --snapshot-frame')
    if args.jobs<1:ap.error('--jobs 必须 >= 1')
    return args


def sequence_name(path):
    """Takes are named <dataset>/takes/<take> so that different datasets' takes never collide."""
    return f'{path.parent.parent.name}/takes/{path.name}' if path.parent.name=='takes' else path.name


def collect(args):
    items={}
    for path in args.sequences:
        path=path.expanduser().resolve();items.setdefault(path,sequence_name(path))
    if args.under:
        base=args.under.expanduser().resolve();found=find_sequences(base)
        if not found:raise SystemExit(f'{base} 下没有找到带 {EXPORT}/ 的序列')
        for name in found:items.setdefault(base/name,name)
    names=list(items.values());duplicates=sorted({n for n in names if names.count(n)>1})
    if duplicates:raise SystemExit(f'不同目录的序列同名，输出会互相覆盖：{duplicates}；请分批渲染')
    return [(name,path) for path,name in items.items()]


def mano_folder(args):
    if args.mano_dir:folder=args.mano_dir.expanduser().resolve()
    else:
        value=Path(json.loads(args.render_settings.read_text())['mano_dir']).expanduser()
        candidates=[value] if value.is_absolute() else [args.render_settings.resolve().parent/value,SCRIPTS.parents[1]/value]
        folder=next((p for p in candidates if p.is_dir()),candidates[0])
    missing=[n for n in ['MANO_LEFT.pkl','MANO_RIGHT.pkl'] if not (folder/n).is_file()]
    if missing and not args.no_hands:
        raise SystemExit(f'缺少 MANO 模型：{[str(folder/n) for n in missing]}。请从 https://mano.is.tue.mpg.de 下载，用 --mano-dir 指定目录')
    return folder


def prepare(args,run_id):
    """Validate every sequence and write its config; returns (jobs, skipped)."""
    mano=mano_folder(args);jobs=[];skipped=[]
    cal=args.calib_size.lower().split('x')
    if len(cal)!=2:raise SystemExit('--calib-size 格式为 宽x高，如 640x360')
    preview=args.preview or args.mode=='snapshot'
    for name,data in collect(args):
        try:
            start,end,note=args.start,args.end,''
            if args.mode!='figure':
                ids=np.load(data/EXPORT/'source_frame_ids.npy',allow_pickle=False)
                start,end,note=batch_frame_range(ids,start,end,'full' if args.full_range else 'inherit')
            config=sequence_config(data,args.camera,not args.no_hands,args.material,cal[0],cal[1],args.frames,not args.no_input,
                                   args.novel_yaw,args.novel_pitch,start,end,args.resolution,args.framing,args.save_frames,args.preview_frames,args.mode,
                                   args.view,args.orbit_yaw,args.orbit_pitch,args.orbit_scale,args.offset_right,args.offset_up,args.offset_forward,
                                   args.position,args.look_at,args.roll,args.snapshot_frame)
        except (OSError,ValueError) as exc:
            skipped.append({'sequence':name,'path':str(data),'error':str(exc)});continue
        run=(args.output.expanduser().resolve()/name/run_id);run.mkdir(parents=True)
        config['output']=str(run)
        config['render_settings']=write_render_settings(run/'render_settings.json',args.render_settings,mano)
        (run/'media_config.json').write_text(json.dumps(config,indent=2))
        jobs.append({'sequence':name,'path':str(data),'note':note,'config':str(run/'media_config.json'),
                     'output':str(run/('preview' if preview else 'final')/args.mode),'log':str(run/'log.txt'),'preview':preview})
    return jobs,skipped


def command(args,job,gpu):
    cmd=[args.python,str(SCRIPTS/'single_sequence.py'),'--config',job['config'],'--mode',args.mode]
    if job['preview']:cmd.append('--preview')
    if gpu is not None:cmd+=['--egl-device',str(gpu)]
    return cmd


def run_all(args,jobs):
    gpus=[int(g) for g in args.gpus.replace(' ','').split(',') if g!='']
    env={**os.environ};env.setdefault('PYOPENGL_PLATFORM','egl');env.setdefault('NUMPY_MADVISE_HUGEPAGE','0')
    env.setdefault('OMP_NUM_THREADS','2');env.setdefault('OPENBLAS_NUM_THREADS','2')
    lock=threading.Lock();procs=[]

    def run(index):
        job=jobs[index];cmd=command(args,job,gpus[index%len(gpus)] if gpus else None);job['command']=cmd
        started=time.time()
        with open(job['log'],'w') as log,subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,cwd=SCRIPTS) as proc:
            with lock:procs.append(proc)
            for line in proc.stdout:
                log.write(line)
                with lock:print(f"[{job['sequence']}] {line}",end='',flush=True)
            job['returncode']=proc.wait()
        try:status=json.loads((Path(job['output'])/'status.json').read_text())
        except (OSError,ValueError):status={'status':'error','error':f"single_sequence.py 退出码 {job['returncode']}，见 {job['log']}"}
        job['status']='complete' if job['returncode']==0 and status.get('status')=='complete' else 'error'
        job['error']=status.get('error','');job['seconds']=round(time.time()-started,1)

    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:list(pool.map(run,range(len(jobs))))
    except KeyboardInterrupt:
        with lock:
            for proc in procs:proc.kill()
        raise


def main(argv=None):
    args=parse_args(argv)
    run_id=time.strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(2)
    jobs,skipped=prepare(args,run_id)
    for s in skipped:print(f"[跳过] {s['sequence']}: {s['error']}",flush=True)
    for job in jobs:
        if job['note']:print(f"[{job['sequence']}] {job['note']}",flush=True)
    if args.dry_run:
        for job in jobs:print(' '.join(command(args,job,None)))
        print(f'dry-run：{len(jobs)} 个序列的配置已写出，{len(skipped)} 个跳过。')
        return 1 if skipped or not jobs else 0
    if jobs:run_all(args,jobs)
    output=args.output.expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
    summary=output/f'batch_{run_id}.json'
    summary.write_text(json.dumps({'run_id':run_id,'argv':sys.argv if argv is None else argv,'jobs':jobs,'skipped':skipped},indent=2,ensure_ascii=False))
    print('\n结果：')
    for job in jobs:
        where=Path(job['output'])/RESULT[args.mode] if job['status']=='complete' else job['error']
        print(f"  {'OK ' if job['status']=='complete' else 'ERR'} {job['sequence']}: {where}")
    for s in skipped:print(f"  SKIP {s['sequence']}: {s['error']}")
    print(f'汇总：{summary}')
    return 0 if jobs and all(j['status']=='complete' for j in jobs) and not skipped else 1


if __name__=='__main__':sys.exit(main())
