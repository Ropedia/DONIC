"""Validated render configs for single_sequence.py, shared by the web panel and the render_hoi.py CLI.

No Gradio or rendering imports: this only checks the inputs of one sequence and writes the JSON config
that single_sequence.py consumes.
"""
import json
import os
import math
import re
from pathlib import Path
import numpy as np

SCRIPTS=Path(__file__).resolve().parent
EXPORT='standardized_hoi_30fps'
# The packaged copy ships only the render look (render_settings.json); the repository keeps it in config_textured_glb.json.
RENDER_SETTINGS=next((p for p in [SCRIPTS/'render_settings.json',SCRIPTS/'config_textured_glb.json'] if p.is_file()),SCRIPTS/'render_settings.json')
RENDER_KEYS=['mano_dir','crop_margin','supersampling','left_color','right_color','object_color',
             'object_material','ambient_light','light_intensity','light_offsets_degrees']
VIEW_MODES=['real','orbit','world']


def integer(value,label,minimum=0):
    try:n=float(value)
    except (TypeError,ValueError):raise ValueError(f'{label} 必须是整数')
    if not math.isfinite(n) or n!=int(n) or n<minimum:raise ValueError(f'{label} 必须是 >= {minimum} 的整数')
    return int(n)


def number(value,label):
    try:n=float(value)
    except (TypeError,ValueError):raise ValueError(f'{label} 必须是数值')
    if not math.isfinite(n):raise ValueError(f'{label} 必须是有限数值')
    return n


def point(text,label,optional=False):
    tokens=[t for t in re.split(r'[,，\s]+',str(text or '').strip()) if t]
    if not tokens and optional:return None
    if len(tokens)!=3:raise ValueError(f'{label} 需要 3 个数值 x,y,z（米）')
    return [number(t,label) for t in tokens]


def custom_camera(view_mode,orbit_yaw,orbit_pitch,orbit_scale,offset_right,offset_up,offset_forward,world_position,world_target,roll=0):
    if view_mode not in VIEW_MODES:raise ValueError('无效相机位置模式')
    roll=number(0 if roll is None or str(roll).strip()=='' else roll,'roll')
    if view_mode=='real':spec={'type':'real'}
    elif view_mode=='orbit':
        scale=number(orbit_scale,'距离倍数')
        if scale<=0:raise ValueError('距离倍数必须 > 0')
        spec={'type':'orbit','yaw_degrees':number(orbit_yaw,'环绕 yaw'),'pitch_degrees':number(orbit_pitch,'环绕 pitch'),'distance_scale':scale,
              'offset_right_up_forward_m':[number(offset_right,'右移'),number(offset_up,'上移'),number(offset_forward,'前移')]}
    else:spec={'type':'world','position':point(world_position,'相机位置'),'look_at':point(world_target,'注视点',optional=True)}
    if roll:spec['roll_degrees']=roll
    return spec


def video_resolution(text):
    match=re.fullmatch(r'\s*(\d+)\s*[xX×]\s*(\d+)\s*',str(text or ''))
    if not match:raise ValueError('无效视频分辨率（格式 宽x高，如 1920x1080）')
    size=[int(n) for n in match.groups()]
    if min(size)<=0 or any(n%2 for n in size):raise ValueError('视频分辨率宽高必须是正偶数（H.264 yuv420p）')
    return size


def sequence_config(data,cam,hands,material,cal_w,cal_h,frames,include_input,yaw,pitch,start,end,resolution,framing,save_frames,preview_frames,mode,
                    view_mode='real',orbit_yaw=0,orbit_pitch=0,orbit_scale=1,offset_right=0,offset_up=0,offset_forward=0,world_position='',world_target='',roll=0,snapshot_frame=None):
    """Config of one sequence folder (intri.yml, extri.yml, standardized_hoi_30fps/) for one media mode."""
    if mode not in ['figure','video','snapshot']:raise ValueError('Invalid media mode')
    data=Path(data);export=data/EXPORT
    required=['metadata.json','source_frame_ids.npy','obj_pose.npz','object_mesh.ply']
    if hands:required+=['mano_left.npz','mano_right.npz']
    for filename in required:
        if not (export/filename).is_file():raise ValueError(f'Missing: {export/filename}')
    for filename in ['intri.yml','extri.yml']:
        if not (data/filename).is_file():raise ValueError(f'Missing: {data/filename}')
    cam=str(cam or '').strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',cam):raise ValueError('无效相机号')
    ids=np.load(export/'source_frame_ids.npy',allow_pickle=False)
    if ids.ndim!=1 or len(ids)==0 or not np.all(np.diff(ids)>0):raise ValueError('帧号必须唯一且递增')
    meta=json.loads((export/'metadata.json').read_text())
    if meta.get('resampled'):raise ValueError('暂不支持重采样的位姿导出')
    glb=data/'meshes/textured.glb'
    if material not in ['texture','vertex','gray']:raise ValueError('无效物体材质')
    if material=='texture' and not (glb.is_file() and glb.with_suffix('.json').is_file()):raise ValueError('Missing: textured.glb 或 textured.json；请选择现有顶点颜色/灰模，或先生成贴图模型')
    config={'dataset':str(data),'export':str(export),'object_glb':str(glb) if material=='texture' else None,
            'camera':cam,'include_hands':bool(hands),'calibration_resolution':[integer(cal_w,'标定宽度',1),integer(cal_h,'标定高度',1)],
            'render_settings':str(RENDER_SETTINGS),
            'render_overrides':{'object_material':'gray' if material=='gray' else 'source'}}
    # Each mode validates only its own parameters.
    if mode=='figure':
        tokens=re.split(r'[,，\s]+',str(frames).strip());chosen=[integer(t,'帧号') for t in tokens if t]
        if not chosen or len(set(chosen))!=len(chosen):raise ValueError('请输入不重复的制图帧号')
        missing=sorted(set(chosen)-set(map(int,ids)))
        if missing:raise ValueError(f'Missing source frames: {missing}')
        for angle in [yaw,pitch]:
            if not math.isfinite(float(angle)):raise ValueError('新视角角度必须是有限数值')
        config['figure']={'frames':chosen,'include_input':bool(include_input),'cell_size':1000 if include_input else 1500,'novel_yaw_pitch_degrees':[float(yaw),float(pitch)]}
    else:
        first=integer(start,'起始帧');stop=None if end is None or str(end).strip()=='' else integer(end,'结束帧',1)
        effective=int(ids[-1])+1 if stop is None else stop
        if effective<=first or first<int(ids[0]) or effective>int(ids[-1])+1:raise ValueError('视频帧段超出已有结果或为空（结束帧不包含在内）')
        selected_ids=ids[(ids>=first)&(ids<effective)]
        if len(selected_ids)!=effective-first:raise ValueError('Missing: 视频帧段有缺帧，不能跳帧补齐')
        if framing not in ['fixed_crop','full_frame']:raise ValueError('无效构图方式')
        config['video']={'start_frame':first,'end_frame_exclusive':stop,'resolution':video_resolution(resolution),
                         'framing':framing,'save_frames':bool(save_frames),'reject_clipping':True,'preview_frames':integer(preview_frames,'预览帧数',1),
                         'custom_camera':custom_camera(view_mode,orbit_yaw,orbit_pitch,orbit_scale,offset_right,offset_up,offset_forward,world_position,world_target,roll)}
        if mode=='snapshot':
            frame=integer(snapshot_frame,'预览帧号')
            if frame not in set(map(int,ids)):raise ValueError(f'Missing source frame: {frame}')
            config['snapshot']={'frame':frame}
    return config


def batch_frame_range(ids,start,end,frame_range):
    """(start, end, note) of one batch sequence: the given range clipped to the sequence's export, or the whole export."""
    first,last=int(ids[0]),int(ids[-1])+1
    if frame_range=='full':return first,'',''
    blank=end is None or str(end).strip()==''
    start=integer(start,'起始帧');stop=last if blank else integer(end,'结束帧',1)
    s,e=max(start,first),min(stop,last)
    if e<=s:raise ValueError(f'起止帧 {start}–{stop} 与该序列已有帧段 {first}–{last} 无交集')
    note='' if (s,e)==(start,stop) else f'帧段裁剪为 {s}–{e}（该序列已有 {first}–{last}）'
    return s,('' if blank else e),note


def find_sequences(base):
    """Relative names of the sequences (<name>) and takes (<name>/takes/<take>) under base with a standardized export."""
    base=Path(base);names=[]
    for folder in sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []:
        if (folder/EXPORT/'metadata.json').is_file():names.append(folder.name)
        names+=[f'{folder.name}/takes/{take.name}' for take in sorted((folder/'takes').glob('*')) if (take/EXPORT/'metadata.json').is_file()]
    return names


def write_render_settings(path,base=RENDER_SETTINGS,mano_dir=None):
    """Snapshot of the render look (optionally with an explicit MANO folder) so a run is reproducible later."""
    source=json.loads(Path(base).read_text());settings={k:source[k] for k in RENDER_KEYS}
    settings['mano_dir']=str(mano_dir or os.getenv('HOI_MANO_DIR') or (SCRIPTS.parents[1]/'.models/mano'))
    Path(path).write_text(json.dumps(settings,indent=2))
    return str(path)
