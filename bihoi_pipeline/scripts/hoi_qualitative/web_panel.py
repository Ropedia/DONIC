"""Shared Gradio panel backed by the existing detached HOI task scheduler."""
import hashlib
import importlib.util
import json
import re
import secrets
import shlex
import sys
import time
import zipfile
from pathlib import Path
import gradio as gr
import numpy as np
from media_config import batch_frame_range,find_sequences,integer,sequence_config,write_render_settings

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
# run_mv_sam3d_metric_scale.py parameters recorded in meshes/metric_scale_info.json; replayed
# verbatim so the texture run reproduces the original reconstruction settings.
RECON_ARGS=['images_dir_name','masks_dir_name','depth_dir_name','intri_name','extri_name','mesh_cam_ids','depth_cam_ids',
            'model_tag','seed','stage1_steps','stage2_steps','depth_unit','depth_scale','extrinsic_translation_scale',
            'calib_width','calib_height','min_depth','max_depth','pixel_stride','mask_erode','max_points_per_cam',
            'mesh_sample_points','max_size_mb']
MODE_LABELS={'figure':'制图','video':'白底视频','snapshot':'指定帧相机预览','texture':'贴图模型生成'}
OUTPUT_DIR='output_reconstruction'


def scheduler(namespace):
    if '_launch_detached' in namespace:
        return namespace
    name = '_hoi_media_scheduler'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT/'webui/interaction_capture_webui.py')
        module = importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return vars(sys.modules[name])


def dataset_path(root, prior, selected):
    if not prior or not re.fullmatch(r'[A-Za-z0-9_.-]+',str(prior)) or prior in ['.','..']:
        raise ValueError('请先选择 Hand prior')
    base=(Path(root).expanduser().resolve()/'rigid'/str(prior)).resolve()
    if not selected:raise ValueError('请选择已完成标准化导出的序列')
    path=(base/str(selected)).resolve()
    if path==base or base not in path.parents or not path.is_dir():raise ValueError('无效的序列路径')
    return path


def output_root(root,prior,selected):
    """<Data Root>/output_reconstruction/<prior>/<sequence>/ — mirrors rigid/<prior>/<sequence>/."""
    root=Path(root).expanduser().resolve();data=dataset_path(root,prior,selected)
    return root/OUTPUT_DIR/str(prior)/data.relative_to((root/'rigid'/str(prior)).resolve())


def sequence_choices(root,prior,names):
    """The given datasets and their takes that have a standardized export."""
    choices=[]
    for name in names:
        try:
            path=dataset_path(root,prior,name)
            if (path/'standardized_hoi_30fps/metadata.json').is_file():choices.append(str(name))
            for meta in sorted(path.glob('takes/*/standardized_hoi_30fps/metadata.json')):
                choices.append(str(Path(name)/meta.parent.parent.relative_to(path)))
        except ValueError:pass
    return choices


def all_sequences(root,prior):
    """Every sequence / take under rigid/<prior>/ with a standardized export."""
    if not prior or not re.fullmatch(r'[A-Za-z0-9_.-]+',str(prior)) or prior in ['.','..']:return []
    return find_sequences(Path(root).expanduser().resolve()/'rigid'/str(prior))


def discover(root,prior,selected,current=None,preferred=None):
    choices=sequence_choices(root,prior,list(dict.fromkeys(list(selected or [])+([preferred] if preferred else []))))
    value=(preferred if preferred in choices else None) if preferred else (current if current in choices else (choices[0] if len(choices)==1 else None))
    return gr.Dropdown(choices=choices,value=value)


def group_dir(data):
    """Takes share the dataset-level meshes/, so the textured model belongs to the dataset group."""
    data=Path(data)
    return data.parents[1] if data.parent.name=='takes' else data


def textured_status(meshes):
    """Sidecar info of a verified textured model in meshes/, or None (missing, incomplete or stale)."""
    glb=Path(meshes)/'textured.glb';ply=Path(meshes)/'compressed.ply'
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        info=json.loads(glb.with_suffix('.json').read_text())
        if info.get('status')=='complete' and info['glb_sha256']==sha(glb) and info['tracking_ply_sha256']==sha(ply):return info
    except (OSError,ValueError,KeyError):pass
    return None


def texture_command(group,code_root):
    """Shell command that re-runs the group's MV-SAM3D reconstruction with UV texture baking.

    Only one textured model per data group: a verified meshes/textured.glb is never regenerated.
    The previous compressed.ply / metric_scale_info.json are kept in meshes/before_texture/.
    """
    group=Path(group);meshes=group/'meshes';q=shlex.quote
    if textured_status(meshes):raise ValueError('该数据组已有校验通过的贴图模型 meshes/textured.glb；每个数据组只生成一次，请直接选择“GLB UV 贴图”材质')
    for rel in ['images1','mask','depth','intri.yml','extri.yml','meshes/compressed.ply']:
        if not (group/rel).exists():raise ValueError(f'Missing: {group/rel}；请先完成该数据组的 Stage 2 物体重建')
    if not any((group/'images1').glob('*.jpg')):raise ValueError(f'Missing: {group/"images1"} 中没有参考帧图像')
    script=Path(code_root)/'MV-SAM3D/run_mv_sam3d_metric_scale.py'
    if not script.is_file() or not script.with_name('texture_export.py').is_file():raise ValueError(f'Missing: {script.parent} 缺少贴图导出支持')
    try:recorded=json.loads((meshes/'metric_scale_info.json').read_text()).get('args',{})
    except (OSError,ValueError,AttributeError):recorded={}
    args=[f'--{key} {q(str(recorded[key]))}' for key in RECON_ARGS if recorded.get(key) is not None]
    if recorded.get('save_debug_pointcloud'):args.append('--save_debug_pointcloud')
    prefix=''
    helper=ROOT/'webui/mask_views.py'
    if recorded.get('mesh_cam_ids') is None and helper.is_file():
        prefix=f'eval "$({q(sys.executable)} {q(str(helper))} select --data_dir {q(str(group))})" && '
        args+=['--mesh_cam_ids "$MESH_CAMS"','--depth_cam_ids "$DEPTH_CAMS"']
    backup=meshes/'before_texture'
    keep=' && '.join(f'{{ [ -e {q(str(backup/n))} ] || cp {q(str(meshes/n))} {q(str(backup/n))}; }}'
                     for n in ['compressed.ply','metric_scale_info.json'] if (meshes/n).is_file())
    return (f'mkdir -p {q(str(backup))} && {keep} && {prefix}CUDA_VISIBLE_DEVICES="$GPU_PICK" python {q(script.name)} '
            f'--data_root {q(str(group.parent))} --setting_name {q(group.name)} {" ".join(args)} --generate_texture'),recorded


def build_config(root,prior,selected,*args,**kwargs):
    return sequence_config(dataset_path(root,prior,selected),*args,**kwargs)


RESULT_FILES=['figure.png','figure.pdf','camera_white.mp4','snapshot.png','snapshot_compare.png','camera.json','manifest.json','config.json','ffmpeg.log']


def render_title(mode,preview):
    return '指定帧相机拍摄预览' if mode=='snapshot' else f'{"预览" if preview else "正式生成"} · {"单序列制图" if mode=="figure" else "单相机白底视频"}'


def render_complete(out):
    try:return json.loads((Path(out)/'status.json').read_text())['status']=='complete'
    except (OSError,ValueError,KeyError,TypeError):return False


def render_media(out,mode):
    """(preview image, video) of a completed figure/video/snapshot output folder."""
    out=Path(out);names={'figure':['figure.png'],'video':['first_frame.png'],'snapshot':['snapshot_compare.png','snapshot.png']}[mode]
    image=next((str(out/n) for n in names if (out/n).is_file()),None)
    return image,(str(out/'camera_white.mp4') if (out/'camera_white.mp4').is_file() else None)


def camera_summary(out):
    """Markdown lines with the rendering camera position, so a view can be refined numerically."""
    try:info=json.loads((Path(out)/'camera.json').read_text())
    except (OSError,ValueError):return ''
    fmt=lambda v:'None' if v is None else ', '.join(f'{x:.4f}' for x in v)
    lines=[f"\n\n渲染相机中心（世界系，米）：`{fmt(info['camera_center_world_m'])}`；视线方向：`{fmt(info['view_direction_world'])}`",
           f"真实相机 {info['reference_camera']} 中心：`{fmt(info['reference_camera_center_world_m'])}`"]
    if info.get('scene_center_world_m') is not None:lines.append(f"场景中心（环绕/默认注视点）：`{fmt(info['scene_center_world_m'])}`")
    if info.get('near_plane_retreat_m'):lines.append(f"⚠️ 近平面保护：相机沿视线后退 {info['near_plane_retreat_m']:.4f} m")
    return '  \n'.join(lines)


class Controller:
    def __init__(self,backend):self.b=backend

    def choices(self,dataset=None):
        return [(f"{', '.join(m.get('datasets',[]))} · {m['run_id']} · {m['media_export']['mode']} · {m['state'].get('status','queued')}",m['run_id'])
                for m in self.b['_list_runs']() if m.get('media_export') and
                (dataset is None or self.matches_dataset(m,dataset))]

    @staticmethod
    def matches_dataset(meta,dataset):
        dataset=Path(dataset).resolve();targets={dataset}
        # A textured model is generated once per data group and shared by its takes.
        if meta.get('media_export',{}).get('mode')=='texture':targets.add(group_dir(dataset))
        return any(step.get('dataset_path') and Path(step['dataset_path']).resolve() in targets
                   for step in meta.get('steps',[]))

    def guard_active(self,data):
        for active in self.b['_active_runs']():
            for step in active.get('steps',[]):
                if not step.get('dataset_path'):continue
                busy=Path(step['dataset_path']).resolve()
                if busy==data or busy in data.parents or data in busy.parents:
                    raise ValueError(f"该序列有运行中的任务 {active['run_id']}，请等待完成或先停止该任务")

    def make_step(self,title,cwd,command,conda_env):
        try:return self.b['Step'](title,cwd,command,conda_env,gpu='single')
        except TypeError:return self.b['Step'](title,cwd,command,conda_env)

    def record(self,step,mode,title,command,logical,data):
        return {'stage':'media_'+mode,'label':logical+' · '+title,'title':title,'logical_data_name':logical,
                'dataset_path':str(data),'text_prompt':'','cwd':str(step.cwd),'command':command,'conda_env':step.conda_env,
                'gpu':'single','after':getattr(step,'after',()),'shell':self.b['_shell_for_step'](step)}

    def render_step(self,mode,preview,config,path,logical):
        """Write one render config; returns its scheduler step record and output folder."""
        path.write_text(json.dumps(config,indent=2));title=render_title(mode,preview)
        command=f'bash {shlex.quote(str(SCRIPTS/"run_single.sh"))} --config {shlex.quote(str(path))} --mode {mode} --egl-device "$GPU_PICK"'
        if preview:command+=' --preview'
        step=self.make_step(title,ROOT,command,None)
        return self.record(step,mode,title,command,logical,config['dataset']),Path(config['output'])/('preview' if preview else 'final')/mode

    @staticmethod
    def render_settings(run_dir):
        return write_render_settings(run_dir/'render_settings.json')

    def launch(self,run_dir,title,mode,prior,names,root,hands,steps,vram,allowed,parallel=1,**export):
        run_id=run_dir.name
        meta={'run_id':run_id,'session':'hoi_'+run_id,'created':time.time(),'stage_sequence':['media_'+mode],
              'hand_prior_name':str(prior),'hand_side':'bimanual' if hands else 'none',
              'datasets':names,'takes':names,'data_root':str(root),
              'scheduler':{'parallel_datasets':parallel,'vram_per_step_mb':vram,'cpu_parallel':1,'oom_max_retries':0,'allowed_gpus':allowed},
              'steps':steps,**export}
        write=self.b['_write_json_atomic']
        write(run_dir/'state.json',{'status':'queued','total_steps':len(steps)});write(run_dir/'meta.json',meta)
        (run_dir/'log.txt').write_text(f'{title}\n')
        try:
            meta['backend']=self.b['_launch_detached'](run_dir,meta['session']);write(run_dir/'meta.json',meta)
        except Exception as exc:
            write(run_dir/'state.json',{'status':'failed','error':str(exc)});raise
        return run_id

    def new_run_dir(self):
        return self.b['RUNS_DIR']/(time.strftime('%Y%m%d-%H%M%S')+'-media-'+secrets.token_hex(3))

    def submit(self,mode,preview,values,gpu):
        root=Path(values[0]).expanduser().resolve();prior=str(values[1])
        allowed=[] if str(gpu).strip().lower()=='auto' else [integer(gpu,'GPU')]
        run_dir=self.new_run_dir()
        if mode=='texture':
            data=group_dir(dataset_path(*values[:3]));self.guard_active(data)
            code_root=Path(self.b.get('EASYMOCAP_ROOT',ROOT))
            command,recorded=texture_command(data,code_root)
            title='生成贴图模型 GLB（重跑物体重建 + UV 贴图烘焙）'
            run_dir.mkdir(parents=True);path=run_dir/'texture_config.json'
            path.write_text(json.dumps({'group':str(data),'meshes':str(data/'meshes'),'code_root':str(code_root),
                                        'recorded_recon_args':recorded,'command':command},indent=2))
            step=self.make_step(title,code_root/'MV-SAM3D',command,'sam3d-objects')
            vram=int(self.b.get('GPU_MIN_FREE_VRAM_MB',32768));hands=False;output=data/'meshes'
            name=str(data.relative_to(root/'rigid'/prior))
            rec=self.record(step,mode,title,command,'rigid/'+prior+'/'+name,data)
        else:
            config=build_config(*values[:18],mode,*values[18:]);self.guard_active(Path(config['dataset']))
            run_dir.mkdir(parents=True);name=str(values[2])
            # Rendered media of every run are collected under one tree next to rigid/ (task bookkeeping stays in run_dir).
            config['output']=str(output_root(root,prior,name)/run_dir.name)
            config['render_settings']=self.render_settings(run_dir)
            path=run_dir/'media_config.json';title=render_title(mode,preview)
            rec,output=self.render_step(mode,preview,config,path,'rigid/'+prior+'/'+name)
            vram=8192;hands=config['include_hands']
        return self.launch(run_dir,title,mode,prior,[name],root,hands,[rec],vram,allowed,
                           media_export={'mode':mode,'preview':preview,'output':str(output),'config':str(path)})

    def submit_batch(self,mode,preview,values,sequences,gpu,parallel,frame_range='inherit'):
        """One run with one render step per sequence (each its own scheduler chain, up to `parallel` at once).

        Every sequence uses the same parameters. The video frame range is the start/end fields clipped to each
        sequence's export ('inherit'), or each sequence's whole export ('full'). Sequences failing validation
        are skipped and returned."""
        if frame_range not in ['inherit','full']:raise ValueError('无效帧段方式')
        if mode not in ['figure','video','snapshot']:raise ValueError('批量任务只支持制图、白底视频和指定帧预览')
        preview=bool(preview) or mode=='snapshot'
        names=list(dict.fromkeys(str(s) for s in sequences or []))
        if not names:raise ValueError('请先勾选要批量生成的序列')
        root=Path(values[0]).expanduser().resolve();prior=str(values[1])
        allowed=self.b['_parse_gpu_list'](gpu);parallel=integer(parallel,'并行序列数',1)
        configs=[];skipped=[]
        for name in names:
            v=list(values);v[2]=name;note=''
            try:
                if mode!='figure':
                    ids=np.load(dataset_path(root,prior,name)/'standardized_hoi_30fps/source_frame_ids.npy',allow_pickle=False)
                    v[12],v[13],note=batch_frame_range(ids,v[12],v[13],frame_range)
                config=build_config(*v[:18],mode,*v[18:]);self.guard_active(Path(config['dataset']))
                configs.append((name,config,note))
            except Exception as exc:skipped.append({'dataset':name,'error':str(exc)})
        if not configs:raise ValueError('没有可提交的序列：'+'；'.join(f"{s['dataset']}: {s['error']}" for s in skipped))
        run_dir=self.new_run_dir();(run_dir/'configs').mkdir(parents=True)
        settings=self.render_settings(run_dir);steps=[];items=[]
        for index,(name,config,note) in enumerate(configs):
            config['output']=str(output_root(root,prior,name)/run_dir.name);config['render_settings']=settings
            path=run_dir/'configs'/f'{index:03d}.json'
            rec,output=self.render_step(mode,preview,config,path,'rigid/'+prior+'/'+name)
            steps.append(rec);items.append({'dataset':name,'output':str(output),'config':str(path),'note':note})
        title=f'批量 · {render_title(mode,preview)} · {len(items)} 个序列'
        run_id=self.launch(run_dir,title,mode,prior,[i['dataset'] for i in items],root,configs[0][1]['include_hands'],steps,8192,allowed,
                           parallel=min(parallel,len(items)),media_batch={'mode':mode,'preview':preview,'items':items,'skipped':skipped})
        return run_id,items,skipped

    def batch_choices(self):
        return [(f"批量{MODE_LABELS[m['media_batch']['mode']]}{'预览' if m['media_batch']['preview'] else ''} · "
                 f"{len(m['media_batch']['items'])} 个序列 · {m['run_id']} · {m['state'].get('status','queued')}",m['run_id'])
                for m in self.b['_list_runs']() if m.get('media_batch')]

    def batch_items(self,meta):
        """[(index, item, step status, complete)] of a batch run."""
        steps=meta['state'].get('steps',{})
        return [(i,item,steps.get(str(i),{}).get('status','pending'),
                 steps.get(str(i),{}).get('status')=='done' and render_complete(item['output']))
                for i,item in enumerate(meta['media_batch']['items'])]

    def batch_poll(self,run_id,previous):
        """(status markdown, completed-item dropdown, signature); the dropdown is left alone while nothing changed."""
        meta=self.b['_find_run'](run_id) if run_id else None
        if not meta or not meta.get('media_batch'):return '请选择批量任务。',(gr.skip() if previous=='' else gr.Dropdown(choices=[],value=None)),''
        items=self.batch_items(meta);icons=self.b.get('STATUS_ICON',{})
        rows=['| # | 序列 | 状态 | 输出目录 |','|---|---|---|---|']
        for i,item,state,complete in items:
            extra=(' · 输出不完整' if state=='done' and not complete else '')+(f" · {item['note']}" if item.get('note') else '')
            rows.append(f"| {i+1} | {item['dataset']} | {icons.get(state,'')} {state}{extra} | `{item['output']}` |")
        for s in meta['media_batch'].get('skipped',[]):
            rows.append(f"| – | {s['dataset']} | 未提交 | {s['error'].replace('|','/')} |")
        status=self.b['_run_status_md'](meta)+'\n\n'+'\n'.join(rows)
        signature=run_id+':'+','.join(f'{i}{state}{complete}' for i,_,state,complete in items)
        if signature==previous:return status,gr.skip(),previous
        return status,gr.Dropdown(choices=[item['dataset'] for _,item,_,complete in items if complete],value=None),signature

    def batch_item_media(self,run_id,dataset):
        meta=self.b['_find_run'](run_id) if run_id else None
        if not meta or not meta.get('media_batch'):return None,None
        for _,item,_,complete in self.batch_items(meta):
            if complete and item['dataset']==dataset:return render_media(item['output'],meta['media_batch']['mode'])
        return None,None

    def batch_zip(self,run_id):
        """Zip of the completed items' result files, stored as <sequence>/<file>."""
        meta=self.b['_find_run'](run_id) if run_id else None
        if not meta or not meta.get('media_batch'):raise ValueError('请选择批量任务')
        done=[item for _,item,_,complete in self.batch_items(meta) if complete]
        if not done:raise ValueError('该批量任务还没有已完成的序列')
        path=Path(meta['dir'])/'batch_outputs.zip';tmp=path.with_suffix('.zip.tmp')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_STORED) as z:
            for item in done:
                for name in RESULT_FILES:
                    if (Path(item['output'])/name).is_file():z.write(Path(item['output'])/name,f"{item['dataset']}/{name}")
        tmp.replace(path)
        return str(path),len(done)

    def poll(self,run_id,previous,expected_dataset=None):
        meta=self.b['_find_run'](run_id)
        if not meta or not meta.get('media_export'):return '请选择媒体导出任务。','',None,None,[],''
        if expected_dataset is not None and not self.matches_dataset(meta,expected_dataset):
            return '所选任务不属于当前导出序列，旧预览已清空。','',None,None,[],''
        state=meta['state']['status'];out=Path(meta['media_export']['output']);mode=meta['media_export']['mode']
        status=self.b['_run_status_md'](meta)+'\n\n结果所属序列：**'+', '.join(meta.get('datasets',[]))+'**'
        log=self.b['_run_log_view'](meta,0)
        signature=f'{run_id}:{state}'
        if signature==previous:return status,log,gr.skip(),gr.skip(),gr.skip(),previous
        complete=False;info=None
        if mode=='texture':info=textured_status(out) if state=='done' else None;complete=info is not None
        else:complete=state=='done' and render_complete(out)
        image=video=None;files=[meta['media_export']['config']]
        if complete and mode=='texture':
            files+=[str(out/n) for n in ['textured.json','textured.glb'] if (out/n).is_file()]
            status+=(f'\n\n贴图模型已生成并通过校验：`{out/"textured.glb"}`（贴图分辨率 {info.get("texture_resolution")}）。'
                     f'原 compressed.ply 备份在 `{out/"before_texture"}`。制图时选择“GLB UV 贴图”材质即可使用；该数据组无需再次生成。')
        elif complete:
            image,video=render_media(out,mode)
            files += [str(out/n) for n in RESULT_FILES if (out/n).is_file()]
            status+=f'\n\n输出：`{out}`'+camera_summary(out)
        return status,log,image,video,files,signature


def mount_media_panel(data_root,hand_prior,datasets,namespace,visualization_dataset=None):
    controller=Controller(scheduler(namespace))
    with gr.Accordion('单序列媒体导出 · 制图 / 单相机白底视频',open=False):
        gr.Markdown('使用已完成的标准化重建结果。任务在后台运行，关闭页面后可继续；预览和正式结果分别保存，每次提交独立目录。')
        with gr.Row():
            sequence=gr.Dropdown(choices=[],value=None,label='当前导出序列 / take（跟随 Visualizations，可单独选择）')
            scan=gr.Button('读取已选序列')
            camera=gr.Textbox(value='3',label='真实相机号')
        with gr.Row():
            hands=gr.Checkbox(value=True,label='显示双手（左红、右蓝）')
            material=gr.Dropdown(choices=[('GLB UV 贴图','texture'),('PLY 顶点颜色','vertex'),('物体灰模','gray')],value='texture',label='物体材质')
            gpu=gr.Textbox(value='auto',label='GPU（auto 或设备号）')
        with gr.Accordion('相机标定参考尺寸',open=False):
            gr.Markdown('填写内参对应的原始分辨率；它不是最终输出分辨率。')
            with gr.Row():
                cal_w=gr.Number(value=640,precision=0,label='标定宽度');cal_h=gr.Number(value=360,precision=0,label='标定高度')
        with gr.Tabs():
            with gr.Tab('制图 · Camera + Novel'):
                frames=gr.Textbox(value='60,140,220',label='帧号（逗号分隔）')
                include_input=gr.Checkbox(value=True,label='包含 Input Frames 列')
                with gr.Row():
                    yaw=gr.Number(value=50,label='Novel View yaw（度）');pitch=gr.Number(value=15,label='Novel View pitch（度）')
                with gr.Row():fig_preview=gr.Button('预览制图（首个指定帧）');fig_run=gr.Button('生成完整制图',variant='primary')
                with gr.Accordion('贴图模型 GLB（“GLB UV 贴图”材质需要）',open=False):
                    gr.Markdown('缺少 meshes/textured.glb 时在此生成：重跑该数据组的 MV-SAM3D 物体重建并烘焙 1024² UV 贴图（同一数据组的 takes 共用）。'
                                '原 compressed.ply / metric_scale_info.json 先备份到 meshes/before_texture/。每个数据组只生成一次：已有校验通过的贴图模型时拒绝重复生成。')
                    texture_run=gr.Button('生成贴图模型 GLB（每个数据组一次）')
            with gr.Tab('白底视频 · 仅 Camera View'):
                gr.Markdown('单真实相机、整段固定投影和裁剪，按原始帧率编码。不生成 Novel View。')
                with gr.Row():
                    start=gr.Number(value=0,precision=0,label='起始帧');end=gr.Textbox(value='',label='结束帧（不包含；留空到末帧）')
                with gr.Row():
                    resolution=gr.Dropdown(choices=['1920x1080','1280x720','640x360'],value='1920x1080',label='视频分辨率')
                    framing=gr.Dropdown(choices=[('整段固定裁剪','fixed_crop'),('完整传感器画幅','full_frame')],value='fixed_crop',label='构图')
                with gr.Row():
                    save_frames=gr.Checkbox(value=False,label='保存逐帧 PNG');preview_frames=gr.Number(value=12,precision=0,label='预览帧数')
                with gr.Accordion('自定义相机位置（默认使用真实相机）',open=False):
                    gr.Markdown('内参沿用上方“真实相机号”的标定焦距；整段视频使用同一个固定相机。场景中心 = 所选帧段全部手物几何包围盒中心。'
                                '提交指定帧预览后，结果状态会给出渲染相机与场景中心的世界坐标，便于微调。')
                    view_mode=gr.Radio(choices=[('真实相机','real'),('绕场景环绕（相对真实相机）','orbit'),('世界坐标指定','world')],value='real',label='相机位置')
                    with gr.Row():
                        orbit_yaw=gr.Number(value=0,label='环绕 yaw（度，绕相机上方向）');orbit_pitch=gr.Number(value=0,label='环绕 pitch（度；正值=相机升高向下俯视，约 90−相机俯角 即正上方）')
                        orbit_scale=gr.Number(value=1,label='到场景中心距离倍数（<1 拉近）')
                    with gr.Row():
                        offset_right=gr.Number(value=0,label='相机系右移（米）');offset_up=gr.Number(value=0,label='相机系上移（米）');offset_forward=gr.Number(value=0,label='相机系前移（米）')
                    with gr.Row():
                        world_position=gr.Textbox(value='',label='相机位置 x,y,z（世界系，米）',placeholder='0.5, -0.3, 1.2')
                        world_target=gr.Textbox(value='',label='注视点 x,y,z（留空 = 场景中心）')
                    roll=gr.Number(value=0,label='画面旋转 roll（度，绕相机光轴；正值 = 画面顺时针；三种相机模式均生效）')
                with gr.Row():
                    snapshot_frame=gr.Number(value=0,precision=0,label='指定帧号（源帧号）')
                    snapshot_run=gr.Button('指定帧相机拍摄预览')
                gr.Markdown('指定帧预览只渲染一帧，使用与正式视频完全相同的相机、近平面保护和固定裁剪（按上方起止帧段计算）；'
                            '真实相机时左侧并排显示同一裁剪的原始 RGB，用于核对对齐。')
                with gr.Row():video_preview=gr.Button('预览白底视频');video_run=gr.Button('生成完整白底视频',variant='primary')
        with gr.Row():
            job=gr.Dropdown(choices=[],value=None,label='当前序列的媒体任务记录')
            refresh=gr.Button('刷新任务记录');stop=gr.Button('停止所选媒体任务',variant='stop')
        submit_notice=gr.Markdown('选择序列后，点击预览或正式生成；不会自动展示其他序列的历史结果。')
        status=gr.Markdown('就绪');logs=gr.Textbox(label='媒体生成日志',lines=10,interactive=False)
        with gr.Row():image=gr.Image(label='制图 / 视频首帧 / 指定帧预览',height=360);video=gr.Video(label='白底视频',height=360)
        downloads=gr.File(label='下载结果和配置',file_count='multiple')
        with gr.Accordion('批量生成 · 多序列（沿用上方相机号、材质、制图 / 白底视频参数与自定义相机）',open=False):
            gr.Markdown('所选序列作为一个后台任务提交，每个序列一个步骤，按“并行序列数”同时渲染；结果照常写入各序列的 `output_reconstruction/` 目录。'
                        '校验不通过的序列（缺帧、缺文件、有运行中的任务等）不提交，原因列在状态表中。批量任务只在此处显示。')
            batch_sequences=gr.CheckboxGroup(choices=[],value=[],label='批量序列 / take')
            with gr.Row():
                batch_list=gr.Button('列出全部已导出序列');batch_sync=gr.Button('只勾选上方已选数据集')
                batch_all=gr.Button('全选');batch_none=gr.Button('清空')
            with gr.Row():
                batch_mode=gr.Radio(choices=[('白底视频','video'),('制图','figure'),('指定帧预览','snapshot')],value='video',label='批量生成内容')
                batch_preview=gr.Checkbox(value=False,label='仅预览（白底视频只渲染“预览帧数”帧，制图只渲染首个指定帧）')
            with gr.Row():
                batch_range=gr.Radio(choices=[('沿用上方起止帧（超出某序列已有帧段时裁剪到该序列范围）','inherit'),('每个序列完整帧段（忽略上方起止帧）','full')],
                                     value='inherit',label='白底视频 / 指定帧预览帧段')
                batch_parallel=gr.Number(value=4,precision=0,label='并行序列数')
                batch_gpu=gr.Textbox(value='auto',label='GPU（auto 或列表，如 0,1,4-7）')
            batch_run=gr.Button('提交批量任务',variant='primary')
            batch_notice=gr.Markdown('勾选序列后提交。')
            with gr.Row():
                batch_job=gr.Dropdown(choices=[],value=None,label='批量任务记录')
                batch_refresh=gr.Button('刷新批量任务');batch_stop=gr.Button('停止所选批量任务',variant='stop')
            batch_status=gr.Markdown('')
            with gr.Row():
                batch_item=gr.Dropdown(choices=[],value=None,label='查看已完成序列');batch_zip=gr.Button('打包下载已完成结果')
            with gr.Row():batch_image=gr.Image(label='制图 / 视频首帧 / 指定帧预览',height=360);batch_video=gr.Video(label='白底视频',height=360)
            batch_file=gr.File(label='批量结果 zip（<序列>/<文件>）')
            batch_signature=gr.State('')
        signature=gr.State('');timer=gr.Timer(3)
        params=[data_root,hand_prior,sequence,camera,hands,material,cal_w,cal_h,frames,include_input,yaw,pitch,start,end,resolution,framing,save_frames,preview_frames,
                view_mode,orbit_yaw,orbit_pitch,orbit_scale,offset_right,offset_up,offset_forward,world_position,world_target,roll,snapshot_frame]
        poll_outputs=[status,logs,image,video,downloads,signature]
        scope=[data_root,hand_prior,sequence]
        def current_choices(root,prior,seq):
            try:return controller.choices(dataset_path(root,prior,seq))
            except ValueError:return []
        def poll_selected(root,prior,seq,rid,previous):
            try:data=dataset_path(root,prior,seq)
            except ValueError:return '请选择当前导出序列。','',None,None,[],''
            if not rid:return f'当前导出序列：**{seq}**。请生成新结果或选择此序列的历史任务。','',None,None,[],''
            return controller.poll(rid,previous,data)
        def reset_scope(root,prior,seq):
            choices=current_choices(root,prior,seq)
            return (gr.Dropdown(choices=choices,value=None),
                    f'当前导出序列：**{seq or "未选择"}**。切换序列后需生成或选择对应任务。',
                    '',None,None,[],'','')
        reset_outputs=[job,status,logs,image,video,downloads,signature,submit_notice]
        def callback(mode,preview):
            def submit(*args):
                values=args[:-1]
                try:
                    rid=controller.submit(mode,preview,values,args[-1])
                    choices=current_choices(*values[:3])
                    return gr.Dropdown(choices=choices,value=rid),f'已提交 **{values[2]}** 的{MODE_LABELS[mode]}任务。',None,None,[],''
                except Exception as exc:
                    # This notice is separate from polled status, so errors remain visible.
                    return gr.Dropdown(choices=current_choices(*values[:3]),value=None),f'提交失败：{exc}',None,None,[],''
            return submit
        for button,mode,preview in [(fig_preview,'figure',True),(fig_run,'figure',False),(texture_run,'texture',False),(video_preview,'video',True),(video_run,'video',False),(snapshot_run,'snapshot',True)]:
            button.click(callback(mode,preview),params+[gpu],[job,submit_notice,image,video,downloads,signature],concurrency_id='hoi_media_submit',concurrency_limit=1)
        def refresh_jobs(root,prior,seq,rid):
            choices=current_choices(root,prior,seq)
            return gr.Dropdown(choices=choices,value=rid if rid in [v for _,v in choices] else None)
        def stop_job(root,prior,seq,rid):
            try:data=dataset_path(root,prior,seq)
            except ValueError:return '请选择当前导出序列。'
            meta=controller.b['_find_run'](rid)
            if not meta or not meta.get('media_export') or not controller.matches_dataset(meta,data):return '请选择当前序列的媒体任务。'
            controller.b['stop_run'](rid)
            return '已请求停止所选媒体任务。'
        discovery_inputs=[data_root,hand_prior,datasets,sequence]
        scan.click(discover,discovery_inputs,sequence)
        datasets.change(discover,discovery_inputs,sequence)
        if visualization_dataset is not None:
            # Visualizations lists all datasets, not just the top-level selection.
            # Follow explicit user input; automatic selector refresh must not undo
            # a new export choice made by the dataset-selection callback.
            visualization_dataset.input(discover,discovery_inputs+[visualization_dataset],sequence)
        sequence.change(reset_scope,scope,reset_outputs,queue=False)
        data_root.change(lambda:gr.Dropdown(choices=[],value=None),outputs=sequence).then(reset_scope,scope,reset_outputs)
        hand_prior.change(lambda:gr.Dropdown(choices=[],value=None),outputs=sequence).then(reset_scope,scope,reset_outputs)
        refresh.click(refresh_jobs,scope+[job],job)
        job.change(poll_selected,scope+[job,signature],poll_outputs)
        timer.tick(poll_selected,scope+[job,signature],poll_outputs,queue=False)
        stop.click(stop_job,scope+[job],submit_notice,concurrency_limit=None).then(poll_selected,scope+[job,signature],poll_outputs)

        def list_all(root,prior,chosen):
            choices=all_sequences(root,prior)
            return gr.CheckboxGroup(choices=choices,value=[c for c in chosen or [] if c in choices])
        def sync_selected(root,prior,selected):
            choices=all_sequences(root,prior);picked=sequence_choices(root,prior,list(selected or []))
            return gr.CheckboxGroup(choices=choices,value=[c for c in picked if c in choices])
        def select_all(root,prior):
            choices=all_sequences(root,prior)
            return gr.CheckboxGroup(choices=choices,value=choices)
        def clear_batch_scope(root,prior):
            return gr.CheckboxGroup(choices=all_sequences(root,prior),value=[])
        def submit_batch(*args):
            *values,chosen,mode,preview,frame_range,parallel,gpu_text=args
            try:
                rid,items,skipped=controller.submit_batch(mode,preview,values,chosen,gpu_text,parallel,frame_range)
                note=f'已提交批量{MODE_LABELS[mode]}任务 **{rid}**：{len(items)} 个序列。'
                clipped=sum(1 for item in items if item.get('note'))
                if clipped:note+=f' {clipped} 个序列的帧段按其已有范围裁剪。'
                if skipped:note+=f' {len(skipped)} 个序列未提交。'
                if clipped or skipped:note+='详见状态表。'
                return gr.Dropdown(choices=controller.batch_choices(),value=rid),note
            except Exception as exc:
                return gr.Dropdown(choices=controller.batch_choices()),f'批量提交失败：{exc}'
        def refresh_batches(rid):
            choices=controller.batch_choices()
            return gr.Dropdown(choices=choices,value=rid if rid in [v for _,v in choices] else None)
        def stop_batch(rid):
            meta=controller.b['_find_run'](rid) if rid else None
            if not meta or not meta.get('media_batch'):return '请选择批量任务。'
            controller.b['stop_run'](rid)
            return '已请求停止所选批量任务；已完成的序列保留。'
        def zip_batch(rid):
            try:path,n=controller.batch_zip(rid)
            except Exception as exc:return None,f'打包失败：{exc}'
            return path,f'已打包 {n} 个已完成序列的结果。'
        batch_outputs=[batch_status,batch_item,batch_signature]
        batch_list.click(list_all,[data_root,hand_prior,batch_sequences],batch_sequences)
        batch_sync.click(sync_selected,[data_root,hand_prior,datasets],batch_sequences)
        batch_all.click(select_all,[data_root,hand_prior],batch_sequences)
        batch_none.click(lambda:[],outputs=batch_sequences)
        data_root.change(clear_batch_scope,[data_root,hand_prior],batch_sequences)
        hand_prior.change(clear_batch_scope,[data_root,hand_prior],batch_sequences)
        batch_run.click(submit_batch,params+[batch_sequences,batch_mode,batch_preview,batch_range,batch_parallel,batch_gpu],[batch_job,batch_notice],
                        concurrency_id='hoi_media_submit',concurrency_limit=1)
        batch_refresh.click(refresh_batches,batch_job,batch_job)
        # previous=None forces a full refresh (clears the completed-item list) when the selected batch changes.
        batch_job.change(lambda rid:controller.batch_poll(rid,None),batch_job,batch_outputs).then(lambda:(None,None,None),outputs=[batch_image,batch_video,batch_file])
        timer.tick(controller.batch_poll,[batch_job,batch_signature],batch_outputs,queue=False)
        batch_stop.click(stop_batch,batch_job,batch_notice,concurrency_limit=None).then(lambda rid:controller.batch_poll(rid,None),batch_job,batch_outputs)
        batch_item.change(controller.batch_item_media,[batch_job,batch_item],[batch_image,batch_video])
        batch_zip.click(zip_batch,batch_job,[batch_file,batch_notice])
    return controller
