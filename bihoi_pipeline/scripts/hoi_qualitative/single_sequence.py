#!/usr/bin/env python3
"""One-sequence figure + fixed-real-camera white-background video workflows."""
import os
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import render_comparison as core


def camera_for_crop(k, box, resolution, aa):
    x0,y0,x1,y1=box
    if x1<=x0 or y1<=y0: raise ValueError('Empty crop')
    out=k.copy();out[0,2]-=x0;out[1,2]-=y0
    out[0]*=resolution[0]*aa/(x1-x0);out[1]*=resolution[1]*aa/(y1-y0)
    return core.pyrender.IntrinsicsCamera(out[0,0],out[1,1],out[0,2],out[1,2],znear=.001,zfar=100),out


class ProjectionError(ValueError):
    def __init__(self,frame,camera,parts):
        self.details={'source_frame':frame,'camera':camera,'near_plane_m':.001,'invalid_parts':parts,
                      'reason':'Geometry crosses or lies behind the real camera near plane. A crop cannot fix this.',
                      'remedy':'Choose a verified camera/frame range, or inspect the saved reconstruction. Do not translate geometry or silently drop frames.'}
        description='; '.join(f"{p['part']}: z=[{p['min_z_m']:.6f}, {p['max_z_m']:.6f}] m" for p in parts)
        super().__init__(f'Camera {camera}, source frame {frame}: {description}. Geometry is behind or crosses the camera near plane (0.001 m); fixed cropping cannot fix it. Inspect projection_error.json; select another verified camera/frame range or check the reconstruction.')


def project_bounds(meshes,k,e,frame=None,camera=None):
    parts=[]
    for mesh,kind in meshes:
        z=mesh.vertices@e[2,:3]+e[2,3]
        if not np.isfinite(z).all() or z.min()<=.001:
            parts.append({'part':kind,'min_z_m':float(z.min()),'max_z_m':float(z.max()),
                          'vertices_behind_near_plane':int(np.count_nonzero(z<=.001))})
    if parts:raise ProjectionError(frame,camera,parts)
    vertices=np.concatenate([m.vertices for m,_ in meshes])
    vc=core.transform(vertices,e)
    uv=vc@k.T;uv=uv[:,:2]/uv[:,2:]
    return uv.min(0),uv.max(0),vertices


def crop_box(lo,hi,aspect,margin):
    span=(hi-lo)*margin
    height=max(span[1],span[0]/aspect)
    width=height*aspect
    center=(lo+hi)/2
    return [float(center[0]-width/2),float(center[1]-height/2),
            float(center[0]+width/2),float(center[1]+height/2)]


def retreat_camera(extrinsic,min_depth,clearance=.1):
    if not np.isfinite(min_depth) or not np.isfinite(clearance) or clearance<=.001:
        raise ValueError('Camera retreat requires finite geometry and clearance > 0.001 m')
    distance=max(0.,clearance-min_depth) if min_depth<=.001 else 0.
    adjusted=extrinsic.copy()
    adjusted[2,3]+=distance
    return adjusted,float(distance)


def custom_extrinsic(e,spec,pivot):
    """World-to-camera (OpenCV) matrix of a user-defined camera; intrinsics stay those of the real camera.

    orbit: rotate the real camera about `pivot` by yaw (around its up axis) / pitch (around its right
    axis), scale its distance to the pivot, then shift it in its own frame (right / up / forward, m).
    Zero angles, scale 1 and zero offsets reproduce the real camera exactly.
    world: camera at `position` looking at `look_at` (default: pivot), horizon from the real camera's up.
    roll_degrees (any type, applied last): rotation about the optical axis; positive turns the picture clockwise.
    """
    c2w=np.linalg.inv(e);r=c2w[:3,:3];c=c2w[:3,3];pivot=np.asarray(pivot,float)
    kind=spec.get('type','real')
    if kind=='real':new_r,new_c=r,c
    elif kind=='orbit':
        gl=r@np.diag([1.,-1.,-1.])  # same yaw/pitch convention as the figure Novel View
        local=core.Rotation.from_euler('yx',[float(spec.get('yaw_degrees',0)),float(spec.get('pitch_degrees',0))],degrees=True).as_matrix()
        world=gl@local@gl.T
        scale=float(spec.get('distance_scale',1.))
        if not math.isfinite(scale) or scale<=0:raise ValueError('distance_scale must be > 0')
        new_r=world@r;new_c=pivot+scale*(world@(c-pivot))
        right,up,forward=[float(x) for x in spec.get('offset_right_up_forward_m',[0,0,0])]
        new_c=new_c+new_r@np.array([right,-up,forward])
    elif kind=='world':
        new_c=np.asarray(spec['position'],float)
        target=pivot if spec.get('look_at') is None else np.asarray(spec['look_at'],float)
        z=target-new_c;n=np.linalg.norm(z)
        if not np.isfinite(n) or n<1e-6:raise ValueError('Camera position must differ from the look-at point')
        z/=n;x=np.cross(z,-r[:,1]);n=np.linalg.norm(x)
        if n<1e-6:raise ValueError('View direction is parallel to the reference camera up axis; move the camera sideways')
        x/=n;new_r=np.stack([x,np.cross(z,x),z],axis=1)
    else:raise ValueError(f'Unknown camera type: {kind}')
    if not np.isfinite(new_c).all():raise ValueError('Nonfinite custom camera position')
    roll=float(spec.get('roll_degrees',0))
    if not math.isfinite(roll):raise ValueError('roll_degrees must be finite')
    # Image-space rotation R_z (x right, y down) applied after world->camera: clockwise on screen.
    new_r=new_r@core.Rotation.from_euler('z',roll,degrees=True).as_matrix().T
    c2w=np.eye(4);c2w[:3,:3]=new_r;c2w[:3,3]=new_c
    return np.linalg.inv(c2w)


def video_ids(ids,start,end):
    # End is exclusive; absent result frames are an error, never silently skipped.
    stop=int(ids[-1])+1 if end is None else int(end)
    selected=list(range(int(start),stop))
    if not selected: raise ValueError('Empty video range')
    missing=sorted(set(selected)-set(map(int,ids)))
    if missing: raise ValueError(f'Missing video source frames: {missing[:12]}')
    return selected


def compose_figure(images,rows,size,output,title,include_input):
    labels=['Input Frames','Camera View','Novel View'] if include_input else ['Camera View','Novel View']
    gap=24;outer=48;top=170;row_gap=60
    width=outer*2+len(labels)*size+(len(labels)-1)*gap
    canvas=Image.new('RGB',(width,top+len(rows)*(size+row_gap)+50),'white');d=ImageDraw.Draw(canvas)
    d.text((width/2,45),title,fill='black',font=core.font(42),anchor='mm')
    for j,label in enumerate(labels):
        d.text((outer+j*(size+gap)+size/2,120),label,fill='black',font=core.font(36),anchor='mm')
    for i,(cells,row) in enumerate(zip(images,rows)):
        y=top+i*(size+row_gap)
        for j,im in enumerate(cells):canvas.paste(im,(outer+j*(size+gap),y))
        d.text((width/2,y+size+28),f"Frame {row['frame']:06d} · Camera {row['camera']}",fill='black',font=core.font(28),anchor='mm')
    canvas.save(output/'figure.png',dpi=(300,300));canvas.save(output/'figure.pdf',resolution=300)
    canvas.thumbnail((1500,1800));canvas.save(output/'figure_review.jpg',quality=95)


class Workflow:
    def __init__(self,cfg,config_path):
        self.cfg=cfg
        self.data=core.resolve(cfg['dataset']);self.export=core.resolve(cfg['export'])
        self.render_cfg=json.loads(core.resolve(cfg['render_settings']).read_text())
        self.render_cfg.update(cfg.get('render_overrides',{}))
        self.hands=cfg.get('include_hands',True)
        mano=core.resolve_near(self.render_cfg['mano_dir'],core.resolve(cfg['render_settings']))
        self.models={side:core.smplx.MANO(str(mano/f'MANO_{side.upper()}.pkl'),is_rhand=side=='right',use_pca=False,flat_hand_mean=True).eval() for side in (['left','right'] if self.hands else [])}
        self.used={str(config_path),str(core.resolve(cfg['render_settings'])),str(Path(__file__).resolve()),str(Path(core.__file__).resolve())}
        self.used.update(str(mano/f'MANO_{s.upper()}.pkl') for s in self.models)
        self.meta=json.loads((self.export/'metadata.json').read_text())
        self.ids=np.load(self.export/'source_frame_ids.npy')
        if self.meta['resampled'] or len(self.ids)==0 or not np.all(np.diff(self.ids)>0):
            raise ValueError('Need non-resampled, unique increasing source frame IDs')
        self.cam=str(cfg['camera'])
        self.k,self.e,self.dist=core.calibration(self.data,self.cam,cfg)
        self.original_e=self.e.copy()
        self.camera_adjustment=None
        self.pose=np.linalg.inv(self.e)@core.FLIP
        self.used.update(str(self.data/n) for n in ['intri.yml','extri.yml'])
        self.glb=core.resolve(cfg['object_glb']) if cfg.get('object_glb') else None
        self.sensor=tuple(cfg['calibration_resolution'])
        self.used.add(str(self.export/'metadata.json'))
        self.used.add(str(self.export/'source_frame_ids.npy'))

    def load(self,frame):
        meshes,state=core.load_result(self.export,frame,self.models,self.used,self.glb,self.hands)
        if meshes is None:raise ValueError(f'Frame {frame}: {state}')
        return meshes,state

    def manifest(self,out,mode,records,settings):
        value={'mode':mode,'config':self.cfg,'egl_device_id':int(os.environ.get('EGL_DEVICE_ID','0')),'effective_render_settings':self.render_cfg,
               'camera_K_original':self.k.tolist(),'world_to_camera':self.e.tolist(),
               'world_to_camera_original':self.original_e.tolist(),'camera_adjustment':self.camera_adjustment,
               'distortion':self.dist.tolist(),'records':records,'settings':settings,
               'sources_sha256':{p:core.sha(p) for p in sorted(self.used)}}
        (out/'manifest.json').write_text(json.dumps(value,indent=2))

    def figure(self,out,preview):
        options=self.cfg['figure'];frames=options['frames'][:1] if preview else options['frames']
        if not frames or len(set(frames))!=len(frames):raise ValueError('Select unique figure frames')
        include_input=options.get('include_input',True)
        size=int(options.get('cell_size',1000 if include_input else 1500))
        if size*(3 if include_input else 2)<3000:raise ValueError('Figure cells must total at least 3000 pixels across')
        aa=self.render_cfg['supersampling'];renderer=core.pyrender.OffscreenRenderer(size*aa,size*aa)
        rows=[];images=[]
        try:
            for frame in frames:
                meshes,state=self.load(frame)
                lo,hi,verts=project_bounds(meshes,self.k,self.e,frame,self.cam)
                # Integer square crop shared exactly by RGB and camera rendering.
                side=math.ceil(max(hi-lo)*self.render_cfg['crop_margin']);xy=np.floor((lo+hi-side)/2).astype(int)
                box=[int(xy[0]),int(xy[1]),int(xy[0]+side),int(xy[1]+side)]
                camera,kout=camera_for_crop(self.k,box,(size,size),aa)
                center=(verts.min(0)+verts.max(0))/2
                r=self.pose[:3,:3]@core.Rotation.from_euler('yx',options['novel_yaw_pitch_degrees'],degrees=True).as_matrix()
                v=(verts-center)@r;extent=np.ptp(v[:,:2],axis=0).max()*self.render_cfg['crop_margin']/2
                center+=r[:,:2]@((v[:,:2].min(0)+v[:,:2].max(0))/2)
                novel_pose=np.eye(4);novel_pose[:3,:3]=r;novel_pose[:3,3]=center+r[:,2]*max(1.,np.ptp(v[:,2])*3)
                novel=core.pyrender.OrthographicCamera(xmag=extent,ymag=extent,znear=.001,zfar=100)
                cell=out/'cells'/f'{frame:06d}';cell.mkdir(parents=True,exist_ok=True)
                cells=[]
                if include_input:
                    candidates=[self.data/f'images/{self.cam}/{frame:06d}{ext}' for ext in ['.jpg','.png','.jpeg']]
                    path=next((p for p in candidates if p.is_file()),None)
                    if path is None:raise FileNotFoundError(f'Missing RGB frame {frame}')
                    with Image.open(path) as im:
                        if im.size!=self.sensor:raise ValueError('RGB/calibration resolution mismatch')
                        rgb=Image.new('RGB',(side,side),'white');rgb.paste(im.convert('RGB'),(-box[0],-box[1]));rgb=rgb.resize((size,size),Image.Resampling.LANCZOS)
                    rgb.save(cell/'input.png');cells.append(rgb);self.used.add(str(path))
                for name,cam,pose in [('camera',camera,self.pose),('novel',novel,novel_pose)]:
                    im=core.render(meshes,cam,pose,size,self.render_cfg,renderer);im.save(cell/f'{name}.png');cells.append(im)
                row={'frame':frame,'camera':self.cam,'crop_xyxy':box,'raster_K':kout.tolist(),'novel_camera_to_world':novel_pose.tolist(),'novel_orthographic_half_extent_m':float(extent),'result':state}
                rows.append(row);images.append(cells);print(f'Figure frame {frame}: Camera + Novel',flush=True)
        finally:renderer.delete()
        compose_figure(images,rows,size,out,self.data.name,include_input)
        self.manifest(out,'figure',rows,options)

    def video(self,out,preview):
        options=self.cfg['video'];resolution=tuple(options['resolution'])
        if len(resolution)!=2 or any(int(n)!=n or n<=0 or n%2 for n in resolution):
            raise ValueError('H.264 yuv420p resolution must contain two positive even integers')
        if shutil.which('ffmpeg') is None:raise RuntimeError('ffmpeg is required')
        frames=video_ids(self.ids,options['start_frame'],options.get('end_frame_exclusive'))
        if preview:frames=frames[:int(options.get('preview_frames',12))]
        if not frames:raise ValueError('No preview frames')
        fps=float(self.meta['source_fps'])
        if not np.isfinite(fps) or fps<=0:raise ValueError('Invalid source FPS')
        timestamps=self.export/'timestamps_sec.npy'
        if timestamps.exists():
            self.used.add(str(timestamps));times=np.load(timestamps)
            if len(times)!=len(self.ids):raise ValueError('Timestamp length mismatch')
            selected=times[np.searchsorted(self.ids,frames)]
            if len(selected)>1 and not np.allclose(np.diff(selected),1/fps,atol=1e-5):raise ValueError('Nonuniform timing requires a VFR encoder; refusing silent retiming')
        camera,kout,box,mode=self.setup_video_camera(frames,out,options,resolution)
        aa=self.render_cfg['supersampling']
        output=out/'camera_white.mp4';temp=out/'camera_white.pending.mp4'
        command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{resolution[0]}x{resolution[1]}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
        records=[];renderer=core.pyrender.OffscreenRenderer(resolution[0]*aa,resolution[1]*aa)
        with (out/'ffmpeg.log').open('wb') as log:
            proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
            try:
                for index,frame in enumerate(frames):
                    meshes,state=self.load(frame);lo,hi,_=project_bounds(meshes,self.k,self.e,frame,self.cam)
                    outside=bool((lo<np.array(box[:2])).any() or (hi>np.array(box[2:])).any())
                    if outside and options.get('reject_clipping',True):raise ValueError(f'Frame {frame} is clipped: choose fixed_crop or another camera')
                    im=core.render(meshes,camera,self.pose,resolution,self.render_cfg,renderer)
                    if options.get('save_frames',False):
                        folder=out/'frames';folder.mkdir(exist_ok=True);im.save(folder/f'{frame:06d}.png')
                    if index==0:im.save(out/'first_frame.png')
                    proc.stdin.write(im.tobytes())
                    records.append({'video_index':index,'source_frame':frame,'time_sec':index/fps,'source_time_sec':frame/fps,'outside_crop':outside,'result':state})
                    print(f'Video {index+1}/{len(frames)}: source {frame}',flush=True)
                proc.stdin.close()
                if proc.wait()!=0:raise RuntimeError(f'ffmpeg failed; see {out/"ffmpeg.log"}')
                temp.replace(output)
            except BaseException:
                proc.kill();proc.wait();temp.unlink(missing_ok=True)
                raise
            finally:renderer.delete()
        self.manifest(out,'video',records,{'fps':fps,'resolution':resolution,'framing':mode,'fixed_crop_xyxy':box,'raster_K':kout.tolist(),'camera_to_world':self.pose.tolist(),'novel_view':False,'encoder_command':command,'duration_sec':len(frames)/fps})

    def setup_video_camera(self,frames,out,options,resolution):
        """Fix one camera (real or custom), near-plane retreat and crop for the whole frame range."""
        self.e=self.original_e.copy();self.camera_adjustment=None;self.pivot=None
        spec=options.get('custom_camera') or {'type':'real'}
        if spec.get('type','real')!='real':
            lo=np.full(3,np.inf);hi=-lo
            for i,frame in enumerate(frames):
                if i%30==0:print(f'Scene bounds preflight {i+1}/{len(frames)}',flush=True)
                meshes,_=self.load(frame)
                for mesh,_ in meshes:lo=np.minimum(lo,mesh.vertices.min(0));hi=np.maximum(hi,mesh.vertices.max(0))
            self.pivot=(lo+hi)/2
        # Any non-real type or a roll makes this a custom view; the input RGB no longer matches it.
        self.custom_view=spec.get('type','real')!='real' or float(spec.get('roll_degrees',0))!=0
        if self.custom_view:
            self.e=custom_extrinsic(self.original_e,spec,self.pivot if self.pivot is not None else np.zeros(3))
            scene='' if self.pivot is None else f', scene center {np.round(self.pivot,4).tolist()} m'
            print(f'Custom camera ({spec.get("type","real")}, roll {float(spec.get("roll_degrees",0)):g}°): center {np.round(np.linalg.inv(self.e)[:3,3],4).tolist()} m{scene}',flush=True)
        self.pose=np.linalg.inv(self.e)@core.FLIP
        policy=options.get('near_plane_policy','retreat')
        if policy not in ('retreat','error'):raise ValueError('near_plane_policy must be retreat or error')
        if policy=='retreat':
            minimum=np.inf;affected=[]
            for i,frame in enumerate(frames):
                if i%30==0:print(f'Camera depth preflight {i+1}/{len(frames)}: camera {self.cam}',flush=True)
                meshes,_=self.load(frame)
                depth=min(float((mesh.vertices@self.e[2,:3]+self.e[2,3]).min()) for mesh,_ in meshes)
                if not np.isfinite(depth):raise ValueError(f'Nonfinite geometry depth at source frame {frame}')
                minimum=min(minimum,depth)
                if depth<=.001:affected.append(int(frame))
            self.e,distance=retreat_camera(self.e,minimum,float(options.get('retreat_clearance_m',.1)))
            self.pose=np.linalg.inv(self.e)@core.FLIP
            if distance:
                self.camera_adjustment={'type':'backward_translation','distance_m':distance,
                    'original_min_depth_m':minimum,'adjusted_min_depth_m':minimum+distance,
                    'affected_source_frames':affected,'fixed_for_entire_video':True,
                    'world_to_camera_original':self.original_e.tolist(),'world_to_camera_adjusted':self.e.tolist(),
                    'warning':'Adjusted camera view: orientation and intrinsics retained; position differs from the calibrated input camera.'}
                (out/'camera_adjustment.json').write_text(json.dumps(self.camera_adjustment,indent=2))
                print(f'WARNING: Camera {self.cam} retreated {distance:.6f} m for all {len(frames)} frames; {len(affected)} frames cross the original near plane. Orientation and intrinsics unchanged; this is an adjusted camera view. See camera_adjustment.json.',flush=True)
        mode=options.get('framing','full_frame')
        if mode=='full_frame':box=[0.,0.,float(self.sensor[0]),float(self.sensor[1])]
        elif mode=='fixed_crop':
            lo=np.full(2,np.inf);hi=-lo
            for i,f in enumerate(frames):
                if i%30==0:print(f'Crop preflight {i+1}/{len(frames)}: camera {self.cam}, source {f}',flush=True)
                meshes,_=self.load(f);a,b,_=project_bounds(meshes,self.k,self.e,f,self.cam);lo=np.minimum(lo,a);hi=np.maximum(hi,b)
            box=crop_box(lo,hi,resolution[0]/resolution[1],self.render_cfg['crop_margin'])
        else:raise ValueError('framing must be full_frame or fixed_crop')
        aa=self.render_cfg['supersampling'];camera,kout=camera_for_crop(self.k,box,resolution,aa)
        self.write_camera_info(out,spec)
        return camera,kout,box,mode

    def write_camera_info(self,out,spec):
        c2w=np.linalg.inv(self.e);ref=np.linalg.inv(self.original_e)
        info={'reference_camera':self.cam,'custom_camera':spec,'camera_center_world_m':c2w[:3,3].tolist(),
              'view_direction_world':c2w[:3,2].tolist(),'up_direction_world':(-c2w[:3,1]).tolist(),
              'reference_camera_center_world_m':ref[:3,3].tolist(),'reference_view_direction_world':ref[:3,2].tolist(),
              'scene_center_world_m':None if self.pivot is None else self.pivot.tolist(),
              'world_to_camera':self.e.tolist(),'near_plane_retreat_m':(self.camera_adjustment or {}).get('distance_m',0.)}
        (out/'camera.json').write_text(json.dumps(info,indent=2))

    def snapshot(self,out,preview):
        """One frame shot by the configured video camera, with the same fixed crop as the full video."""
        options=self.cfg['video'];resolution=tuple(options['resolution'])
        frame=int(self.cfg['snapshot']['frame'])
        if frame not in set(map(int,self.ids)):raise ValueError(f'Missing snapshot source frame {frame}')
        # Crop/retreat are fixed over the configured video range (plus this frame), exactly as the video will be.
        frames=sorted(set(video_ids(self.ids,options['start_frame'],options.get('end_frame_exclusive')))|{frame})
        camera,kout,box,mode=self.setup_video_camera(frames,out,options,resolution)
        aa=self.render_cfg['supersampling'];renderer=core.pyrender.OffscreenRenderer(resolution[0]*aa,resolution[1]*aa)
        try:
            meshes,state=self.load(frame);lo,hi,_=project_bounds(meshes,self.k,self.e,frame,self.cam)
            im=core.render(meshes,camera,self.pose,resolution,self.render_cfg,renderer)
        finally:renderer.delete()
        im.save(out/'snapshot.png')
        outside=bool((lo<np.array(box[:2])).any() or (hi>np.array(box[2:])).any())
        compare=None
        # The input RGB is only meaningful when the rendering camera is the unmodified real camera.
        if not self.custom_view and not self.camera_adjustment:
            candidates=[self.data/f'images/{self.cam}/{frame:06d}{ext}' for ext in ['.jpg','.png','.jpeg']]
            path=next((p for p in candidates if p.is_file()),None)
            if path is not None:
                with Image.open(path) as src:
                    if src.size==self.sensor:
                        rgb=Image.new('RGB',(int(round(box[2]-box[0])),int(round(box[3]-box[1]))),'white')
                        rgb.paste(src.convert('RGB'),(-int(round(box[0])),-int(round(box[1]))))
                        rgb=rgb.resize(resolution,Image.Resampling.LANCZOS)
                        compare=Image.new('RGB',(resolution[0]*2,resolution[1]),'white');compare.paste(rgb,(0,0));compare.paste(im,(resolution[0],0))
                        compare.save(out/'snapshot_compare.png');self.used.add(str(path))
        print(f'Snapshot source frame {frame}: camera {self.cam}'+(' (custom)' if self.custom_view else '')+(' — geometry outside the crop' if outside else ''),flush=True)
        self.manifest(out,'snapshot',[{'source_frame':frame,'outside_crop':outside,'rgb_comparison':compare is not None,'result':state}],
                      {'resolution':resolution,'framing':mode,'fixed_crop_xyxy':box,'crop_frames':[frames[0],frames[-1]],'raster_K':kout.tolist(),'camera_to_world':self.pose.tolist()})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--mode',choices=['figure','video','snapshot','all'],required=True);ap.add_argument('--preview',action='store_true');ap.add_argument('--egl-device',type=int,default=None);args=ap.parse_args()
    if args.egl_device is not None:
        if args.egl_device < 0:raise ValueError('EGL device index must be nonnegative')
        os.environ['EGL_DEVICE_ID']=str(args.egl_device)
    path=core.resolve(args.config);cfg=json.loads(path.read_text());workflow=Workflow(cfg,path)
    root=core.resolve(cfg['output'])/('preview' if args.preview else 'final')
    for mode in (['figure','video'] if args.mode=='all' else [args.mode]):
        out=root/mode;out.mkdir(parents=True,exist_ok=True)
        (out/'config.json').write_text(json.dumps(cfg,indent=2))
        try:getattr(workflow,mode)(out,args.preview)
        except Exception as exc:
            if isinstance(exc,ProjectionError):
                (out/'projection_error.json').write_text(json.dumps(exc.details,indent=2))
            (out/'status.json').write_text(json.dumps({'status':'error','error':str(exc)},indent=2));raise
        (out/'status.json').write_text(json.dumps({'status':'complete','mode':mode},indent=2))

if __name__=='__main__':main()
