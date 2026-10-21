#!/usr/bin/env python3
"""Render measured cameras and a shared novel view from standardized HOI exports.
No fitting, smoothing, mesh repair, pose editing, or missing-result substitution.
"""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import argparse, hashlib, json, math, sys
from functools import lru_cache
from scipy.spatial import cKDTree
from pathlib import Path
import cv2
import numpy as np
import torch
import smplx
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
FLIP = np.diag([1., -1., -1., 1.])
def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

def resolve_near(p, anchor):
    """Relative p next to the config file `anchor` when it exists there (standalone package), else under ROOT (repository)."""
    p = Path(p)
    near = Path(anchor).resolve().parent / p
    return p if p.is_absolute() else near if near.exists() else ROOT / p

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def transform(v, t):
    return v @ t[:3,:3].T + t[:3,3]

@lru_cache(maxsize=16)
def font(n):
    # Other machines may lack the Debian font path; any bold sans is fine for labels.
    for name in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf','DejaVuSans-Bold.ttf','Arial Bold.ttf','arialbd.ttf']:
        try:return ImageFont.truetype(name, n)
        except OSError:pass
    return ImageFont.load_default(n)

def label(text, size):
    im = Image.new('RGB', (size,size), 'white')
    ImageDraw.Draw(im).text((size/2,size/2), text, fill='black', font=font(44), anchor='mm')
    return im

def calibration(data, camera, cfg):
    fs = cv2.FileStorage(str(data/'intri.yml'), cv2.FILE_STORAGE_READ)
    k = fs.getNode('K_'+camera).mat(); dist = fs.getNode('dist_'+camera).mat(); fs.release()
    if k is None or dist is None: raise ValueError('Missing calibration')
    if np.any(np.abs(dist)>1e-12): raise ValueError('Nonzero distortion requires explicit distortion-aware rendering adapter')
    fs = cv2.FileStorage(str(data/'extri.yml'), cv2.FILE_STORAGE_READ)
    e=np.eye(4); e[:3,:3]=fs.getNode('Rot_'+camera).mat();e[:3,3]=fs.getNode('T_'+camera).mat().ravel();fs.release()
    return k,e,dist

@lru_cache(maxsize=8)
def load_textured_object(glb, exported_ply):
    glb, exported_ply = Path(glb), Path(exported_ply)
    sidecar = glb.with_suffix('.json')
    tracking = glb.with_name('compressed.ply')
    info = json.loads(sidecar.read_text())
    if info['status'] != 'complete' or sha(glb) != info['glb_sha256'] or sha(tracking) != info['tracking_ply_sha256']:
        raise ValueError('Texture GLB provenance checksum mismatch')
    alignment = np.array(info['canonical_to_tracking_local'])
    if alignment.shape != (4,4) or not np.allclose(alignment, np.eye(4)):
        raise ValueError('This adapter requires the documented shared metric canonical frame')
    current = trimesh.load(tracking, process=False)
    exported = trimesh.load(exported_ply, process=False)
    distances = np.r_[cKDTree(current.vertices).query(exported.vertices)[0],
                      cKDTree(exported.vertices).query(current.vertices)[0]]
    bound_error = float(np.max(np.abs(current.bounds-exported.bounds)))
    q99 = float(np.quantile(distances, .99))
    # Check compatibility, never estimate a transform or deform either mesh.
    if bound_error > .001 or q99 > .001:
        raise ValueError('New tracking mesh differs materially from pose-export mesh; require verified alignment/new poses')
    scene = trimesh.load(glb, force='scene', process=False)
    objects = list(scene.dump())
    if len(objects) != 1:
        raise ValueError('Expected one textured object; multi-part GLB needs a separate adapter')
    mesh = objects[0]
    if mesh.visual.kind != 'texture' or mesh.visual.uv is None or mesh.visual.material.baseColorTexture is None:
        raise ValueError('GLB lacks UV coordinates or embedded base-color texture')
    audit = {'glb':str(glb),'sidecar':str(sidecar),'appearance_source':info['appearance_source'],
             'texture_resolution':list(mesh.visual.material.baseColorTexture.size),
             'alignment':'identity in documented metric canonical frame; no fitted alignment',
             'tracking_export_mesh_identical':sha(tracking)==sha(exported_ply),
             'tracking_vs_export_bounds_max_m':bound_error,
             'tracking_vs_export_nearest_vertex_q99_m':q99,
             'tracking_vs_export_nearest_vertex_max_m':float(distances.max()),
             'geometry_note':'Uses the actual new textured GLB geometry with existing exported poses; mesh versions are not identical.'}
    return mesh, audit, (str(glb),str(sidecar),str(tracking))


def load_result(folder, frame, models, used, object_glb=None, include_hands=True):
    required=['metadata.json','source_frame_ids.npy','obj_pose.npz','object_mesh.ply']
    if include_hands:
        required += ['mano_left.npz','mano_right.npz']
    missing=[str(folder/n) for n in required if not (folder/n).is_file()]
    if missing: return None, {'status':'Missing','missing':missing}
    meta=json.loads((folder/'metadata.json').read_text())
    if meta['resampled']: raise ValueError('Resampled outputs need an explicit original-frame adapter')
    ids=np.load(folder/'source_frame_ids.npy');idx=np.flatnonzero(ids==frame)
    if len(idx)==0:return None,{'status':'Missing','reason':'source frame absent'}
    if len(idx)!=1:raise ValueError('Ambiguous source frame')
    i=int(idx[0]); t=np.linalg.inv(np.array(meta['world_frame']['T_export_world_from_original_easymocap_world']))
    if meta['units']['length']!='meter':raise ValueError('Unsupported units')
    meshes=[]
    for side in (['left','right'] if include_hands else []):
        if not meta['mano_'+side]['hand_pose']['flat_hand_mean']:raise ValueError('Unexpected MANO mean policy')
        d=np.load(folder/f'mano_{side}.npz')
        kw={k:torch.tensor(d[k][i:i+1],dtype=torch.float32) for k in ['global_orient','hand_pose','transl']}
        b=d['betas'];kw['betas']=torch.tensor(b[None] if b.ndim==1 else b[i:i+1],dtype=torch.float32)
        with torch.no_grad():v=models[side](**kw).vertices[0].numpy()
        meshes.append((trimesh.Trimesh(vertices=transform(v,t),faces=models[side].faces,process=False),side))
    object_audit = None
    if object_glb:
        m, object_audit, sources = load_textured_object(str(object_glb), str(folder/'object_mesh.ply'))
        m = m.copy()
        used.update(sources)
    else:
        m=trimesh.load(folder/'object_mesh.ply',process=False)
    d=np.load(folder/'obj_pose.npz');q=d['quat'][i]
    if meta['object']['quat']['order']!='wxyz':raise ValueError('Unexpected quaternion convention')
    p=np.eye(4);p[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();p[:3,3]=d['trans'][i]
    m.apply_transform(t@p);meshes.append((m,'object'))
    for mesh,_ in meshes:
        if not np.isfinite(mesh.vertices).all():raise ValueError('Nonfinite geometry; investigate instead of declaring Failed')
    used.update(str(folder/n) for n in required)
    return meshes, {'status':'OK','source_index':i,'export_to_original_world':t.tolist(),'object_source_color':m.visual.kind,'object_asset_audit':object_audit,'vertices':[len(m.vertices) for m,_ in meshes]}

def render(meshes, camera, pose, size, cfg, renderer=None):
    aa=cfg['supersampling']
    width, height = (size, size) if isinstance(size, int) else size
    own_renderer = renderer is None
    if own_renderer:
        renderer=pyrender.OffscreenRenderer(width*aa,height*aa)
    # Four nearby directional sources, rendered separately then averaged: soft visibility shadows.
    samples=[]
    for offset in cfg['light_offsets_degrees']:
        scene=pyrender.Scene(bg_color=[1.0,1.0,1.0,1.0],ambient_light=cfg['ambient_light'])
        for m,kind in meshes:
            if kind=='object' and cfg['object_material']=='source':
                if m.visual.kind not in ['vertex','face','texture']:
                    raise ValueError('Source-colored object requested but mesh has no color or texture')
                pm=pyrender.Mesh.from_trimesh(m,smooth=True)
                for primitive in pm.primitives:
                    primitive.material.metallicFactor=0
                    primitive.material.roughnessFactor=1
                    primitive.material.alphaMode='OPAQUE'
            else:
                color=cfg[kind+'_color']
                mat=pyrender.MetallicRoughnessMaterial(baseColorFactor=color,metallicFactor=0,roughnessFactor=1,alphaMode='OPAQUE',doubleSided=True)
                display_mesh=trimesh.Trimesh(vertices=m.vertices,faces=m.faces,process=False)
                pm=pyrender.Mesh.from_trimesh(display_mesh,material=mat,smooth=True)
            scene.add(pm)
        scene.add(camera,pose=pose)
        light_pose=pose.copy();light_pose[:3,:3]=pose[:3,:3]@Rotation.from_euler('xy',[25+offset[0],-30+offset[1]],degrees=True).as_matrix()
        scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=cfg['light_intensity']),pose=light_pose)
        try:rgb,_=renderer.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
        except BaseException:
            if own_renderer: renderer.delete()
            raise
        samples.append(rgb.astype(np.float32))
    if own_renderer: renderer.delete()
    rgb=np.clip(np.mean(samples,axis=0),0,255).astype(np.uint8)
    return Image.fromarray(rgb).resize((width,height),Image.Resampling.LANCZOS)

def compose(rows, cfg, out, name):
    s=cfg['cell_size'];gap=cfg['gap'];left=140;top=170;cols=1+2*len(cfg['methods']);w=left+cols*s+(cols+1)*gap
    h=top+len(rows)*(s+70)+110
    canvas=Image.new('RGB',(w,h),'white');draw=ImageDraw.Draw(canvas)
    x0=left+gap
    draw.text((x0+s/2,100),'Input Frames',font=font(42),fill='black',anchor='mm')
    for j,method in enumerate(cfg['methods']):
        x=x0+(1+2*j)*(s+gap)
        draw.text((x+s+gap/2,48),method['label'],font=font(42),fill='black',anchor='mm')
        for k,title in enumerate(['Camera View','Novel View']):draw.text((x+k*(s+gap)+s/2,112),title,font=font(38),fill='black',anchor='mm')
    groups={}
    for i,row in enumerate(rows):
        y=top+i*(s+70)
        for j,im in enumerate(row['images']):canvas.paste(im,(x0+j*(s+gap),y))
        draw.text((x0+s/2,y+s+27),f"Frame {row['frame']:06d} · Camera {row['camera']}",font=font(28),fill='black',anchor='mm')
        groups.setdefault(row['sequence'],[]).append(i)
    for group,indices in groups.items():
        tile=Image.new('RGB',(max(700,len(group)*40),100),'white');ImageDraw.Draw(tile).text((tile.width/2,50),group,font=font(38),fill='black',anchor='mm');tile=tile.rotate(90,expand=True)
        center=top+(min(indices)+max(indices)+1)*(s+70)/2
        canvas.paste(tile,(15,int(center-tile.height/2)))
    object_label=cfg.get('object_caption', 'source colors / texture' if cfg['object_material']=='source' else 'gray')
    caption=cfg.get('caption',f'Contact-refined export (method identity pending). Left: red. Right: blue. Object: {object_label}.')
    draw.text((w/2,h-60),caption,font=font(29),fill='black',anchor='mm')
    canvas.save(out/f'{name}.png',dpi=(300,300));canvas.save(out/f'{name}.pdf',resolution=300)
    small=canvas.copy();small.thumbnail((1500,1800));small.save(out/f'{name}_review.jpg',quality=95)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--preview',action='store_true');args=ap.parse_args()
    config_path=resolve(args.config);cfg=json.loads(config_path.read_text());out=resolve(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    mano=resolve_near(cfg['mano_dir'],config_path)
    models={side:smplx.MANO(str(mano/f'MANO_{side.upper()}.pkl'),is_rhand=side=='right',use_pca=False,flat_hand_mean=True).eval() for side in ['left','right']}
    used={str(config_path)};used.update(str(mano/f'MANO_{s.upper()}.pkl') for s in models)
    rows=[];records=[]
    entries=cfg['rows'][:1] if args.preview else cfg['rows']
    for entry in entries:
        seq=entry['sequence'];frame=entry['frame'];cam=str(entry['camera']);data=resolve(cfg['data_root'])/seq
        k,e,dist=calibration(data,cam,cfg);src=data/f'images/{cam}/{frame:06d}.jpg';im=Image.open(src).convert('RGB')
        if list(im.size)!=cfg['calibration_resolution']:raise ValueError('Image/calibration resolution mismatch')
        used.update(str(p) for p in [src,data/'intri.yml',data/'extri.yml'])
        results=[];states=[]
        for method in cfg['methods']:
            failure=method.get('failures',{}).get(f'{seq}/{frame:06d}')
            if failure:
                if not failure.get('evidence'):raise ValueError('Failed requires evidence')
                meshes,state=None,{'status':'Failed',**failure}
            else:
                glb = resolve(method['object_glb'].format(sequence=seq)) if method.get('object_glb') else None
                meshes,state=load_result(resolve(method['path'].format(sequence=seq)),frame,models,used,glb)
            results.append(meshes);states.append(state)
        valid=[m.vertices for result in results if result for m,_ in result]
        if not valid:raise ValueError('No geometry to define a shared crop; configure a reference ROI')
        verts=np.concatenate(valid);vc=transform(verts,e)
        if vc[:,2].min()<=0:raise ValueError('Geometry behind input camera')
        uv=vc@k.T;uv=uv[:,:2]/uv[:,2:];lo=uv.min(0);hi=uv.max(0)
        side=int(math.ceil(max(hi-lo)*cfg['crop_margin']));xy=np.floor((lo+hi-side)/2).astype(int);box=[int(xy[0]),int(xy[1]),int(xy[0]+side),int(xy[1]+side)]
        # White padding only outside the original sensor; RGB values inside are untouched.
        rgb=Image.new('RGB',(side,side),'white');rgb.paste(im,(-box[0],-box[1]));rgb=rgb.resize((cfg['cell_size'],)*2,Image.Resampling.LANCZOS)
        s=cfg['cell_size'];aa=cfg['supersampling'];kc=k.copy();kc[0,2]-=box[0];kc[1,2]-=box[1];kc[:2]*=s*aa/side
        camera=pyrender.IntrinsicsCamera(kc[0,0],kc[1,1],kc[0,2],kc[1,2],znear=.001,zfar=100)
        pose=np.linalg.inv(e)@FLIP
        center=(verts.min(0)+verts.max(0))/2
        novel_r=pose[:3,:3]@Rotation.from_euler('yx',cfg['novel_yaw_pitch_degrees'],degrees=True).as_matrix()
        v=(verts-center)@novel_r;extent=np.max(np.ptp(v[:,:2],axis=0))*cfg['crop_margin']/2
        # Recenter in the novel image plane using the UNION of all method bounds.
        center=center+novel_r[:,:2]@((v[:,:2].min(0)+v[:,:2].max(0))/2)
        novel_pose=np.eye(4);novel_pose[:3,:3]=novel_r;novel_pose[:3,3]=center+novel_r[:,2]*max(1.,np.ptp(v[:,2])*3)
        novel_camera=pyrender.OrthographicCamera(xmag=extent,ymag=extent,znear=.001,zfar=100)
        folder=out/'cells'/seq/f'{frame:06d}_cam{cam}';folder.mkdir(parents=True,exist_ok=True);rgb.save(folder/'input.png');ims=[rgb]
        for method,result,state in zip(cfg['methods'],results,states):
            if result:
                a=render(result,camera,pose,s,cfg);b=render(result,novel_camera,novel_pose,s,cfg)
            else:a=b=label(state['status'],s)
            a.save(folder/f"{method['id']}_camera.png");b.save(folder/f"{method['id']}_novel.png");ims += [a,b]
        rec={**entry,'input':str(src),'camera_K':k.tolist(),'world_to_camera':e.tolist(),'distortion':dist.tolist(),'crop_xyxy':box,'geometry_outside_sensor':bool((lo<0).any() or (hi>np.array(im.size)).any()),'novel_camera_to_world_opengl':novel_pose.tolist(),'novel_orthographic_half_extent_m':float(extent),'methods':states}
        records.append(rec);rows.append({**entry,'images':ims});print(seq,frame,states[0]['status'],'outside_sensor',rec['geometry_outside_sensor'],flush=True)
    name='preview' if args.preview else 'available_results'
    compose(rows,cfg,out,name)
    manifest={'config':cfg,'rows':records,'sources_sha256':{p:sha(p) for p in sorted(used)},'limitations':cfg.get('limitations',['Method names and baseline results were not supplied; this is a single existing-output visualization, not a completed multi-method comparison.','Reference attachment was not available in this conversation.','Input RGB is 640x360; output enlargement adds no input detail.'])}
    (out/f'{name}_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
