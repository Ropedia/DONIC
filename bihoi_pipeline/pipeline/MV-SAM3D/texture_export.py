"""Export a genuine UV/texture asset alongside the tracking PLY, in its metric frame."""
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh


def export_textured_metric(mesh, scale, tracking_ply):
    tracking_ply = Path(tracking_ply)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Texture scale must be finite and positive')
    if not isinstance(mesh, trimesh.Trimesh) or mesh.visual.kind != 'texture':
        raise ValueError('Texture baking did not produce a UV-textured mesh')
    uv = mesh.visual.uv
    material = mesh.visual.material
    image = getattr(material, 'baseColorTexture', None)
    if image is None:
        image = getattr(material, 'image', None)
    if uv is None or len(uv) != len(mesh.vertices) or not np.isfinite(uv).all() or image is None:
        raise ValueError('Texture baking output lacks valid UV coordinates or texture image')
    metric = mesh.copy()
    metric.vertices = np.asarray(metric.vertices) * float(scale)
    output = tracking_ply.with_name('textured.glb')
    temporary = output.with_name('textured.pending.glb')
    try:
        metric.export(str(temporary))
        check = trimesh.load(str(temporary), force='scene', process=False)
        if not check.geometry or any(g.visual.kind != 'texture' for g in check.geometry.values()):
            raise ValueError('GLB export lost texture attributes')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    info = {
        'status': 'complete', 'path': str(output),
        'appearance_source': 'MV-SAM3D reconstructed Gaussian appearance baked into UV texture; not direct RGB reprojection',
        'texture_resolution': list(image.size), 'scale_to_meter': float(scale),
        'canonical_to_tracking_local': np.eye(4).tolist(),
        'coordinate_policy': 'Same SAM3D GLB axes and origin; same isotropic origin scaling as compressed.ply. UV seam splitting allowed; tracking PLY may be simplified.',
        'tracking_ply_sha256': hashlib.sha256(tracking_ply.read_bytes()).hexdigest(),
        'glb_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    sidecar = output.with_suffix('.json')
    tmp = sidecar.with_suffix('.pending.json')
    tmp.write_text(json.dumps(info, indent=2), encoding='utf-8')
    tmp.replace(sidecar)
    return info
