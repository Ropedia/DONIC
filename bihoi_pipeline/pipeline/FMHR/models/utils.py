import os
from os.path import join
import numpy as np
import cv2
import skimage
from skimage import measure
import plyfile
import trimesh
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F

def convert_sdf_to_mesh(sdf_values, voxel_origin, voxel_size, file_name, level=0.):
    if isinstance(sdf_values, torch.Tensor):
        sdf_values = sdf_values.detach().cpu().numpy()
    verts, faces, normals, values = measure.marching_cubes_lewiner(sdf_values,
                                level=level, spacing=voxel_size)

    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]

    mesh = trimesh.Trimesh(mesh_points, faces)
    mesh.export(file_name)


def load_K_Rt_from_P(P):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    # c2w
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    # convert to w2c
    pose = np.linalg.inv(pose)

    return intrinsics, pose


def meshcleaning(file_name):

    mesh = trimesh.load(file_name, process=False, maintain_order=True)
    cc = mesh.split(only_watertight=False)    

    out_mesh = cc[0]
    bbox = out_mesh.bounds
    area = (bbox[1,0] - bbox[0,0]) * (bbox[1,1] - bbox[0,1])
    for c in cc:
        bbox = c.bounds
        if area < (bbox[1,0] - bbox[0,0]) * (bbox[1,1] - bbox[0,1]):
            area = (bbox[1,0] - bbox[0,0]) * (bbox[1,1] - bbox[0,1])
            out_mesh = c
    
    out_mesh.export(file_name)

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat2mat(quat)


def batch_global_rigid_transformation(Rs, Js, parent, rotate_base=False):
    '''
    Rs: B 24 3 3
    Js: B 24 3
    '''
    N = Rs.shape[0]
    num_joints = Rs.shape[1]
    device = Rs.device
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x = torch.from_numpy(np_rot_x).float().to(device)
        root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        t_homo = torch.cat([t, torch.ones(N, 1, 1).to(device)], dim=1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    results = torch.stack(results, dim=1)

    new_J = results[:, :, :3, 3]
    Js_w0 = torch.cat([Js, torch.zeros(N, num_joints, 1, 1).to(device)], dim=2)
    init_bone = torch.matmul(results, Js_w0)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    A = results - init_bone

    return new_J, A


# TODO need check the dimension
def hierarchical_softmax(x):
    n_batch, n_point, n_dim = x.shape
    x = x.flatten(0,1)

    prob_all = torch.ones(n_batch * n_point, 24, device=x.device)

    prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * torch.sigmoid(x[:, [0]]) * F.softmax(x[:, [1, 2, 3]])
    prob_all[:, [0]] = prob_all[:, [0]] * (1 - torch.sigmoid(x[:, [0]]))

    prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (torch.sigmoid(x[:, [4, 5, 6]]))
    prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - torch.sigmoid(x[:, [4, 5, 6]]))

    prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (torch.sigmoid(x[:, [7, 8, 9]]))
    prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - torch.sigmoid(x[:, [7, 8, 9]]))

    prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (torch.sigmoid(x[:, [10, 11]]))
    prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - torch.sigmoid(x[:, [10, 11]]))

    prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * torch.sigmoid(x[:, [24]]) * F.softmax(x[:, [12, 13, 14]])
    prob_all[:, [9]] = prob_all[:, [9]] * (1 - torch.sigmoid(x[:, [24]]))

    prob_all[:, [15]] = prob_all[:, [12]] * (torch.sigmoid(x[:, [15]]))
    prob_all[:, [12]] = prob_all[:, [12]] * (1 - torch.sigmoid(x[:, [15]]))

    prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (torch.sigmoid(x[:, [16, 17]]))
    prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - torch.sigmoid(x[:, [16, 17]]))

    prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (torch.sigmoid(x[:, [18, 19]]))
    prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - torch.sigmoid(x[:, [18, 19]]))

    prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (torch.sigmoid(x[:, [20, 21]]))
    prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - torch.sigmoid(x[:, [20, 21]]))

    prob_all[:, [22, 23]] = prob_all[:, [20, 21]] * (torch.sigmoid(x[:, [22, 23]]))
    prob_all[:, [20, 21]] = prob_all[:, [20, 21]] * (1 - torch.sigmoid(x[:, [22, 23]]))

    prob_all = prob_all.reshape(n_batch, n_point, prob_all.shape[-1])
    return prob_all


def get_matrix(normal, degree=3):
    if isinstance(normal, np.ndarray):
        matrix = np.zeros((normal.shape[0], degree**2))
    elif isinstance(normal, torch.Tensor):
        matrix = torch.zeros(normal.shape[0], degree**2, device=normal.device)

    matrix[:,0] = 1
    if degree > 1:
        matrix[:,1] = normal[:,1]
        matrix[:,2] = normal[:,2]
        matrix[:,3] = normal[:,0]
    if degree > 2:
        matrix[:,4] = normal[:,0] * normal[:,1]
        matrix[:,5] = normal[:,1] * normal[:,2]
        matrix[:,6] = (2*normal[:,2]*normal[:,2]-normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])
        matrix[:,7] = normal[:,2] * normal[:,0]
        matrix[:,8] = (normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])

    return matrix

def get_radiance(coeff, normal, degree=3):
    '''
    coeff 9 or n 9
    normal n 3
    '''

    radiance = coeff[...,0]
    if degree > 1:
        radiance = radiance + coeff[...,1] * normal[:,1]
        radiance = radiance + coeff[...,2] * normal[:,2]
        radiance = radiance + coeff[...,3] * normal[:,0]
    if degree > 2:
        radiance = radiance + coeff[...,4] * normal[:,0] * normal[:,1]
        radiance = radiance + coeff[...,5] * normal[:,1] * normal[:,2]
        radiance = radiance + coeff[...,6] * (2*normal[:,2]*normal[:,2]-normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])
        radiance = radiance + coeff[...,7] * normal[:,2] * normal[:,0]
        radiance = radiance + coeff[...,8] * (normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])

    return radiance


def eval_sh(deg, sh, dirs):
    '''
    deg int
    sh [b C (deg+1)**2]
    dirs B 3
    '''

    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]
    C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]
    
    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3] # b 1
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])
            if deg > 2:
                result = (result +
                        C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                        C3[1] * xy * z * sh[..., 10] +
                        C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                        C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                        C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                        C3[5] * z * (xx - yy) * sh[..., 14] +
                        C3[6] * x * (xx - 3 * yy) * sh[..., 15])
                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])

    return result


def depth2normal(depth):
    zy, zx = np.gradient(depth)  
    ones = np.ones_like(zx)

    normal = np.stack([-zx, -zy, ones], axis=2)
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    norm[norm==0] = 1

    normal = normal / norm

    return normal


def normal2depth(dx,dy, init_depth=0):
    h,w = dx.shape
    depth = np.zeros([h,w])
    depth[:,0] = init_depth + 2 * np.c_[np.r_[0,dy[1:-1:2,0].cumsum()], dy[::2,0].cumsum() - dy[0,0] / 2].ravel()[:h]

    for i in range(h):
        depth[i] = depth[i,0] + 2 * np.c_[np.r_[0,dx[i,1:-1:2].cumsum()], dx[i,::2].cumsum() - dx[i,0] / 2].ravel()[:w]

    return depth


def grid_smooth_loss(grid):
    '''
    grid
    '''
    h,w,d,_ = grid.shape
    hh = []
    ww = []
    dd = []
    for i in range(3):
        hh.append(range(i,h-2+i))
        ww.append(range(i,w-2+i))
        dd.append(range(i,d-2+i))

    loss = (grid[hh[1],ww[1],dd[1]]-grid[hh[0],ww[1],dd[1]]).abs() + (grid[hh[1],ww[1],dd[1]]-grid[hh[2],ww[1],dd[1]]).abs() + (grid[hh[1],ww[1],dd[1]]-grid[hh[1],ww[0],dd[1]]).abs() +\
            (grid[hh[1],ww[1],dd[1]]-grid[hh[1],ww[2],dd[1]]).abs() + (grid[hh[1],ww[1],dd[1]]-grid[hh[1],ww[1],dd[0]]).abs() + (grid[hh[1],ww[1],dd[1]]-grid[hh[1],ww[1],dd[2]]).abs()

    return loss.mean()



def create_meshgrid(height, width, normalized_coordinates=False):
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width, dtype=torch.float)
        ys = torch.linspace(-1, 1, height, dtype=torch.float)
    else:
        xs = torch.linspace(0, width - 1, width, dtype=torch.float)
        ys = torch.linspace(0, height - 1, height, dtype=torch.float)

    # generate grid by stacking coordinates
    base_grid = torch.stack(torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW

    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2

def get_ray_directions(H, W, focal, c=None, minus_z=True, minus_y=True):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5
    i, j = grid.unbind(-1)

    if isinstance(focal, float):
        fx = focal
        fy = focal
    elif isinstance(focal, list):
        fx, fy = focal

    if minus_z:
        z = -torch.ones_like(i)
    else:
        z = torch.ones_like(i)

    if c is None:
        c = [W/2, H/2]

    x = (i-c[0])/fx
    if minus_y:
        y = -(j-c[1])/fy
    else:
        y = (j-c[1])/fy

    directions = torch.stack([x,y,z], -1)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    # rays_d = rays_d.view(-1, 3)
    # rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


class NormalConsistency(nn.Module):
    def __init__(self, faces, average=False):
        super(NormalConsistency, self).__init__()

        self.nf = faces.shape[0]
        self.average = average
        if torch.is_tensor(faces):
            faces = faces.detach().cpu().numpy()
        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))

        tmp = dict()
        for face in faces:
            f1 = np.sort(face[:2])
            f2 = np.sort(face[1:])
            f3 = np.sort(face[::2])
            k1 = int(f1[0] * self.nf + f1[1])
            k2 = int(f2[0] * self.nf + f2[1])
            k3 = int(f3[0] * self.nf + f3[1])

            if k1 not in tmp.keys():
                tmp[k1] = []
            tmp[k1].append(face[2])

            if k2 not in tmp.keys():
                tmp[k2] = []
            tmp[k2].append(face[0])

            if k3 not in tmp.keys():
                tmp[k3] = []
            tmp[k3].append(face[1])

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []

        for v0, v1 in zip(v0s, v1s):
            k = int(v0 * self.nf + v1)
            v2s.append(tmp[k][0])
            if len(tmp[k]) < 2:
                v3s.append(tmp[k][0])
            else:
                v3s.append(tmp[k][1])

        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')

        self.register_buffer('v0s', torch.from_numpy(v0s).long())
        self.register_buffer('v1s', torch.from_numpy(v1s).long())
        self.register_buffer('v2s', torch.from_numpy(v2s).long())
        self.register_buffer('v3s', torch.from_numpy(v3s).long())

    def forward(self, vertices):
        batch_size = vertices.size(0)

        v0s = vertices[:, self.v0s, :]
        v1s = vertices[:, self.v1s, :]
        v2s = vertices[:, self.v2s, :]
        v3s = vertices[:, self.v3s, :]

        a1 = v1s - v0s
        b1 = v2s - v0s
        b2 = v3s - v0s

        n1 = torch.cross(b1, a1)
        n2 = torch.cross(a1, b2)

        loss = 1. - F.cosine_similarity(n1, n2)

        return loss.sum()


def mynormalize(tensor, p=2, dim=1, eps=1e-12):
    denom = tensor.norm(p, dim, keepdim=True).clamp_min(eps).expand_as(tensor).detach()
    return tensor / denom

'''
code adapted from pytorch3d https://github.com/facebookresearch/pytorch3d
'''

def get_normals(vertices, faces):
    '''
    vertices b n 3
    faces f 3
    '''
    verts_normals = torch.zeros_like(vertices)

    vertices_faces = vertices[:, faces] # b f 3 3

    verts_normals.index_add_(
        1,
        faces[:, 1],
        torch.cross(
            vertices_faces[:, :, 2] - vertices_faces[:, :, 1],
            vertices_faces[:, :, 0] - vertices_faces[:, :, 1],
            dim=2,
        ),
    )
    verts_normals.index_add_(
        1,
        faces[:, 2],
        torch.cross(
            vertices_faces[:, :, 0] - vertices_faces[:, :, 2],
            vertices_faces[:, :, 1] - vertices_faces[:, :, 2],
            dim=2,
        ),
    )
    verts_normals.index_add_(
        1,
        faces[:, 0],
        torch.cross(
            vertices_faces[:, :, 1] - vertices_faces[:, :, 0],
            vertices_faces[:, :, 2] - vertices_faces[:, :, 0],
            dim=2,
        ),
    )

    verts_normals = F.normalize(verts_normals, p=2, dim=2, eps=1e-6)
    # verts_normals = mynormalize(verts_normals, p=2, dim=2, eps=1e-6)

    return verts_normals


def get_edges(verts, faces):
    V = verts.shape[0]
    f = faces.shape[0]
    device = verts.device
    v0, v1, v2 = faces.chunk(3, dim=1)
    e01 = torch.cat([v0, v1], dim=1)  # (sum(F_n), 2)
    e12 = torch.cat([v1, v2], dim=1)  # (sum(F_n), 2)
    e20 = torch.cat([v2, v0], dim=1)  # (sum(F_n), 2)

    edges = torch.cat([e12, e20, e01], dim=0)  # (sum(F_n)*3, 2)
    edges, _ = edges.sort(dim=1)
    edges_hash = V * edges[:, 0] + edges[:, 1]
    u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)
    sorted_hash, sort_idx = torch.sort(edges_hash, dim=0)
    unique_mask = torch.ones(edges_hash.shape[0], dtype=torch.bool, device=device)
    unique_mask[1:] = sorted_hash[1:] != sorted_hash[:-1]
    edges = torch.stack([u // V, u % V], dim=1)

    faces_to_edges = inverse_idxs.reshape(3, f).t()

    return edges, faces_to_edges

# from torch.utils.cpp_extension import load

# module_path = os.path.dirname(__file__)
# find_verts = load(
#     'find_verts',
#     sources=[join(module_path, 'csrc/normal_consistency.cpp')],
# )

# def normal_consistency(verts, faces):

#     edges, face_to_edge = get_edges(verts, faces)

#     E = edges.shape[0]
#     f = faces.shape[0]

#     with torch.no_grad():
#         edge_idx = face_to_edge.reshape(f * 3)
#         vert_idx = (faces.view(1, f, 3).expand(3, f, 3).transpose(0, 1).reshape(3 * f, 3))
#         edge_idx, edge_sort_idx = edge_idx.sort()
#         vert_idx = vert_idx[edge_sort_idx]

#     edge_num = edge_idx.bincount(minlength=E)

#     vert_edge_pair_idx = find_verts.find_verts(edge_num.cpu()).to(edge_num.device)

#     if vert_edge_pair_idx.shape[0] == 0:
#         return torch.tensor([0.0], dtype=torch.float32, device=verts.device)

#     v0_idx = edges[edge_idx, 0]
#     v0 = verts[v0_idx]
#     v1_idx = edges[edge_idx, 1]
#     v1 = verts[v1_idx]

#     # two of the following cross products are zeros as they are cross product
#     # with either (v1-v0)x(v1-v0) or (v1-v0)x(v0-v0)
#     n_temp0 = (v1 - v0).cross(verts[vert_idx[:, 0]] - v0, dim=1)
#     n_temp1 = (v1 - v0).cross(verts[vert_idx[:, 1]] - v0, dim=1)
#     n_temp2 = (v1 - v0).cross(verts[vert_idx[:, 2]] - v0, dim=1)
#     n = n_temp0 + n_temp1 + n_temp2
#     n0 = n[vert_edge_pair_idx[:, 0]]
#     n1 = -n[vert_edge_pair_idx[:, 1]]
#     loss = 1 - F.cosine_similarity(n0, n1, dim=1)

#     return loss.mean()

def laplacian_cot(verts, faces):
    '''
    verts n 3
    faces f 3
    '''
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces] # f 3 3
    v0, v1, v2 = face_verts[:,0], face_verts[:,1], face_verts[:,2]

    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    s = 0.5 * (A + B + C)
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot = cot / 4.0

    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]

    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    L = L + L.t()

    idx = faces.view(-1)
    inv_areas = torch.zeros(V, dtype=torch.float32, device=verts.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    inv_areas.scatter_add_(0, idx, val)
    idx = inv_areas > 0
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.view(-1, 1)

    return L, inv_areas


def compute_laplacian(verts, faces):
    # first compute edges

    V = verts.shape[0]
    device = verts.device

    edges, faces_to_edges = get_edges(verts, faces)

    e0, e1 = edges.unbind(1)
    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L


def laplacian_smoothing(verts, faces, method="uniform"):
    weights = 1.0 / verts.shape[0]

    with torch.no_grad():
        if method == "uniform":
            L = compute_laplacian(verts, faces)
        else:
            L, inv_areas = laplacian_cot(verts, faces)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas

    if method == "uniform":
        loss = L.mm(verts)
    elif method == "cot":
        loss = L.mm(verts) * norm_w - verts
    else:
        loss = (L.mm(verts) - L_sum * verts) * norm_w
    
    loss = loss.norm(dim=1)

    loss = loss * weights
    return loss.sum()