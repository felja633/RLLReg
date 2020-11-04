import torch
import math
import numpy as np
from torch_geometric.data import Data, Batch
from lib.tensorlist import TensorList, TensorListList
from scipy.spatial import KDTree
from torch_geometric.nn import knn
import MinkowskiEngine as ME
from robust_loss_pytorch import lossfun

def global_to_relative(R, t, ref_index):
    R_rel = TensorListList()
    t_rel = TensorListList()
    for Ri, ti in zip(R,t):
        R_ref = Ri[ref_index]
        t_ref = ti[ref_index]
        R_reli = R_ref.permute(1,0) @ Ri
        t_reli = R_ref.permute(1,0) @ ti - R_ref.permute(1,0) @ t_ref
        R_rel.append(R_reli)
        t_rel.append(t_reli)


    return R_rel, t_rel


def L2sqr(R1,R2):
    M = R1.shape[1]
    err = []
    for ind1 in range(M - 1):
        for ind2 in range(ind1 + 1, M):
            Rdiff = per_matrix_dot(R1[:, ind1:ind1 + 1, :, :].permute(0, 1, 3, 2),
                                   R1[:, ind2:ind2 + 1, :, :]) - per_matrix_dot(
                R2[:, ind1:ind1 + 1, :, :].permute(0, 1, 3, 2), R2[:, ind2:ind2 + 1, :, :])
            err.append((Rdiff * Rdiff).sum(dim=-1).sum(dim=-1))

    return torch.cat(err, dim=-1)

def L2sqrTrans(t1,t2,R1,R2):
    M = t1.shape[1]
    err = []
    for ind1 in range(M - 1):
        for ind2 in range(ind1 + 1, M):
            # estimated relative translation from ind2 to ind1
            test = per_matrix_dot(R1[:, ind1:ind1 + 1, :, :].permute(0, 1, 3, 2),
                                   t1[:, ind2:ind2 + 1, :, :]) - per_matrix_dot(R1[:, ind1:ind1 + 1, :, :].permute(0, 1, 3, 2),
                                   t1[:, ind1:ind1 + 1, :, :])
            # ground truth translation from ind2 to ind1
            tgt = per_matrix_dot(R2[:, ind1:ind1 + 1, :, :].permute(0, 1, 3, 2),
                                   t2[:, ind2:ind2 + 1, :, :]) - per_matrix_dot(R2[:, ind1:ind1 + 1, :, :].permute(0, 1, 3, 2),
                                   t2[:, ind1:ind1 + 1, :, :])
            diff = test-tgt
            err.append((diff * diff).sum(dim=-2).squeeze(-1))

    return torch.cat(err, dim=-1)


def L2sqrList(R1_list, R2_list):
    err = TensorListList()
    for R1, R2 in zip(R1_list, R2_list):
        M = len(R1)
        err1 = TensorList()
        for ind1 in range(M - 1):
            for ind2 in range(ind1 + 1, M):
                Rdiff = R1[ind1].permute(1,0) @ R1[ind2] - R2[ind1].permute(1,0) @ R2[ind2]
                err1.append((Rdiff * Rdiff).sum(dim=-1).sum(dim=-1).unsqueeze(0))
        err.append(err1)

    return err

def L2sqrTransList(t1_list,t2_list,R1_list,R2_list):
    err = TensorListList()
    for t1, t2, R1, R2 in zip(t1_list, t2_list, R1_list, R2_list):
        M = len(t1)
        err1 = TensorList()
        for ind1 in range(M - 1):
            for ind2 in range(ind1 + 1, M):
                # estimated relative translation from ind2 to ind1
                test = R1[ind1].permute(1,0) @ t1[ind2] - R1[ind1].permute(1,0) @ t1[ind1]
                # ground truth translation from ind2 to ind1
                tgt = R2[ind1].permute(1,0) @ t2[ind2] - R2[ind1].permute(1,0) @ t2[ind1]

                diff = test-tgt
                err1.append((diff * diff).sum(dim=-2))
        err.append(err1)
    return err

# RLL Loss
def GARPointList(t1_list,t2_list,R1_list,R2_list, V, c, alpha):
    err = TensorListList()

    for t1, t2, R1, R2, Vb in zip(t1_list, t2_list, R1_list, R2_list, V):
        M = len(t1)
        err1 = TensorList()
        for ind1 in range(M - 1):
            for ind2 in range(ind1 + 1, M):
                Rtest = R1[ind1].permute(1,0) @ R1[ind2]
                # estimated relative translation from ind2 to ind1
                test = R1[ind1].permute(1,0) @ t1[ind2] - R1[ind1].permute(1,0) @ t1[ind1]

                Rgt = R2[ind1].permute(1, 0) @ R2[ind2]
                # ground truth translation from ind2 to ind1
                tgt = R2[ind1].permute(1,0) @ t2[ind2] - R2[ind1].permute(1,0) @ t2[ind1]

                diff = Rtest@Vb[ind2]+test - (Rgt@Vb[ind2]+tgt)

                err_i = (diff * diff).sum(dim=-2).sqrt()
                err1.append(lossfun(err_i, alpha=alpha, scale=c).mean())

        err.append(err1)

    return err

def compute_rotation_errors(Rs, R_gt):
    if isinstance(Rs, TensorListList):
        err = L2sqrList(Rs, R_gt)
    else:
        err = L2sqr(Rs, R_gt)
    return err.sqrt()

def compute_translation_errors(ts, tgt, Rs, R_gt):
    # Rs: batch, point cloud, 3,3 takes points from point set to world
    if isinstance(ts, TensorListList):
        err = L2sqrTransList(ts, tgt, Rs, R_gt)
    else:
        err = L2sqrTrans(ts, tgt, Rs, R_gt)
    return err.sqrt()


def corrL2sqr(point_clouds_est, info, th=0.5):
    M = point_clouds_est.shape[1]
    inds_nn = info['correspondences']["indices"]
    d = info['correspondences']["distances"]
    loss = 0
    batch_size = point_clouds_est.shape[0]
    for ind1 in range(M - 1):
        for ind2 in range(ind1 + 1, M):
            for pb, indsb, db in zip(point_clouds_est, inds_nn, d):
                mask = db < th
                diff = pb[ind1, :, indsb[0, mask]] - pb[ind2, :, indsb[1, mask]]
                loss = loss + (diff * diff).sum(dim=0).mean()

    return loss/batch_size

def compute_angular_error(R_err):
    return 360.0/2.0/math.pi*2.0*torch.asin(R_err/math.sqrt(8.0))

def get_random_rotation_matrix(ang_range):
    ang = ang_range*torch.rand(1)
    axis = torch.rand(3)-0.5
    axis/=axis.norm()
    R=torch.zeros([3,3])
    R[0,1], R[0,2], R[1,2] = -axis[2], axis[1], -axis[0]
    R[1,0], R[2,0], R[2,1] = axis[2], -axis[1], axis[0]
    R = torch.eye(3,3) + math.sin(ang) * R + (1. - math.cos(ang)) * R @ R
    return R

def get_random_translation_vector(t_range):
    t = torch.rand(3)-0.5
    t = t / t.norm()
    return t_range*torch.rand(1)*t.unsqueeze(dim=1)

def get_translation_vector(t_range, direction):
    t = direction-0.5
    t = t / t.norm()
    return t_range * t.unsqueeze(dim=1)

def get_rotation_matrix(ang_range, r, ax):
    ang = ang_range*r
    axis = ax-0.5
    axis/=axis.norm()
    R=torch.zeros([3,3])
    R[0,1], R[0,2], R[1,2] = -axis[2], axis[1], -axis[0]
    R[1,0], R[2,0], R[2,1] = axis[2], -axis[1], axis[0]
    R = torch.eye(3,3) + math.sin(ang) * R + (1. - math.cos(ang)) * R @ R
    return R

def per_matrix_dot(a, b):
    # a dims: batch, point set, d1, d2
    # b dims: batch, point set, d1, d2
    c = a.view(a.shape[0]*a.shape[1], *a.shape[2:]).bmm(b.view(b.shape[0]*b.shape[1], *b.shape[2:]))
    return c.view(a.shape[0],a.shape[1], *c.shape[-2:])


def convert_data_to_batch(x):
    data_list = []
    for xx in x:
        data_list.append(Data(pos=xx))

    batch = Batch()
    return batch.from_data_list(data_list)

## resampling ##
def random_sampling(pts, num_points_left):
    def rnd_samp(nump, s, device):
        if nump < s:
            return torch.arange(0, nump).to(device)

        arr = torch.randperm(nump)
        return arr[0:s]

    if isinstance(pts, (list,)):
        sh = pts[0].shape
        arr = rnd_samp(sh[-1], num_points_left, pts[0].device)
        return [p[arr,:] for p in pts]

    else:
        sh = pts.shape
        arr = rnd_samp(sh[-1], num_points_left, pts.device)
        return pts[:, arr], arr


def farthest_point_sampling(pts, num_points_left):
    """
    Select a subset of K points from pts using farthest point sampling
    from: https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    """

    def calc_distances(p0, points):
        return (torch.sum((p0.unsqueeze_(1) - points) ** 2, dim=0))

    farthest_pts = torch.zeros((pts.size()[0], num_points_left))
    f_ind = np.random.RandomState().randint(pts.shape[1])
    #f_ind = torch.from_numpy(f_ind)
    farthest_pts[:, 0] = pts[:, f_ind]
    distances = calc_distances(farthest_pts[:, 0], pts)
    inds = []
    inds.append(f_ind)
    for i in range(1, num_points_left):
        amax_ind = np.argmax(distances)
        farthest_pts[:, i] = pts[:, amax_ind]
        inds.append(amax_ind)
        distances = np.minimum(distances, calc_distances(farthest_pts[:, i], pts))

    return farthest_pts, inds


def extract_correspondences(coords, Rs, ts):
    coords = Rs @ coords + ts

    M = len(coords)
    ind_nns = []
    dists = []
    for ind1 in range(M - 1):
        for ind2 in range(ind1 + 1, M):
            point1 = coords[ind1].cpu().numpy().T
            point2 = coords[ind2].cpu().numpy().T
            tree = KDTree(point1)
            d, inds_nn1 = tree.query(point2, k=1)
            inds_nn2 = torch.tensor([n for n in range(0, point2.shape[0])])
            inds_nn1 = torch.from_numpy(inds_nn1)

            ind_nns.append(torch.stack([inds_nn1, inds_nn2], dim=0))
            dists.append(torch.from_numpy(d))

    return dict(indices=TensorList(ind_nns).to(coords[0][0].device), distances=TensorList(dists).to(coords[0][0].device))

def extract_correspondences_gpu(coords, Rs, ts, device=None):
    coords = Rs @ coords + ts

    M = len(coords)
    ind_nns = []
    dists = []
    if not device is None:
        coords = coords.to(device)
    for ind1 in range(M - 1):
        for ind2 in range(ind1 + 1, M):
            point1 = coords[ind1].permute(1,0)
            point2 = coords[ind2].permute(1,0)
            inds_nn = knn(point2, point1, 1)
            d = point1[inds_nn[0,:]] - point2[inds_nn[1,:]]
            d = (d * d).sum(dim=1).sqrt()
            dists.append(d)
            ind_nns.append(inds_nn)

    return dict(indices=TensorList(ind_nns), distances=TensorList(dists))

def estimate_overlap(ps1, ps2, R1, R2, t1, t2, voxel_size, device):
    # voxel grid downsampling
    quantized_coords1 = torch.floor(ps1.permute(1,0) / voxel_size)
    inds1 = ME.utils.sparse_quantize(quantized_coords1, return_index=True)
    quantized_coords2 = torch.floor(ps2.permute(1,0) / voxel_size)
    inds2 = ME.utils.sparse_quantize(quantized_coords2, return_index=True)

    ps1_v = ps1[:, inds1]
    ps2_v = ps2[:, inds2]

    corrs = extract_correspondences_gpu(TensorList([ps1_v, ps2_v]), TensorList([R1, R2]),
                            TensorList([t1, t2]), device)

    d = corrs["distances"]
    rates = []
    thresh = voxel_size
    for di in d:
        rate = float((di < thresh).sum())/di.shape[0]
        rates.append(rate)

    return rates
