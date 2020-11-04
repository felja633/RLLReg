import numpy as np
import torch
from lib.tensorlist import TensorList, TensorListList
from torch_geometric.nn import nearest


def get_default_cluster_priors(num_clusters, gamma):
    pk = 1 / (num_clusters + gamma) * torch.ones((num_clusters, 1), dtype=torch.float32)
    return pk

def get_randn_sphere_cluster_means_list(data, num_clusters, scale=1.0):
    """ Create random cluster means, distributed on a sphere.
        The standard deviation of all point-cloud points is the sphere radius.
    :param data: [ X1, X2, ... ]. Xi = D x Ni points [np.array].
    :param num_clusters: Number of clusters to generate
    :return: cluster means, (3, num_clusters) [np.array]
    """
    N = len(data)
    # Sample the the unit sphere and scale with data standard deviation
    X = np.random.randn(data[0].shape[0], num_clusters).astype(np.float32)
    X = torch.from_numpy(X).to(data[0].device)
    X = X / torch.norm(X, dim=-2,keepdim=True)
    v = 0
    m = 0
    for d in data:
        v = v + torch.var(d, dim=-1, keepdim=True)
        m = m + d.mean(dim=1, keepdim=True)

    #M = data.shape[1]
    means = X * scale * torch.sqrt(v.mean(dim=-2, keepdim=True))/N + m / N

    return means

def get_randn_box_cluster_means_list(data, num_clusters):

    X = np.random.random_sample((data[0].shape[0], num_clusters)).astype(np.float32) - 0.5

    min_xyz = [d.min(dim=-1)[0] for d in data]
    max_xyz = [d.max(dim=-1)[0] for d in data]
    min_xyz = torch.stack(min_xyz).min(dim=0)[0]
    max_xyz = torch.stack(max_xyz).max(dim=0)[0]

    diff = (max_xyz-min_xyz).mean()

    X = torch.from_numpy(X).to(data[0].device)
    means = X * diff
    t, b = data.permute(1,0).cat_tensors()
    means = means + t.t().mean(dim=-1, keepdim=True)

    return means

def get_scaled_cluster_precisions_list(data, cluster_means, scale):
    # Minimum coordinates in point clouds and clusters
    #torch.min(point_clouds, dim=)
    N = len(data)
    min_xyz_X = None
    min_xyz_X = cluster_means.min(dim=-1, keepdim=True)[0]
    max_xyz_X = cluster_means.max(dim=-1, keepdim=True)[0]
    for d in data:
        min_xyz_i = d.min(dim=-1, keepdim=True)[0]
        min_xyz_X = torch.min(torch.cat([min_xyz_X, min_xyz_i], dim=1), dim=1, keepdim=True)[0]

        max_xyz_i = d.max(dim=-1, keepdim=True)[0]
        max_xyz_X = torch.min(torch.cat([max_xyz_X, max_xyz_i], dim=1), dim=1, keepdim=True)[0]

    d = min_xyz_X - max_xyz_X
    s = torch.sum(d * d)
    q = scale / s

    Q = q * torch.ones(1,cluster_means.shape[-1]).to(data[0].device)
    return Q.float()

def get_default_start_poses(point_clouds, cluster_means):
    """ Create default start poses
    :param cluster_means:
    :param point_clouds:
    :return:
    """
    I = torch.eye(3, dtype=torch.float32)  # Identity rotation
    mu = torch.mean(cluster_means, 0)  # Mean of cluster means
    poses = [(I, mu - torch.mean(pcl, 0)) for pcl in point_clouds]
    return poses


def get_default_beta(cluster_precisions, gamma):
    h = 2 / cluster_precisions.mean(dim=-1, keepdim=True)
    beta = gamma / (h * (gamma + 1))
    return beta

def get_init_transformation(pcds):
    m_pcds = pcds.mean(dim=3, keepdim=True)
    m_target = m_pcds[:,0,:,:].unsqueeze(dim=1)
    init_t = (m_target - m_pcds)
    init_R = torch.eye(3, 3).to(pcds.device)
    init_R = init_R.unsqueeze(dim=0).unsqueeze(dim=0).repeat(pcds.shape[0], pcds.shape[1], 1, 1)
    return init_R, init_t

def get_init_transformation_list(pcds, mean_init=True):
    if mean_init:
        m_pcds = pcds.mean(dim=-1, keepdim=True)
        m_target = TensorList([m[0] for m in m_pcds])
        init_t = -(m_pcds - m_target)
    else:
        tt = []
        for i in range(len(pcds)):
            tt.append(TensorList([torch.zeros(3, 1).to(pcds[0][0].device) for i in range(len(pcds[i]))]))
        init_t = TensorListList(tt)

    rr = []
    for i in range(len(pcds)):
        rr.append(TensorList([torch.eye(3, 3).to(pcds[0][0].device) for i in range(len(pcds[i]))]))

    init_R = TensorListList(rr)
    return init_R, init_t

def get_init_transformation_list_dgr(pcds, dgr_init_model):
    if isinstance(pcds, TensorListList):
        Rs = TensorListList()
        ts = TensorListList()
    else:
        # B, P, N, M = point_clouds.shape
        # assert P == 2
        Rs = []
        ts = []

    target_R = torch.eye(3, 3).to(pcds[0][0].device)
    target_t = torch.zeros(3, 1).to(pcds[0][0].device)
    for pc in pcds:
        assert len(pc) == 2
        R,t=dgr_init_model.register_point_sets(pc[0].permute(1,0), pc[1].permute(1,0))

        if isinstance(pcds, TensorListList):
            Rs.append(TensorList([R, target_R]))
            ts.append(TensorList([t.unsqueeze(dim=1), target_t]))
        else:
            Rs.append(torch.stack([R, target_R]))
            ts.append(torch.stack([t.unsqueeze(dim=1), target_t]))

    return Rs, ts

