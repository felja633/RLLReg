import open3d as o3d
import numpy as np
import torch
import visdom
from lib.utils import L2sqr, corrL2sqr, L2sqrTrans, per_matrix_dot
from pathlib import Path

def my_hot_colormap(scalars):
    scalars_n = scalars/scalars.max()
    scalars_q = (2*scalars_n*255).floor().astype(int)

    map = np.zeros(2*256,3)
    map[:255,0] = np.arange(0,256)
    map[256:,0] = 255
    map[256:,1] = np.arange(0,256)

    colors = map[scalars_q,:]
    return colors

def draw_point_clouds(pcds, X=None, colors=None, weights=None):

    M = pcds.shape[0]
    pcd_list = []
    for idx in range(M):
        pc = pcds[idx]
        o3dpcd = o3d.geometry.PointCloud()
        o3dpcd.points = o3d.utility.Vector3dVector(pc.T)
        if not weights is None:
            o3dpcd.colors = my_hot_colormap(weights)
        elif not colors is None:
            o3dpcd.colors = colors

        pcd_list.append(o3dpcd)

    if not X is None:
        Xs = X.squeeze()
        pcd_list.append(Xs)

    o3d.visualization.draw_geometries(pcd_list)
    input("Press Enter to continue...")

def save_point_clouds_open3d(pcds_info, info, root_path, X=None, visualize_weights=True):
    # pcds: num_pcds, coords, num pts; numpy
    names = info["names"]
    pcds_orig = pcds_info["coordinates_ds"]
    pcds=pcds_info["Vs"]
    if "att" in pcds_info.keys() and visualize_weights:
        feature_w = pcds_info["att"]
    else:
        feature_w=None

    Path(root_path).mkdir(parents=True, exist_ok=True)

    import os
    M = len(pcds)
    for idx in range(M):
        cnt=0
        sample_ind = len(list(os.walk(root_path)))
        sample_name =root_path+"/pcds"+str(sample_ind)
        Path(sample_name).mkdir(parents=True, exist_ok=True)
        for i, pc in enumerate(pcds[idx]):

            o3dpcd = o3d.geometry.PointCloud()
            o3dpcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy().T)
            fname=sample_name + "/"+names[idx][cnt].split("/")[-1] +".ply"
            o3d.io.write_point_cloud(filename=fname, pointcloud=o3dpcd)

            o3dpcd2 = o3d.geometry.PointCloud()
            o3dpcd2.points = o3d.utility.Vector3dVector(pcds_orig[idx][i].cpu().numpy().T)
            fname2 = sample_name + "/" + names[idx][cnt].split("/")[-1] + "_orig.ply"
            o3d.io.write_point_cloud(filename=fname2, pointcloud=o3dpcd2)
            if not feature_w is None and len(feature_w)>0 and len(feature_w[0])>0:
                fname_w = sample_name + "/" + names[idx][cnt].split("/")[-1]+"_weights.npz"
                np.save(fname_w, feature_w[idx][i].cpu().numpy())

            R_gt = info['R_gt'][idx][i]
            t_gt = info['t_gt'][idx][i]
            fname_r = sample_name + "/" + names[idx][cnt].split("/")[-1] + "_Rgt.npz"
            np.save(fname_r, R_gt.cpu().numpy())
            fname_t = sample_name + "/" + names[idx][cnt].split("/")[-1] + "_tgt.npz"
            np.save(fname_t, t_gt.cpu().numpy())
            cnt+=1

    if not X is None:
        Xs = X.squeeze()
        fname = root_path + "/pcd_X" + ".ply"
        o3d.io.write_point_cloud(filename=fname, pointcloud=Xs.cpu().numpy())


def draw_point_clouds_visdom(pcds, X=None, c=None):
    cfg = {"server": "localhost",
           "port": 8102}
    vis = visdom.Visdom('http://' + cfg["server"], port=cfg["port"])

    N = pcds.shape[-1]
    M = pcds.shape[0]
    pcds_all = pcds.permute(0,2,1).contiguous().view(-1,3)

    if c is None:
        c = []
        for idx in range(M):
            c.append(torch.zeros((N,), dtype=torch.int8) + idx)

        c = torch.stack(c).contiguous().view(-1)

    if not X is None:
        Xs = X.squeeze()
        pcds_all = torch.cat([pcds_all, Xs])
        c = torch.cat([c, torch.zeros((Xs.shape[0],), dtype=torch.int8) + M])

    vis.scatter(X=pcds_all, Y=c+1, opts=dict(markersize=1))

def draw_point_clouds_list_visdom(pcds, X=None, c=None):
    cfg = {"server": "localhost",
           "port": 8102}
    vis = visdom.Visdom('http://' + cfg["server"], port=cfg["port"])

    M = len(pcds)

    pcds_all = torch.cat(pcds.permute(1,0))
    if c is None:
        c = []
        for idx in range(M):
            N = pcds[idx].shape[-1]
            c.append(torch.zeros((N,), dtype=torch.int8) + idx)

        c = torch.cat(c).contiguous()

    if not X is None:
        Xs = X.squeeze()
        pcds_all = torch.cat([pcds_all, Xs])
        c = torch.cat([c, torch.zeros((Xs.shape[0],), dtype=torch.int8) + M])

    vis.scatter(X=pcds_all, Y=c+1, opts=dict(markersize=1))

class display_training_info:
    def __init__(self, port=8102, server='127.0.0.1'):
        cfg = {"server": server,
               "port": port}

        self.vis = visdom.Visdom(server=cfg["server"], port=cfg["port"])

    def __call__(self, data, info):
        Rinit = info['R_init'].cpu()
        Rgt = info['R_gt'].cpu()
        Rs = data['Rs'].cpu().detach()
        Riter = data['Riter']

        Riter = torch.stack(Riter).cpu().detach()

        R = torch.cat((Rinit.unsqueeze(dim=0), Riter, Rs.unsqueeze(dim=0)))
        err = []
        for n in range(R.shape[0]):
            err.append(L2sqr(R[n], Rgt).sqrt().mean(dim=0))

        err = torch.cat(err)
        data_x = torch.arange(err.shape[0])

        self.vis.line(err, data_x, opts={'title': 'R err vs iter'}, win='R_err')

        tinit = info['t_init'].cpu()
        tgt = info['t_gt'].cpu()
        ts = data['ts'].cpu().detach()
        titer = data['titer']

        titer = torch.stack(titer).cpu().detach()

        t = torch.cat((tinit.unsqueeze(dim=0), titer, ts.unsqueeze(dim=0)))
        err = []
        for n in range(t.shape[0]):
            err.append(L2sqrTrans(t[n], tgt, R[n], Rgt).sqrt().mean(dim=0))

        err = torch.cat(err)
        data_x = torch.arange(err.shape[0])

        self.vis.line(err, data_x, opts={'title': 't err vs iter'}, win='t_err')

        if 'correspondences' in info.keys():
            Vs_iter = data["Vs_iter"]
            Vs_iter.append(data["Vs"])
            err_corr = []
            for Vs in Vs_iter:
                err_corr.append(corrL2sqr(Vs.detach().cpu(), info, th=0.5))

            data_x = torch.arange(len(err_corr))
            self.vis.line(err_corr, data_x, opts={'title': 'corr_err vs iter'}, win='corr_err')


class display_reg_info:
    def __init__(self, vis):
        self.vis = vis

    def __call__(self, input, info, data):
        Rinit = info['R_init'].cpu()
        Rgt = info['R_gt'].cpu()
        Rs = input['Rs'].cpu().detach()
        Riter = input['Riter']

        Riter = torch.stack(Riter).cpu().detach()

        R = torch.cat((Rinit.unsqueeze(dim=0), Riter, Rs.unsqueeze(dim=0)))
        err = []
        for n in range(R.shape[0]):
            err.append(L2sqr(R[n], Rgt).sqrt().mean(dim=0))

        err = torch.cat(err)
        data_x = torch.arange(err.shape[0])
        self.vis.register(err, 'lineplot', 2, 'R_err')

        tinit = info['t_init'].cpu()
        tgt = info['t_gt'].cpu()
        ts = input['ts'].cpu().detach()
        titer = input['titer']

        titer = torch.stack(titer).cpu().detach()

        t = torch.cat((tinit.unsqueeze(dim=0), titer, ts.unsqueeze(dim=0)))
        err = []
        for n in range(t.shape[0]):
            err.append(L2sqrTrans(t[n], tgt, R[n], Rgt).sqrt().mean(dim=0))

        err = torch.cat(err)
        data_x = torch.arange(err.shape[0])

        #self.vis.line(err, data_x, opts={'title': 't err vs iter'}, win='t_err')
        self.vis.register(err, 'lineplot', 2, 't_err')

        if 'correspondences' in info.keys():
            Vs_iter = input["Vs_iter"]
            Vs_iter.append(input["Vs"].detach())
            err_corr = []
            for Vs in Vs_iter:
                err_corr.append(corrL2sqr(Vs.detach().cpu(), info, th=0.5))

            data_x = torch.arange(len(err_corr))
            self.vis.register(torch.stack(err_corr), 'lineplot', 2, 'corr_err')

        TVs = input["Vs"].detach()
        self.vis.register(dict(pcds=TVs[0].cpu(), X=None, c=None), 'point_clouds', 2, 'registration')

        TVs = input["Vs_iter"][1].detach()
        self.vis.register(dict(pcds=TVs[0].cpu(), X=None, c=None), 'point_clouds', 2, 'registration-iter1')

        coords = per_matrix_dot(Rgt, data["coordinates"]) + tgt
        self.vis.register(dict(pcds=coords[0].cpu(), X=None, c=None), 'point_clouds', 2, 'ground_truth')