import torch
import torch.nn as nn
from registration.fppsr import PSREG_features_list
from lib.tensorlist import TensorListList, TensorList
from parameter_settings.registration.dare_settings import get_default_feature_dare_parameters
from easydict import EasyDict as edict
from lib.resampling import random_indices
import numpy as np
import time
from sklearn.cluster import KMeans
import open3d as o3d

def load_weights(name="FPPSRModelFPFH", device = "cuda:0", num_iters=100, K=200, feature_model="vonmises",
                 voxel_size=0.4, use_attention=True, epsilon=float(1e-5),
                 use_dare_weights=True, ds_rate=1.0, mean_init=True, debug=False,
                 cluster_init="default", cluster_precision_scale=1.0, max_num_points=14000,
                 cluster_mean_scale=1.0, num_feature_clusters=10, downsample_online=True):

    params = get_default_feature_dare_parameters()
    params.name = name
    params.device = device
    params["feature_distr_parameters"] = edict(model=feature_model,
                                               num_feature_clusters=num_feature_clusters)
    params.layer = "final"
    params.num_iters = num_iters
    params.backprop_iter = range(-100, -2)
    params.K = K
    params.cluster_init = cluster_init
    params.cluster_precision_init = 'scaled'
    params.cluster_precision_scale = cluster_precision_scale
    params.fix_cluster_pos_iter = 2
    params.use_dare_weighting = use_dare_weights
    params.debug = debug
    params.epsilon = epsilon
    params.gamma = float(0.005)
    params.voxel_size = voxel_size
    params.use_attention = use_attention
    params.downsample_online = downsample_online
    params.downsample_rate = ds_rate
    params.ds_method = "random"
    params.max_num_points = max_num_points
    params.mean_init = mean_init
    params.cluster_mean_scale = cluster_mean_scale

    fmodel = feature_reg_model(params)

    return fmodel

class feature_reg_model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.registration = PSREG_features_list(params).to(params.device)

    def train(self, mode=True):
        return

    def eval(self):
        self.train(mode=False)

    def cluster_features(self, features, num_clusters):
        feature_labels_LL = TensorListList()
        for f in features:
            feature_labels_L = TensorList()
            fcat = torch.cat([fi for fi in f])
            fcat = fcat.to("cpu").numpy()
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(fcat)
            labels = torch.from_numpy(kmeans.labels_)
            onehot = torch.nn.functional.one_hot(labels.long(), num_clusters).to(self.params.device)
            cnt = 0
            for fi in f:
                feature_labels_L.append(onehot[cnt:cnt+fi.shape[0]])
                cnt+=fi.shape[0]

            feature_labels_LL.append(feature_labels_L.permute(1,0).float())

        return feature_labels_LL

    def preprocess_point_cloud(self, coords, voxel_size):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords.cpu().numpy().T)

        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

        radius_normal = voxel_size * 2

        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5

        pcd_fpfh = o3d.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        pc = np.asarray(pcd_down.points)
        p2_torch = torch.from_numpy(pc).permute(1, 0).float()
        f = pcd_fpfh.data
        f_torch = torch.from_numpy(f).permute(1, 0).float()
        return p2_torch, f_torch

    def forward(self, x):
        coords = x['coordinates'].clone()
        t_tot = time.time()

        features_LL = TensorListList()
        att_LL = TensorListList()
        coords_ds_LL = TensorListList()
        inds_LL = []
        ind_cnt = 0
        for coords_b in coords:
            features_L = TensorList()
            att_L = TensorList()
            coords_ds_L = TensorList()
            inds_L = []
            for coords_s in coords_b:
                pcd_down, f = self.preprocess_point_cloud(coords_s, self.params.voxel_size)

                features_L.append(f)
                coords_ds_L.append(pcd_down)

                ind_cnt = ind_cnt + 1

            features_LL.append(features_L)
            att_LL.append(att_L)
            coords_ds_LL.append(coords_ds_L)
            inds_LL.append(inds_L)

        x = dict()
        x['features'] = self.cluster_features(features_LL, self.params.feature_distr_parameters.num_feature_clusters)
        x['att'] = att_LL
        x['coordinates_ds'] = coords_ds_LL.to(self.params.device)
        x['indices'] = inds_LL

        out = self.registration(x)
        tot_time = time.time() - t_tot
        print("tot time: %.1f ms" % (tot_time * 1000))

        out["time"] = tot_time
        out["features"] = features_LL
        out["indices"] = inds_LL
        out["coordinates_ds"] = coords_ds_LL

        return out


    def downsample(self, f, rate, num_points, att, max_num_points):
        if rate is None or rate <= 0.0 or rate >=1.0:
            return np.random.permutation(np.arange(f))

        method = self.params.get("ds_method", "random")
        if method == "random":
            if not num_points is None:
                ds_inds = random_indices(f, num_points=num_points, max_num_points=max_num_points)
            else:
                ds_inds = random_indices(f, rate=rate, max_num_points=max_num_points)

        elif method == "highest_att":
            n_points = int(f * rate)
            att_sorted, ds_inds = torch.sort(att[:,0])
            ds_inds = ds_inds[-n_points:]
        else:
            ds_inds = random_indices(f.shape[0], rate=rate, max_num_points=max_num_points)

        return ds_inds