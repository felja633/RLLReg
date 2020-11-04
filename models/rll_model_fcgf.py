import torch
import torch.nn as nn
from registration.rll_reg import PSREG_features_list
from models.feature_extraction.fcgf import resunet
import MinkowskiEngine as ME
from lib.tensorlist import TensorListList, TensorList
from actors import reg_loss_actors
from parameter_settings.registration.dare_settings import get_default_feature_dare_parameters
from easydict import EasyDict as edict
from lib.resampling import random_indices
import numpy as np
import time

def load_weights(path=None, name="RLLRegModel", device = "cuda:0", num_iters=100, K=200, feature_model="vonmises",
                 voxel_size=0.3, use_attention=True, epsilon=float(1e-5), s=0.4,
                 use_dare_weights=False, ds_rate=1.0, mean_init=True, debug=False,
                 cluster_init="default", cluster_precision_scale=1.0, downsample_online=True, max_num_points=14000,
                 cluster_mean_scale=1.0, num_channels=16, conv1_ksize=3, update_priors=False, train_s=False):

    params = get_default_feature_dare_parameters()
    params.name = name
    params.device = device
    params["feature_distr_parameters"] = edict(num_channels=num_channels,
                                               s=s,
                                               model=feature_model,
                                               )
    params.layer = "final"
    params.num_iters = num_iters
    params.backprop_iter = range(0, num_iters+1)
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
    params.conv1_ksize = conv1_ksize
    params.update_priors = update_priors
    params.train_s=train_s

    fmodel = feature_reg_model(params)
    if not path is None:
        act = reg_loss_actors.RLLActor(fmodel, weight=1)
        ch = torch.load(path, map_location="cpu")
        act.load_state_dict(ch["actor"])

        fmodel = act.model
        for x in fmodel.parameters():
            x.requires_grad_(False)
        fmodel.eval()
        fmodel.to(device)
        fmodel.params = params

    return fmodel

class feature_reg_model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.registration = PSREG_features_list(params).to(params.device)
        k_size = self.params.get("conv1_ksize", 3)
        self.feature_extractor = resunet.ResUNetBN2C(in_channels=1,out_channels=params.feature_distr_parameters.num_channels,
                                                     normalize_feature=True, conv1_kernel_size=k_size, D=3).to(params.device)

    def train(self, mode=True):

        self.feature_extractor.train(mode)
        if mode:
            for x in self.parameters():
                x.requires_grad_(True)

            self.feature_extractor.train()
        else:
            for x in self.parameters():
                x.requires_grad_(False)

            self.feature_extractor.eval()

    def eval(self):
        self.train(mode=False)

    def sparse_quantize_tensors(self, coords, voxel_size):
        quantized_coords = torch.floor(coords / voxel_size)
        inds = ME.utils.sparse_quantize(quantized_coords, return_index=True)

        if self.params.get("downsample_online", False):
            rate = self.params.get("downsample_rate", 0.2)
            max_num_points = self.params.get("max_num_points", None)
            inds_ds = self.downsample(inds.shape[0], rate, num_points=None, att=None, max_num_points=max_num_points)
            inds = inds[inds_ds]

        return quantized_coords[inds], inds

    def create_sparse_tensors(self, coords):

        quantized_coords = []
        inds_list = []
        feats_list = []
        for coords_b in coords:

            voxel_size = self.params.voxel_size

            for coords_s in coords_b:
                quantized, inds = self.sparse_quantize_tensors(coords_s.permute(1,0), voxel_size)
                quantized_coords.append(quantized)
                feats_list.append(torch.ones(quantized.shape[0],1))
                inds_list.append(inds)

        coords_sparse, feats_sparse = ME.utils.sparse_collate(quantized_coords, feats_list)
        sinput = ME.SparseTensor(
            feats_sparse, coords=coords_sparse).to(self.params.device)
        return sinput, inds_list

    def forward(self, x_in):
        coords = x_in['coordinates'].clone()
        t_tot = time.time()
        sinput, inds_list = self.create_sparse_tensors(coords)
        resample_time = time.time() - t_tot
        print("resample time: %.1f ms" % ((resample_time) * 1000))
        if not self.params.feature_distr_parameters.model=='none':
            features_dict = self.feature_extractor(sinput)
            features = features_dict["features"]
            if "attention" in features_dict.keys():
                att = features_dict["attention"]

            extract_time=time.time()-t_tot
            print("extract time: %.1f ms" % ((extract_time) * 1000))
            batch_indices = list(features.coords_man.get_batch_indices())
        else:
            features_dict=None
            extract_time=0

        time_preprocess = time.time()

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
                if not features_dict is None:
                    mask = features.C[:, 0] == batch_indices[ind_cnt]

                    # hacky way of finding batch channel dimension
                    if mask.int().sum() != inds_list[ind_cnt].shape[0]:
                        mask = features.C[:, -1] == batch_indices[ind_cnt]

                    f = features.F[mask]
                    assert f.shape[0] == inds_list[ind_cnt].shape[0]
                    if "attention" in features_dict.keys():
                        a = att.F[mask]
                        assert a.shape[0] == inds_list[ind_cnt].shape[0]
                        att_L.append(a)

                    features_L.append(f.permute(1, 0))

                coords_ds_L.append(coords_s[:, inds_list[ind_cnt]])
                inds_L.append(inds_list[ind_cnt])

                ind_cnt = ind_cnt + 1

            features_LL.append(features_L)
            att_LL.append(att_L)
            coords_ds_LL.append(coords_ds_L)
            inds_LL.append(inds_L)

        x = dict()
        x['features'] = features_LL
        x['att'] = att_LL
        x['coordinates_ds'] = coords_ds_LL.to(self.params.device)
        x['indices'] = inds_LL
        x['coordinates'] = x_in['coordinates']

        print("preprocess time: %.1f ms" % ((time.time()-time_preprocess) * 1000))

        reg_time=time.time()
        out = self.registration(x)
        reg_time2 = time.time() - reg_time
        print("reg time: %.1f ms" % ((time.time() - reg_time) * 1000))
        tot_time = time.time() - t_tot
        print("tot time: %.1f ms" % (tot_time * 1000))

        out["time"] = tot_time
        out["reg_time"] = reg_time2
        out["extract_time"] =extract_time
        out["resample_time"] = resample_time
        out["coordinates_ds"] = coords_ds_LL

        return out

    def extract_features(self, x):
        coords = x['coordinates']

        sinput, inds_list = self.create_sparse_tensors(coords)
        features_dict = self.feature_extractor(sinput)
        features = features_dict["features"]
        if "attention" in features_dict.keys():
            att = features_dict["attention"]

        batch_indices = list(features.coords_man.get_batch_indices())
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
                mask = features.C[:, 0] == batch_indices[ind_cnt]

                # hacky way of finding batch channel dimension
                if mask.int().sum() != inds_list[ind_cnt].shape[0]:
                    mask = features.C[:, -1] == batch_indices[ind_cnt]

                f = features.F[mask]
                assert f.shape[0] == inds_list[ind_cnt].shape[0]
                if "attention" in features_dict.keys():
                    a = att.F[mask]
                    assert a.shape[0] == inds_list[ind_cnt].shape[0]
                    att_L.append(a)

                features_L.append(f.permute(1, 0))

                coords_ds_L.append(coords_s[:, inds_list[ind_cnt]])
                inds_L.append(inds_list[ind_cnt])

                ind_cnt = ind_cnt + 1

            features_LL.append(features_L)
            att_LL.append(att_L)
            coords_ds_LL.append(coords_ds_L)
            inds_LL.append(inds_L)

        out = dict()
        out['features'] = features_LL
        out['att'] = att_LL
        out['coordinates_ds'] = coords_ds_LL.to(self.params.device)
        out['indices'] = inds_LL
        out['coordinates'] = x['coordinates']

        return out

    def register(self, x):
        out = self.registration(x)
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