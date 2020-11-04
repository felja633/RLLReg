from pathlib import Path
from models.rll_model_fcgf import load_weights
from evaluation.evaluate import benchmark
from parameter_settings.registration.icp_settings import get_default_icp_parameters
from parameter_settings.registration.fgr_settings import get_default_fgr_parameters
from parameter_settings.dataset_settings import get_default_threedmatch_dataset_parameters
from registration.icp import ICP
from registration.fgr import FGR
from registration.dgr import DGR
import models.fppsr_model_fcgf as fppsr_reg
import models.fppsr_model_fpfh as fpfh_fppsr_reg
from datasets import threedmatch
from datasets.processing import NoProcessing
from datasets.data_reader import collate_tensorlist
import torch
import os, sys
import models.feature_extraction.fcgf.resunet_fcgf as fcgf_net
from easydict import EasyDict as edict
import config

envsettings = config.EnvironmentSettings()
filename=os.path.split(sys.argv[0])[1].split('.')[0]
workspace = envsettings.workspace_dir +filename

def load_fcgf_model(model_path, device):
    checkpoint = torch.load(model_path)
    config = checkpoint['config']

    num_feats = 1
    model = fcgf_net.ResUNetBN2C(
        num_feats,
        config.model_n_out,
        bn_momentum=0.05,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        D=3)
    model.load_state_dict(checkpoint['state_dict'])
    for x in model.parameters():
        x.requires_grad_(False)
    model.eval()

    return model.to(device)

def configure_methods():
    device = "cuda:0"

    mean_init = True
    jrmpc = load_weights(name="JRMPC", feature_model="none", use_attention=False, device=device, voxel_size=0.05,
                          use_dare_weights=False, mean_init=mean_init, K=100, downsample_online=False)

    path = envsettings.pretrained_networks + "/RLLReg_threedmatch.pth"
    rll = load_weights(path=path, name="RLLReg", feature_model="vonmises", use_attention=True,
                                          device=device, voxel_size=0.05, use_dare_weights=False, mean_init=mean_init, 
                                          num_channels=16, K=100, conv1_ksize=5, downsample_online=False)

    path = envsettings.pretrained_networks + "/2019-08-16_19-21-47.pth"
    contrastive = load_weights(name="Contrastive", feature_model="vonmises", use_attention=False,
                                              device=device, voxel_size=0.05, use_dare_weights=False,
                                                  mean_init=mean_init, num_channels=32, K=100, downsample_online=False)
    contrastive.feature_extractor = load_fcgf_model(path, device)

    path = envsettings.pretrained_networks + "/2019-08-16_19-21-47.pth"
    fcgf_fppsr = fppsr_reg.load_weights(name="FPPSR+FCGF", use_attention=False, num_channels=32,
                                              device=device, voxel_size=0.05, use_dare_weights=False,
                                              mean_init=mean_init, K=100,
                                              num_feature_clusters=10, downsample_online=False)
    fcgf_fppsr.feature_extractor = load_fcgf_model(path, device)


    fpfh_fppsr = fpfh_fppsr_reg.load_weights(name="FPPSR+FPFH", use_attention=False,
                                              device=device, voxel_size=0.05, use_dare_weights=False,
                                              mean_init=mean_init, K=100, num_feature_clusters=10, downsample_online=False)

    fgr_params = get_default_fgr_parameters()
    fgr_params.metric = "p2pl"
    fgr_params.threshold = 0.2
    fgr_params.voxel_size = 0.05
    fgr_params.radius_normal = 2*fgr_params.voxel_size
    fgr_params.radius_feature = 4*fgr_params.voxel_size
    fgr_params.name = "FGR"
    fgr_params.refine=False
    fgr_test = FGR(fgr_params)


    dgr_params = edict()
    dgr_params.name="DGR"
    dgr_params.weights=envsettings.pretrained_networks+"/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
    dgr_params.use_icp=False
    dgr_model = DGR(dgr_params)


    icp_params_p2p= get_default_icp_parameters()
    icp_params_p2p.metric = "p2p"
    icp_params_p2p.name = "ICP-pt2p"
    icp_params_p2p.threshold = 0.3
    icp_params_p2p.mean_init = mean_init
    icp_params_p2p.voxel_size = 0.05

    icp_params_p2pl = get_default_icp_parameters()
    icp_params_p2pl.metric = "p2pl"
    icp_params_p2pl.name = "ICP-p2plane"
    icp_params_p2pl.threshold = 0.3
    icp_params_p2pl.radius_normal = 0.2
    icp_params_p2pl.mean_init = mean_init
    icp_params_p2pl.voxel_size = 0.05

    return [rll, contrastive, jrmpc, fcgf_fppsr, fpfh_fppsr, fgr_test, dgr_model, ICP(icp_params_p2p), ICP(icp_params_p2pl)]


def configure_datasets():
    params = get_default_threedmatch_dataset_parameters()
    threedmatch_params = params

    threedmatch_params.dataset_parameters.num_samples = 1000
    threedmatch_params.dataset_parameters.num_views = 2
    threedmatch_params.dataset_parameters.dist_threshold = 2
    threedmatch_params.dataset_parameters.range_limit = 50

    threedmatch_params.data_loader_parameters.with_correspondences = False
    threedmatch_params.data_loader_parameters.correspondence_thresh = 0.3
    threedmatch_params.data_loader_parameters.augment=False
    threedmatch_params.data_loader_parameters.ang_range = 0
    threedmatch_params.data_loader_parameters.t_range = 0
    threedmatch_params.data_loader_parameters.deterministic = True

    threedmatch_params["workspace"] = workspace
    path_threedmatch= envsettings.threedmatch_dir
    downsampler = NoProcessing()
    processor = NoProcessing()
    return threedmatch.ThreeDMatchDataset(path_threedmatch, threedmatch_params, downsampler, processor,
                                          training=False, correspondence_rate=0.3)


if __name__ == '__main__':
    Path(workspace).mkdir(parents=True, exist_ok=True)
    methods = configure_methods()
    dataset = configure_datasets()

    data = benchmark(methods, dataset, workspace, batch_size=16, vis=None, num_workers=0, collate_fn=collate_tensorlist,
              max_err_R=4.0, max_err_t=0.3, success_R=4, success_t=0.1, save_errors=True, plot=False)

    import pickle

    with open(workspace+'/results.txt', 'wb') as handle:
        pickle.dump(data, handle)
