from pathlib import Path
import models.rll_model_fcgf as rll
from evaluation.evaluate import benchmark
from parameter_settings.dataset_settings import get_default_kitti_dataset_parameters
from datasets import kitti
import models.fppsr_model_fcgf as fppsr_reg
import models.fppsr_model_fpfh as fpfh_fppsr_reg
from datasets.processing import NoProcessing
from datasets.data_reader import collate_tensorlist
import torch
import os, sys
import models.feature_extraction.fcgf.resunet_fcgf as fcgf_net
import config

envsettings = config.EnvironmentSettings()
filename=os.path.split(sys.argv[0])[1].split('.')[0]
workspace = envsettings.workspace_dir +filename

def load_fcgf_model(model_path, device, channels):
    checkpoint = torch.load(model_path)

    num_feats = 1
    model = fcgf_net.ResUNetBN2C(
        num_feats,
        channels,
        bn_momentum=0.05,
        normalize_feature=True,
        conv1_kernel_size=5,
        D=3)
    model.load_state_dict(checkpoint['state_dict'])
    for x in model.parameters():
        x.requires_grad_(False)
    model.eval()

    return model.to(device)

def configure_methods():
    device = "cuda:0"

    mean_init = False
    dare = rll.load_weights(name="DARE", feature_model="none", use_attention=False, device=device, K=100, voxel_size=0.3,
                                use_dare_weights=True, mean_init=mean_init, downsample_online=False)

    path = envsettings.pretrained_networks + "/KITTI-v0.3-ResUNetBN2C-conv1-5-nout16.pth"
    contrastive = rll.load_weights(name="Contrastive", feature_model="vonmises", use_attention=False, mean_init=mean_init,
                              device=device, voxel_size=0.3, use_dare_weights=True,
                              num_channels=16, s=0.4, K=100, downsample_online=False)
    contrastive.feature_extractor = load_fcgf_model(path, device, 16)

    path = envsettings.pretrained_networks + "/RLLReg_kitti.pth"
    pairwise = rll.load_weights(path=path, name="pairwise", feature_model="vonmises", num_channels=16,
                                 mean_init=mean_init, device=device, use_attention=True, K=100,
                                 voxel_size=0.3, s=0.4, conv1_ksize=5, use_dare_weights=False, downsample_online=False)

    path = envsettings.pretrained_networks + "/RLLReg_kitti_multi.pth"
    multi = rll.load_weights(path=path, name="multi", feature_model="vonmises", num_channels=16, mean_init=mean_init,
                              device=device, use_attention=True, K=100, voxel_size=0.3, s=0.4, conv1_ksize=5,
                              use_dare_weights=False, downsample_online=False)

    path = envsettings.pretrained_networks + "/KITTI-v0.3-ResUNetBN2C-conv1-5-nout16.pth"
    fcgf_fppsr_dare = fppsr_reg.load_weights(name="FPPSR+FCGF", use_attention=False, num_channels=16,
                                              device=device, voxel_size=0.3, use_dare_weights=True,
                                              mean_init=mean_init, K=100, num_feature_clusters=10,
                                              downsample_online=False)
    fcgf_fppsr_dare.feature_extractor = load_fcgf_model(path, device, 16)


    fpfh_fppsr_dare = fpfh_fppsr_reg.load_weights(name="FPPSR+FPFH", use_attention=False,
                                              device=device, voxel_size=0.3, use_dare_weights=True,
                                              mean_init=mean_init, K=100, num_feature_clusters=10,
                                              downsample_online=False)

    return [pairwise, multi, dare, contrastive, fcgf_fppsr_dare, fpfh_fppsr_dare]

def configure_datasets():

    kitti_params = get_default_kitti_dataset_parameters()
    kitti_params.dataset_parameters.sequences = ['08', '09', '10']
    kitti_params.dataset_parameters.range_limit = 100
    kitti_params.dataset_parameters.dist_threshold = 15.0
    kitti_params.dataset_parameters.num_samples = 512
    kitti_params.dataset_parameters.num_views = 4
    kitti_params.data_loader_parameters.with_correspondences = False
    kitti_params.data_loader_parameters.correspondence_thresh = 0.5
    kitti_params.data_loader_parameters.augment = False
    kitti_params.data_loader_parameters.ang_range = 0
    kitti_params.data_loader_parameters.t_range = 0
    kitti_params.data_loader_parameters.deterministic = True
    path_kitti = envsettings.kitti_dir
    kitti_params["workspace"] = workspace
    downsampler = NoProcessing()
    processor = NoProcessing()
    return kitti.KittiDataset(path_kitti, kitti_params, downsampler, processor, training=False, data_type='list',
                              correspondence_rate=0.3)

if __name__ == '__main__':
    Path(workspace).mkdir(parents=True, exist_ok=True)
    methods = configure_methods()
    dataset = configure_datasets()

    data = benchmark(methods, dataset, workspace, batch_size=16, vis=None, num_workers=0, collate_fn=collate_tensorlist,
                     max_err_R=4.0, max_err_t=0.6, success_R=4, success_t=0.3, save_errors=True, plot=False)

    import pickle

    with open(workspace+'/results.txt', 'wb') as handle:
        pickle.dump(data, handle)
