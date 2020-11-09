from pathlib import Path
from models.rll_model_fcgf import feature_reg_model
from parameter_settings.registration.dare_settings import get_default_feature_dare_parameters
from parameter_settings.dataset_settings import get_default_threedmatch_dataset_parameters
from datasets import threedmatch
from easydict import EasyDict as edict
from training.trainer import trainer
import torch.optim as optim
from actors import reg_loss_actors
from datasets.processing import RandomDownsampler, NoProcessing
from datasets.data_reader import collate_tensorlist
import math
import os, sys
import config

envsettings = config.EnvironmentSettings()
filename=os.path.split(sys.argv[0])[1].split('.')[0]
workspace = envsettings.workspace_dir +filename

def configure_methods():

    params_feat = get_default_feature_dare_parameters()
    params_feat.name = "RLLReg_threedmatch"
    params_feat.device="cuda:0"
    params_feat.feature_distr_parameters = edict(num_channels=16,
                                      s=float(0.4),
                                      init_prec_scale=0.1,
                                      model="vonmises",
                                      )
    params_feat.layer = "lin3_out"
    params_feat.num_iters=40
    params_feat.backprop_iter = range(0, 24)
    params_feat.K = 50
    params_feat.cluster_init = 'sphere'
    params_feat.cluster_precision_init = 'scaled'
    params_feat.cluster_precision_scale = 1.0
    params_feat.fix_cluster_pos_iter = 2
    params_feat.use_dare_weighting = False
    params_feat.debug = False
    params_feat.use_features_iter = 0
    params_feat.use_attention = True
    params_feat.voxel_size = 0.05
    params_feat.downsample_online = True
    params_feat.downsample_rate = 1.0
    params_feat.ds_method = "random"
    params_feat.max_num_points = 12000
    params_feat.conv1_ksize = 5

    fmodel = feature_reg_model(params_feat)

    fmodel.train()
    return fmodel.to(params_feat.device)

def configure_datasets():
    params = get_default_threedmatch_dataset_parameters()
    threedmatch_params = params
    threedmatch_params.dataset_parameters.sequences = []
    threedmatch_params.dataset_parameters.val_sequences = ['sun3d-harvard_c3-hv_c3_1/seq-01', 'rgbd-scenes-v2-scene_05/seq-01',
                                                       'bundlefusion-apt1/seq-01',
                                                       'analysis-by-synthesis-apt2-living/seq-01', '7-scenes-fire/seq-01']
    threedmatch_params.dataset_parameters.num_samples = 2000
    threedmatch_params.dataset_parameters.num_views = 4
    threedmatch_params.data_loader_parameters.with_correspondences = False
    threedmatch_params.data_loader_parameters.correspondence_thresh = 0.3
    threedmatch_params.data_loader_parameters.augment=True
    threedmatch_params.data_loader_parameters.ang_range=math.pi/8.
    threedmatch_params.data_loader_parameters.t_range=0.8
    threedmatch_params["workspace"] = workspace
    path_threedmatch= envsettings.threedmatch_dir
    params["workspace"] = workspace
    downsampler = RandomDownsampler(num_points=100000)
    processor = NoProcessing()
    return threedmatch.ThreeDMatchDataset(path_threedmatch, threedmatch_params, downsampler, processor,
                                          training=True, correspondence_rate=0.0)

def configure_val_datasets():
    params = get_default_threedmatch_dataset_parameters()
    threedmatch_params = params
    threedmatch_params.dataset_parameters.sequences = ['sun3d-harvard_c3-hv_c3_1/seq-01', 'rgbd-scenes-v2-scene_05/seq-01',
                                                       'bundlefusion-apt1/seq-01',
                                                       'analysis-by-synthesis-apt2-living/seq-01', '7-scenes-fire/seq-01']
    threedmatch_params.dataset_parameters.num_samples = 100
    threedmatch_params.data_loader_parameters.with_correspondences = False
    threedmatch_params.data_loader_parameters.correspondence_thresh = 0.1
    threedmatch_params.data_loader_parameters.augment=False
    threedmatch_params.data_loader_parameters.ang_range = 0.0
    threedmatch_params.data_loader_parameters.t_range = 0.0
    threedmatch_params.data_loader_parameters.deterministic = True
    threedmatch_params["workspace"] = workspace
    path_threedmatch= envsettings.threedmatch_dir
    downsampler = RandomDownsampler(num_points=100000)
    processor = NoProcessing()
    return threedmatch.ThreeDMatchDataset(path_threedmatch, threedmatch_params, downsampler, processor, training=True,
                                          correspondence_rate=0.0)


def configure_trainer(model):
    # paper versions are trained with adam, but better performance is acheived using AdamW
    optimizer = optim.AdamW([{'params': model.feature_extractor.parameters(), 'lr': 1e-4}], lr=1e-4)

    base_vs_weight = 3*(40.0-24.0)/2.0
    Vs_iter = [base_vs_weight/(model.params.num_iters-i) for i in range(model.params.num_iters-1)]
    for i in range(model.params.num_iters-1):
        if i not in model.params.backprop_iter:
            Vs_iter[i] = 0.0

    weights = dict(Vs=0.0, Vs_iter=Vs_iter)


    compute_loss_iter = 4
    # init actor
    actor = reg_loss_actors.RLLActor(model=model, dist_thresh=0.05, weight=weights, c=0.4,
                                     compute_loss_iter=compute_loss_iter, vis=None, alpha=-2.0)

    step_size = 10
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4*step_size, gamma=0.2)
    training_module = trainer(num_epochs=181,
                              optimizer=optimizer, scheduler=lr_scheduler,
                              actor=actor, job_name=workspace, collate_fn=collate_tensorlist)
    return training_module

if __name__ == '__main__':
    Path(workspace).mkdir(parents=True, exist_ok=True)
    method = configure_methods()
    dataset = configure_datasets()
    val_dataset = configure_val_datasets()
    training_module = configure_trainer(method)
    training_module.train(dataset, batch_size=6, num_workers=0, valdata=val_dataset)
