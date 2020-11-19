from easydict import EasyDict as edict
import math

def get_default_data_loader_parameters():
    params = edict(
        with_features=False,
        use_ground_truth=False,
        augment="with_gt",
        ang_range=math.pi/4.,
        t_range=1.0,
        num_points=10000,
        save_resampled=0,
        with_correspondences=False,
        correspondence_thresh=0.2,
        deterministic=False
    )
    return params

def get_default_dataset_parameters():
    data_loader_parameters = get_default_data_loader_parameters()

    dataset_parameters = edict(
        sequences=[], # use all
        num_views=2,
        with_features=data_loader_parameters.with_features,
        num_samples=128,
    )
    params = edict(
        data_loader_parameters=data_loader_parameters,
        dataset_parameters=dataset_parameters,
    )
    return params

def get_default_kitti_dataset_parameters():
    data_loader_parameters = get_default_data_loader_parameters()

    dataset_parameters = edict(
        sequences = ['00', '01', '02', '03', '04', '05',],
        num_views=2,
        with_features=data_loader_parameters.with_features,
        num_samples=2000,
        range_limit=100,
        dist_threshold=20,
    )
    params = edict(
        data_loader_parameters=data_loader_parameters,
        dataset_parameters=dataset_parameters,
    )
    return params

def get_default_threedmatch_dataset_parameters():
    data_loader_parameters = get_default_data_loader_parameters()

    dataset_parameters = edict(
        sequences = [],
        val_sequences=[],
        num_views=2,
        with_features=data_loader_parameters.with_features,
        num_samples=1000,
        range_limit=50,
        dist_threshold=2,
    )
    params = edict(
        data_loader_parameters=data_loader_parameters,
        dataset_parameters=dataset_parameters,
    )
    return params
