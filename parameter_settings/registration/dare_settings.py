from easydict import EasyDict as edict
from registration.observation_weights import empirical_estimate

def get_default_dare_parameters():
    params = edict(
        epsilon=float(1e-5),
        gamma=float(0.005),
        num_neighbors=20,
        K=100,
        num_iters=50,
        observation_weight_function=empirical_estimate,
        ow_args=10,
        fix_cluster_pos_iter=2,
        use_dare_weighting=True,
        cluster_init='default',
        cluster_precision_init='default',
        debug=False,
        cluster_precision_scale = 1.0,
        name="DARE",
        use_features_iter=-1,
    )
    return params

def get_default_feature_dare_parameters():
    params = edict(
        epsilon=float(1e-5),
        gamma=float(0.005),
        num_neighbors=10,
        train_epsilon=False,
        K=50,
        num_iters=50,
        ow_args=10,
        fix_cluster_pos_iter=2,
        use_dare_weighting=True,
        cluster_init='default',
        cluster_precision_init='default',
        debug=False,
        cluster_precision_scale = 1.0,
        name="DARE_features",
        use_features_iter=-1,
        feature_distr_parameters=edict(num_classes=16,
                                      epsilon=float(1.0),
                                      init_prec_scale=0.1,
                                      model="hack",
                                      num_iters=2,
                                      radius=1.0,
                                      fixed_feature_var=False),
        layer = "lin3_out",
        backprop_iter = range(-1, 1000),
        use_attention = True,
    )
    return params
