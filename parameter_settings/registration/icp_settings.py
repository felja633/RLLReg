from easydict import EasyDict as edict

def get_default_icp_parameters():
    params = edict(
        metric="p2p",
        threshold=5.0,
        name="ICP_p2p",
        radius_normal=1.0,
    )

    return params

def get_default_icp_lite_parameters():
    params = edict(
        step_length=0.0005,
        momentum=(0.9, 0.999),
        num_iters=50,
        num_inner_iters=10,
        num_neighbors=10,
        sigma=1.0,
        c=0.1,
        rho=0.5,
        max_bt_iters = 5,
        use_observation_weights = False,
        name="ICP_Lite",
        debug=False,
        thresh=5.0,
        num_icp_neighbors=10,
        compute_correspendence_weights=True
    )

    return params