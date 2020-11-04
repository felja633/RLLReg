from easydict import EasyDict as edict


def get_default_fgr_parameters():
    params = edict(
        metric="p2pl",
        threshold=5.0,
        voxel_size=0.5,
        radius_normal=1.0,
        radius_feature=2.0,
        name="FGR",
    )

    return params