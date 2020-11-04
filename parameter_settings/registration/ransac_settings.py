from easydict import EasyDict as edict


def get_default_ransac_parameters():
    params = edict(
        metric="p2pl",
        threshold=5.0,
        voxel_size=0.3,
        radius_normal=1.0,
        radius_feature=2.0,
        name="RANSAC",
    )

    return params