import numpy as np
import torch
import open3d as o3d
from datasets.minipykitti import MiniOdometry
from PIL import Image

class ETHDataLoader:
    def __call__(self, name, sample_info):
        print("load point clouds from ",
              sample_info["meta_info"]["dset_path"] + "/" + sample_info["sequence"] + "/" + name + ".ply")
        v = o3d.io.read_point_cloud(
            sample_info["meta_info"]["dset_path"] + "/" + sample_info["sequence"] + "/" + name + ".ply")
        v = np.asarray(v.points).astype(np.float32)
        R = torch.eye(3, 3)
        t = torch.zeros(3, 1)

        if name != 's1':
            T = np.loadtxt(
                sample_info["meta_info"]["dset_path"] + "/" + sample_info["sequence"] + "/" + sample_info["meta_info"][
                    "groundtruth"] + "/" + name + "-s1.tfm")
            R = torch.from_numpy(T[0:3, 0:3].astype(np.float32))
            t = torch.from_numpy(T[0:3, 3:].astype(np.float32))

        if sample_info["with_features"]:
            return dict(coordinates=torch.from_numpy(v[:, 0:3]).t(), features=torch.from_numpy(v[:, 3:]), R_gt=R,
                        t_gt=t)
        else:
            return dict(coordinates=torch.from_numpy(v[:, 0:3]).t(), R_gt=R, t_gt=t)

class KittiDataLoader:
    def __call__(self, name, sample_info, device="cpu"):

        basedir = sample_info["meta_info"]["dset_path"]

        # Specify the dataset to load
        sequence = sample_info["sequence"]
        frame = int(name)

        # Load the data. Optionally, specify the frame range to load.
        dataset = MiniOdometry(basedir, sequence)
        calib_cam0 = torch.from_numpy(dataset.calib["T_cam0_velo"].astype(np.float32))
        pose = dataset.poses[frame]
        velo = dataset.get_velo(frame)
        velo = velo[:,0:3]
        R = pose[0:3,0:3] @ calib_cam0[0:3,0:3]
        t = pose[0:3, 3:] + pose[0:3,0:3] @ calib_cam0[0:3, 3:]
        return dict(coordinates=torch.from_numpy(velo.T).to(device), R_gt=R.to(device), t_gt=t.to(device))

    def get_poses(self, basedir, sequence):
        dataset = MiniOdometry(basedir, sequence)
        return dataset.poses

    def get_pose(self, basedir, sequence, frame):
        dataset = MiniOdometry(basedir, sequence)
        calib_cam0 = torch.from_numpy(dataset.calib["T_cam0_velo"].astype(np.float32))
        pose = dataset.poses[frame]
        R = pose[0:3, 0:3] @ calib_cam0[0:3, 0:3]
        t = pose[0:3, 3:].astype(np.float32) + pose[0:3, 0:3] @ calib_cam0[0:3, 3:]
        return dict(R_gt=R, t_gt=t, pos=t)

class ThreeDMatchDataLoader:
    def __call__(self, name, sample_info, device="cpu"):

        basedir = sample_info["meta_info"]["dset_path"]

        # Specify the dataset to load
        sample_dict = sample_info["sample_dict"][name]

        # Load the data
        calib_cam0 = np.loadtxt(basedir+"/"+sample_dict["intrinsics"])
        pose = np.loadtxt(basedir+"/"+sample_dict["pose"])
        depth_image = np.asarray(Image.open(basedir+"/"+sample_dict["depth"]))/1000.0

        # create point cloud from depth image
        x = np.arange(depth_image.shape[1])
        x = np.expand_dims(x, axis=0)
        x = x.repeat(depth_image.shape[0], axis=0)
        y = np.arange(depth_image.shape[0])
        y = np.expand_dims(y, axis=1)
        y = y.repeat(depth_image.shape[1], axis=1)
        im_coords = (np.linalg.inv(calib_cam0) @ np.vstack([x.flatten(), y.flatten(), np.ones(y.flatten().shape)]))
        pcd = np.expand_dims(depth_image.flatten(), axis=0) * im_coords
        pcd = pcd[:,np.linalg.norm(pcd, axis=0)>0]
        pcd = pcd[:,np.linalg.norm(pcd, axis=0)<30.0]
        R = torch.from_numpy(pose[0:3,0:3].astype(np.float32))
        t = torch.from_numpy(pose[0:3, 3:].astype(np.float32))

        return dict(coordinates=torch.from_numpy(pcd).float().to(device), R_gt=R.to(device), t_gt=t.to(device))

    def get_pose(self, sample_info, name):
        basedir = sample_info["meta_info"]["dset_path"]
        sample_dict = sample_info["sample_dict"][name]
        pose = np.loadtxt(basedir + "/" + sample_dict["pose"])
        R = torch.from_numpy(pose[0:3,0:3].astype(np.float32))
        t = torch.from_numpy(pose[0:3, 3:].astype(np.float32))
        return dict(R_gt=R, t_gt=t, pos=t)

