import open3d as o3d
import numpy as np
import torch
from lib.tensorlist import TensorListList, TensorList
import time

class ICP:
    def __init__(self, params):

        self.params = params
        if self.params.metric == "p2p":
            self.metric = o3d.registration.TransformationEstimationPointToPoint()
        elif self.params.metric == "p2pl":
            self.metric = o3d.registration.TransformationEstimationPointToPlane()

    def __call__(self, point_clouds):
        return self.register_point_sets(point_clouds)

    def train(self):
        return

    def eval(self):
        return

    def register_point_sets(self, x):
        point_clouds = x["coordinates"].clone()
        t_tot = time.time()

        if isinstance(point_clouds, TensorListList):
            Rs = TensorListList()
            ts = TensorListList()
        else:
        #B, P, N, M = point_clouds.shape
        #assert P == 2
            Rs = []
            ts = []

        target_R = torch.eye(3,3)
        target_t = torch.zeros(3,1)
        for pcds in point_clouds:
            if self.params.get("mean_init", True):
                m_target = pcds[1].mean(dim=1)
                m_source = pcds[0].mean(dim=1)
                init_t = -(m_target-m_source).cpu().numpy()
            else:
                init_t = [0,0,0]

            trans_init = np.asarray([[1., 0., 0., init_t[0]],
                                     [0., 1., 0., init_t[1]],
                                     [0., 0., 1., init_t[2]], [0.0, 0.0, 0.0, 1.0]])


            source = o3d.geometry.PointCloud()
            target = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(pcds[0].cpu().numpy().T)
            target.points = o3d.utility.Vector3dVector(pcds[1].cpu().numpy().T)

            voxel_size = self.params.voxel_size
            max_correspondence_distance=self.params.threshold
            radius_normal=float(self.params.radius_normal)

            if self.params.metric == "p2pl":
                source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius_normal, max_nn=30))
                target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius_normal, max_nn=30))
                # o3d.geometry.estimate_normals(source)
                # o3d.geometry.estimate_normals(target)

            if self.params.get("downsample", True):
                source = source.voxel_down_sample(voxel_size=voxel_size)
                target = target.voxel_down_sample(voxel_size=voxel_size)

            reg_p2p = o3d.registration.registration_icp(
                    source, target, max_correspondence_distance=max_correspondence_distance,
                    estimation_method=self.metric, init=trans_init)

            T = reg_p2p.transformation
            R = T[0:3,0:3]
            t = T[0:3,3:]

            if isinstance(point_clouds, TensorListList):
                Rs.append(TensorList([torch.from_numpy(R).float(), target_R]))
                ts.append(TensorList([torch.from_numpy(t).float(), target_t]))
            else:
                Rs.append(torch.stack([torch.from_numpy(R).float(), target_R]))
                ts.append(torch.stack([torch.from_numpy(t).float(), target_t]))

        tot_time = time.time() - t_tot
        print("ICP tot time: %.1f ms" % (tot_time * 1000))

        if isinstance(point_clouds, TensorListList):
            return dict(Rs=Rs, ts=ts, Vs=Rs @ point_clouds + ts, time=tot_time)
        else:
            return dict(Rs=torch.stack(Rs), ts=torch.stack(ts), Vs=point_clouds, time=tot_time)