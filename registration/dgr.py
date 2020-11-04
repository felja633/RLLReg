import open3d as o3d
import torch
from lib.tensorlist import TensorListList, TensorList
from external.DeepGlobalRegistration.config import get_config
from external.DeepGlobalRegistration.core.deep_global_registration import DeepGlobalRegistration
import time

class DGR:
    def __init__(self, params):
        self.params = params
        config = get_config()
        config.weights=self.params.weights
        self.dgr = DeepGlobalRegistration(config, use_icp=self.params.use_icp)

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
            # B, P, N, M = point_clouds.shape
            # assert P == 2
            Rs = []
            ts = []

        target_R = torch.eye(3,3)
        target_t = torch.zeros(3,1)
        for pcds in point_clouds:
            source = o3d.geometry.PointCloud()
            target = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(pcds[0].cpu().numpy().T)
            target.points = o3d.utility.Vector3dVector(pcds[1].cpu().numpy().T)

            T = self.dgr.register(source, target)
            R = T[0:3, 0:3]
            t = T[0:3, 3:]

            if isinstance(point_clouds, TensorListList):
                Rs.append(TensorList([torch.from_numpy(R).float(), target_R]))
                ts.append(TensorList([torch.from_numpy(t).float(), target_t]))
            else:
                Rs.append(torch.stack([torch.from_numpy(R).float(), target_R]))
                ts.append(torch.stack([torch.from_numpy(t).float(), target_t]))

        tot_time = time.time() - t_tot
        if isinstance(point_clouds, TensorListList):
            return dict(Rs=Rs, ts=ts, Vs=Rs @ point_clouds + ts, time=tot_time)
        else:
            return dict(Rs=torch.stack(Rs), ts=torch.stack(ts), Vs=point_clouds, time=tot_time)
