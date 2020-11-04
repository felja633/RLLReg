import open3d as o3d
import torch
from lib.tensorlist import TensorListList, TensorList
import time

class FGR:
    def __init__(self, params):
        self.params = params

    def __call__(self, point_clouds):
        return self.register_point_sets(point_clouds)

    def train(self):
        return

    def eval(self):
        return

    def preprocess_point_cloud(self, pcd, voxel_size):
        #print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)#o3d.voxel_down_sample(pcd, voxel_size)

        radius_normal = voxel_size * 2
        #print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def prepare_dataset(self, source, target, voxel_size):
        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)
        return source_down, target_down, source_fpfh, target_fpfh

    def execute_fast_global_registration(self, source_down, target_down, source_fpfh,
                                         target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.5
        #print(":: Apply fast global registration with distance threshold %.3f" \
        #      % distance_threshold)
        result = o3d.registration.registration_fast_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        return result

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

            voxel_size = self.params.voxel_size

            source_down, target_down, source_fpfh, target_fpfh = \
                self.prepare_dataset(source, target, voxel_size)

            result_fast = self.execute_fast_global_registration(source_down, target_down,
                                                           source_fpfh, target_fpfh,
                                                           voxel_size)

            T = result_fast.transformation
            R = T[0:3, 0:3]
            t = T[0:3, 3:]

            if self.params.get("refine", True):
                radius_normal = voxel_size * 2
                if self.params.metric == "p2pl":
                    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
                    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

                distance_threshold = voxel_size * 0.5
                reg_p2p = o3d.registration.registration_icp(
                        source, target, max_correspondence_distance=distance_threshold,
                        estimation_method=o3d.registration.TransformationEstimationPointToPlane(),
                    init=result_fast.transformation)

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
        print("FGR tot time: %.1f ms" % (tot_time * 1000))

        if isinstance(point_clouds, TensorListList):
            return dict(Rs=Rs, ts=ts, Vs=Rs @ point_clouds + ts, time=tot_time)
        else:
            return dict(Rs=torch.stack(Rs), ts=torch.stack(ts), Vs=point_clouds, time=tot_time)
