from lib.utils import global_to_relative
from torch.utils.data import DataLoader
from collections import OrderedDict
import open3d as o3d
import os

def register_sequences(method, batch):
    data, info = batch
    print("register", info["names"])
    out = method(data)
    R, t = out["Rs"], out["ts"]
    out["Rs"], out["ts"] = global_to_relative(R, t, ref_index=0)
    out["names"] = info["names"]
    return out

def run_sequence(methods, dataset, job_name, batch_size=1, num_workers=0,
                 origin_ref_names = [], collate_fn=None):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    output_data = dict()
    for m in methods:
        m.eval()
        output_data[m.params.name] = []

    for batch in loader:
        print("register batch...")
        for m in methods:
            print(m.params.name)
            output_data[m.params.name].append(register_sequences(m, batch))



    for m in output_data.keys():
        out_list = output_data[m]

        print("create bundle matrix...")
        # create bundle matrix
        bundle_dict = OrderedDict()
        for out in out_list:
            current_names = out["names"]
            Vs = out["Vs"] # Vs
            batch_size = len(Vs)
            sequence_size = len(Vs[0])
            ref_names = list(c[0] for c in current_names)
            for b in range(batch_size):
                if ref_names[b] not in bundle_dict.keys():
                    bundle_dict[ref_names[b]] = OrderedDict()
                for s in range(sequence_size):
                    name_list = list(current_names[b])
                    point_cloud_name = name_list[s]
                    bundle_dict[ref_names[b]][point_cloud_name] = {"Vs": Vs[b][s], "Rs": out["Rs"][b][s], "ts":out["ts"][b][s]}


        print("map point clouds to reference origin...")
        # map point clouds to reference origin
        if len(origin_ref_names) == 0:
            ref_names = list(out_list[0]["names"][0])
            origin_ref_names = [ref_names[0],]

        for origin_ref_name in origin_ref_names:
            ref_found = True

            while ref_found:
                ref_found = False
                current_name_list = [n for n in bundle_dict[origin_ref_name].keys() if n != origin_ref_name]
                print(origin_ref_name)
                for n in current_name_list:
                    if n in bundle_dict.keys():
                        print("extract ref transformation from origin_ref_name ", origin_ref_name, " for ", n)
                        Rref, tref = bundle_dict[origin_ref_name][n]["Rs"], bundle_dict[origin_ref_name][n]["ts"]
                        for ns in bundle_dict[n].keys():
                            print("Transform ", ns, " from ", n, "frame to ", origin_ref_name, " frame")
                            Rs = bundle_dict[n][ns]["Rs"]
                            ts = bundle_dict[n][ns]["ts"]
                            bundle_dict[n][ns]["Rs"] = Rref @ Rs
                            bundle_dict[n][ns]["ts"] = Rref @ ts + tref

                        print("Set ", n, " to new origin_ref_name")
                        origin_ref_name = n
                        ref_found = True
                        continue

        stored_point_clouds = []
        for n in bundle_dict.keys():
            for ns in bundle_dict[n].keys():
                if ns not in stored_point_clouds:
                    Vs = bundle_dict[n][ns]["Rs"] @ bundle_dict[n][ns]["Vs"].cuda() + bundle_dict[n][ns]["ts"]
                    source = o3d.geometry.PointCloud()
                    source.points = o3d.utility.Vector3dVector(Vs.cpu().numpy().T)
                    name_split = ns.split("/")
                    dirname = job_name + "/registered_sequences/" + m + "/" + name_split[0]
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)

                    fname = dirname + "/" + name_split[1] + ".ply" # setup name
                    print("save point cloud to:", fname)
                    o3d.io.write_point_cloud(filename=fname, pointcloud=source)
                    stored_point_clouds.append(ns)









