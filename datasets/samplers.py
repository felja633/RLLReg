import random
from easydict import EasyDict as edict
import numpy as np
from lib.utils import estimate_overlap
import torch

def generate_random_samples(parameters, meta, deterministic=False):
    # num samples
    # num views per sample
    # Metadata: sequence name, num views and view names
    if deterministic:
        random.seed(a=0)

    sequences = meta["sequences"]
    if len(parameters.sequences) > 0:
        sequences = list(set(sequences).intersection(parameters.sequences))

    sample_specs = []
    num_samples_per_sequence = int(round(float(parameters.num_samples)/float(len(sequences))))
    for s in sequences:
        for count in range(num_samples_per_sequence):
            views = sequences[s]
            tot_num_views = len(views)
            nums = [x for x in range(tot_num_views)]
            random.shuffle(nums)
            sample_views = []
            if parameters.num_views == 0:
                # use all (only during evaluation)
                sample_views = views
            else:
                sample_views = [views[idx] for idx in nums[0:parameters.num_views]]

            sample = edict()
            sample["view_names"] = sample_views
            sample["sequence"] = s
            sample["with_features"] = parameters.with_features
            sample["meta_info"] = meta["meta_info"]
            sample_specs.append(sample)

    return sample_specs

def generate_random_samples_kitti(parameters, meta, loader, deterministic=False, correspondence_rate=0.0):
    print("generate samples...")
    if deterministic:
        random.seed(a=0)

    sequences = meta["sequences"]
    sequences_filt = dict()
    if len(parameters.sequences) > 0:
        sequences_keys = list(set(sequences).intersection(parameters.sequences))
        sequences_keys.sort()
        for s in sequences_keys:
            sequences_filt[s] = sequences[s]

        sequences = sequences_filt

    sample_specs = []
    num_samples_per_sequence = max(1,int(round(float(parameters.num_samples) / float(len(sequences)))))
    for s in sequences:
        poses = loader.get_poses(meta["meta_info"]["dset_path"], s)
        for count in range(num_samples_per_sequence):
            views = sequences[s]
            tot_num_views = len(views)
            nums = [x for x in range(tot_num_views)]
            random.shuffle(nums)
            sample_views = []
            if parameters.num_views == 0:
                # use all (only during evaluation)
                sample_views = views
            else:
                center_view = nums[0]
                pc = poses[center_view][0:3, 3:]
                if correspondence_rate > 0.0:
                    tmp_dict = {"meta_info": meta["meta_info"], "sequence": s}
                    pc_c = loader(center_view, tmp_dict)

                min_view = max(center_view-parameters.range_limit,0)
                max_view = min(center_view+parameters.range_limit, tot_num_views-1)
                range_list = [n for n in range(min_view, max_view)]
                random.shuffle(range_list)
                sample_views = [center_view,]
                cnt=1
                for idx in range_list:
                    if cnt >= parameters.num_views:
                        break

                    if idx != center_view:
                        ps = poses[idx][0:3, 3:]
                        d = torch.sqrt(torch.sum((pc-ps)**2))
                        if d < parameters.dist_threshold:
                            if correspondence_rate > 0.0:
                                tmp_dict = {"meta_info": meta["meta_info"], "sequence": s}
                                pc_i = loader(idx, tmp_dict)
                                rate = estimate_overlap(pc_c["coordinates"], pc_i["coordinates"],
                                                        pc_c["R_gt"], pc_i["R_gt"],
                                                        pc_c["t_gt"], pc_i["t_gt"], voxel_size=2.0,
                                                        device="cuda:0")

                                if rate[0] >= correspondence_rate:
                                    sample_views.append(idx)
                                    cnt = cnt + 1
                            else:
                                sample_views.append(idx)
                                cnt=cnt+1

                sample_views = [views[idx] for idx in sample_views]


            sample = edict()
            sample["view_names"] = sample_views
            sample["sequence"] = s
            sample["with_features"] = parameters.with_features
            sample["meta_info"] = meta["meta_info"]
            sample_specs.append(sample)

    return sample_specs

def generate_random_samples_threedmatch(parameters, meta, loader, deterministic=False,
                                        correspondence_rate=0.0, view_list=None):
    print("generate samples...")
    if deterministic:
        random.seed(a=22)

    sequences = meta["sequences"]
    basedir = meta["meta_info"]["dset_path"]
    sequences_filt = dict()
    if len(parameters.sequences) > 0:
        sequences_keys = list(set(sequences.keys()).intersection(parameters.sequences))
        sequences_keys.sort()
        for s in sequences_keys:
            sequences_filt[s] = sequences[s]

        sequences = sequences_filt

    if parameters.get("val_sequences", False):
        sequences_filt = dict()
        for s in sequences.keys():
            if s not in parameters.val_sequences:
                sequences_filt[s] = sequences[s]

        sequences = sequences_filt

    sample_specs = []
    tot_count=0
    num_samples_per_sequence = max(1,int(round(float(parameters.num_samples) / float(len(sequences)))))
    for s in sequences.keys():
        for count in range(num_samples_per_sequence):

            view_dicts = sequences[s]
            if view_list is None:
                names = sorted([n for n in view_dicts.keys()])
                tot_num_views = len([k for k in view_dicts.keys()])
                nums = [x for x in range(tot_num_views)]
                random.shuffle(nums)
                sample_views = []
                if parameters.num_views == 0:
                    # use all (only during evaluation)
                    sample_views = names
                else:
                    center_view = nums[0]
                    tmp_dict = {"meta_info": meta["meta_info"], "sample_dict": view_dicts}
                    c_pose = loader.get_pose(tmp_dict, names[center_view])
                    pc = c_pose["pos"]
                    if correspondence_rate > 0.0:
                        pc_c = loader(names[center_view], tmp_dict)

                    min_view = max(center_view-parameters.range_limit,0)
                    max_view = min(center_view+parameters.range_limit, tot_num_views-1)
                    range_list = [n for n in range(min_view, max_view)]
                    random.shuffle(range_list)
                    sample_views = [center_view,]
                    cnt=1
                    for idx in range_list:
                        if cnt >= parameters.num_views:
                            break

                        if idx != center_view:
                            tmp_dict = {"meta_info": meta["meta_info"], "sample_dict": view_dicts}
                            i_pose = loader.get_pose(tmp_dict, names[idx])
                            ps = i_pose["pos"]

                            d = torch.sqrt(torch.sum((pc-ps)**2))
                            if d < parameters.dist_threshold:
                                if correspondence_rate > 0.0:
                                    tmp_dict = {"meta_info": meta["meta_info"], "sample_dict": view_dicts}
                                    pc_i = loader(names[idx], tmp_dict)
                                    rate = estimate_overlap(pc_c["coordinates"], pc_i["coordinates"],
                                                            c_pose["R_gt"], i_pose["R_gt"],
                                                            c_pose["t_gt"], i_pose["t_gt"],
                                                            voxel_size=0.2, device="cuda:0")
                                    if rate[0] >= correspondence_rate:
                                        sample_views.append(idx)
                                        cnt = cnt + 1
                                else:
                                    sample_views.append(idx)
                                    cnt = cnt + 1

                    sample_views = [names[idx] for idx in sample_views]
            else:
                sample_views = view_list[tot_count]

            tot_count+=1

            if len(sample_views) == parameters.num_views:
                samle_dict = dict()
                for n in sample_views:
                    samle_dict[n] = view_dicts[n]

                sample = edict()
                sample["view_names"] = sample_views
                sample["sample_dict"] = samle_dict
                sample["sequence"] = s
                sample["with_features"] = parameters.with_features
                sample["meta_info"] = meta["meta_info"]
                sample_specs.append(sample)

    return sample_specs

# for evaluation only
def generate_sequential_samples(parameters, meta):
    sequences = meta["sequences"]
    num_init_frames = parameters.num_init_frames

    sample_specs = []

    overlap = parameters.overlap
    num_views_per_sample = parameters.num_views
    stride = num_views_per_sample - overlap
    for s in (s for s in sequences.keys() if s in parameters.sequences):
        views = sequences[s]
        tot_num_views = len(views)
        num_samples = min(parameters.num_samples, (tot_num_views-num_init_frames)//stride)
        for count in range(num_samples):
            start_frame = (num_init_frames+count*stride)
            sample_views = views[start_frame:start_frame+num_views_per_sample]

            sample = edict()
            sample["view_names"] = sample_views
            sample["sequence"] = s
            sample["with_features"] = parameters.with_features
            sample["meta_info"] = meta["meta_info"]
            sample_specs.append(sample)

    return sample_specs