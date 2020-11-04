import torch
from lib.utils import get_rotation_matrix, get_translation_vector
from scipy.spatial import KDTree
from lib.tensorlist import TensorList, TensorListList

class BaseDataReaderList:
    def __init__(self, parameters, workspace, loader, downsampler, preprocesser):
        self.parameters = parameters
        self.workspace = workspace
        self.downsampler = downsampler
        self.preprocesser = preprocesser
        self.loader = loader

        if self.parameters.get("deterministic", False):
            torch.manual_seed(0)

    def read_sample(self, specs):
        coords = []
        R_gt = []
        t_gt = []
        R_init = []
        t_init = []
        time_stamps = []
        names = []

        if self.parameters.with_features:
            features = []

        for name in specs.view_names:
            names.append(specs["sequence"] + "/" + name)

            data = self.loader(name, specs)
            data = self.downsampler(data, self.parameters.with_features)
            data = self.preprocesser(data, self.parameters.with_features)

            if self.parameters.augment == True:
                ## generate random transformation as ground truths
                r = torch.rand(1)
                ax = torch.rand(3)
                direction = torch.rand(3)
                t_scale = torch.rand(1)
                R = get_rotation_matrix(self.parameters.ang_range, r=r, ax=ax)
                t = get_translation_vector(self.parameters.t_range*t_scale, direction=direction)
                # transform coordinates
                data['coordinates'] = R.t() @ data['coordinates'] - R.t() @ t  # transform with inverse augmented ground truth
                R_gt.append(data['R_gt'] @ R)
                t_gt.append(data['R_gt'] @ t + data['t_gt'])
            else:
                R_gt.append(data['R_gt'])
                t_gt.append(data['t_gt'])

            R_init.append(torch.eye(3, 3))
            t_init.append(torch.zeros(3, 1))

            coords.append(data['coordinates'])
            if self.parameters.with_features:
                features.append(data['features'])

            if "time_stamps" in data.keys():
                time_stamps.append(data["time_stamps"])

        out = {'coordinates': TensorList(coords)}

        info = {'R_gt': TensorList(R_gt),
                't_gt': TensorList(t_gt),
                'R_init': TensorList(R_init),
                't_init': TensorList(t_init),
                'sequence': specs.sequence,
                'names': names}

        if self.parameters.with_features:
            out['features'] = TensorList(features)

        if self.parameters.with_correspondences:
            info['correspondences'] = self.extract_correspondences(out, info)

        if len(time_stamps) > 0:
            out['time_stamps'] = TensorList(time_stamps)

        return out, info

    def extract_correspondences(self, out, info):
        coords = out['coordinates']
        Rs = info['R_gt']
        ts = info['t_gt']

        coords = Rs @ coords + ts

        M = len(coords)
        ind_nns = []
        dists = []
        for ind1 in range(M - 1):
            for ind2 in range(ind1 + 1, M):
                point1 = coords[ind1].cpu().numpy().T
                point2 = coords[ind2].cpu().numpy().T
                tree = KDTree(point1)
                d, inds_nn1 = tree.query(point2, k=1)
                inds_nn2 = torch.tensor([n for n in range(0, point2.shape[0])])
                inds_nn1 = torch.from_numpy(inds_nn1)

                ind_nns.append(torch.stack([inds_nn1, inds_nn2], dim=0))
                dists.append(torch.from_numpy(d))

        return dict(indices=TensorList(ind_nns).to(coords[0][0].device), distances=TensorList(dists).to(coords[0][0].device))


def collate_tensorlist(batch):
    out = dict()
    info = dict()
    tensorlistkeysout = []
    tensorlistkeysinfo = []
    for b in batch:
        cb, ib = b
        for k, v in cb.items():
            if isinstance(v, TensorList):
                if not k in out:
                    out[k] = TensorList()
                    tensorlistkeysout.append(k)
            else:
                if not k in out:
                    out[k] = []

            out[k].append(v)

        for k, v in ib.items():
            if isinstance(v, TensorList):
                if not k in info:
                    info[k] = TensorList()
                    tensorlistkeysinfo.append(k)
            else:
                if not k in info:
                    info[k] = []

            info[k].append(v)

    for k in tensorlistkeysout:
        out[k] = TensorListList(out[k])

    for k in tensorlistkeysinfo:
        info[k] = TensorListList(info[k])

    return out, info