from lib.utils import random_sampling, farthest_point_sampling
from torch_geometric.nn import fps
from lib.utils import convert_data_to_batch

class NoProcessing:
    def __init__(self, remove_zeros=True, coordinates_key="coordinates"):
        self.remove_zeros = remove_zeros
        self.coordinates_key = coordinates_key
    def __call__(self, data, with_features=False, remove_zeros=True):
        if self.remove_zeros:
            mask = data[self.coordinates_key].norm(dim=-2) > 0.0001
            data[self.coordinates_key] = data[self.coordinates_key][..., mask]
            if with_features:
                data['features'] = data['features'][:, mask]
            if "time_stamps" in data.keys():
                data['time_stamps'] = data['time_stamps'][:, mask]

        return data

class RandomDownsampler:
    def __init__(self, num_points, remove_origin=True, coordinates_key="coordinates", remove_zeros=True):
        self.num_points = num_points
        self.remove_origin = remove_origin
        self.coordinates_key = coordinates_key
        self.remove_zeros = remove_zeros

    def __call__(self, data, with_features=False):

        return self.downsample(data, with_features)

    def downsample(self, data, with_features):
        if self.remove_zeros:
            mask = data[self.coordinates_key].norm(dim=-2) > 0.0001
            data[self.coordinates_key] = data[self.coordinates_key][...,mask]
            if with_features:
                data['features'] = data['features'][:,mask]
            if "time_stamps" in data.keys():
                data['time_stamps'] = data['time_stamps'][:,mask]

        data[self.coordinates_key], inds = random_sampling(data[self.coordinates_key], self.num_points)
        if with_features:
            data['features'] = data['features'][:,inds]

        if "time_stamps" in data.keys():
            data['time_stamps'] = data['time_stamps'][:,inds]

        data['ds_inds'] = inds
        return data

class RandomDownsamplerRate:
    def __init__(self, rate, remove_origin=True, coordinates_key="coordinates", remove_zeros=True):
        self.rate = rate
        self.remove_origin = remove_origin
        self.coordinates_key = coordinates_key
        self.remove_zeros = remove_zeros

    def __call__(self, data, with_features=False):
        return self.downsample(data, with_features)

    def downsample(self, data, with_features):
        if self.remove_zeros:
            mask = data[self.coordinates_key].norm(dim=-2) > 0.0001
            data[self.coordinates_key] = data[self.coordinates_key][...,mask]
            if with_features:
                data['features'] = data['features'][:,mask]
            if "time_stamps" in data.keys():
                data['time_stamps'] = data['time_stamps'][:,mask]

        num_points = int(data[self.coordinates_key].shape[-1]*self.rate)

        data[self.coordinates_key], inds = random_sampling(data[self.coordinates_key], num_points)
        if with_features:
            data['features'] = data['features'][:,inds]

        if "time_stamps" in data.keys():
            data['time_stamps'] = data['time_stamps'][:,inds]

        data['ds_inds'] = inds
        return data

class FarthestPointDownsampler:
    def __init__(self, num_points, coordinates_key="coordinates", remove_zeros=True):
        self.num_points = num_points
        self.coordinates_key = coordinates_key
        self.remove_zeros = remove_zeros

    def __call__(self, data, with_features=False):
        return self.downsample(data, with_features)

    def downsample(self, data, with_features):
        if self.remove_zeros:
            mask = data[self.coordinates_key].norm(dim=-2) > 0.0001
            data[self.coordinates_key] = data[self.coordinates_key][...,mask]
            if with_features:
                data['features'] = data['features'][:,mask]
            if "time_stamps" in data.keys():
                data['time_stamps'] = data['time_stamps'][:,mask]
        data[self.coordinates_key], inds = farthest_point_sampling(data[self.coordinates_key], self.num_points)
        if with_features:
            data['features'] = data['features'][:,inds]

        if "time_stamps" in data.keys():
            data['time_stamps'] = data['time_stamps'][:,inds]

        data['ds_inds'] = inds
        return data

class FarthestPointDownsamplerTorchGeo:
    def __init__(self, num_points, device="cpu", coordinates_key="coordinates", remove_zeros=True):
        self.num_points = num_points
        self.device = device
        self.coordinates_key = coordinates_key
        self.remove_zeros = remove_zeros

    def __call__(self, data, with_features=False):
        return self.downsample(data, with_features)

    def downsample(self, data, with_features):
        if self.remove_zeros:
            mask = data[self.coordinates_key].norm(dim=-2) > 0.0001
            data[self.coordinates_key] = data[self.coordinates_key][...,mask]
            if with_features:
                data['features'] = data['features'][:,mask]
            if "time_stamps" in data.keys():
                data['time_stamps'] = data['time_stamps'][:,mask]

        coords = data[self.coordinates_key].to(self.device)
        #pcds = coords.view(coords.shape[0] * coords.shape[1], *coords.shape[2:]).permute(0,2,1)
        b = convert_data_to_batch(coords.permute(1,0).unsqueeze(dim=0))
        ratio = float(self.num_points+1) / coords.shape[-1]
        inds = fps(b.pos, batch=b.batch.to(b.pos.device), ratio=ratio)
        if inds.shape[0]>self.num_points:
            inds = inds[:self.num_points]

        data[self.coordinates_key] = data[self.coordinates_key][:,inds.to(data[self.coordinates_key].device)]
        if with_features:
            data['features'] = data['features'][:,inds.to(data['features'].device)]

        if "time_stamps" in data.keys():
            data['time_stamps'] = data['time_stamps'][:,inds.to(data['features'].device)]

        data['ds_inds'] = inds
        return data
