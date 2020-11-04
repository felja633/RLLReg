from pathlib import Path
import torch
from torch.utils.data import Dataset
from datasets.data_reader import BaseDataReaderList
from datasets.samplers import generate_random_samples
from datasets.processing import NoProcessing
from datasets.basedataset import BaseDataset
from datasets.data_loaders import ETHDataLoader

class ETHDataset(BaseDataset):
    def __init__(self, dset_path, parameters, downsampler, preprocessor, training=True):
        """
        :param dset_path: Dataset root path
        :param sampler: (tuple) name of sampler method in DatasetMeta, sampler configuration dict
        :param parameters: Dataset configuration dict
        :param training: Select training/validation sequences
        """
        super().__init__(dset_path, parameters, training)
        loader = ETHDataLoader()
        self.sample_specs = generate_random_samples(parameters.dataset_parameters, self.meta)
        self.data_reader = BaseDataReaderList(parameters.data_loader_parameters, self.parameters.workspace, loader, downsampler, preprocessor)

    def generate_meta(self, dset_path):
        import os
        sequences = os.listdir(dset_path)
        meta = dict()
        sequences_dict = dict()
        for s in sequences:
            names = [os.path.splitext(f)[0] for f in os.listdir(dset_path + "/" + s) if f.endswith('.ply')]
            sequences_dict[s] = names

        meta["sequences"] = sequences_dict
        meta["meta_info"] = {"dset_path": dset_path, "dataset": "eth", "groundtruth": "groundtruth"}
        return meta

    def __len__(self):
        return len(self.sample_specs)

    def __getitem__(self, item):
        return self.data_reader.read_sample(self.sample_specs[item])
