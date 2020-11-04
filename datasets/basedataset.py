from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, dset_path, parameters, training=True):
        """
        :param dset_path: Dataset root path
        :param sampler: (tuple) name of sampler method in DatasetMeta, sampler configuration dict
        :param parameters: Dataset configuration dict
        :param training: Select training/validation sequences
        """
        super().__init__()
        self.training = training
        self.parameters = parameters
        self.meta = self.generate_meta(dset_path)



    def generate_meta(self, dset_path):
        return NotImplemented

    def __len__(self):
        return len(self.sample_specs)

    def __getitem__(self, item):
        return NotImplemented
