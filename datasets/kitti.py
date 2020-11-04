from datasets.data_reader import BaseDataReaderList
from datasets.samplers import generate_random_samples_kitti
from datasets.basedataset import BaseDataset
from datasets.data_loaders import KittiDataLoader

class KittiDataset(BaseDataset):
    def __init__(self, dset_path, parameters, downsampler, preprocessor, training=True,
                 data_type="tensor", correspondence_rate=0.3):
        """
        :param dset_path: Dataset root path
        :param parameters: Dataset configuration dict
        :param downsampler: downsapling function applied on all point sets
        :param preprocessor: additional processing after augmentation
        :param training: Select training/validation sequences
        :param correspondence_rate: the minimum rate of correspondences between pairs for samples to be accepted
        :param sample_views: list of samples
        """
        super().__init__(dset_path, parameters, training)
        loader = KittiDataLoader()
        self.correspondence_rate = correspondence_rate
        self.deterministic = parameters.data_loader_parameters.get("deterministic", False)
        self.sample_specs = generate_random_samples_kitti(parameters.dataset_parameters, self.meta, loader,
                                                          self.deterministic, correspondence_rate=correspondence_rate)

        self.data_reader = BaseDataReaderList(parameters.data_loader_parameters, self.parameters.workspace, loader, downsampler, preprocessor)

    def generate_samples(self):
        self.sample_specs = generate_random_samples_kitti(self.parameters.dataset_parameters, self.meta, self.data_reader.loader,
                                                          self.deterministic, self.correspondence_rate)
        return

    def __getitem__(self, item):
        return self.data_reader.read_sample(self.sample_specs[item])

    def generate_meta(self, dset_path):
        import os
        sequences = os.listdir(dset_path + "/sequences/")
        sequences.sort()
        meta = dict()
        sequences_dict = dict()
        for s in sequences:
            names = [os.path.splitext(f)[0] for f in os.listdir(dset_path + "/sequences/" + s + "/velodyne") if f.endswith('.bin')]
            sequences_dict[s] = sorted(names)

        meta["sequences"] = sequences_dict
        meta["meta_info"] = {"dset_path": dset_path, "dataset": "kitti", "groundtruth": "groundtruth"}
        return meta
