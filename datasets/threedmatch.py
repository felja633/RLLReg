from datasets.data_reader import BaseDataReaderList
from datasets.samplers import generate_random_samples_threedmatch
from datasets.basedataset import BaseDataset
from datasets.data_loaders import ThreeDMatchDataLoader
import os

class ThreeDMatchDataset(BaseDataset):
    def __init__(self, dset_path, parameters, downsampler, preprocessor, training=True,
                 correspondence_rate=0.3, sample_views=None):
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
        loader = ThreeDMatchDataLoader()
        self.correspondence_rate = correspondence_rate
        self.deterministic = parameters.data_loader_parameters.get("deterministic", False)
        self.sample_views=sample_views
        self.sample_specs = generate_random_samples_threedmatch(parameters.dataset_parameters, self.meta, loader,
                                                                self.deterministic, correspondence_rate=correspondence_rate, view_list=sample_views)

        self.data_reader = BaseDataReaderList(parameters.data_loader_parameters, self.parameters.workspace, loader, downsampler, preprocessor)

    def generate_samples(self):
        self.sample_specs = generate_random_samples_threedmatch(self.parameters.dataset_parameters, self.meta, self.data_reader.loader,
                                                                self.deterministic, self.correspondence_rate, view_list=self.sample_views)
        return

    def __getitem__(self, item):
        return self.data_reader.read_sample(self.sample_specs[item])

    def generate_meta(self, dset_path):

        if self.training:
            filename= dset_path + "/train.txt"
        else:
            filename=dset_path + "/test.txt"

        with open(filename) as f:
            content = f.readlines()
        sequences = [x.strip() for x in content]
        sequences.sort()

        meta = dict()
        sequences_dict = dict()
        for s in sequences:
            sub_seq = [f for f in os.listdir(dset_path + "/" + s) if not
                     f.endswith('.txt')]
            sub_seq.sort()
            for ss in sub_seq:
                sub_sequence_dict = dict()
                names = [f.split('.')[0] for f in os.listdir(dset_path + "/" + s + "/" + ss) if f.endswith('depth.png')]

                for n in names:
                    frame_dict = dict(depth=s + "/" + ss + "/"+n+".depth.png", color=s + "/" + ss + "/"+n+".color.png",
                                      pose=s + "/" + ss + "/"+n+".pose.txt", intrinsics=s+"/camera-intrinsics.txt")
                    sub_sequence_dict[s + "/" + ss + "/" + n] = frame_dict

                sequences_dict[s + "/" + ss] = sub_sequence_dict

        meta["sequences"] = sequences_dict
        meta["meta_info"] = {"dset_path": dset_path, "dataset": "threedmatch", "groundtruth": "groundtruth"}
        return meta
