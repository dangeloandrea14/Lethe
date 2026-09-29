import os

import torch
from torch.utils.data import TensorDataset, ConcatDataset, Subset

from lethe.utils.config.global_ctx import Global
from lethe.utils.config.local_ctx import Local
from lethe.data.data_sources.datasource import DataSource
from lethe.data.datasets.Dataset import DatasetWrapper


# torch.serialization.add_safe_globals([TensorDataset, ConcatDataset, Subset])


class TorchFileDataSource(DataSource):
    """ Load Dataset from a Torch file """

    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.path = self.params['path']

    def get_name(self):
        return os.path.basename(self.path)

    def create_data(self) -> DatasetWrapper:
        torch_dataset = torch.load(self.path, weights_only=False)
        return TorchFileDataset(torch_dataset)

    def get_wrapper(self, data):
        return DatasetWrapper(data, self.preprocess)


class TorchFileDataset(DatasetWrapper):
    def get_n_classes(self):
        return self.data.n_classes
