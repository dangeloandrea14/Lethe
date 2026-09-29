import torch
from torch_geometric.data import Data
from torch_geometric.utils import stochastic_blockmodel_graph

from lethe.data.data_sources.datasource import DataSource
from lethe.data.data_sources.TwitchGamersDataSource import SingleGraphDataset
from lethe.data.data_sources.EdgeFileDataSource import GeometricWrapper
from lethe.utils.config.global_ctx import Global
from lethe.utils.config.local_ctx import Local


class SBMDataSource(DataSource):
    """Structure-reliant synthetic dataset based on the Stochastic Block Model."""

    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        p = self.local_config["parameters"]
        self.num_classes     = p.get("num_classes",     2)
        self.nodes_per_class = p.get("nodes_per_class", 500)
        self.p_in            = p.get("p_in",            0.05)
        self.p_out           = p.get("p_out",           0.001)
        self.num_features    = p.get("num_features",    64)
        self.name            = p.get("name",            "SBM")

    def get_name(self):
        return self.name

    def create_data(self):
        k = self.num_classes
        n = self.nodes_per_class

        block_sizes = [n] * k
        edge_probs  = [
            [self.p_in if i == j else self.p_out for j in range(k)]
            for i in range(k)
        ]

        edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)

        # pure noise features
        total_nodes = k * n
        x = torch.randn(total_nodes, self.num_features)

        # Labels = block membership
        y = torch.repeat_interleave(
            torch.arange(k, dtype=torch.long),
            torch.tensor([n] * k, dtype=torch.long)
        )

        data = Data(x=x, edge_index=edge_index, y=y)
        return GeometricWrapper(SingleGraphDataset(data), self.preprocess)

    def get_simple_wrapper(self, data):
        return GeometricWrapper(data, self.preprocess)

    def check_configuration(self):
        super().check_configuration()
