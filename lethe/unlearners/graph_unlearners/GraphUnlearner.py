from abc import ABCMeta, abstractmethod
import copy
import torch
from lethe.core.base import Configurable
from lethe.data.datasets.Dataset import DatasetWrapper
from lethe.utils.config.global_ctx import Global
from lethe.utils.config.local_ctx import Local
from lethe.unlearners.torchunlearner import TorchUnlearner
from lethe.utils.graph_ops import khop_infected, edge_endpoints


class GraphUnlearner(TorchUnlearner):

    def __init__(self, global_ctx: Global, local_ctx):
        super().__init__(global_ctx, local_ctx)  

        self.hops = len(self.predictor.model.hidden_channels) + 1
        self.removal_type = self.global_ctx.removal_type

        og_graph =  self.dataset.partitions['all'] 
        
        self.x = og_graph[0][0].x.to(self.device).float()
        self.edge_index = og_graph[0][0].edge_index.to(self.device).long()
        self.labels = og_graph[0][1].to(self.device).long()
        self.labels = torch.tensor(self.labels)
        self.forget_part = self.local.config['parameters']['forget_part']
        self.retain_part = self.local.config['parameters']['retain_part']
        self.train_part = self.local.config['parameters']['train_part']

        self.is_link_prediction = hasattr(self.predictor, 'task_loss')
        if self.is_link_prediction:
            _, _, mp_ei, _ = self.predictor.lp_context()
            self.edge_index = mp_ei.to(self.device).long()

    def task_loss(self, node_subset=None, edge_subset=None):
        """The training objective of the predictor's task, over a subset of samples."""
        if self.is_link_prediction:
            return self.predictor.task_loss(node_subset=node_subset,
                                            edge_subset=edge_subset)

        pred = self.predictor.model(self.x, self.edge_index)[node_subset]
        return self.predictor.loss_fn(pred, self.labels[node_subset])


    def infected_nodes(self, edges_to_forget, hops):
        graph = self.dataset.partitions['all'][0][0]
        return khop_infected(graph.edge_index,
                             edge_endpoints(edges_to_forget),
                             hops, graph.num_nodes)
    

    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['forget_part'] = self.local.config['parameters'].get('forget_part','forget')
        self.local.config['parameters']['retain_part'] = self.local.config['parameters'].get('retain_part','retain')
        self.local.config['parameters']['train_part'] = self.local.config['parameters'].get('train_part','train')

