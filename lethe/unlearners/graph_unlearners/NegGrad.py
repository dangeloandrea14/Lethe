from lethe.unlearners.graph_unlearners.GraphUnlearner import GraphUnlearner
from fractions import Fraction
import torch
from lethe.core.factory_base import get_instance_kvargs

class NegGrad(GraphUnlearner):
    def init(self):
        """Initializes the NegGrad class with global and local contexts."""

        super().init()

        self.epochs = self.local.config['parameters']['epochs']  
        self.ref_data = self.local.config['parameters']['ref_data'] 
        self.predictor.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.predictor.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})


    def __unlearn__(self):

        self.info(f'Starting NegGrad with {self.epochs} epochs')

        forget_edges = self.dataset.partitions[self.ref_data]
        forget_set = forget_edges

        if self.removal_type == 'edge':
            forget_set = self.infected_nodes(forget_edges, self.hops)
        else:
            forget_edges = None

        for epoch in range(self.epochs):
            losses = []
            self.predictor.model.train()


            self.predictor.optimizer.zero_grad() 

            loss = -self.task_loss(node_subset=forget_set, edge_subset=forget_edges)

            losses.append(loss.to('cpu').detach().numpy())

            loss.backward()

            self.predictor.optimizer.step()

            epoch_loss = sum(losses) / len(losses)
            self.info(f'NegGrad - epoch = {epoch} ---> var_loss = {epoch_loss:.4f}')

            self.predictor.lr_scheduler.step()
        
        return self.predictor
    

    
    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 5)  # Default 5 epoch
        self.local.config['parameters']['ref_data'] = self.local.config['parameters'].get("ref_data", 'forget')  # Default reference data is forget
        self.local.config['parameters']['optimizer'] = self.local.config['parameters'].get("optimizer", {'class':'torch.optim.Adam', 'parameters':{}})  # Default optimizer is Adam