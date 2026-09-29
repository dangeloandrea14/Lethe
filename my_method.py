from lethe.unlearners.graph_unlearners.GraphUnlearner import GraphUnlearner
from lethe.core.factory_base import get_instance_kvargs


class MyUnlearner(GraphUnlearner):

    def init(self):
        super().init()
        self.ascent_epochs = self.local.config['parameters']['ascent_epochs']
        self.repair_epochs = self.local.config['parameters']['repair_epochs']
        self.predictor.optimizer = get_instance_kvargs(
            self.local_config['parameters']['optimizer']['class'],
            {'params': self.predictor.model.parameters(),
             **self.local_config['parameters']['optimizer']['parameters']})

    def __unlearn__(self):
        forget_edges = self.dataset.partitions[self.forget_part]
        affected = self.infected_nodes(forget_edges, self.hops)
        repair_nodes = self.dataset.partitions[self.train_part]

        self.predictor.model.train()

        for epoch in range(self.ascent_epochs):
            self.predictor.optimizer.zero_grad()
            loss = -self.task_loss(node_subset=affected)
            loss.backward()
            self.predictor.optimizer.step()
            self.info(f'MyUnlearner ascent epoch={epoch} loss={loss.item():.4f}')

        for epoch in range(self.repair_epochs):
            self.predictor.optimizer.zero_grad()
            loss = self.task_loss(node_subset=repair_nodes)
            loss.backward()
            self.predictor.optimizer.step()
            self.info(f'MyUnlearner repair epoch={epoch} loss={loss.item():.4f}')

        return self.predictor

    def check_configuration(self):
        super().check_configuration()
        p = self.local.config['parameters']
        p['ascent_epochs'] = p.get('ascent_epochs', 3)
        p['repair_epochs'] = p.get('repair_epochs', 5)
        p['optimizer'] = p.get('optimizer', {'class': 'torch.optim.Adam',
                                             'parameters': {'lr': 0.0001}})
