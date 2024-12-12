from .gnnv3 import gnn
from torch.nn import functional as F


class Reweight(gnn):
    def loss(self, output, y):
        return F.cross_entropy(output, y, weight=self.reweight('train'), reduction='none')
    