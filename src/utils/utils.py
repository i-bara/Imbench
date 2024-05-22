import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, input, target, weight=None, reduction='mean'):
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class Neighbors:
    def __init__(self, edge_index):
        self.next_neighbors = dict()
        self.prev_neighbors = dict()
        for i in range(edge_index.shape[1]):
            row = edge_index[0][i].item()
            col = edge_index[1][i].item()
            self.add_neighbor(row, col)

    def add_neighbor(self, x, y):
        if x in self.next_neighbors:
            self.next_neighbors[x].append(y)
        else:
            self.next_neighbors[x] = [y]
        if y in self.prev_neighbors:
            self.prev_neighbors[y].append(x)
        else:
            self.prev_neighbors[y] = [x]

    def get_neighbors(self, x):
        if x in self.next_neighbors:
            next_neighbors = self.next_neighbors[x]
        else:
            next_neighbors = []
        if x in self.prev_neighbors:
            prev_neighbors = self.prev_neighbors[x]
        else:
            prev_neighbors = []
        return next_neighbors, prev_neighbors
