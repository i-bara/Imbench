import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)


class GCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias=False, batch_norm=False):
        super(GCNConv, self).__init__()
        self.weight = torch.Tensor(in_channels, out_channels)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_channels)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_channels) if batch_norm else None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, batch_norm=False):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(edge_index, support)
        if self.bias is not None:
            output = output + self.bias
        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)
        return output


    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())
