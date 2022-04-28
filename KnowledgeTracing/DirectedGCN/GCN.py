import torch.nn as nn
import torch.nn.functional as F
from KnowledgeTracing.DirectedGCN.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)


    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x2 = F.relu(self.gc2(x1, adj))
        return x2
