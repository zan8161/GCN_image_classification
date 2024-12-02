# This model is designed as a JK-Net like model, inspired by
# https://arxiv.org/abs/1811.01287
""" Towards Sparse Hierarchical Graph Classifiers """
""" Catalina Cangea, Petar Velickovic, Nikola Jovanovic, Thomas Kipf, Pietro Liò """




import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, BatchNorm, TopKPooling, MLP, global_mean_pool, global_max_pool
from torch.nn import ReLU
class MyModel(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(MyModel, self).__init__()
        torch.manual_seed(8161)
        self.conv1 = GCNConv(in_dim, 64)
        self.bn1 = BatchNorm(64)
        self.relu1 = ReLU()
        self.pool1 = TopKPooling(64, ratio=0.8)

        self.conv2 = GCNConv(64, 64)
        self.bn2 = BatchNorm(64)
        self.relu2 = ReLU()
        self.pool2 = TopKPooling(64, ratio=0.8)

        self.conv3 = GCNConv(64, 64)
        self.bn3 = BatchNorm(64)
        self.relu3 = ReLU()
        self.pool3 = TopKPooling(64, ratio=0.8)

        self.mlp = MLP([(64 * 2), out_dim])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Convolution layer + pooling layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu1(x)
        x, edge_index, _, batch, _, score1 = self.pool1(
            x, edge_index, None, batch, None
        )

        # concatenate the mean vector and the max vector
        readout1 = torch.cat(
            [global_mean_pool(x=x, batch=batch),
             global_max_pool(x=x, batch=batch)],
            dim=1,
        )

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu2(x)
        x, edge_index, _, batch, _, score2 = self.pool2(
            x, edge_index, None, batch, None
        )

        readout2 = torch.cat(
            [global_mean_pool(x=x, batch=batch),
             global_max_pool(x=x, batch=batch)],
            dim=1,
        )

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.relu3(x)
        x, edge_index, _, batch, _, score3 = self.pool3(
            x, edge_index, None, batch, None
        )

        readout3 = torch.cat(
            [global_mean_pool(x=x, batch=batch),
             global_max_pool(x=x, batch=batch)],
            dim=1,
        )

        # readout layer
        readout = readout1 + readout2 + readout3

        # fully connected network
        readout = self.mlp(readout)

        return readout
