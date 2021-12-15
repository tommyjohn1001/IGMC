import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Linear
from torch_geometric.nn import GCNConv, RGCNConv, global_add_pool, global_sort_pool
from torch_geometric.utils import dropout_adj
from util_functions import *


class GNN(torch.nn.Module):
    # a base GNN class, GCN message passing + sum_pooling
    def __init__(
        self,
        dataset,
        gconv=GCNConv,
        latent_dim=[32, 32, 32, 1],
        regression=False,
        adj_dropout=0.2,
        force_undirected=False,
    ):
        super(GNN, self).__init__()
        self.regression = regression
        self.adj_dropout = adj_dropout
        self.force_undirected = force_undirected
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0]))
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i + 1]))
        self.lin1 = Linear(sum(latent_dim), 128)
        if self.regression:
            self.lin2 = Linear(128, 1)
        else:
            self.lin2 = Linear(128, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index,
                edge_type,
                p=self.adj_dropout,
                force_undirected=self.force_undirected,
                num_nodes=len(x),
                training=self.training,
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        x = global_add_pool(concat_states, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0]
        else:
            return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class DGCNN(GNN):
    # DGCNN from [Zhang et al. AAAI 2018], GCN message passing + SortPooling
    def __init__(
        self,
        dataset,
        gconv=GCNConv,
        latent_dim=[32, 32, 32, 1],
        k=30,
        regression=False,
        adj_dropout=0.2,
        force_undirected=False,
    ):
        super(DGCNN, self).__init__(
            dataset, gconv, latent_dim, regression, adj_dropout, force_undirected
        )
        if k < 1:  # transform percentile to number
            node_nums = sorted([g.num_nodes for g in dataset])
            k = node_nums[int(math.ceil(k * len(node_nums))) - 1]
            k = max(10, k)  # no smaller than 10
        self.k = int(k)
        print("k used in sortpooling is:", self.k)
        conv1d_channels = [16, 32]
        conv1d_activation = nn.ReLU()
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws = [self.total_latent_dim, 5]
        self.conv1d_params1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(self.dense_dim, 128)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d_params1.reset_parameters()
        self.conv1d_params2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index,
                edge_type,
                p=self.adj_dropout,
                force_undirected=self.force_undirected,
                num_nodes=len(x),
                training=self.training,
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        x = global_sort_pool(concat_states, batch, self.k)  # batch * (k*hidden)
        x = x.unsqueeze(1)  # batch * 1 * (k*hidden)
        x = F.relu(self.conv1d_params1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv1d_params2(x))
        x = x.view(len(x), -1)  # flatten
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0]
        else:
            return F.log_softmax(x, dim=-1)


class DGCNN_RS(DGCNN):
    # A DGCNN model using RGCN convolution to take consideration of edge types.
    def __init__(
        self,
        dataset,
        gconv=RGCNConv,
        latent_dim=[32, 32, 32, 1],
        k=30,
        num_relations=5,
        num_bases=2,
        regression=False,
        adj_dropout=0.2,
        force_undirected=False,
    ):
        super(DGCNN_RS, self).__init__(
            dataset,
            GCNConv,
            latent_dim,
            k,
            regression,
            adj_dropout=adj_dropout,
            force_undirected=force_undirected,
        )
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0], num_relations, num_bases))
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i + 1], num_relations, num_bases))

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index,
                edge_type,
                p=self.adj_dropout,
                force_undirected=self.force_undirected,
                num_nodes=len(x),
                training=self.training,
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index, edge_type))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        x = global_sort_pool(concat_states, batch, self.k)  # batch * (k*hidden)
        x = x.unsqueeze(1)  # batch * 1 * (k*hidden)
        x = F.relu(self.conv1d_params1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv1d_params2(x))
        x = x.view(len(x), -1)  # flatten
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0]
        else:
            return F.log_softmax(x, dim=-1)


class Queue:
    def __init__(self) -> None:
        self.queue = list()

    def enqueue(self, item):
        self.queue.append(item)

    def remove_last(self):
        del self.queue[-1]["accumulated"]
        del self.queue[-1]

    def get_last(self):
        if len(self.queue) == 0:
            return None

        return self.queue[-1]


class CustomContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()

        self.temperature = temperature

    def forward(self, rate_embds, pred, ys):
        # rate_embds: [n, 128]
        # pred: [bz, 128]
        # ys: [bz]

        n = len(rate_embds)

        dot_result = pred @ rate_embds.T / self.temperature
        # for numerical stability
        logits_max, _ = torch.max(dot_result, dim=1, keepdim=True)
        logits = dot_result - logits_max.detach()
        logits = torch.exp(logits)
        # [bz, n]

        onehots = F.one_hot(ys, num_classes=n)
        # [bz, n]

        positive = (logits * onehots).sum(-1)
        # [bz]
        negatives = (logits * (1 - onehots)).sum(-1)

        loss = -torch.log(positive / negatives)
        # [bz]
        loss = torch.mean(loss)

        return loss
