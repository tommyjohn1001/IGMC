import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch.nn import Conv1d, Linear
from torch_geometric.nn import GCNConv, RGCNConv, global_add_pool, global_sort_pool
from torch_geometric.utils import dropout_adj
from torch_scatter import scatter

from util_functions import *


class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    """
    GatedGCN layer
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """

    def __init__(self, in_dim, out_dim, dropout=0.1, residual=True, **kwargs):
        super().__init__(**kwargs)
        self.A = nn.Linear(in_dim, out_dim, bias=True)
        self.B = nn.Linear(in_dim, out_dim, bias=True)
        self.C = nn.Linear(in_dim, out_dim, bias=True)
        self.D = nn.Linear(in_dim, out_dim, bias=True)
        self.E = nn.Linear(in_dim, out_dim, bias=True)

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual
        self.e = None

    def forward(self, x, e, edge_index):
        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        x, e = self.propagate(edge_index, Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce, e=e, Ax=Ax)

        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = F.relu(x)
        e = F.relu(e)

        if self.residual:
            x = x_in + x
            e = e_in + e

        x = F.dropout(x, self.dropout, training=self.training)
        # edge_attr = F.dropout(e, self.dropout, training=self.training)

        return x

    def message(self, Dx_i, Ex_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """

        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)
        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """

        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size, reduce="sum")

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size, reduce="sum")

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """

        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out


class GNN(torch.nn.Module):
    # a base GNN class, GCN message passing + sum_pooling
    def __init__(
        self,
        dataset,
        gconv=GatedGCNLayer,
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
        # for conv in self.convs:
        #     conv.reset_parameters()
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
