import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.utils import dropout_adj

from layers import *
from util_functions import *


class IGMC(GNN):
    # The GNN model of Inductive Graph-based Matrix Completion.
    # Use RGCN convolution + center-nodes readout.
    def __init__(
        self,
        dataset,
        gconv=GatedGCNLayer,
        latent_dim=[32, 32, 32, 32],
        num_relations=5,
        num_bases=2,
        regression=False,
        adj_dropout=0.2,
        force_undirected=False,
        side_features=False,
        n_side_features=0,
        multiply_by=1,
        pe_dim=20,
        n_nodes=3000,
    ):
        super(IGMC, self).__init__(
            dataset, gconv, latent_dim, regression, adj_dropout, force_undirected
        )
        self.multiply_by = multiply_by

        self.edge_embd = nn.Embedding(num_relations, latent_dim[0])

        self.convs = torch.nn.ModuleList()

        # NOTE: If for the 3rd scenario, use the followings instead of the above
        self.node_feat_dim = dataset.num_features - pe_dim - 1
        self.lin_node_feat = nn.Linear(self.node_feat_dim, latent_dim[0])
        # self.lin_pe = nn.Linear(pe_dim, 64)
        # self.node_embds = nn.Embedding(n_nodes, 64)
        # self.ff1 = nn.Sequential(
        #     nn.Linear(64 * 3, 64 * 3), nn.Dropout(0.2), nn.Tanh(), nn.Linear(64 * 3, latent_dim[0])
        # )
        self.convs.append(gconv(latent_dim[0], latent_dim[0]))
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i + 1]))
        self.lin1 = Linear(2 * sum(latent_dim), 128)
        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(2 * sum(latent_dim) + n_side_features, 128)

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

        edge_embd = self.edge_embd(edge_type)

        # NOTE: If for the 3rd scenario, enable the followings
        node_indx, node_subgraph_feat, pe = (
            x[:, :1].long(),
            x[:, 1 : self.node_feat_dim + 1],
            x[:, self.node_feat_dim + 1 :],
        )
        node_subgraph_feat = self.lin_node_feat(node_subgraph_feat)
        # pe = self.lin_pe(pe)
        # node_global_feat = self.node_embds(node_indx)
        # x = torch.cat((node_subgraph_feat, pe, node_global_feat.squeeze(1)), dim=-1)
        # x = self.ff1(x)

        x = node_subgraph_feat

        concat_states = []
        for conv in self.convs:
            x = conv(x, edge_embd, edge_index)
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        users = data.x[:, 1] == 1
        items = data.x[:, 2] == 1
        x = torch.cat([concat_states[users], concat_states[items]], 1)
        if self.side_features:
            x = torch.cat([x, data.u_feature, data.v_feature], 1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0] * self.multiply_by
        else:
            return F.log_softmax(x, dim=-1)
