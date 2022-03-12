import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
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
        latent_dim=None,
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
        scenario=1,
    ):
        super(IGMC, self).__init__(
            dataset, gconv, latent_dim, regression, adj_dropout, force_undirected
        )

        if latent_dim is None:
            latent_dim = [32, 32, 32, 32]

        self.multiply_by = multiply_by
        self.scenario = scenario

        ## Declare modules to convert edge index to hidden edge vector
        self.edge_embd = nn.Linear(1, latent_dim[0])

        ## Declare modules to convert node feat, pe to hidden vectors
        self.node_feat_dim = dataset.num_features - pe_dim - 1
        self.lin_node_feat = nn.Linear(self.node_feat_dim + pe_dim, latent_dim[0])

        ## Declare GNN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(latent_dim[0], latent_dim[0]))
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i + 1]))

        self.lin1 = Linear(2 * sum(latent_dim), 128)
        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(2 * sum(latent_dim) + n_side_features, 128)

        self.graphsizenorm = pyg_nn.norm.graph_size_norm.GraphSizeNorm()

        ## init weights
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, GatedGCNLayer):
                m.initialize_weights()

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

        edge_type = edge_type.unsqueeze(-1).float()
        edge_embd = self.edge_embd(edge_type)

        ## Extract node feature, RWPE info and global node index
        node_indx, node_subgraph_feat, pe = (
            x[:, :1].long(),
            x[:, 1 : self.node_feat_dim + 1],
            x[:, self.node_feat_dim + 1 :],
        )

        ## Convert node feat, pe to suitable dim before passing thu GNN layers
        x = torch.cat((node_subgraph_feat, pe), -1)
        x = self.lin_node_feat(x)

        ## Apply graph size norm
        if self.scenario in [3, 4]:
            x = self.graphsizenorm(x, batch)

        ## Pass node feat thru GNN layers
        concat_states = []
        for conv in self.convs:
            x, edge_embd = conv(x, edge_embd, edge_index)
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
