import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.utils import dropout_adj

from layers import *
from util_functions import *
from hypermixer import HyperMixerLayer

class IGMC(GNN):
    # The GNN model of Inductive Graph-based Matrix Completion.
    # Use RGCN convolution + center-nodes readout.
    def __init__(
        self,
        dataset,
        gconv=None,
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
        class_values=None,
        scenario=1,
    ):
        # gconv = GatedGCNLayer GatedGCNLSPELayer RGatedGCNLayer RGCNConv
        if scenario in [1, 2, 3, 4, 5, 6, 7, 8]:
            # gconv = RGCNConvLSPE GatedGCNLSPELayer
            gconv = OldRGCNConvLSPE
        elif scenario in [9, 10, 11, 12, 13, 14, 15, 16]:
            gconv = GatedGCNLSPELayer
        else:
            raise NotImplementedError()

        super(IGMC, self).__init__(
            dataset,
            gconv,
            latent_dim,
            regression,
            adj_dropout,
            force_undirected,
            num_relations,
            num_bases,
        )

        if latent_dim is None:
            latent_dim = [32, 32, 32, 32]

        self.multiply_by = multiply_by
        self.scenario = scenario
        self.class_values = class_values

        ## Declare modules to convert node feat, pe to hidden vectors
        self.node_feat_dim = dataset.num_features - pe_dim - 1
        if self.scenario in [1, 2, 3, 4, 5, 6, 7, 8]:
            self.lin_pe = nn.Linear(pe_dim, self.node_feat_dim)
        elif self.scenario in [9, 10, 11, 12, 13, 14, 15, 16]:
            self.lin_pe = nn.Linear(pe_dim, latent_dim[0])
            self.edge_embd = nn.Linear(1, latent_dim[0])
            self.lin_x = nn.Linear(self.node_feat_dim, latent_dim[0])

        ## Declare GNN layers
        self.convs = torch.nn.ModuleList()
        if scenario in [1, 2, 3, 4, 5, 6, 7, 8]:
            kwargs = {"num_relations": num_relations, "num_bases": num_bases, "is_residual": True}
            self.convs.append(gconv(self.node_feat_dim, latent_dim[0], **kwargs))
            for i in range(0, len(latent_dim) - 1):
                self.convs.append(gconv(latent_dim[i], latent_dim[i + 1], **kwargs))
        elif scenario in [9, 10, 11, 12, 13, 14, 15, 16]:
            self.convs.append(gconv(latent_dim[0], latent_dim[0]))
            for i in range(0, len(latent_dim) - 1):
                self.convs.append(gconv(latent_dim[i], latent_dim[i + 1]))

        ## NOTE: Temporarily disabled
        # self.trans_encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=4, nhead=2), 4
        # )
        self.hyper_mixer = nn.Sequential(*[HyperMixerLayer(N=420, hid_dim=4) for _ in range(4)])
        # self.hyper_mixer = HyperMixerLayer(N=420, hid_dim=4)

        if scenario in [1, 2, 3, 4, 9, 10, 11, 12]:
            self.lin1 = Linear(2 * sum(latent_dim), 128)
        elif scenario in [5, 6, 7, 8, 13, 14, 15, 16]:
            self.lin1 = Linear(4 * sum(latent_dim), 128)
        else:
            raise NotImplementedError()

        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(2 * sum(latent_dim) + n_side_features, 128)

        self.graphsizenorm = GraphSizeNorm()

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
            elif isinstance(m, (GatedGCNLayer, GatedGCNLSPELayer)):
                m.initialize_weights()

    def conv_score2class(self, score, device=None):
        if not isinstance(self.class_values, torch.Tensor):
            self.class_values = torch.tensor(self.class_values, device=device)

        classes_ = self.class_values.unsqueeze(0).repeat(score.shape[0], 1)
        indices = torch.abs((score - classes_)).argmin(-1)

        return indices

    def create_trans_mask(self, batch, dtype, device, batch_size=50):
        masks = []
        for i in range(batch_size):
            n_nodes_batch = torch.sum(batch == i)
            mask_batch = torch.ones((n_nodes_batch, n_nodes_batch), dtype=dtype, device=device)
            masks.append(mask_batch)

        mask = torch.block_diag(*masks)

        return mask

    def forward(self, data, epoch=-1, is_training=True):
        x, edge_index, edge_type, batch, non_edges = (
            data.x,
            data.edge_index,
            data.edge_type,
            data.batch,
            data.non_edge_index,
        )

        device, dtype = x.device, x.dtype

        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index,
                edge_type,
                p=self.adj_dropout,
                force_undirected=self.force_undirected,
                num_nodes=len(x),
                training=self.training,
            )

        ## Extract node feature, RWPE info and global node index
        node_indx, node_subgraph_feat, pe = (
            x[:, :1].long(),
            x[:, 1 : self.node_feat_dim + 1],
            x[:, self.node_feat_dim + 1 :],
        )

        ## NOTE: Randomize node feat
        # node_subgraph_feat = torch.rand_like(node_subgraph_feat)

        ## Node-set random feat
        user_idx, item_idx = torch.where(node_subgraph_feat[:, 0] == 1)[0], torch.where(node_subgraph_feat[:, 1] == 1)[0]

        rand_feats = torch.ones_like(node_subgraph_feat)
        feats_notS = torch.rand(node_subgraph_feat.size(-1), device=device, dtype=dtype)
        feats_S = torch.rand(node_subgraph_feat.size(-1), device=device, dtype=dtype)

        rand_feats = rand_feats * feats_notS
        rand_feats[user_idx] = rand_feats[item_idx] = feats_S

        # NOTE: Temporarily disable this
        # mask = self.create_trans_mask(batch, x.dtype, x.device)
        # node_subgraph_feat = self.trans_encoder(node_subgraph_feat.unsqueeze(1), mask).squeeze(1)
        for hyper_mixer in self.hyper_mixer:
            node_subgraph_feat = hyper_mixer(node_subgraph_feat, batch)

        ## Convert node feat, pe to suitable dim before passing thu GNN layers
        pe = self.lin_pe(pe)
        x = node_subgraph_feat

        if self.scenario in [9, 10, 11, 12, 13, 14, 15, 16]:
            edge_type = edge_type.unsqueeze(-1).float()
            edge_embd = self.edge_embd(edge_type)
            x = self.lin_x(x)

        ## Apply graph size norm
        if self.scenario in [3, 4, 11, 12, 15, 16]:
            x = self.graphsizenorm(x, batch)

        ## Pass node feat thru GNN layers
        concat_states = []
        # x_before = x
        for conv in self.convs:
            if self.scenario in [1, 2, 3, 4, 5, 6, 7, 8]:
                x, pe = conv(x, pe, edge_index, edge_type)
            elif self.scenario in [9, 10, 11, 12, 13, 14, 15, 16]:
                x, edge_embd, pe = conv(x, edge_embd, edge_index, pe)
            else:
                raise NotImplementedError()

            if self.scenario in [5, 6, 7, 8, 13, 14, 15, 16]:
                concat_states.append(torch.cat((x, pe), dim=-1))
            elif self.scenario in [1, 2, 3, 4, 9, 10, 11, 12]:
                concat_states.append(x)
            else:
                raise NotImplementedError()

        concat_states = torch.cat(concat_states, 1)

        ## NOTE: Enable the following if using EdgeAugment
        # if epoch > 25:
        #     with torch.no_grad():
        #         users, items = non_edges[0], non_edges[1]
        #         x_ = torch.cat([concat_states[users], concat_states[items]], 1)
        #         x_ = F.relu(self.lin1(x_))
        #         x_ = F.dropout(x_, p=0.5, training=self.training)
        #         score_non_edges = self.lin2(x_)

        #         ## Convert predicted score to class
        #         trg_class = self.conv_score2class(score_non_edges, x.device).to(
        #             dtype=edge_type.dtype
        #         )

        #         edge_index = torch.cat((edge_index, non_edges), dim=-1)
        #         edge_type = torch.cat((edge_type, trg_class), dim=-1)

        #     ## Pass node feat thru GNN layers
        #     x = x_before
        #     concat_states = []
        #     for conv in self.convs:
        #         if self.scenario in [1, 2, 3, 4, 5, 6, 7, 8]:
        #             # torch.save(x, "pkls/x_after.pkl")
        #             # torch.save(edge_index, "pkls/edge_index_after.pkl")
        #             # torch.save(edge_type, "pkls/edge_type_after.pkl")
        #             # exit()
        #             x = conv(x, edge_index, edge_type)

        #         elif self.scenario in [9, 10, 11, 12]:
        #             x, pe = conv(x, pe, edge_index, edge_type)
        #         else:
        #             raise NotImplementedError()

        #         concat_states.append(x)
        #     concat_states = torch.cat(concat_states, 1)

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
