from torch_geometric.nn import RGCNConv
from torch_geometric.utils import dropout_adj

from all_packages import *
from hypermixer import HyperMixerLayer
from layers import *
from regularization.mlp import MLP
from util_functions import *


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
        dropout=0.2,
        mixer=None,
        mode=None,
        dataname=None,
    ):
        # gconv = GatedGCNLayer GatedGCNLSPELayer RGatedGCNLayer RGCNConv
        if scenario in [1, 2, 3, 4, 5, 6, 7, 8]:
            # gconv = RGCNConvLSPE GatedGCNLSPELayer NewFastRGCNConv OldRGCNConvLSPE
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
        self.mode = mode
        self.gconv = gconv
        self.side_features = side_features
        self.dataname = dataname
        self.pe_dim = pe_dim

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
            # kwargs = {"num_relations": num_relations, "num_bases": num_bases, "is_residual": True}
            kwargs = {"num_relations": num_relations, "num_bases": num_bases}
            self.convs.append(gconv(self.node_feat_dim, latent_dim[0], **kwargs))
            for i in range(0, len(latent_dim) - 1):
                self.convs.append(gconv(latent_dim[i], latent_dim[i + 1], **kwargs))
        elif scenario in [9, 10, 11, 12, 13, 14, 15, 16]:
            self.convs.append(gconv(latent_dim[0], latent_dim[0]))
            for i in range(0, len(latent_dim) - 1):
                self.convs.append(gconv(latent_dim[i], latent_dim[i + 1]))

        ## Declare Mixer: TransEncoder ; HyperMixer
        if mixer == "trans_encoder":
            self.mixer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=4, nhead=2), 4)
        elif self.mixer == "hyper_mixer":
            self.mixer = nn.Sequential(*[HyperMixerLayer(N=420, hid_dim=4) for _ in range(4)])
        else:
            raise NotImplementedError()

        ## Define final FF
        if scenario in [1, 2, 3, 4, 9, 10, 11, 12]:
            final_dim = 2 * sum(latent_dim)
        elif scenario in [5, 6, 7, 8, 13, 14, 15, 16]:
            final_dim = 4 * sum(latent_dim)
        else:
            raise NotImplementedError()
        ## NOTE: If using PE as node feat, enable the following
        # final_dim = 2 * sum(latent_dim)
        self.final_ff = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(final_dim, 1),
        )

        self.graphsizenorm = GraphSizeNorm()

        ## init weights
        self.initialize_weights()

        ## Load weights of Contrastive MLP layer
        if self.mode == "coop":
            self.load_pretrained_weights()

    def save_pretrained_weights(self):
        """Save weights of gconv layers and mixer"""
        path_pretrained = f"weights/{self.dataname}/{self.scenario}_{self.pe_dim}_{self.gconv.__name__}_{type(self.mixer).__name__}.pkl"

        pretrained_weights = {"convs": None, "mixer": None}
        pretrained_weights["mixer"] = self.mixer.state_dict()
        pretrained_weights["convs"] = [conv.state_dict() for conv in self.convs]

        os.makedirs(osp.dirname(path_pretrained), exist_ok=True)
        torch.save(pretrained_weights, path_pretrained)

    def load_pretrained_weights(self):
        """Load weights of gconv layers and mixer"""

        path_pretrained = f"weights/{self.dataname}/{self.scenario}_{self.pe_dim}_{self.gconv.__name__}_{type(self.mixer).__name__}.pkl"
        if not osp.isfile(path_pretrained):
            logger.error(f"Path to pretrained GNN and Mixer not valid: {path_pretrained}")
            exit(1)

        ## Load gconvs and mixer
        pretrained_weights = torch.load(path_pretrained)
        self.mixer.load_state_dict(pretrained_weights["mixer"])
        for ith, conv in enumerate(self.convs):
            conv.load_state_dict(pretrained_weights["convs"][ith])

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

        if isinstance(self.mixer, nn.TransformerEncoder):
            mask = self.create_trans_mask(batch, x.dtype, x.device)
            node_subgraph_feat = self.mixer(node_subgraph_feat.unsqueeze(1), mask).squeeze(1)
        else:
            for hyper_mixer in self.mixer:
                node_subgraph_feat = hyper_mixer(node_subgraph_feat, batch)

        ## Convert node feat, pe to suitable dim before passing thu GNN layers
        if self.mode == "coop":
            pe = self.lin_pe(self.mlp_contrastive(pe))
        else:
            pe = self.lin_pe(pe)
        x = node_subgraph_feat
        # NOTE: If using PE as node_feat, enable the following
        # x = pe

        if self.scenario in [9, 10, 11, 12, 13, 14, 15, 16]:
            edge_type = edge_type.unsqueeze(-1).float()
            edge_embd = self.edge_embd(edge_type)
            x = self.lin_x(x)

        ## Apply graph size norm
        if self.scenario in [3, 4, 7, 8, 11, 12, 15, 16]:
            x = self.graphsizenorm(x, batch)

        ## Pass node feat thru GNN layers
        concat_states = []
        for conv in self.convs:
            if self.scenario in [1, 2, 3, 4, 5, 6, 7, 8]:
                # NOTE: If using PE as node_feat, enable the following
                # x = conv(x, edge_index, edge_type)
                x, pe = conv(x, pe, edge_index, edge_type)
            elif self.scenario in [9, 10, 11, 12, 13, 14, 15, 16]:
                # NOTE: If using PE as node_feat, enable the following
                # x, edge_embd = conv(x, edge_embd, edge_index)
                x, edge_embd, pe = conv(x, edge_embd, edge_index, pe)
            else:
                raise NotImplementedError()

            if self.scenario in [5, 6, 7, 8, 13, 14, 15, 16]:
                # NOTE: If using PE as node_feat, enable the following
                # concat_states.append(x)
                concat_states.append(torch.cat((x, pe), dim=-1))
            elif self.scenario in [1, 2, 3, 4, 9, 10, 11, 12]:
                concat_states.append(x)
            else:
                raise NotImplementedError()

        concat_states = torch.cat(concat_states, 1)

        users = data.x[:, 1] == 1
        items = data.x[:, 2] == 1
        x = torch.cat([concat_states[users], concat_states[items]], 1)
        if self.side_features:
            x = torch.cat([x, data.u_feature, data.v_feature], 1)

        x = self.final_ff(x)
        if self.regression:
            return x[:, 0] * self.multiply_by
        else:
            return F.log_softmax(x, dim=-1)
