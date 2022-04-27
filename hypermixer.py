import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parameter import Parameter


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        pe = self.pe[: x.size(1)]

        pe = pe.transpose(0, 1)

        return pe


class HyperTokenMixer(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()

        self.pe = PositionalEncoding(hid_dim)
        self.mlp = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.GELU())

    def forward(self, x):
        # [bz, N, d]

        ## Prepare W for MLP
        absolute_pe = self.pe(x)
        # [1, N, d]

        x_ = x + absolute_pe
        # [bz, N, d]

        w = self.mlp(x_)
        # [bz, N, d]

        ## Use W into MLP
        w1, w2 = w, w.transpose(-1, -2)
        x = F.gelu(w2 @ x)
        # [bz, d, d]
        x = w1 @ x
        # [bz, N, d]

        return x


class MLP(nn.Module):
    def __init__(self, N, hid_dim):
        super().__init__()

        self.w1 = Parameter(torch.empty((N, hid_dim)))
        self.w2 = Parameter(torch.empty((N, hid_dim)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x):
        # x: [bz, N, d]

        N = x.shape[1]

        w1, w2 = self.w1[:N], self.w2[:N]

        x = w2.transpose(-1, -2) @ x
        # [bz, d, d]
        x = F.gelu(x)
        x = w1 @ x
        # [bz, N, d]

        return x


class HyperMixerLayer(nn.Module):
    def __init__(self, N=500, hid_dim=128):
        super().__init__()

        self.N = N
        self.hid_dim = hid_dim

        self.lnorm = nn.LayerNorm(hid_dim)
        self.token_mixer = HyperTokenMixer(hid_dim)
        self.mlp = MLP(N, hid_dim)

    def pad_subgraph(self, inp, batch, batch_size=50):
        padded = []
        for b in range(batch_size):
            node_features = inp[batch == b]

            n_padded = abs(self.N - node_features.shape[0])
            pad0_tensor = torch.zeros((n_padded, self.hid_dim), device=inp.device, dtype=inp.dtype)
            padded_subgraph = torch.cat((node_features, pad0_tensor), dim=0)
            # [N, d]

            padded.append(padded_subgraph.unsqueeze(0))

        padded = torch.cat(padded, dim=0)

        return padded

    def unpad_subgraph(self, inp, batch, batch_size=50):
        unpadded = []
        for b in range(batch_size):
            n_nodes_batch = torch.sum(batch == b)
            node_feat = inp[b, :n_nodes_batch]

            unpadded.append(node_feat)

        unpadded = torch.cat(unpadded, dim=0)

        return unpadded

    def forward(self, inp, batch):
        # inp: [m, d]

        padded = self.pad_subgraph(inp, batch)
        # [bz, N, d]

        x = self.lnorm(padded)
        # [m, d]
        x_before_mixing = x

        x = self.token_mixer(x)
        # [m, d]

        ## Skip connection
        x = x + x_before_mixing
        x_before_mlp = x

        x = self.lnorm(x)

        ## Apply MLP
        x = self.mlp(x)
        # [bz, m, d]

        x = x + x_before_mlp
        # [bz, m, d]

        x = self.unpad_subgraph(x, batch)
        # [m, d]

        return x


# hyper_mixer = HyperMixerLayer(10, 128)
