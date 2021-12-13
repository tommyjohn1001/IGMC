import math
import pdb
import time

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


class IGMC(GNN):
    # The GNN model of Inductive Graph-based Matrix Completion.
    # Use RGCN convolution + center-nodes readout.
    def __init__(
        self,
        dataset,
        gconv=RGCNConv,
        latent_dim=[32, 32, 32, 32],
        num_relations=5,
        num_bases=2,
        regression=False,
        adj_dropout=0.2,
        force_undirected=False,
        side_features=False,
        n_side_features=0,
        multiply_by=1,
        batch_size=4,
        max_neighbors=50,
        max_walks=10,
    ):
        super(IGMC, self).__init__(
            dataset, GCNConv, latent_dim, regression, adj_dropout, force_undirected
        )

        self.batch_size = batch_size

        self.multiply_by = multiply_by
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0], num_relations, num_bases))
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i + 1], num_relations, num_bases))
        # self.lin1 = Linear(2 * sum(latent_dim), 128)
        self.lin1 = Linear(128, 128)
        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(2 * sum(latent_dim) + n_side_features, 128)

        ## TODO: Think about the architecture/components of network
        self.max_neighbors = max_neighbors
        self.max_walks = max_walks
        self.naive_walking_lin1 = nn.Linear(128, self.max_neighbors)
        self.naive_walking_lin2 = nn.Linear(128, 1)
        # self.edge_embd = nn.Linear(1, 128)
        # self.lin_embd = nn.Sequential(nn.Linear(256, 256), nn.Tanh())
        # self.lin_embd2 = nn.Linear(256, 128, bias=False)

        ## Apply new embedding and LSTMCell instead of simple Linear
        self.edge_embd = nn.Embedding(num_relations, 128)
        self.lin_embd = nn.LSTMCell(256, 128)

    def naive_reasoning_walking(self, x, edge_index, edge_type, max_walks, start_node, end_node):
        # x: [n_nodes, 128]
        # edge_index: [2: n_edges]

        ## Do something to get start and end nodes

        ## start looping

        n_walks = 0
        node = start_node
        traversed_node = set([node.item()])
        accumulated = x[node]
        is_reaching_target = False
        h, c = 0, 0
        queue = Queue()

        ## Init queueu
        queue.enqueue(
            {
                "h": h,
                "c": c,
                "node": node,
                "best": 0,
                "accumulated": accumulated,
                "traversed_node": traversed_node,
            }
        )
        longest_walk = {"n_walks": 1, "accumulated": accumulated}

        ## Start walking
        while not is_reaching_target and n_walks < max_walks:
            item = queue.get_last()
            h, c, node, best, accumulated, traversed_node = (
                item["h"],
                item["c"],
                item["node"],
                item["best"],
                item["accumulated"],
                item["traversed_node"],
            )

            edge_type_ = edge_type[edge_index[0] == node]
            neighbors = edge_index[1][edge_index[0] == node]

            if best == min(self.max_neighbors, len(neighbors)):
                if node == start_node:
                    break
                queue.remove_last()
                last = queue.get_last()
                if last is None:
                    break

                last["best"] += 1

                continue

            x_neighbors = x[neighbors][: self.max_neighbors]
            if self.max_neighbors > len(x_neighbors):
                pad_0s = torch.zeros(
                    (self.max_neighbors - len(x_neighbors), x_neighbors.size(-1)),
                    dtype=x_neighbors.dtype,
                    device=x_neighbors.device,
                )
                x_neighbors = torch.cat((x_neighbors, pad_0s), 0)

            tau = 0.25
            selection_dist = self.naive_walking_lin1(accumulated)
            selection_dist = selection_dist / tau

            ## Thiết lập các mask
            ## Do masking to selection_dist : lý do là vì network thì là cố định output, mà số lượng neightbor
            ## của mỗi node là không cố định
            mask_neighbors = torch.zeros_like(selection_dist)
            mask_neighbors[: len(neighbors)] = 1

            mask_traversed_nodes = torch.ones_like(selection_dist)
            for i, neighbor in enumerate(neighbors):
                if i >= len(mask_traversed_nodes):
                    break

                if neighbor.item() in traversed_node:
                    mask_traversed_nodes[i] = 0

            mask = (mask_neighbors * mask_traversed_nodes) == 0

            if mask.sum() == len(mask) or best == self.max_neighbors - mask.sum():
                ## Nếu tất cả các ô đều bị mask thì back về state phía trước
                queue.remove_last()

                item = queue.get_last()
                if item is None:
                    break

                item["best"] += 1

                continue

            selection_dist = selection_dist.masked_fill_(mask, float("-inf"))
            selected_neighbor_softmax = torch.softmax(selection_dist, -1)
            _, indices = torch.sort(selected_neighbor_softmax, -1, descending=True)
            selected_neighbor = indices[best]
            if selected_neighbor >= len(neighbors):
                queue.remove_last()

                item = queue.get_last()
                if item is None:
                    break

                item["best"] += 1

                continue
            selected_node = neighbors[selected_neighbor]
            selected_neighbor_softmax = selected_neighbor_softmax.unsqueeze(0)
            selected_node_embd = selected_neighbor_softmax @ x_neighbors
            selected_edge_type = edge_type_[selected_neighbor]
            selected_edge_embd = self.edge_embd(selected_edge_type.unsqueeze(0))

            ## dùng cơ chế cộng để accumulate hoặc áp dụng memorynet
            total = torch.cat((selected_node_embd, selected_edge_embd), -1)
            # [b, 256]
            if not torch.is_tensor(h):
                h, c = self.lin_embd(total)
            else:
                h, c = self.lin_embd(total, (h, c))
            # [bsz, 128]
            accumulated = accumulated + h.squeeze(0)

            node = selected_node
            new_traversed_node = traversed_node.copy()
            new_traversed_node.add(node.item())
            n_walks += 1

            ## Save state
            queue.enqueue(
                {
                    "h": h,
                    "c": c,
                    "node": node,
                    "best": 0,
                    "accumulated": accumulated,
                    "traversed_node": new_traversed_node,
                }
            )

            if n_walks > longest_walk["n_walks"]:
                longest_walk["n_walks"] = n_walks
                longest_walk["accumulated"] = accumulated

            if selected_node == end_node:
                is_reaching_target = True

                longest_walk = {"n_walks": n_walks + 1, "accumulated": accumulated}

        return longest_walk["accumulated"] / longest_walk["n_walks"]

    def forward(self, data):
        start = time.time()
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

        ## Lọc ra vị trí user của các batch, user này là user cần predict link, chứ không phải user's neighbor
        users = data.x[:, 0] == 1
        ## Lọc ra vị trí movie của các batch, movie này là movie cần predict
        items = data.x[:, 1] == 1
        ## vì mỗi batch chỉ cần predict một cặp user-item, nên có 50 batch thì sẽ có 50 user được lọc ra, 50 movie được lọc ra
        ## mỗi user sẽ được

        ## Bắt đầu NRW
        accum_n_nodes = 0
        max_walks = 20
        start_nodes, end_nodes = torch.where(users)[0], torch.where(items)[0]
        accumulations = []
        for bsz, (start_node, end_node) in enumerate(zip(start_nodes, end_nodes)):
            concat_states_b = concat_states[batch == bsz]

            n_nodes = concat_states_b.size(0)
            edge_b = torch.where(
                (accum_n_nodes <= edge_index[0]) & (edge_index[0] < accum_n_nodes + n_nodes)
            )[0]
            edge_index_b = edge_index[:, edge_b] - accum_n_nodes
            edge_type_b = edge_type[edge_b]
            start_node = start_node - accum_n_nodes
            end_node = end_node - accum_n_nodes

            accumulation = self.naive_reasoning_walking(
                concat_states_b, edge_index_b, edge_type_b, max_walks, start_node, end_node
            )
            accumulations.append(accumulation.unsqueeze(0))

            accum_n_nodes += n_nodes

        ## Trích xuất các vector biểu diễn user (chỉ user thôi, không lấy user's neighbor) và
        ## biểu diễn item (chỉ item thôi chứ không lấy item's neighbors)
        ## FIXME: Đoạn này tạm thời bỏ qua
        # x = torch.cat([concat_states[users], concat_states[items]], 1)
        # if self.side_features:
        #     x = torch.cat([x, data.u_feature, data.v_feature], 1)
        x = torch.cat(accumulations, 0)
        # [bz, 128]

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0] * self.multiply_by
        else:
            return F.log_softmax(x, dim=-1)
