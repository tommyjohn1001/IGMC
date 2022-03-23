import torch
import torch.nn as nn
import torch.nn.functional as F
from all_packages import *
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, RGCNConv
from torch_geometric.utils import dropout_adj
from util_functions import *

from utils_model.layers import *
from utils_model.losses import *
from utils_model.utils import *


class IGMC(GNN):
    # The GNN model of Inductive Graph-based Matrix Completion.
    # Use RGCN convolution + center-nodes readout.
    def __init__(
        self,
        dataset,
        gconv=GatedGCNLayer,
        latent_dim=[32, 32, 32, 32],
        hid_dim=128,
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
        class_values=None,
        ARR=0.01,
        temperature=0.1,
    ):
        super(IGMC, self).__init__(
            dataset, GCNConv, latent_dim, regression, adj_dropout, force_undirected, hid_dim
        )

        self.batch_size = batch_size
        self.ARR = ARR
        self.num_relations = num_relations
        self.temperature = temperature
        self.multiply_by = multiply_by
        self.map_edgetype2id = {v: i for i, v in enumerate(class_values)}
        self.class_values = class_values

        self.convs = torch.nn.ModuleList()
        if gconv in [RGCNConv]:
            self.convs.append(gconv(latent_dim[0], latent_dim[0]))
            for i in range(0, len(latent_dim) - 1):
                self.convs.append(gconv(latent_dim[i], latent_dim[i + 1]))
        elif gconv is GATConv:
            self.convs.append(gconv(dataset.num_features, latent_dim[0], edge_dim=128))
            for i in range(0, len(latent_dim) - 1):
                self.convs.append(gconv(latent_dim[i], latent_dim[i + 1], edge_dim=128))
        else:
            raise NotImplementedError()
        self.lin1 = Linear(hid_dim, hid_dim)
        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(2 * sum(latent_dim) + n_side_features, hid_dim)

        self.max_neighbors = max_neighbors
        self.max_walks = max_walks
        self.naive_walking_lin1 = nn.Linear(hid_dim, self.max_neighbors)
        # self.edge_embd = nn.Linear(1, 128)
        # self.lin_embd = nn.Sequential(nn.Linear(256, 256), nn.Tanh())
        # self.lin_embd2 = nn.Linear(256, 128, bias=False)

        ## Apply new embedding and LSTMCell instead of simple Linear
        self.edge_embd = nn.Embedding(num_relations, hid_dim)
        self.lin_embd = nn.LSTMCell(hid_dim * 2, hid_dim)

        self.contrastive_criterion = CustomContrastiveLoss(self.temperature, num_relations)

    def naive_reasoning_walking(self, x, edge_index, edge_type, max_walks, start_node, end_node):
        # x: [n_nodes, 128]
        # edge_index: [2: n_edges]

        # Get
        edge_embeddings = []
        for gconv in self.convs:
            w = torch.matmul(gconv.att, gconv.basis.view(gconv.num_bases, -1)).view(
                gconv.num_relations, gconv.in_channels, gconv.out_channels
            )
            # [n_relations, n_in, n_out]
            edge_embeddings.append(w.mean(1))
        edge_embeddings = torch.cat(edge_embeddings, -1)
        # [n_relations, 128]

        ## start looping

        n_walks = 0
        node = start_node
        traversed_node = set([node.item()])
        accumulated = x[node]
        is_reaching_target = False
        h, c = 0, 0
        while not is_reaching_target and n_walks < max_walks:
            # print(f"==> node: {node}")
            edge_type_ = edge_type[edge_index[0] == node]
            neighbors = edge_index[1][edge_index[0] == node]

            if len(neighbors) == 0:
                accumulated += x[end_node]
                break

            selection_dist = self.naive_walking_lin1(accumulated)

            ## Thiết lập các mask
            ## Do masking to selection_dist : lý do là vì network thì là cố định output, mà số lượng neightbor
            ## của mỗi node là không cố định
            mask_neighbors = torch.ones_like(selection_dist) * 0
            mask_neighbors[: len(neighbors)] = 1

            mask_traversed_nodes = torch.ones_like(selection_dist)
            for i, neighbor in enumerate(neighbors):
                if i >= len(mask_traversed_nodes):
                    break

                if neighbor.item() in traversed_node:
                    mask_traversed_nodes[i] = 0

            mask = (mask_neighbors * mask_traversed_nodes) == 0

            selection_dist = selection_dist.masked_fill_(mask, float("-inf"))

            ## ở đây có thể áp dụng tiếp softmax rồi argmax hoặc soft argmax
            selected_neighbor = torch.argmax(torch.softmax(selection_dist, -1), -1)
            # [1]
            selected_node = neighbors[selected_neighbor]
            selected_node_embd = x[selected_node].unsqueeze(0)
            selected_edge_type = edge_type_[selected_neighbor]
            selected_edge_type = F.one_hot(
                selected_edge_type.unsqueeze(0), self.num_relations
            ).float()
            selected_edge_embd = selected_edge_type @ edge_embeddings

            ## dùng cơ chế cộng để accumulate hoặc áp dụng memorynet
            total = torch.cat((selected_node_embd, selected_edge_embd), -1)
            # accumulated_nodes.append(total.unsqueeze(1))

            # [b, 256]
            if not torch.is_tensor(h):
                h, c = self.lin_embd(total)
            else:
                h, c = self.lin_embd(total, (h, c))
            # [bsz, 128]
            accumulated += h.squeeze(0)

            if selected_node == end_node:
                is_reaching_target = True

            node = selected_node
            traversed_node.add(node.item())

            n_walks += 1

        # accumulated_nodes = torch.cat(accumulated_nodes, 1)
        # # [bz, n, 256]
        # pad_zeros = torch.zeros(
        #     (bz, max_walks - accumulated_nodes.shape[1], 256),
        #     dtype=accumulated_nodes.dtype,
        #     device=accumulated_nodes.device,
        # )
        # accumulated_nodes = torch.cat((accumulated_nodes, pad_zeros))
        # # [bz, max_walks, 256]

        # This ensures end_node is always appened into accumulated
        n_traversed_nodes = len(traversed_node)

        if len(neighbors) != 0 and is_reaching_target:
            accumulated += x[end_node]
            n_traversed_nodes += 1

        return accumulated / n_traversed_nodes

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
            if isinstance(conv, GATConv):
                edge_features = self.edge_embd(edge_type)
                x = torch.tanh(conv(x, edge_index, edge_features))
            else:
                x = conv(x, edge_index, edge_type)
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
                concat_states_b, edge_index_b, edge_type_b, self.max_walks, start_node, end_node
            )
            accumulations.append(accumulation.unsqueeze(0))

            accum_n_nodes += n_nodes

        ## Trích xuất các vector biểu diễn user (chỉ user thôi, không lấy user's neighbor) và
        ## biểu diễn item (chỉ item thôi chứ không lấy item's neighbors)
        ## FIXME: Đoạn này tạm thời bỏ qua
        # x = torch.cat([concat_states[users], concat_states[items]], 1)
        # if self.side_features:
        #     x = torch.cat([x, data.u_feature, data.v_feature], 1)
        x_128 = torch.cat(accumulations, 0)
        # [bz, 128]

        x = F.relu(self.lin1(x_128))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        ## Calculate loss
        loss = 0

        ## MSE loss
        loss_mse = F.mse_loss(x[:, 0] * self.multiply_by, data.y.view(-1))

        ## ARR loss
        loss_arr = 0
        if self.ARR > 0:
            if isinstance(conv, GATConv):
                loss_arr = torch.sum((self.edge_embd.weight[1:] - self.edge_embd.weight[:-1]) ** 2)
            else:
                for gconv in self.convs:
                    w = torch.matmul(gconv.att, gconv.basis.view(gconv.num_bases, -1)).view(
                        gconv.num_relations, gconv.in_channels, gconv.out_channels
                    )
                    reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :]) ** 2)
                    loss_arr += reg_loss

        loss = loss_mse + self.ARR * loss_arr
        if torch.isnan(loss):
            print(f"NaN: {loss_mse} - {loss_arr}")
            sys.exit(1)
        if loss < 0:
            print(f"below 0: {loss_mse} - {loss_arr}")

        return x[:, 0], loss


class IGMC2(GNN):
    # The GNN model of Inductive Graph-based Matrix Completion.
    # Use RGCN convolution + center-nodes readout.
    def __init__(
        self,
        dataset,
        gconv=RGCNConv,
        latent_dim=[32, 32, 32, 32],
        hid_dim=128,
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
        class_values=None,
        ARR=0.01,
        temperature=0.1,
        pe_dim=20,
        n_nodes=3000,
        scenario=1,
    ):
        super(IGMC2, self).__init__(
            dataset,
            gconv,
            latent_dim,
            regression,
            adj_dropout,
            force_undirected,
            num_relations=num_relations,
            num_bases=num_bases,
        )

        gconv = OldRGCNConvLSPELayer

        self.batch_size = batch_size
        self.ARR = ARR
        self.num_relations = num_relations
        self.temperature = temperature
        self.map_edgetype2id = {v: i for i, v in enumerate(class_values)}
        self.scenario = scenario
        self.multiply_by = multiply_by

        ## Declare modules to convert node feat, pe to hidden vectors
        self.node_feat_dim = dataset.num_features - pe_dim - 1
        self.lin_pe = nn.Linear(pe_dim, self.node_feat_dim)

        ## Declare GNN layers
        self.convs = torch.nn.ModuleList()
        kwargs = {"num_relations": num_relations, "num_bases": num_bases, "is_residual": True}
        self.convs.append(gconv(self.node_feat_dim, latent_dim[0], **kwargs))
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i + 1], **kwargs))

        # self.lin1 = Linear(2 * sum(latent_dim), 128)
        self.lin1 = Linear(hid_dim, hid_dim)
        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(2 * sum(latent_dim) + n_side_features, hid_dim)

        self.max_neighbors = max_neighbors
        self.max_walks = max_walks
        self.naive_walking_lin1 = nn.Linear(hid_dim, self.max_neighbors)

        ## Apply new embedding and LSTMCell instead of simple Linear
        self.lin_embd = nn.LSTMCell(hid_dim * 2, hid_dim)

        self.contrastive_criterion = CustomContrastiveLoss(self.temperature, num_relations)

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

    def naive_reasoning_walking(self, x, edge_index, edge_type, max_walks, start_node, end_node):
        # x: [n_nodes, 128]
        # edge_index: [2: n_edges]

        # Get edge emmbedding from gconv
        edge_embeddings = []
        for gconv in self.convs:
            w = torch.matmul(gconv.att, gconv.basis.view(gconv.num_bases, -1)).view(
                gconv.num_relations, gconv.in_channels, gconv.out_channels
            )
            # [n_relations, n_in, n_out]
            edge_embeddings.append(w.mean(1))
        edge_embeddings = torch.cat(edge_embeddings, -1)

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

            selection_dist = self.naive_walking_lin1(accumulated)
            selection_dist = selection_dist / self.temperature

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
            selected_edge_type = F.one_hot(
                selected_edge_type.unsqueeze(0), self.num_relations
            ).float()
            selected_edge_embd = selected_edge_type @ edge_embeddings
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

        if not is_reaching_target:
            longest_walk["accumulated"] += x[end_node]
            longest_walk["n_walks"] += 1

        return longest_walk["accumulated"] / longest_walk["n_walks"]

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

        ## Extract node feature, RWPE info and global node index
        node_indx, node_subgraph_feat, pe = (
            x[:, :1].long(),
            x[:, 1 : self.node_feat_dim + 1],
            x[:, self.node_feat_dim + 1 :],
        )

        ## Convert node feat, pe to suitable dim before passing thu GNN layers
        pe = self.lin_pe(pe)
        x = node_subgraph_feat

        ## Apply graph size norm
        if self.scenario in [13, 14]:
            x = self.graphsizenorm(x, batch)

        ## Pass node feat thru GNN layers
        concat_states = []
        for conv in self.convs:
            x, pe = conv(x, pe, edge_index, edge_type)
            concat_states.append(x)

        concat_states = torch.cat(concat_states, 1)

        ## Lọc ra vị trí user của các batch, user này là user cần predict link, chứ không phải user's neighbor
        users = data.x[:, 1] == 1
        ## Lọc ra vị trí movie của các batch, movie này là movie cần predict
        items = data.x[:, 2] == 1
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
        x_128 = torch.cat(accumulations, 0)
        # [bz, 128]

        x = F.relu(self.lin1(x_128))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        ## MSE loss
        loss_mse = F.mse_loss(x[:, 0] * self.multiply_by, data.y.view(-1))

        ## ARR loss
        loss_arr = 0
        if self.ARR > 0:
            for gconv in self.convs:
                w = torch.matmul(gconv.att, gconv.basis.view(gconv.num_bases, -1)).view(
                    gconv.num_relations, gconv.in_channels, gconv.out_channels
                )
                reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :]) ** 2)
                loss_arr += reg_loss

        loss = loss_mse + self.ARR * loss_arr

        if torch.isnan(loss):
            print(f"NaN: {loss}")
            sys.exit(1)
        if loss < 0:
            print(f"below 0: {loss}")

        return x[:, 0], loss


###################################################################################


class IGMCLitModel(LightningModule):
    def __init__(self, model, hps):
        super().__init__()

        self._hparams = hps
        self.model = model

    def training_step(self, batch, batch_idx):
        _, loss = self.model(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

    def training_epoch_end(self, outputs):
        self.log("epoch", self.current_epoch, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        out, _ = self.model(batch)
        mse = F.mse_loss(out, batch.y.view(-1), reduction="sum").item()

        return mse, len(batch.y)

    def validation_epoch_end(self, outputs) -> None:
        mse, total = 0, 0
        for output in outputs:
            mse += output[0]
            total += output[1]
        mse_loss = mse / total
        rmse = math.sqrt(mse_loss)

        self.log("val_loss", rmse, on_epoch=True)

        return rmse

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        out, _ = self.model(batch)

        return out, batch.y

    def on_predict_epoch_end(self, results):
        preds, trgs = [], []
        for output in results[0]:
            preds.append(output[0])
            trgs.append(output[1])

        preds, trgs = torch.cat(preds), torch.cat(trgs)
        return preds.cpu(), trgs.cpu()

    def configure_optimizers(self):
        optimizer = Adam(
            self.model.parameters(),
            lr=self._hparams["lr"],
            weight_decay=self._hparams["weight_decay"],
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=self._hparams["lr"])

        scheduler = {
            "scheduler": get_custom_lr_scheduler(optimizer, self._hparams),
            "interval": "epoch",  # or 'epoch'
            "frequency": 1,
        }

        return [optimizer], [scheduler]
