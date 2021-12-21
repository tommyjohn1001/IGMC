import torch
import torch.nn as nn
import torch.nn.functional as F
from all_packages import *
from torch.nn import Linear
from torch_geometric.nn import GCNConv, RGCNConv
from torch_geometric.utils import dropout_adj
from train_eval import get_linear_schedule_with_warmup
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
        class_values=None,
        ARR=0.01,
        contrastive=True,
        temperature=0.1,
    ):
        super(IGMC, self).__init__(
            dataset, GCNConv, latent_dim, regression, adj_dropout, force_undirected
        )

        self.batch_size = batch_size
        self.ARR = ARR
        self.contrastive = contrastive
        self.temperature = temperature
        self.multiply_by = multiply_by
        self.map_edgetype2id = {v: i for i, v in enumerate(class_values)}
        self.class_values = class_values

        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0], num_relations, num_bases))
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i + 1], num_relations, num_bases))
        self.lin1 = Linear(128, 128)
        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(2 * sum(latent_dim) + n_side_features, 128)

        self.max_neighbors = max_neighbors
        self.max_walks = max_walks
        self.naive_walking_lin1 = nn.Linear(128, self.max_neighbors)
        # self.edge_embd = nn.Linear(1, 128)
        # self.lin_embd = nn.Sequential(nn.Linear(256, 256), nn.Tanh())
        # self.lin_embd2 = nn.Linear(256, 128, bias=False)

        ## Apply new embedding and LSTMCell instead of simple Linear
        self.edge_embd = nn.Embedding(num_relations, 128)
        self.lin_embd = nn.LSTMCell(256, 128)

        self.contrastive_criterion = CustomContrastiveLoss(self.temperature, num_relations)

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
            selected_edge_embd = self.edge_embd(selected_edge_type.unsqueeze(0))

            ## dùng cơ chế cộng để accumulate hoặc áp dụng memorynet
            total = torch.cat((selected_node_embd, selected_edge_embd), -1)
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

        return accumulated / len(traversed_node)

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
            for gconv in self.convs:
                w = torch.matmul(gconv.comp, gconv.weight.view(gconv.num_bases, -1)).view(
                    gconv.num_relations, gconv.in_channels, gconv.out_channels
                )
                reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :]) ** 2)
                loss_arr = reg_loss

        ## Custom Contrastive loss
        loss_contrastive = 0
        if self.contrastive > 0:
            edgetype_indx = [self.map_edgetype2id[edgetype.item()] for edgetype in data.y]
            edgetype_indx = torch.tensor(edgetype_indx, dtype=torch.int64, device=data.y.device)
            loss_contrastive = self.contrastive_criterion(
                self.edge_embd.weight, x_128, edgetype_indx
            )

        loss = loss_mse + self.ARR * loss_arr + self.contrastive * loss_contrastive
        if torch.isnan(loss):
            print(f"NaN: {loss_mse} - {loss_arr} - {loss_contrastive}")
            exit(1)
        if loss < 0:
            print(f"below 0: {loss_mse} - {loss_arr} - {loss_contrastive}")

        return x[:, 0], loss


class IGMC2(GNN):
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
        class_values=None,
        ARR=0.01,
        contrastive=True,
        temperature=0.1,
    ):
        super(IGMC2, self).__init__(
            dataset, GCNConv, latent_dim, regression, adj_dropout, force_undirected
        )

        self.batch_size = batch_size
        self.ARR = ARR
        self.contrastive = contrastive
        self.temperature = temperature
        self.map_edgetype2id = {v: i for i, v in enumerate(class_values)}

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

        self.max_neighbors = max_neighbors
        self.max_walks = max_walks
        self.naive_walking_lin1 = nn.Linear(128, self.max_neighbors)
        # self.edge_embd = nn.Linear(1, 128)
        # self.lin_embd = nn.Sequential(nn.Linear(256, 256), nn.Tanh())
        # self.lin_embd2 = nn.Linear(256, 128, bias=False)

        ## Apply new embedding and LSTMCell instead of simple Linear
        self.edge_embd = nn.Embedding(num_relations, 128)
        self.lin_embd = nn.LSTMCell(256, 128)

        self.contrastive_criterion = CustomContrastiveLoss(self.temperature, num_relations)

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
            for gconv in self.convs:
                w = torch.matmul(gconv.comp, gconv.weight.view(gconv.num_bases, -1)).view(
                    gconv.num_relations, gconv.in_channels, gconv.out_channels
                )
                reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :]) ** 2)
                loss_arr = reg_loss

        ## Custom Contrastive loss
        loss_contrastive = 0
        if self.contrastive > 0:
            edgetype_indx = [self.map_edgetype2id[edgetype.item()] for edgetype in data.y]
            edgetype_indx = torch.tensor(edgetype_indx, dtype=torch.int64, device=data.y.device)
            loss_contrastive = self.contrastive_criterion(
                self.edge_embd.weight, x_128, edgetype_indx
            )

        loss = loss_mse + self.ARR * loss_arr + self.contrastive * loss_contrastive
        if torch.isnan(loss):
            print(f"NaN: {loss_mse} - {loss_arr} - {loss_contrastive}")
            exit(1)
        if loss < 0:
            print(f"below 0: {loss_mse} - {loss_arr} - {loss_contrastive}")

        return x[:, 0], loss


###################################################################################


class IGMCLitModel(LightningModule):
    def __init__(self, model, hps):
        super().__init__()

        self._hparams = hps
        self.model = model

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        _, loss = self.model(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

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
        aList = []
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
            "scheduler": get_custom_lr_scheduler(
                optimizer,
                self._hparams["percent_warmup"],
                self._hparams["percent_latter"],
                self._hparams["num_training_steps"],
                self._hparams["lr"],
                self._hparams["init_lr"],
                self._hparams["latter_lr"],
            ),
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }

        return [optimizer], [scheduler]
