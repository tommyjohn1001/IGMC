from all_packages import *

from regularization.mlp import MLP


class ContrastiveDistance(nn.Module):
    def __init__(self, tau=0.07, eps=1e-8):
        super().__init__()

        self.tau = tau
        self.eps = eps

    def sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def forward(self, x_permuted: Tensor):
        # [n_nodes, n_perm, d]

        device, dtype = x_permuted.device, x_permuted.dtype
        n_nodes, n_perms, _ = x_permuted.size()

        distances = []
        # TODO: need to improve this to eliminate for loop
        for i in range(n_perms):
            x = x_permuted[:, i]
            # [n_nodes, d]

            ## Calculate sim matrix
            sim_matrix = self.sim_matrix(x, x, self.eps)
            # [n_nodes, n_nodes]

            ## Divide by temperature
            sim_matrix = torch.div(sim_matrix, self.tau)
            # [n_nodes, n_nodes]

            ## Numerically stable trick
            sim_matrix_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - sim_matrix_max.detach()
            # [n_nodes, n_nodes]

            ## Mask out self-contrasted cases
            mask = 1 - torch.ones(n_nodes, device=device, dtype=dtype).diag()
            sim_matrix_masked = mask * sim_matrix
            # [n_nodes, n_nodes]

            ## Calculate distance matrix
            sim_matrix_logsumexp = torch.logsumexp(sim_matrix_masked, dim=1)
            # [n_nodes, n_nodes]
            distance_matrix = sim_matrix_logsumexp - sim_matrix
            # [n_nodes, n_nodes]

            distances.append(distance_matrix[None])

        distances = torch.cat(distances, dim=0)
        # [n_perms, n_nodes, n_nodes]

        return distances


class Regularization(nn.Module):
    def __init__(self, d_pe, tau=0.07, eps=1e-8, dropout=0.25):
        super().__init__()

        self.ConDistance = ContrastiveDistance(tau=tau, eps=eps)
        self.mlp = MLP(d=d_pe, dropout=dropout)
        self.criterion_mse = nn.MSELoss()

    def create_trg_regu_matrix(targets_permuted: Tensor, n_nodes: int):
        """Create target matrix for training regularization loss which is achieved by ContrastiveLoss

        Args:
            n_nodes (int): no. nodes
            targets_permuted (Tensor): tensor containing target user and item index of each permutation
        """
        # targets_permuted: [n_perm, 2]

        n_perm = targets_permuted.size(0)
        trg_matrix = torch.ones((n_perm, n_nodes, n_nodes))

        for n in range(n_perm):
            trg_user_idx = int(targets_permuted[n][0].item())
            trg_item_idx = int(targets_permuted[n][1].item())
            S = set([trg_user_idx, trg_item_idx])

            for i in range(n_nodes):
                for j in range(n_nodes):
                    condition1 = i not in S and j in S
                    condition2 = i in S and j not in S
                    if condition1 or condition2:
                        trg_matrix[i, j] = 0

        return trg_matrix

    def create_trg_distance(targets_permuted: Tensor, batch: Tensor):
        # targets_permuted: [n_perm, 2]

        batch_indx, _ = torch.unique(batch).sort()

        for b in batch_indx:
            n_nodes += torch.sum(batch == b).item()

            # target

    def forward(self, x_permuted: Tensor, targets_permuted: Tensor, batch: Tensor):
        # x_permuted: [N, n_perm, d]
        # targets_permuted: [bz * n_perm, 2]
        # batch: [N]

        n_perm = x_permuted.shape[1]
        targets_permuted = targets_permuted.reshape(-1, n_perm, 2)
        # [bz, n_perm, 2]

        ## Apply MLP
        x_permuted = self.mlp(x_permuted)
        # [N, n_perm, d]

        loss_mse = 0
        N = 0
        for b in range(batch.unique()):
            x_permuted_ = x_permuted[batch == b]
            targets_permuted_ = targets_permuted[b]
            # [n_nodes, n_perm, d]
            # [n_perm, 2]

            ## 1. Calculate distace matrix L_hat
            L_hat = self.ConDistance(x_permuted_)
            # [n_perms, n_nodes, n_nodes]

            ## 2. Create target distance L
            L = self.create

            ## 3. Calculate MSE loss
            loss_mse += self.criterion_mse(L_hat, L)

            N += 1

        loss_mse = 1 / N * loss_mse

        return loss_mse
