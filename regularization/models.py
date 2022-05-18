from all_packages import *
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Data
from tqdm import tqdm

import regularization.utils as reg_utils
from regularization.mlp import MLP


def get_linear_schedule_with_warmup(optimizer, num_warmup_epochs, num_train_epochs, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_epochs (`int`):
            The number of steps for the warmup phase.
        num_train_epochs (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_epoch: int):
        if current_epoch < num_warmup_epochs:
            return float(current_epoch) / float(max(1, num_warmup_epochs))
        return max(
            0.0,
            float(num_train_epochs - current_epoch)
            / float(max(1, num_train_epochs - num_warmup_epochs)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class ContrastiveDistance(nn.Module):
    def __init__(self, tau=0.07, eps=1e-8, metric="L2"):
        super().__init__()

        self.tau = tau
        self.eps = eps
        self.metric = metric

    def get_sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        # a, b: [n_max, d]

        a_n, b_n = a.norm(dim=-1, keepdim=True), b.norm(dim=-1, keepdim=True)

        if self.metric == "cosine":
            a_norm = a / torch.clamp(a_n, min=eps)
            b_norm = b / torch.clamp(b_n, min=eps)
            sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
        elif self.metric == "L1":
            sim_mt = torch.cdist(a_n, b_n, p=1)
        elif self.metric == "L2":
            sim_mt = torch.cdist(a_n, b_n, p=2)
        else:
            raise NotImplementedError()

        # sim_mt: [n_max, n_max]

        return sim_mt

    def forward(self, x: Tensor):
        # x: [n_max, d]

        n_max = x.shape[0]

        device, dtype = x.device, x.dtype

        ## Calculate sim matrix
        sim_matrix = self.get_sim_matrix(x, x, self.eps)
        # [n_max, n_max]

        ## Divide by temperature
        sim_matrix = torch.div(sim_matrix, self.tau)

        ## Numerically stable trick
        sim_matrix_max, _ = torch.max(sim_matrix, dim=-1, keepdim=True)
        sim_matrix = sim_matrix - sim_matrix_max.detach()
        # [n_max, n_max]

        ## Mask out self-contrasted cases
        mask_selfcontrs = torch.ones(n_max, device=device, dtype=dtype).diag()
        mask_selfcontrs.masked_fill_(mask_selfcontrs == 1, float("-inf"))
        # [n_max, n_max]
        sim_matrix_masked = sim_matrix + mask_selfcontrs
        # [n_max, n_max]

        ## Calculate distance matrix
        sim_matrix_logsumexp = torch.logsumexp(sim_matrix_masked, dim=-1, keepdim=True)
        # [n_max, n_max]
        distance = sim_matrix_logsumexp - sim_matrix
        # [n_max, n_max]

        return distance


class ContrastiveModel(nn.Module):
    def __init__(
        self,
        pe_dim=40,
        tau=0.07,
        eps=1e-8,
        dropout=0.25,
        metric="L2",
        n_perm_subgraphs=4,
        n_samples_equi=4,
        alpha=0,
        beta=2,
    ):
        super().__init__()

        self.metric = metric
        self.n_perm_subgraphs, self.n_samples_equi = n_perm_subgraphs, n_samples_equi
        self.alpha, self.beta = alpha, beta
        self.pe_dim = pe_dim

        self.ConDistance = ContrastiveDistance(tau=tau, eps=eps, metric=metric)
        self.mlp = MLP(d=pe_dim, dropout=dropout)
        self.criterion_mse = nn.MSELoss()

    def forward(self, permuted_graphs, device, dtype):
        loss_target_dis = 0
        loss_equi = 0

        for entry in permuted_graphs:
            entry = entry[0]

            for subgraph in entry:
                pe_origin, L_trg_origin, pe_transforms, transforms = subgraph
                pe_origin = pe_origin.to(device=device, dtype=dtype)
                L_trg_origin = L_trg_origin.to(device=device, dtype=dtype)

                pe_origin = self.mlp(pe_origin)

                L_hat_origin = self.ConDistance(pe_origin)
                # [n_max, n_max]

                loss_target_dis = self.criterion_mse(L_hat_origin, L_trg_origin)

                ## For Equivariance
                loss_equi_ = 0
                for pe_transf, transf in zip(pe_transforms, transforms):
                    pe_transf = pe_transf.to(device=device, dtype=dtype)
                    transf = transf.to(device=device, dtype=dtype)

                    L_hat_transform = self.ConDistance(pe_transf)
                    # [n_max, n_max]

                    loss_equi_ += self.criterion_mse(L_hat_transform, transf @ L_hat_origin)

                loss_equi += 1 / self.n_samples_equi * loss_equi_

            loss_contrastive = 1 / self.n_perm_subgraphs * (loss_target_dis + loss_equi)

        return loss_contrastive
