from all_packages import *
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR

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
    def __init__(self, tau=0.07, eps=1e-8):
        super().__init__()

        self.tau = tau
        self.eps = eps

    def get_sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        # a, b: [bz, n_max, d]

        a_n, b_n = a.norm(dim=-1, keepdim=True), b.norm(dim=-1, keepdim=True)
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
        # [bz, n_max, n_max]

        return sim_mt

    def forward(self, x: Tensor):
        # x: [bz, n_max, d]

        bz, n_max, _ = x.shape

        device, dtype = x.device, x.dtype

        ## Calculate sim matrix
        sim_matrix = self.get_sim_matrix(x, x, self.eps)
        # [bz, n_max, n_max]

        ## Divide by temperature
        sim_matrix = torch.div(sim_matrix, self.tau)

        ## Numerically stable trick
        sim_matrix_max, _ = torch.max(sim_matrix, dim=-1, keepdim=True)
        sim_matrix = sim_matrix - sim_matrix_max.detach()
        # [bz, n_max, n_max]

        ## Mask out self-contrasted cases
        mask_selfcontrs = torch.ones(n_max, device=device, dtype=dtype).diag()
        mask_selfcontrs = mask_selfcontrs.unsqueeze(0).repeat(bz, 1, 1)
        mask_selfcontrs.masked_fill_(mask_selfcontrs == 1, float("-inf"))
        # [bz, n_max, n_max]
        sim_matrix_masked = sim_matrix + mask_selfcontrs
        # [bz, n_max, n_max]

        ## Calculate distance matrix
        sim_matrix_logsumexp = torch.logsumexp(sim_matrix_masked, dim=-1, keepdim=True)
        # [bz, n_max, n_max]
        distance = sim_matrix_logsumexp - sim_matrix
        # [bz, n_max, n_max]

        ## This line ensures values lie between 0 and 1
        # distance = torch.sigmoid(distance)

        return distance


class ContrastiveModel(nn.Module):
    def __init__(self, d_pe, tau=0.07, eps=1e-8, dropout=0.25):
        super().__init__()

        self.ConDistance = ContrastiveDistance(tau=tau, eps=eps)
        self.mlp = MLP(d=d_pe, dropout=dropout)
        self.criterion_mse = nn.MSELoss()

    def forward(self, X: Tensor, trgs: Tensor, mask: Tensor):
        # X: [bz, n_max, d]
        # mask: [bz]
        # trgs: [N, N]

        bz = X.size(0)

        ## 1. Apply MLP
        X = self.mlp(X)
        # [bz, n_max, d]

        ## 2. Calculate distace matrix L_hat
        L_hat = self.ConDistance(X)
        # [bz, n_max, n_max]

        ## 3. Do some magic to convert L_hat to the same shape with trgs
        L_hat_ = []
        for b in range(bz):
            L_batch, mask_ = L_hat[b], mask[b]
            L_hat_.append(L_batch[:mask_, :mask_])

        L_hat = torch.block_diag(*L_hat_)

        ## 4. Calculate MSE loss
        loss_mse = 1 / bz * self.criterion_mse(L_hat, trgs)

        return loss_mse


#################################################################
# For LitModel
#################################################################


class ContrasLearnLitModel(plt.LightningModule):
    def __init__(
        self,
        d_pe,
        batch_size=50,
        tau=0.07,
        eps=1e-8,
        dropout=0.25,
        weight_decay=1e-2,
        lr=5e-4,
        num_warmup_epochs=10,
        num_train_epochs=50,
    ):
        super().__init__()

        self._hparams = {
            "d_pe": d_pe,
            "batch_size": batch_size,
            "tau": tau,
            "eps": eps,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "lr": lr,
            "num_warmup_epochs": num_warmup_epochs,
            "num_train_epochs": num_train_epochs,
        }
        self.save_hyperparameters()

        self.model = ContrastiveModel(d_pe=d_pe, tau=tau, eps=eps, dropout=dropout)

    def training_step(self, batch: Any, batch_idx: int):
        loss_mse = self.model(*batch)

        self.log("train_loss", loss_mse, on_step=True, on_epoch=True, batch_size=self._hparams['batch_size'])

        return loss_mse

    def validation_step(self, batch: Any, batch_idx: int):
        loss_mse = self.model(*batch)

        self.log("val_loss", loss_mse, on_step=False, on_epoch=True,batch_size=self._hparams['batch_size'])

        return loss_mse

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self._hparams["lr"],
            weight_decay=self._hparams["weight_decay"],
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=self._hparams["lr"])

        scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer, self._hparams["num_warmup_epochs"], self._hparams["num_train_epochs"]
            ),
            "interval": "epoch",  # 'step' or 'epoch'
            "frequency": 1,
        }

        return [optimizer], [scheduler]
