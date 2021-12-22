from torch.optim.lr_scheduler import LambdaLR


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


def get_custom_lr_scheduler(
    optimizer,
    hparams,
    last_epoch=-1,
):

    # 0 -> 15: warmup with init = 5e-5, peak = 3e-3
    # 15 -> 100: decrease linearly from 1e-3 to 3e-4
    # 100 -> 175: stable at 3e-4
    # 175 -> 250: stable at 7e-5
    init_lr = 5e-5
    peak_lr = hparams["lr"]
    lr_100 = 3e-4
    lr_175 = 7e-5
    epochs_warmup = 20
    epochs_peak = 100
    epochs_100 = 175
    epochs_175 = hparams["max_epochs"]

    def lr_lambda(current_epoch: int):
        if current_epoch < epochs_warmup:
            output_lr = float(current_epoch) / float(max(1, epochs_warmup)) * peak_lr + init_lr
        elif current_epoch < epochs_peak:
            output_lr = (
                max(
                    lr_100,
                    float(epochs_peak - current_epoch)
                    / float(max(1, epochs_peak - epochs_warmup)),
                )
                * peak_lr
            )
        elif current_epoch < epochs_100:
            output_lr = lr_100
        else:
            output_lr = lr_175

        ## Vì mục tiêu của lr scheduler là tạo ra hệ số để sau đó nhân với hparams['lr']
        output_lr = output_lr / peak_lr

        return output_lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)
