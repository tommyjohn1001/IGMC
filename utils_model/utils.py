import numpy as np
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
    def get_lr(range_epochs, lrs, currect_epoch):
        assert len(range_epochs) == 2
        assert len(lrs) == 2

        if lrs[0] == lrs[1]:
            return lrs[0]

        A = np.array([[range_epochs[0], 1], [range_epochs[1], 1]])
        B = np.array([[lrs[0]], [lrs[1]]])
        X = np.dot(np.linalg.inv(A), B)

        currect_epoch = np.array([currect_epoch, 1])

        final_lr = np.dot(currect_epoch, X).item()

        return final_lr

    def lr_lambda(current_epoch: int):
        if current_epoch < 135:
            output_lr = hparams["lr"]
        elif current_epoch < 165:
            output_lr = hparams["lr"] / 2
        else:
            output_lr = hparams["lr"] / 5

        ## Vì mục tiêu của lr scheduler là tạo ra hệ số để sau đó nhân với hparams['lr']
        output_lr = output_lr / hparams["lr"]

        return output_lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)
