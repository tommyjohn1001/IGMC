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
    def get_lr(range_epoch, range_lr, currect_epoch):
        assert len(range_epoch) == 2
        assert len(range_lr) == 2

        if range_lr[0] == range_lr[1]:
            return range_lr[0]

        A = np.array([[range_epoch[0], 1], [range_epoch[1], 1]])
        B = np.array([[range_lr[0]], [range_lr[1]]])
        X = np.dot(np.linalg.inv(A), B)

        currect_epoch = np.array([currect_epoch, 1])

        final_lr = np.dot(currect_epoch, X).item()

        return final_lr

    def lr_lambda(current_epoch: int):
        for sched in hparams["lr_scheduler"]:
            range_epoch = sched["range_epoch"]
            range_lr = sched["range_lr"]

            if range_epoch[0] <= current_epoch < range_epoch[1]:
                output_lr = get_lr(range_epoch, range_lr, current_epoch)

                return output_lr

        return 1e-3

    return LambdaLR(optimizer, lr_lambda, last_epoch)
