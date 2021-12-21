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
    percent_warmup,
    percent_latter,
    num_training_steps,
    lr=8e-3,
    init_lr=5e-4,
    latter_lr=3e-4,
    last_epoch=-1,
):
    num_warmup_steps = int(percent_warmup * num_training_steps)
    num_latter_steps = int((1 - percent_latter) * num_training_steps)

    def lr_lambda(current_step: int):

        if current_step < num_warmup_steps:
            output_lr = float(current_step) / float(max(1, num_warmup_steps)) * lr + init_lr
        elif current_step > num_latter_steps:
            output_lr = latter_lr
        else:
            output_lr = (
                max(
                    0.0,
                    float(num_training_steps - current_step)
                    / float(max(1, num_training_steps - num_warmup_steps)),
                )
                * lr
            )

        ## Vì mục tiêu của lr scheduler là tạo ra hệ số để sau đó nhân với hparams['lr']
        output_lr = output_lr / lr

        return output_lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)
