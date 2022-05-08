from re import L

from all_packages import *

from regularization.create_data import create_data
from regularization.utils import get_args, get_litdata, get_litmodel, get_trainer


def train():
    ## Get args
    args = get_args()

    ## Create data if not available
    if not osp.isdir(args.path_dir_dataset):
        logger.info("Data not created. Creating...")

        create_data(args, num_workers=8)

    ## Create model, Trainer
    logger.info("Start training")

    lit_model = get_litmodel(args)
    lit_data = get_litdata(args)
    trainer = get_trainer(args)

    trainer.fit(lit_model, lit_data)

    ## Save weights of MLP
    logger.info("Save weights of MLP")

    path_dir_weights = osp.dirname(args.path_mlp_weights)
    if not osp.isdir(path_dir_weights):
        os.makedirs(path_dir_weights, exist_ok=True)
    torch.save(lit_model.model.mlp.state_dict(), args.path_mlp_weights)


if __name__ == "__main__":
    train()
