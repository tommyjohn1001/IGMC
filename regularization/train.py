from all_packages import *

from regularization.create_data import create_data
from regularization.utils import get_args, get_litdata, get_litmodel, get_trainer

dotenv.load_dotenv(override=True)


def train():
    ## Get args
    args = get_args()

    path_dir_weights = args.path_dir_mlp_weights
    if not osp.isdir(path_dir_weights):
        os.makedirs(path_dir_weights, exist_ok=True)
    path_mlp_weights = osp.join(
        path_dir_weights, f"mlp_{args.dataset}_{args.pe_dim}_{args.metric}.pt"
    )

    ## Create data if not available
    path_train_dataset = (
        f"regularization/data/train_dataset_{args.dataset}_{args.pe_dim}_{args.metric}.pkl"
    )
    path_val_dataset = (
        f"regularization/data/val_dataset_{args.dataset}_{args.pe_dim}_{args.metric}.pkl"
    )
    if not osp.isfile(path_train_dataset):
        logger.info("Data not created. Creating...")

        data_train, data_val = create_data(args, num_workers=12)

        # Save datasets
        os.makedirs(osp.dirname(path_train_dataset))

        torch.save(data_train, path_train_dataset)
        torch.save(data_val, path_val_dataset)

    ## Create model, Trainer
    logger.info("Start training")

    lit_model = get_litmodel(args)
    lit_data = get_litdata(args)
    trainer = get_trainer(args)

    trainer.fit(lit_model, lit_data)

    ## Save weights of MLP
    logger.info("Save weights of MLP")

    torch.save(lit_model.model.mlp.state_dict(), path_mlp_weights)


if __name__ == "__main__":
    train()
