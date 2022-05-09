from all_packages import *
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from . import ContrasLearnLitData, ContrasLearnLitModel


def get_args():
    parser = argparse.ArgumentParser(description="Train Regularization using Contrastive Learning")
    # general settings

    parser.add_argument("--gpus", "-g", default="0")
    parser.add_argument("--ckpt", "-c", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--superpod", action="store_true")
    parser.add_argument("--path_dir_dataset", type=str, default="regularization/data")
    parser.add_argument("--path_mlp_weights", type=str, default="weights/mlp.pt")

    parser.add_argument("--hop", type=int, default=1)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--max-nodes-per-hop", default=10000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pe_dim", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--num_warmup_epochs", type=int, default=10)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)

    args = parser.parse_args()

    ## Set up some configurations
    args.gpus = [int(x) for x in args.gpus.split(",")]

    return args


def get_litmodel(args):
    litmodel = ContrasLearnLitModel(
        args.pe_dim,
        tau=args.tau,
        eps=args.eps,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        lr=args.lr,
        num_warmup_epochs=args.num_warmup_epochs,
        num_train_epochs=args.num_train_epochs,
    )

    return litmodel


def get_trainer(args):
    root_logging = "logs"
    if args.superpod:
        now = datetime.now().strftime("%b%d_%H-%M-%S")
    else:
        now = (datetime.now() + timedelta(hours=7)).strftime("%b%d_%H-%M-%S")

    additional_info = []
    if args.superpod:
        additional_info.append("superpod")
    if len(args.gpus) > 1:
        additional_info.append("multi")
    additional_info = f"{'_'.join(additional_info)}_" if len(additional_info) > 0 else ""
    name = f"{additional_info}{now}"

    callback_ckpt = ModelCheckpoint(
        dirpath=osp.join(root_logging, "ckpts", name),
        filename="{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    callback_tqdm = TQDMProgressBar(refresh_rate=5)
    callback_lrmornitor = LearningRateMonitor(logging_interval="step")
    logger_tboard = TensorBoardLogger(
        root_logging,
        name=name,
        version=now,
    )
    # logger_wandb = WandbLogger(name, root_logging, project="IGMC_RegTrick")

    trainer = plt.Trainer(
        gpus=args.gpus,
        max_epochs=args.num_train_epochs,
        gradient_clip_val=args.gradient_clip_val,
        strategy="ddp" if len(args.gpus) > 1 else None,
        # log_every_n_steps=5,
        callbacks=[callback_ckpt, callback_tqdm, callback_lrmornitor],
        # logger=logger_wandb if args.wandb else logger_tboard,
        logger=logger_tboard,
    )

    return trainer


def get_litdata(args):
    litdata = ContrasLearnLitData(args.path_dir_dataset, batch_size=args.batch_size)

    return litdata
