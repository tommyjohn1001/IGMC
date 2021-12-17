from all_packages import *
from utils import *

dotenv.load_dotenv(override=True)

if __name__ == "__main__":
    args = get_args()

    hparams = {
        "batch_size": args.batch_size,
        "num_workers": 10,
        "lr": 2e-3,
        "max_epochs": 100,
        "init_lr": 8e-6,
        "gradient_clip_val": 1,
        "ARR": args.ARR,
        "regression": True,
        "weight_decay": 1e-4,
        "use_contrastive_loss": False,
        "temperature": 0.1,
    }

    train_graphs, test_graphs, u_features, v_features, class_values = get_train_val_datasets(args)
    print("All ratings are:")
    print(class_values)
    train_loader = DataLoader(
        train_graphs, hparams["batch_size"], shuffle=True, num_workers=hparams["num_workers"]
    )
    val_loader = DataLoader(
        test_graphs, hparams["batch_size"], shuffle=False, num_workers=hparams["num_workers"]
    )
    model = get_model(args, hparams, train_graphs, u_features, v_features, class_values)

    ## Create things belonging to pytorch lightning
    hparams["num_training_steps"] = len(train_graphs) * hparams["max_epochs"] / args.batch_size
    root_logging = "logs"
    now = (datetime.now() + timedelta(hours=7)).strftime("%b%d_%H-%M-%S")
    name = f"{args.data_name}_{args.exp_name}_{now}"

    path_dir_ckpt = osp.join(root_logging, "ckpts", name)
    callback_ckpt = ModelCheckpoint(
        dirpath=path_dir_ckpt,
        monitor="val_loss",
        filename="{epoch}-{val_loss:.2f}",
        mode="min",
        save_top_k=3,
    )
    callback_tqdm = TQDMProgressBar(refresh_rate=5)
    callback_lrmornitor = LearningRateMonitor(logging_interval="step")
    logger_tboard = TensorBoardLogger(
        root_logging,
        name=name,
        version=now,
    )
    logger_wandb = WandbLogger(name, root_logging)

    lit_model = IGMCLitModel(model, hparams)

    trainer = Trainer(
        gpus=args.gpus,
        max_epochs=hparams["max_epochs"],
        gradient_clip_val=hparams["gradient_clip_val"],
        callbacks=[callback_ckpt, callback_tqdm, callback_lrmornitor],
        logger=logger_wandb if args.use_wandb else logger_tboard,
    )

    trainer.fit(
        lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt
    )

    if args.ensemble:
        final_test_model(path_dir_ckpt, lit_model, trainer, val_loader)
