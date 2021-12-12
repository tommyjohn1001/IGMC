from all_packages import *
from utils import IGMCLitModel, get_args, get_model, get_train_val_datasets

if __name__ == "__main__":
    args = get_args()

    root_logging = "logs"

    hparams = {
        "batch_size": args.batch_size,
        "num_workers": 10,
        "lr": 2e-3,
        "max_epochs": 100,
        "init_lr": 8e-5,
        "gradient_clip_val": 1,
        "ARR": args.ARR,
        "regression": True,
    }

    train_graphs, test_graphs, u_features, v_features, class_values = get_train_val_datasets(args)
    train_loader = DataLoader(
        train_graphs, hparams["batch_size"], shuffle=True, num_workers=hparams["num_workers"]
    )
    val_loader = DataLoader(
        test_graphs, hparams["batch_size"], shuffle=False, num_workers=hparams["num_workers"]
    )
    model = get_model(args, train_graphs, u_features, v_features, class_values)

    ## Create things belonging to pytorch lightning
    now = datetime.now() + timedelta(hours=7)
    callback_ckpt = ModelCheckpoint(
        dirpath=root_logging, monitor="val_acc", filename="{epoch}-{val_acc:.2f}", mode="max"
    )
    callback_tqdm = TQDMProgressBar(refresh_rate=5)
    callback_lrmornitor = LearningRateMonitor(logging_interval="step")
    logger_tboard = TensorBoardLogger(root_logging, version=now.strftime("%b%d-%H:%M:%S"))
    # logger_wandb = WandbLogger()

    lit_model = IGMCLitModel(model, hparams)

    trainer = Trainer(
        gpus=args.gpus,
        max_epochs=hparams["max_epochs"],
        gradient_clip_val=hparams["gradient_clip_val"],
        callbacks=[callback_ckpt, callback_tqdm, callback_lrmornitor],
        # FIXME: uncomment this and remove the following
        # logger=logger_wandb,
        logger=logger_tboard,
    )

    trainer.fit(
        lit_model, train_dataloader=train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt
    )

    # final_test_model(args)
