from all_packages import *
from utils import *

dotenv.load_dotenv(override=True)

if __name__ == "__main__":
    args, config_dataset = get_args()

    logger.info(f"Command: {' '.join(sys.argv)}")
    print(json.dumps(config_dataset, indent=2, ensure_ascii=False))

    hparams = {
        "batch_size": args.batch_size,
        "num_workers": 10,
        "lr": args.lr,
        "max_epochs": args.epochs,
        "gradient_clip_val": 0,
        "regression": True,
        "weight_decay": 0,
        "ARR": args.ARR,
        "contrastive": args.contrastive,
        "temperature": 0.1,
        "lr_scheduler": config_dataset["lr_scheduler"],
        "hid_dim": args.hid_dim,
    }

    (
        train_graphs,
        val_graphs,
        test_graphs,
        u_features,
        v_features,
        class_values,
    ) = get_train_val_datasets(args, combine_trainval=True)
    print("All ratings are:")
    print(class_values)

    train_loader, val_loader, test_loader = get_loaders(
        train_graphs, val_graphs, test_graphs, hparams
    )
    model = get_model(args, hparams, train_graphs, u_features, v_features, class_values)
    trainer_train, trainer_eval, path_dir_ckpt = get_trainer(args, hparams)
    lit_model = IGMCLitModel(model, hparams)

    trainer_train.logger.log_hyperparams(hparams)

    if not args.predict:
        trainer_train.fit(
            lit_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.ckpt,
        )
    else:
        path_dir_ckpt = args.ckpt

    logger.info("Start predicting...")

    rmse = final_test_model(path_dir_ckpt, lit_model, trainer_eval, test_loader)
    logger.info(f"Final ensemble RMSE: {rmse:4f}")
