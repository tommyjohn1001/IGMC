from all_packages import *
from utils import *

dotenv.load_dotenv(override=True)

if __name__ == "__main__":
    args = get_args()

    logger.info(f"Command: {' '.join(sys.argv)}")

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
    }

    train_graphs, test_graphs, u_features, v_features, class_values = get_train_val_datasets(args)
    print("All ratings are:")
    print(class_values)

    train_loader, val_loader = get_loaders(train_graphs, test_graphs, hparams)
    model = get_model(args, hparams, train_graphs, u_features, v_features, class_values)
    trainer, path_dir_ckpt = get_trainer(args, hparams)
    lit_model = IGMCLitModel(model, hparams)

    trainer.fit(
        lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt
    )

    if args.ensemble is True:
        final_test_model(path_dir_ckpt, lit_model, trainer, val_loader)
