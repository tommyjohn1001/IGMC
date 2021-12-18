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
        "use_contrastive_loss": True,
        "temperature": 0.1,
    }

    train_graphs, test_graphs, u_features, v_features, class_values = get_train_val_datasets(args)
    hparams["num_training_steps"] = len(train_graphs) * hparams["max_epochs"] / args.batch_size
    print("All ratings are:")
    print(class_values)

    train_loader, val_loader = get_loaders(train_graphs, test_graphs, hparams)
    model = get_model(args, hparams, train_graphs, u_features, v_features, class_values)
    trainer, path_dir_ckpt = get_trainer(args, hparams)
    lit_model = IGMCLitModel(model, hparams)

    trainer.fit(
        lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt
    )

    if args.ensemble:
        final_test_model(path_dir_ckpt, lit_model, trainer, val_loader)
