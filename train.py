from all_packages import *
from utils import *

dotenv.load_dotenv(override=True)

if __name__ == "__main__":
    args = get_args()

    hparams = {
        "batch_size": args.batch_size,
        "num_workers": 10,
        "lr": args.lr,
        "max_epochs": args.epochs,
        "init_lr": args.init_lr,
        "gradient_clip_val": 0,
        "regression": True,
        "weight_decay": 0,
        "percent_warmup": args.percent_warmup,
        "ARR": args.ARR,
        "contrastive": 0,
        "temperature": 0.1,
    }

    train_graphs, test_graphs, u_features, v_features, class_values = get_train_val_datasets(args)
    hparams["num_training_steps"] = int(
        len(train_graphs) * hparams["max_epochs"] / args.batch_size
    )
    print("All ratings are:")
    print(class_values)

    train_loader, val_loader = get_loaders(train_graphs, test_graphs, hparams)
    model = get_model(args, hparams, train_graphs, u_features, v_features, class_values)
    trainer, path_dir_ckpt = get_trainer(args, hparams)
    path_dir_ckpt = "logs/ckpts/yahoo_music_nrw0.1_Dec19_15-25-09"
    lit_model = IGMCLitModel(model, hparams)

    # trainer.fit(
    #     lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt
    # )

    if args.ensemble:
        final_test_model(path_dir_ckpt, lit_model, trainer, val_loader)
