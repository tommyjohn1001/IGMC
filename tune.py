import optuna
from optuna.integration import PyTorchLightningPruningCallback

from all_packages import *
from utils import *


def objective(trial: optuna.trial.Trial) -> float:
    ## Define to-be-optimized hparams
    batch_size = trial.suggest_categorical("batch_size", [50, 64, 80, 100])
    lr = trial.suggest_float("lr", 5e-4, 5e-3)
    # max_epochs = trial.suggest_categorical("max_epochs", [60, 80, 100, 120, 200])
    arr = trial.suggest_float("ARR", 5e-4, 5e-2)
    hid_dim = trial.suggest_categorical("hid_dim", [64, 100, 120])
    max_walks = trial.suggest_categorical("max_walks", [20, 30, 40, 50])

    args, config_dataset = get_args()

    args.max_walks = max_walks

    hparams = {
        "batch_size": batch_size,
        "num_workers": 10,
        "lr": lr,
        "max_epochs": 20,
        "gradient_clip_val": 0,
        "regression": True,
        "weight_decay": 0,
        "ARR": arr,
        "contrastive": args.contrastive,
        "temperature": 0.1,
        "lr_scheduler": config_dataset["lr_scheduler"],
        "hid_dim": hid_dim,
    }

    (
        train_graphs,
        val_graphs,
        test_graphs,
        u_features,
        v_features,
        class_values,
    ) = get_train_val_datasets(args, tuning=True)

    train_loader, val_loader, _ = get_loaders(train_graphs, val_graphs, test_graphs, hparams)
    model = get_model(args, hparams, train_graphs, u_features, v_features, class_values)
    trainer_train, trainer_eval, path_dir_ckpt = get_trainer(args, hparams)
    lit_model = IGMCLitModel(model, hparams)

    trainer_train.logger.log_hyperparams(hparams)

    trainer_train.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    rmse = final_test_model(path_dir_ckpt, lit_model, trainer_eval, val_loader)

    return rmse


if __name__ == "__main__":

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=20)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
