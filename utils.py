from torch_geometric.data import Batch

import util_functions
from all_packages import *
from data_utils import *
from preprocessing import *
from utils_model.models import *


def get_train_val_datasets(args, combine_trainval=False):
    rating_map, post_rating_map = None, None
    if args.standard_rating:
        if args.data_name in ["flixster", "ml_10m"]:  # original 0.5, 1, ..., 5
            rating_map = {x: int(math.ceil(x)) for x in np.arange(0.5, 5.01, 0.5).tolist()}
        elif args.data_name == "yahoo_music":  # original 1, 2, ..., 100
            rating_map = {x: (x - 1) // 20 + 1 for x in range(1, 101)}
        else:
            rating_map = None

    if args.data_name in ["ml_1m", "ml_10m", "ml_25m"]:
        if args.use_features:
            datasplit_path = (
                "raw_data/"
                + args.data_name
                + "/withfeatures_split_seed"
                + str(args.data_seed)
                + ".pickle"
            )
        else:
            datasplit_path = (
                "raw_data/" + args.data_name + "/split_seed" + str(args.data_seed) + ".pickle"
            )
    elif args.use_features:
        datasplit_path = "raw_data/" + args.data_name + "/withfeatures.pickle"
    else:
        datasplit_path = "raw_data/" + args.data_name + "/nofeatures.pickle"

    if args.data_name in ["flixster", "douban", "yahoo_music"]:
        (
            u_features,
            v_features,
            adj_train,
            train_labels,
            train_u_indices,
            train_v_indices,
            val_labels,
            val_u_indices,
            val_v_indices,
            test_labels,
            test_u_indices,
            test_v_indices,
            class_values,
        ) = load_data_monti(args.data_name, combine_trainval, rating_map, post_rating_map)
    elif args.data_name == "ml_100k":
        print("Using official MovieLens split u1.base/u1.test with 20% validation...")
        (
            u_features,
            v_features,
            adj_train,
            train_labels,
            train_u_indices,
            train_v_indices,
            val_labels,
            val_u_indices,
            val_v_indices,
            test_labels,
            test_u_indices,
            test_v_indices,
            class_values,
        ) = load_official_trainvaltest_split(
            args.data_name, combine_trainval, rating_map, post_rating_map, args.ratio
        )
    else:
        (
            u_features,
            v_features,
            adj_train,
            train_labels,
            train_u_indices,
            train_v_indices,
            val_labels,
            val_u_indices,
            val_v_indices,
            test_labels,
            test_u_indices,
            test_v_indices,
            class_values,
        ) = create_trainvaltest_split(
            args.data_name,
            1234,
            combine_trainval,
            datasplit_path,
            True,
            True,
            rating_map,
            post_rating_map,
            args.ratio,
        )

    train_indices = (train_u_indices, train_v_indices)
    val_indices = (val_u_indices, val_v_indices)
    test_indices = (test_u_indices, test_v_indices)

    mode = "trainval_test" if combine_trainval is True else "train_val_test"
    dataset_class = "MyDynamicDataset" if args.dynamic_train else "MyDataset"
    train_graphs = getattr(util_functions, dataset_class)(
        f"data/{mode}/{args.data_name}/train",
        adj_train,
        train_indices,
        train_labels,
        args.hop,
        args.sample_ratio,
        args.max_nodes_per_hop,
        u_features,
        v_features,
        class_values,
        max_num=args.max_train_num,
    )
    test_graphs = getattr(util_functions, dataset_class)(
        f"data/{mode}/{args.data_name}/test",
        adj_train,
        test_indices,
        test_labels,
        args.hop,
        args.sample_ratio,
        args.max_nodes_per_hop,
        u_features,
        v_features,
        class_values,
        max_num=args.max_test_num,
    )
    val_graphs = getattr(util_functions, dataset_class)(
        f"data/{mode}/{args.data_name}/val",
        adj_train,
        val_indices,
        val_labels,
        args.hop,
        args.sample_ratio,
        args.max_nodes_per_hop,
        u_features,
        v_features,
        class_values,
        max_num=args.max_val_num,
    )

    if not combine_trainval:
        logger.info(
            f"Data info: train: {len(train_graphs)} - val: {len(val_graphs)} - test: {len(test_graphs)}"
        )
    else:
        logger.info(f"Data info: train: {len(train_graphs)} - test: {len(test_graphs)}")

    return train_graphs, val_graphs, test_graphs, u_features, v_features, class_values


def get_args():
    parser = argparse.ArgumentParser(description="Inductive Graph-based Matrix Completion")
    # general settings

    parser.add_argument("--gpus", "-g", default="0")
    parser.add_argument("--ckpt", "-c", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--superpod", action="store_true")
    parser.add_argument("--combine_trainval", action='store_true')
    parser.add_argument(
        "--ARR",
        type=float,
        default=0.001,
        help="The adjacenct rating regularizer. If not 0, regularize the \
                        differences between graph convolution parameters W associated with\
                        adjacent ratings",
    )
    parser.add_argument(
        "--contrastive",
        type=float,
        default=0,
        help="Contrastive loss. If not 0, use constrastive loss",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        choices=[1, 2],
        help="Switch between Naive Reasoning Walking ver 1 and 2",
    )
    parser.add_argument("--hid-dim", type=int, default=64)

    ################################################################################################################

    parser.add_argument(
        "--testing",
        action="store_true",
        default=False,
        help="if set, use testing mode which splits all ratings into train/test;\
                        otherwise, use validation model which splits all ratings into \
                        train/val/test and evaluate on val only",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        default=False,
        help="if set, skip the training and directly perform the \
                        transfer/ensemble/visualization",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="turn on debugging mode which uses a small number of data",
    )
    parser.add_argument("--data-name", default="ml_100k", help="dataset name")
    parser.add_argument(
        "--data-appendix", default="", help="what to append to save-names when saving datasets"
    )
    parser.add_argument(
        "--save-appendix", default="", help="what to append to save-names when saving results"
    )
    parser.add_argument(
        "--max-train-num", type=int, default=None, help="set maximum number of train data to use"
    )
    parser.add_argument(
        "--max-val-num", type=int, default=None, help="set maximum number of val data to use"
    )
    parser.add_argument(
        "--max-test-num", type=int, default=None, help="set maximum number of test data to use"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=1234,
        metavar="S",
        help="seed to shuffle data (1234,2341,3412,4123,1324 are used), \
                        valid only for ml_1m and ml_10m",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        default=False,
        help="if True, reprocess data instead of using prestored .pkl data",
    )
    parser.add_argument(
        "--dynamic-train",
        action="store_true",
        default=False,
        help="extract training enclosing subgraphs on the fly instead of \
                        storing in disk; works for large datasets that cannot fit into memory",
    )
    parser.add_argument("--dynamic-test", action="store_true", default=False)
    parser.add_argument("--dynamic-val", action="store_true", default=False)
    parser.add_argument(
        "--keep-old",
        action="store_true",
        default=False,
        help="if True, do not overwrite old .py files in the result folder",
    )
    parser.add_argument(
        "--save-interval", type=int, default=10, help="save model states every # epochs "
    )
    # subgraph extraction settings
    parser.add_argument("--hop", default=1, metavar="S", help="enclosing subgraph hop number")
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="if < 1, subsample nodes per hop according to the ratio",
    )

    parser.add_argument(
        "--max-nodes-per-hop",
        default=10000,
        help="if > 0, upper bound the # nodes per hop by another subsampling",
    )
    parser.add_argument(
        "--use-features",
        action="store_true",
        default=False,
        help="whether to use node features (side information)",
    )
    # edge dropout settings
    parser.add_argument(
        "--adj-dropout",
        type=float,
        default=0.2,
        help="if not 0, random drops edges from adjacency matrix with this prob",
    )
    parser.add_argument(
        "--force-undirected",
        action="store_true",
        default=False,
        help="in edge dropout, force (x, y) and (y, x) to be dropped together",
    )
    # optimization settings
    parser.add_argument(
        "--continue-from",
        type=int,
        default=None,
        help="from which epoch's checkpoint to continue training",
    )
    parser.add_argument("--max_neighbors", type=int, default=50)
    parser.add_argument("--max_walks", type=int, default=21)
    parser.add_argument(
        "--lr", type=float, default=1, metavar="LR", help="learning rate (default: 1)"
    )
    parser.add_argument(
        "--lr-decay-step-size", type=int, default=50, help="decay lr by factor A every B steps"
    )
    parser.add_argument(
        "--lr-decay-factor", type=float, default=0.1, help="decay lr by factor A every B steps"
    )
    parser.add_argument(
        "--epochs", type=int, default=80, metavar="N", help="number of epochs to train"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, metavar="N", help="batch size during training"
    )
    parser.add_argument(
        "--test-freq", type=int, default=1, metavar="N", help="test every n epochs"
    )

    # transfer learning, ensemble, and visualization settings
    parser.add_argument(
        "--transfer", default="", help="if not empty, load the pretrained models in this path"
    )
    parser.add_argument(
        "--num-relations",
        type=int,
        default=5,
        help="if transfer, specify num_relations in the transferred model",
    )
    parser.add_argument(
        "--multiply-by",
        type=int,
        default=1,
        help="if transfer, specify how many times to multiply the predictions by",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="if True, load a pretrained model and do visualization exps",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        default=False,
        help="if True, load a series of model checkpoints and ensemble the results",
    )
    parser.add_argument(
        "--standard-rating",
        action="store_true",
        default=False,
        help="if True, maps all ratings to standard 1, 2, 3, 4, 5 before training",
    )
    parser.add_argument("--exp_name", type=str)
    # sparsity experiment settings
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="For ml datasets, if ratio < 1, downsample training data to the\
                        target ratio",
    )

    """
        Set seeds, prepare for transfer learning (if --transfer)
    """
    args = parser.parse_args()

    ## Set up some configurations
    path_config_datasets = "config_datasets.yaml"
    assert os.path.isfile(f"./{path_config_datasets}")
    with open(path_config_datasets) as stream:
        try:
            config_dataset = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit()

    if not args.data_name in config_dataset.keys():
        print(f"Err: Config of dataset {args.data_name} not in: config_datasets.yaml")
        exit(1)

    config_dataset = config_dataset[args.data_name]
    for k, v in config_dataset.items():
        if k == "lr_scheduler":
            for sched in v:
                range_epoch = sched["range_epoch"]
                range_lr = sched["range_lr"]

                sched["range_epoch"] = list(map(int, range_epoch.split(",")))
                sched["range_lr"] = list(map(float, range_lr.split(",")))

        setattr(args, k, v)

    args.gpus = [int(x) for x in args.gpus.split(",")]

    return args, config_dataset


def get_model(args, hparams, train_dataset, u_features, v_features, class_values):
    if args.use_features:
        u_features, v_features = u_features.toarray(), v_features.toarray()
        n_features = u_features.shape[1] + v_features.shape[1]
        print(
            "Number of user features {}, item features {}, total features {}".format(
                u_features.shape[1], v_features.shape[1], n_features
            )
        )
    else:
        u_features, v_features = None, None
        n_features = 0

    if args.transfer:
        num_relations = args.num_relations
        multiply_by = args.multiply_by
    else:
        num_relations = len(class_values)
        multiply_by = 1

    if args.version == 1:
        model = "IGMC"
    elif args.version == 2:
        model = "IGMC2"
    else:
        raise NotImplementedError()

    latent_dim_each = hparams["hid_dim"] // 4
    model = eval(model)(
        train_dataset,
        latent_dim=[latent_dim_each, latent_dim_each, latent_dim_each, latent_dim_each],
        # gconv=GATConv,
        num_relations=num_relations,
        num_bases=4,
        hid_dim=hparams["hid_dim"],
        regression=True,
        adj_dropout=args.adj_dropout,
        force_undirected=args.force_undirected,
        side_features=args.use_features,
        n_side_features=n_features,
        multiply_by=multiply_by,
        batch_size=args.batch_size,
        max_neighbors=args.max_neighbors,
        max_walks=args.max_walks,
        class_values=class_values,
        ARR=hparams["ARR"],
        temperature=hparams["temperature"],
    )

    return model


def get_trainer(args, hparams):
    root_logging = "logs"
    if args.superpod:
        now = datetime.now().strftime("%b%d_%H-%M-%S")
    else:
        now = (datetime.now() + timedelta(hours=7)).strftime("%b%d_%H-%M-%S")

    additional_info = []
    additional_info.append(str(args.version))
    if args.contrastive > 0:
        additional_info.append("contrs")
    if args.superpod:
        additional_info.append("superpod")
    if len(args.gpus) > 1:
        additional_info.append("multi")
    additional_info = f"{'_'.join(additional_info)}_" if len(additional_info) > 0 else ""
    name = f"{args.data_name}_{args.exp_name}_{additional_info}{now}"

    path_dir_ckpt = osp.join(root_logging, "ckpts", name)

    callback_ckpt = ModelCheckpoint(
        dirpath=path_dir_ckpt,
        filename="{epoch}-{val_loss:.3f}",
        monitor="epoch",
        mode="max",
        save_top_k=5,
        every_n_epochs=args.save_interval
        # monitor="val_loss",
        # mode="min",
        # save_top_k=3,
        # save_last=True,
    )
    callback_tqdm = TQDMProgressBar(refresh_rate=5)
    callback_lrmornitor = LearningRateMonitor(logging_interval="step")
    logger_tboard = TensorBoardLogger(
        root_logging,
        name=name,
        version=now,
    )
    logger_wandb = WandbLogger(name, root_logging)

    trainer_train = Trainer(
        gpus=args.gpus,
        max_epochs=hparams["max_epochs"],
        gradient_clip_val=hparams["gradient_clip_val"],
        strategy="ddp" if len(args.gpus) > 1 else None,
        # log_every_n_steps=5,
        callbacks=[callback_ckpt, callback_tqdm, callback_lrmornitor],
        logger=logger_wandb if args.wandb else logger_tboard,
    )

    trainer_eval = Trainer(
        gpus=[args.gpus[0]], strategy=None, logger=logger_wandb if args.wandb else logger_tboard
    )

    return trainer_train, trainer_eval, path_dir_ckpt


def get_loaders(train_graphs, val_graphs, test_graphs, hparams):
    train_loader = DataLoader(
        train_graphs,
        hparams["batch_size"],
        shuffle=True,
        num_workers=hparams["num_workers"],
        collate_fn=lambda batch: Batch.from_data_list(batch, []),
    )
    val_loader = DataLoader(
        val_graphs,
        hparams["batch_size"],
        shuffle=False,
        num_workers=hparams["num_workers"],
        collate_fn=lambda batch: Batch.from_data_list(batch, []),
    )
    test_loader = DataLoader(
        test_graphs,
        hparams["batch_size"],
        shuffle=False,
        num_workers=hparams["num_workers"],
        collate_fn=lambda batch: Batch.from_data_list(batch, []),
    )

    return train_loader, val_loader, test_loader


def final_test_model(path_dir_ckpt, model, trainer, test_loader):
    path_ckpts = glob(osp.join(path_dir_ckpt, "*.ckpt"))
    assert len(path_ckpts) > 0, "No ckpt found"

    preds, trgs = 0, None
    for path_ckpt in path_ckpts:
        outputs = trainer.predict(
            model,
            test_loader,
            return_predictions=True,
            ckpt_path=path_ckpt,
        )

        list_preds, list_trgs = [], []
        for output in outputs:
            list_preds.append(output[0])
            list_trgs.append(output[1])
        preds = preds + torch.cat(list_preds)
        if trgs is None:
            trgs = torch.cat(list_trgs)

    mean_preds = preds / len(path_ckpts)

    mse = F.mse_loss(mean_preds, trgs.view(-1))
    rmse = torch.sqrt(mse).item()

    return rmse
