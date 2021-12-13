from all_packages import *
from data_utils import *
from models import *
from preprocessing import *
from train_eval import *
from train_eval import get_linear_schedule_with_warmup
from util_functions import *


def get_train_val_datasets(args):
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
        ) = load_data_monti(args.data_name, args.testing, rating_map, post_rating_map)
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
            args.data_name, args.testing, rating_map, post_rating_map, args.ratio
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
            args.testing,
            datasplit_path,
            True,
            True,
            rating_map,
            post_rating_map,
            args.ratio,
        )

    if args.testing:
        val_test_appendix = "testmode"
    else:
        val_test_appendix = "valmode"
    data_combo = (args.data_name, args.data_appendix, val_test_appendix)
    train_indices = (train_u_indices, train_v_indices)
    val_indices = (val_u_indices, val_v_indices)
    test_indices = (test_u_indices, test_v_indices)

    dataset_class = "MyDynamicDataset" if args.dynamic_train else "MyDataset"
    train_graphs = eval(dataset_class)(
        "data/{}{}/{}/train".format(*data_combo),
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
    dataset_class = "MyDynamicDataset" if args.dynamic_test else "MyDataset"
    test_graphs = eval(dataset_class)(
        "data/{}{}/{}/test".format(*data_combo),
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
    if not args.testing:
        dataset_class = "MyDynamicDataset" if args.dynamic_val else "MyDataset"
        val_graphs = eval(dataset_class)(
            "data/{}{}/{}/val".format(*data_combo),
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

    # Determine testing data (on which data to evaluate the trained model
    if not args.testing:
        test_graphs = val_graphs

    print(
        "Used #train graphs: %d, #test graphs: %d"
        % (
            len(train_graphs),
            len(test_graphs),
        )
    )

    return train_graphs, test_graphs, u_features, v_features, class_values


def get_args():
    parser = argparse.ArgumentParser(description="Inductive Graph-based Matrix Completion")
    # general settings

    parser.add_argument("--gpus", "-g", default="0")
    parser.add_argument("--ckpt", "-c", default=None)
    parser.add_argument("--use_wandb", action="store_true")
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

    parser.add_argument("--percent_warmup", type=float, default=0.15)
    parser.add_argument("--init_lr", type=float, default=5e-4)
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
    parser.add_argument("--max_walks", type=int, default=10)
    parser.add_argument(
        "--lr", type=float, default=1e-3, metavar="LR", help="learning rate (default: 1e-3)"
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
    parser.add_argument(
        "--ARR",
        type=float,
        default=0.001,
        help="The adjacenct rating regularizer. If not 0, regularize the \
                        differences between graph convolution parameters W associated with\
                        adjacent ratings",
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
        if "lr" in k:
            setattr(args, k, float(v))
        else:
            setattr(args, k, v)

    args.gpus = [int(x) for x in args.gpus.split(",")]

    return args


def get_model(args, train_dataset, u_features, v_features, class_values):
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

    model = IGMC(
        train_dataset,
        latent_dim=[32, 32, 32, 32],
        num_relations=num_relations,
        num_bases=4,
        regression=True,
        adj_dropout=args.adj_dropout,
        force_undirected=args.force_undirected,
        side_features=args.use_features,
        n_side_features=n_features,
        multiply_by=multiply_by,
        batch_size=args.batch_size,
        max_neighbors=args.max_neighbors,
        max_walks=args.max_walks,
    )

    return model


class IGMCLitModel(LightningModule):
    def __init__(self, model, hps):
        super().__init__()

        self._hparams = hps
        self.model = model

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        out = self.model(batch)
        if self._hparams["regression"]:
            loss = F.mse_loss(out, batch.y.view(-1))
        else:
            loss = F.nll_loss(out, batch.y.view(-1))

        if self._hparams["ARR"] != 0:
            for gconv in self.model.convs:
                w = torch.matmul(gconv.comp, gconv.weight.view(gconv.num_bases, -1)).view(
                    gconv.num_relations, gconv.in_channels, gconv.out_channels
                )
                reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :]) ** 2)
                loss += self._hparams["ARR"] * reg_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        out = self.model(batch)
        if self._hparams["regression"]:
            loss = F.mse_loss(out, batch.y.view(-1), reduction="sum").item()
        else:
            loss = F.nll_loss(out, batch.y.view(-1), reduction="sum").item()

        return loss, len(batch.y)

    def validation_epoch_end(self, outputs) -> None:
        mse, total = 0, 0
        for output in outputs:
            mse += output[0]
            total += output[1]
        mse_loss = mse / total
        rmse = math.sqrt(mse_loss)

        self.log("val_loss", rmse, on_epoch=True)

        return rmse

    def configure_optimizers(self):
        optimizer = Adam(
            self.model.parameters(),
            lr=self._hparams["lr"],
            weight_decay=self._hparams["weight_decay"],
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=self._hparams["lr"])

        scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer,
                self._hparams["num_training_steps"] * 0.2,
                self._hparams["num_training_steps"],
                self._hparams["init_lr"],
            ),
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }

        return [optimizer], [scheduler]


def final_test_model(path_dir_ckpt, model, trainer, val_loader):
    path_ckpts = glob(osp.join(path_dir_ckpt, "*.ckpt"))
    assert len(path_ckpts) > 0, "No ckpt found"

    rmses = []
    for path_ckpt in path_ckpts:
        rmse = trainer.validate(model, val_loader, path_ckpt)
        rmses.append(rmse)

    rmse = sum(rmses) / len(rmses)

    logger.info(f"Final ensemble RMSE: {rmse:4f}")