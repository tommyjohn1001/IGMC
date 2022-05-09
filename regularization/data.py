from all_packages import *


def collate_fn(batch):
    x, trg = [torch.from_numpy(_[0]) for _ in batch], [torch.from_numpy(_[1]) for _ in batch]
    # x: list of [n*, d]
    # trg: list of [n*, n*]

    device, dtype = x[0].device, x[0].dtype
    n_max = max([_.shape[0] for _ in x])
    d = x[0].size(-1)

    X, trgs, mask = [], None, []

    ## Create mask
    mask = torch.tensor([x_.size(0) for x_ in x])
    # [bz]

    ## Create X
    for x_ in x:
        pad0 = torch.zeros((n_max - x_.size(0), d), device=device, dtype=dtype)
        x_ = torch.cat((x_, pad0), dim=0)
        X.append(x_)
    X = torch.stack(X)
    # [bz, n_max, d]

    trgs = torch.block_diag(*trg)
    # [N, N]

    return X, trgs, mask


class ContrasLearnDataset(Dataset):
    def __init__(self, path_dir_dataset, split) -> None:
        super().__init__()

        self.dataset = None

        self.load_dataset(path_dir_dataset, split)

    def load_dataset(self, path_dir_dataset, split):
        path_dataset = osp.join(path_dir_dataset, f"{split}_dataset.pkl")
        if not osp.isfile(path_dataset):
            logger.error(f"Path to dataset {split} invalid: {path_dataset}")
            sys.exit(1)

        self.dataset = torch.load(path_dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class ContrasLearnLitData(plt.LightningDataModule):
    def __init__(self, path_dir_dataset, batch_size=50):
        super().__init__()

        self.path_dir_dataset = path_dir_dataset
        self.batch_size = batch_size

        self.data_train, self.data_val = None, None

    def setup(self, stage):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if not osp.isdir(self.path_dir_dataset):
            logger.error(f"Path to dir containing dataset incorrect: {self.path_dir_dataset}")
            sys.exit(1)

        self.data_train = ContrasLearnDataset(self.path_dir_dataset, "train")
        self.data_val = ContrasLearnDataset(self.path_dir_dataset, "val")

    def train_dataloader(self):
        """Return DataLoader for training."""
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=8,
        )

    def val_dataloader(self):
        """Return DataLoader for validation."""

        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=8,
        )
