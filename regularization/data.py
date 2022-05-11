from all_packages import *


def collate_fn(batch):
    x, trg = [torch.from_numpy(_[0]) for _ in batch], [torch.from_numpy(_[1]) for _ in batch]
    # x: list of [n*, d]
    # trg: list of [n*, n*]

    device, dtype = x[0].device, x[0].dtype
    n_max = 420  # maximum number of nodes
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
    def __init__(self, path_dataset) -> None:
        super().__init__()

        self.dataset = None

        self.load_dataset(path_dataset)

    def load_dataset(self, path_dataset):
        if not osp.isfile(path_dataset):
            logger.error(f"Path to dataset invalid: {path_dataset}")
            sys.exit(1)

        self.dataset = torch.load(path_dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class ContrasLearnLitData(plt.LightningDataModule):
    def __init__(self, path_train_dataset, path_val_dataset, batch_size=50):
        super().__init__()

        self.path_train_dataset = path_train_dataset
        self.path_val_dataset = path_val_dataset
        self.batch_size = batch_size

        self.data_train, self.data_val = None, None

    def setup(self, stage):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""

        self.data_train = ContrasLearnDataset(self.path_train_dataset)
        self.data_val = ContrasLearnDataset(self.path_val_dataset)

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
