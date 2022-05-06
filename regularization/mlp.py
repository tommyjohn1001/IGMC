from all_packages import *


class MLP(nn.Module):
    def __init__(self, d: int, dropout=0.25) -> None:
        super().__init__()
        self.core = nn.Sequential(
            nn.Linear(d, d),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(d, d),
            nn.Dropout(dropout),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.core(x)

        return out
