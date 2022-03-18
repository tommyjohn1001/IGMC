import argparse
import os
import random
from datetime import datetime, timedelta

SEED = random.randint(0, 1000)


def flixster(pe_dim, scenario):
    now = (datetime.now() + timedelta(hours=7)).strftime("%b%d_%H:%M")

    os.system(
        f"python Main.py\
            --data-name flixster\
            --epochs 40\
            --testing\
            --ensemble\
            --save-appendix _gatedGCN_{pe_dim}_{scenario}\
            --data-appendix _{pe_dim}\
            --seed {SEED}\
            --pe-dim {pe_dim}\
            --scenario {scenario} > logs/flixster_{scenario}_{pe_dim}_{now}.log"
        # --wandb
    )


def douban(pe_dim, scenario):
    now = (datetime.now() + timedelta(hours=7)).strftime("%b%d_%H:%M")

    os.system(
        f"python Main.py\
            --data-name douban\
            --epochs 40\
            --testing\
            --ensemble\
            --save-appendix _gatedGCN_{pe_dim}_{scenario}\
            --data-appendix _{pe_dim}\
            --seed {SEED}\
            --pe-dim {pe_dim}\
            --scenario {scenario} > logs/douban_{scenario}_{pe_dim}_{now}.log"
        # --wandb
    )


def yahoo(pe_dim, scenario):
    now = (datetime.now() + timedelta(hours=7)).strftime("%b%d_%H:%M")

    os.system(
        f"python Main.py\
            --data-name yahoo_music\
            --epochs 100\
            --ensemble\
            --save-appendix _gatedGCN_{pe_dim}_{scenario}\
            --data-appendix _{pe_dim}\
            --seed {SEED}\
            --pe-dim {pe_dim}\
            --scenario {scenario} > logs/yahoo_{scenario}_{pe_dim}_{now}.log"
        # --wandb
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inductive Graph-based Matrix Completion")
    parser.add_argument("--scenario", type=int)

    args = parser.parse_args()

    pe_dims = {
        "yahoo_music": 140,
        "douban": 115,
        "flixster": 86,
    }

    flixster(pe_dims["flixster"], args.scenario)
    douban(pe_dims["douban"], args.scenario)
    yahoo(pe_dims["yahoo_music"], args.scenario)

    # python Main.py --data-name yahoo_music --epochs 40 --testing --ensemble --save-appendix _80 --data-appendix _80 --seed 42 --pe-dim 80
