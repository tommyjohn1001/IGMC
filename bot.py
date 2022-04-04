import argparse
import os
import random
from datetime import datetime, timedelta

PE_DIMS = {"common": 40, "yahoo_music": 140, "douban": 115, "flixster": 86, "ml_100k": 80}
TIMES = 1


def flix_dou_yah(dataset, pe_dim, scenario, cuda_device, ith):
    now = (datetime.now() + timedelta(hours=7)).strftime("%b%d_%H:%M")
    SEED = 1  # random.randint(0, 10)
    epoch = 20 if dataset == "yahoo_music" else 40
    cuda_device = 0

    os.system(
        f"CUDA_VISIBLE_DEVICES={cuda_device} python Main.py\
            --data-name {dataset}\
            --epochs {epoch}\
            --testing --ensemble\
            --save-appendix _{pe_dim}_{scenario}\
            --data-appendix _{pe_dim}\
            --seed {SEED}\
            --pe-dim {pe_dim}\
            --scenario {scenario} > logs/{dataset}_{scenario}_{ith}_{pe_dim}_{now}.log"
        # --wandb
    )


def movielens(dataset, pe_dim, scenario, cuda_device, ith, batch_size=30):
    now = (datetime.now() + timedelta(hours=7)).strftime("%b%d_%H:%M")
    SEED = 1  # random.randint(0, 10)
    cuda_device = 0

    os.system(
        f"CUDA_VISIBLE_DEVICES={cuda_device} python Main.py\
            --data-name {dataset}\
            --epochs 80\
            --testing --ensemble\
            --save-appendix _{pe_dim}_{scenario}\
            --data-appendix _{pe_dim}\
            --max-nodes-per-hop 200\
            --batch-size {batch_size}\
            --seed {SEED}\
            --pe-dim {pe_dim}\
            --scenario {scenario} > logs/{dataset}_{scenario}_{ith}_{pe_dim}_{now}.log"
        # --dynamic-train\
        # --wandb
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inductive Graph-based Matrix Completion")
    parser.add_argument("--scenario", type=int)

    args = parser.parse_args()

    cuda_device = args.scenario % 2

    for dataset in ["ml_100k"]:  # "ml_100k" "yahoo_music", "flixster", "douban"
        for ith in range(TIMES):
            if args.scenario in [1, 2]:
                pe_dim = 1
            elif args.scenario in [3, 5, 7, 9]:
                pe_dim = PE_DIMS["common"]
            elif args.scenario in [4, 6, 8, 10]:
                pe_dim = PE_DIMS[dataset]
            else:
                raise NotImplementedError()

            # if superpod, use batch_size = 40
            movielens(dataset, pe_dim, args.scenario, cuda_device, ith)
