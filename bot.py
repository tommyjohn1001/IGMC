import argparse
import os
import random
from datetime import datetime, timedelta

PE_DIMS = {"common": 40, "yahoo_music": 140, "douban": 115, "flixster": 86, "ml_100k": 80}
TIMES = 2


def flix_dou_yah(dataset, pe_dim, scenario, cuda_device, ith):
    now = (datetime.now() + timedelta(hours=7)).strftime("%b%d_%H:%M")
    SEED = random.randint(0, 20)
    epoch = 40

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
    SEED = 0

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
    parser.add_argument("--gpu", "-g", type=int)

    args = parser.parse_args()

    cuda_device = args.gpu

    for dataset in ["yahoo_music", "flixster"]:  # "ml_100k" "yahoo_music", "flixster", "douban"
        for ith in range(TIMES):
            if args.scenario % 2 == 1:
                pe_dim = PE_DIMS["common"]
            else:
                pe_dim = PE_DIMS[dataset]

            # if superpod, use batch_size = 40
            # movielens(dataset, pe_dim, args.scenario, cuda_device, ith, batch_size=30)
            flix_dou_yah(dataset, pe_dim, args.scenario, cuda_device, ith)
