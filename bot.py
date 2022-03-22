import argparse
import os
import random
from datetime import datetime, timedelta

SEED = random.randint(0, 1000)
PE_DIMS = {
    "common": 20,
    "yahoo_music": 140,
    "douban": 115,
    "flixster": 86,
}
TIMES = 1


def flix_dou_yah(dataset, pe_dim, scenario, cuda_device, ith):
    now = (datetime.now() + timedelta(hours=7)).strftime("%b%d_%H:%M")

    os.system(
        f"CUDA_VISIBLE_DEVICES={cuda_device} python Main.py --data-name {dataset} --epochs 40\
            --testing --ensemble\
            --save-appendix _{pe_dim}_{scenario}\
            --data-appendix _{pe_dim}\
            --seed {SEED}\
            --pe-dim {pe_dim}\
            --scenario {scenario} > logs/{dataset}_{scenario}_{ith}_{pe_dim}_{now}.log"
        # --wandb
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inductive Graph-based Matrix Completion")
    parser.add_argument("--scenario", type=int)

    args = parser.parse_args()

    cuda_device = args.scenario % 2
    for ith in range(TIMES):
        for dataset in ["yahoo_music", "douban", "flixster"]:
            if args.scenario in [1, 2]:
                pe_dim = 1
            if args.scenario in [3, 5, 7, 9]:
                pe_dim = PE_DIMS["common"]
            elif args.scenario in [4, 6, 8, 10]:
                pe_dim = PE_DIMS[dataset]
            else:
                raise NotImplementedError()

            flix_dou_yah(dataset, pe_dim, args.scenario, cuda_device, ith)
