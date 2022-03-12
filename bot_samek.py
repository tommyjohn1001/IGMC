import argparse
import os
import random

SEED = random.randint(0, 1000)


def flix_dou_yah(dataset, pe_dim, scenario):
    os.system(
        f"python Main.py --data-name {dataset} --epochs 40\
            --testing --ensemble\
            --save-appendix _gatedGCN_{pe_dim}_{scenario}\
            --data-appendix _{pe_dim}\
            --seed {SEED}\
            --pe-dim {pe_dim}\
            --scenario {scenario} > logs/{dataset}_{pe_dim}_{scenario}.log"
            # --wandb
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inductive Graph-based Matrix Completion")
    parser.add_argument("--scenario", type=int)

    args = parser.parse_args()

    pe_dims = {
        "yahoo_music": 20,
        "douban": 20,
        "flixster": 20,
    }

    for dataset in ["yahoo_music", "douban", "flixster"]:
        flix_dou_yah(dataset, pe_dims[dataset], args.scenario)

    # python Main.py --data-name yahoo_music --epochs 40 --testing --ensemble --save-appendix _80 --data-appendix _80 --seed 42 --pe-dim 80
