import os

from loguru import logger


def flix_dou_yah(dataset, ith, seed, pe_dim, scenario):
    os.system(
        f"python train.py --data-name {dataset}\
            --wandb -g 0 --version 2\
            --save-appendix _gatedGCN_{pe_dim}_{scenario}_{ith}\
            --data-appendix _pyg2.0_{pe_dim}\
            --seed {seed}\
            --pe-dim {pe_dim}\
            --scenario {scenario}"
    )


if __name__ == "__main__":
    scenario = 7

    pe_dims = {
        "yahoo_music": 20,
        "douban": 20,
        "flixster": 20,
    }

    seeds = [37, 10, 4, 73, 21]

    for dataset in ["yahoo_music", "douban", "flixster"]:
        for ith, seed_val in enumerate(seeds):
            logger.info(f"Test: {ith} - {dataset} - seed: {seed_val:3d}")

            flix_dou_yah(dataset, ith, seed_val, pe_dims[dataset], scenario)

    # python Main.py --data-name yahoo_music --epochs 40 --testing --ensemble --save-appendix _80 --data-appendix _80 --seed 42 --pe-dim 80
