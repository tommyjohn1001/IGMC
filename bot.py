import os

from loguru import logger


def flix_dou_yah(dataset, ith, seed, pe_dim, scenario):
    os.system(
        f"python Main.py --data-name {dataset} --epochs 40 --testing --ensemble\
            --save-appendix _RWPE_{pe_dim}_{scenario}_{ith} --data-appendix _RWPE_{pe_dim}\
            --seed {seed} --pe-dim {pe_dim}"
    )


def ml1M(ith, seed, pe_dim, scenario):
    os.system(
        f"python Main.py --data-name ml_1m --save-appendix _RWPE_{pe_dim}_{scenario}_{ith} --data-appendix _RWPE_{pe_dim}\
            --max-nodes-per-hop 100 --testing --epochs 40 --save-interval 5 --adj-dropout 0\
            --lr-decay-step-size 20 --ensemble --dynamic-train --seed {seed} --pe-dim {pe_dim}"
    )


def ml100k(ith, seed, pe_dim, scenario):
    os.system(
        f"python Main.py --data-name ml_100k --save-appendix _RWPE_{pe_dim}_{scenario}_{ith} --data-appendix _RWPE_{pe_dim}\
            --epochs 80 --max-nodes-per-hop 200 --testing --ensemble\
            --dynamic-train --seed {seed} --pe-dim {pe_dim}"
    )


pe_dims = {
    "ml100k": 50,
    "ml1M": 50,
    "yahoo_music": 140,
    "douban": 115,
    "flixster": 86,
}

seeds = [37, 10, 4, 73, 21]
scenario = 3
for dataset in ["yahoo_music", "douban", "flixster", "ml100k", "ml1M"]:
    for ith, seed_val in enumerate(seeds):
        logger.info(f"Test: {ith} - {dataset} - seed: {seed_val:3d}")
        if dataset == "ml100k":
            ml100k(ith, seed_val, pe_dims[dataset])
        elif dataset == "ml1M":
            ml1M(ith, seed_val, pe_dims[dataset])
        else:
            flix_dou_yah(dataset, ith, seed_val, pe_dims[dataset])


# python Main.py --data-name yahoo_music --epochs 40 --testing --ensemble --save-appendix _RWPE_80 --data-appendix _RWPE_80 --seed 42 --pe-dim 80
