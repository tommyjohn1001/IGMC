import os

from loguru import logger


def flix_dou_yah(dataset, ith, seed, pe_dim):
    os.system(
        f"python Main.py --data-name {dataset} --epochs 40 --testing --ensemble\
            --save-appendix _RWPE_{ith} --seed {seed} --pe-dim {pe_dim}"
    )


def ml1M(ith, seed, pe_dim):
    os.system(
        f"python Main.py --data-name ml_1m --save-appendix _RWPE_{ith}\
            --max-nodes-per-hop 100 --testing --epochs 40 --save-interval 5 --adj-dropout 0\
            --lr-decay-step-size 20 --ensemble --dynamic-train --seed {seed} --pe-dim {pe_dim}"
    )


def ml100k(ith, seed, pe_dim):
    os.system(
        f"python Main.py --data-name ml_100k --save-appendix _RWPE_{ith}\
            --epochs 80 --max-nodes-per-hop 200 --testing --ensemble\
            --dynamic-train --seed {seed} --pe-dim {pe_dim}"
    )


pe_dims = {
    "ml100k": {"k": 50},
    "ml1M": {"k": 50},
    "yahoo_music": {"k": 140},
    "douban": {"k": 115},
    "flixster": {"k": 86},
}

seeds = [37, 10, 4, 73, 21]
for dataset in ["ml100k", "ml1M", "yahoo_music", "douban", "flixster"]:
    for ith, seed_val in enumerate(seeds):
        logger.info(f"Test: {ith} - {dataset} - seed: {seed_val:3d}")
        if dataset == "ml100k":
            ml100k(ith, seed_val, pe_dims[dataset])
        elif dataset == "ml1M":
            ml1M(ith, seed_val, pe_dims[dataset])
        else:
            flix_dou_yah(dataset, ith, seed_val, pe_dims[dataset])
