import os

from loguru import logger


def flix_dou_yah(dataset, ith):
    os.system(
        f"python Main.py --data-name {dataset} --epochs 40 --testing --ensemble --save-appendix _RWPE_{ith}"
    )


def ml1M(ith):
    os.system(
        f"python Main.py --data-name ml_1m --save-appendix _RWPE_{ith} --data-appendix _mnph100 --max-nodes-per-hop 100 --testing --epochs 40 --save-interval 5 --adj-dropout 0 --lr-decay-step-size 20 --ensemble --dynamic-train"
    )


def ml100k(ith):
    os.system(
        f"python Main.py --data-name ml_100k --save-appendix _RWPE_{ith} --data-appendix _mnph200 --epochs 80 --max-nodes-per-hop 200 --testing --ensemble --dynamic-train"
    )


for dataset in ["ml100k", "ml1M", "yahoo_music", "douban", "flixster"]:
    for ith in range(1, 6):
        logger.info(f"Test: {ith} - {dataset}")
        if dataset == "ml100k":
            ml100k(ith)
        elif dataset == "ml1M":
            ml1M(ith)
        else:
            flix_dou_yah(dataset, ith)
