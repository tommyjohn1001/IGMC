import argparse
import os
import os.path as osp
import random
from datetime import datetime, timedelta

from loguru import logger

PE_DIMS = {"common": 40, "yahoo_music": 140, "douban": 115, "flixster": 86, "ml_100k": 80}
TIMES = 2


def flix_dou_yah(dataset, pe_dim, scenario, cuda_device, ith, metric, mixer, mode):
    now = (datetime.now() + timedelta(hours=7)).strftime("%b%d_%H:%M")
    SEED = random.randint(0, 20)
    epoch = 40

    if mode == "pretraining":
        path_log = f"logs/{mode}/{mixer}/{dataset}_{scenario}_{pe_dim}_{now}.log"
    elif mode == "coop":
        path_log = f"logs/{mode}/{mixer}/{metric}/{dataset}_{scenario}_{ith}_{pe_dim}_{now}.log"
    else:
        raise NotImplementedError()
    os.makedirs(osp.dirname(path_log), exist_ok=True)

    os.system(
        f"CUDA_VISIBLE_DEVICES={cuda_device} python Main.py\
            --data-name {dataset}\
            --epochs {epoch}\
            --testing --ensemble\
            --save-appendix _{pe_dim}_{scenario}\
            --data-appendix _{pe_dim}\
            --lr 0.0008\
            --seed {SEED}\
            --pe-dim {pe_dim}\
            --scenario {scenario}\
            --mode {mode}\
            --mixer {mixer} > {path_log}"
    )


def movielens(pe_dim, scenario, cuda_device, ith, batch_size, metric):
    dataset = "ml_100k"
    now = (datetime.now() + timedelta(hours=7)).strftime("%b%d_%H:%M")
    SEED = 0

    path_weight_mlp = f"weights/mlp_{dataset}_{pe_dim}_{metric}.pt"

    os.system(
        f"CUDA_VISIBLE_DEVICES={cuda_device} python Main.py\
            --data-name {dataset}\
            --epochs 80\
            --testing --ensemble\
            --save-appendix _{pe_dim}_{scenario}\
            --data-appendix _{pe_dim}\
            --max-nodes-per-hop 200\
            --lr 0.0008\
            --batch-size {batch_size}\
            --seed {SEED}\
            --pe-dim {pe_dim}\
            --scenario {scenario}\
            --path_weight_mlp {path_weight_mlp} > logs/{dataset}_{scenario}_{ith}_{pe_dim}_{now}.log"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E(n)")
    parser.add_argument("--scenario", type=int)
    parser.add_argument("--gpu", "-g", type=int)
    parser.add_argument("--metric", type=str, default="L1", choices=["cosine", "L1", "L2"])
    parser.add_argument(
        "--mixer", type=str, default="trans_encoder", choices=["hyper_mixer", "trans_encoder"]
    )
    parser.add_argument("--mode", type=str, default="pretraining", choices=["pretraining", "coop"])
    parser.add_argument(
        "--dataset", type=str, default="flixster", choices=["flixster", "douban", "yahoo_music"]
    )

    args = parser.parse_args()

    cuda_device = args.gpu
    if args.mode == "pretraining":
        datasets = ["yahoo_music", "flixster", "douban"]
        times = 1
    else:
        datasets = [args.dataset]
        times = TIMES

    logger.info(f"==> Mode: {args.mode}")

    for dataset in datasets:  # "ml_100k" "yahoo_music", "flixster", "douban"
        logger.info(dataset)

        for ith in range(TIMES):
            if args.scenario % 2 == 1:
                pe_dim = PE_DIMS["common"]
            else:
                pe_dim = PE_DIMS[dataset]

            # if superpod, use batch_size = 40
            if dataset == "ml_100k":
                movielens(pe_dim, args.scenario, cuda_device, ith, 30, args.metric)
            else:
                flix_dou_yah(
                    dataset,
                    pe_dim,
                    args.scenario,
                    cuda_device,
                    ith,
                    args.metric,
                    args.mixer,
                    args.mode,
                )
