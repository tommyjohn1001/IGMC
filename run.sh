#!/bin/bash

## 1. Install env
# pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# TORCH=1.7.1
# CUDA=cu110
# pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
# pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
# pip install torch-geometric

# pip install loguru black isort jupyterlab

## 2. Train

## Train with dataset Flixster
# python -W ignore Main.py --data-name flixster --hop 3 --epochs 200 --testing --ensemble --continue-from 80 > logs/Dec1_flixster.log

# CUDA_VISIBLE_DEVICES=1 python -W ignore Main.py --data-name douban --hop 3 --epochs 100 --testing --ensemble --continue-from 40 > logs/Dec1_douban.log

CUDA_VISIBLE_DEVICES=0 python Main.py --data-name yahoo_music --lr 1e-5 --epochs 100 --testing --ensemble > logs/Dec1_yahoo.log

## Train with dataset MovieLens-100k
# CUDA_VISIBLE_DEVICES=1 python Main.py --data-name ml_100k --save-appendix _mnph200 --data-appendix _mnph200 --epochs 150 --max-nodes-per-hop 200 --testing --ensemble --dynamic-train > logs/Dec1_movie_100k.log

# python Main.py --data-name ml_1m --save-appendix _mnhp100 --data-appendix _mnph100 --max-nodes-per-hop 100 --testing --epochs 100 --save-interval 5  --lr-decay-step-size 20 --ensemble --dynamic-train > logs/Dec1_movie_1M.log