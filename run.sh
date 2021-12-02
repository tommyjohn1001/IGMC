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
# CUDA_VISIBLE_DEVICES=1 python Main.py --data-name flixster > logs/Dec1_flixster.log

# CUDA_VISIBLE_DEVICES=1 python Main.py --data-name douban > logs/Dec1_douban.log

# CUDA_VISIBLE_DEVICES=1 python Main.py --data-name yahoo_music > logs/Dec1_yahoo.log

## Train with dataset MovieLens-100k
# CUDA_VISIBLE_DEVICES=1 python Main.py --data-name ml_100k > logs/Dec1_movie_100k.log

# CUDA_VISIBLE_DEVICES=1 python Main.py --data-name ml_1m  > logs/Dec1_movie_1M.log