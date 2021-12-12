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
# python train.py --data-name flixster --exp_name nrw0.2 -g 0

# python train.py --data-name douban --exp_name nrw0.2 -g 0

# python train.py --data-name yahoo_music --exp_name nrw0.2 -g 0

## Train with dataset MovieLens-100k
# python train.py --data-name ml_100k --exp_name nrw0.2 -g 0

# python train.py --data-name ml_1m  --exp_name nrw0.2 -g 0