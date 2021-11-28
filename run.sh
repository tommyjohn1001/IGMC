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
CUDA_VISIBLE_DEVICES=1 python -W ignore Main.py --data-name flixster --hop 3 --epochs 40 --testing --ensemble

## Train with dataset MovieLens-100k
# python Main.py --data-name ml_100k --save-appendix _mnph200 --data-appendix _mnph200 --epochs 80 --max-nodes-per-hop 200 --testing --ensemble --dynamic-train