# Va vao giai dieu nay

## 1. Conda env

```
conda env create -f environment.yml
```

## 2. Train

### 2.1. Train Regularization trick

```
python -m regularization.train\
    -g 2\
    --dataset yahoo_music\
    --pe_dim 140\
    --n_perm_graphs 6\
    --metric L1
```

Options:

`metric`: `L1`, `L2`, `cosine`

`dataset`: `yahoo_music`, `douban`, `flixster`, `ml_100k`

### 2.2. Train Link prediction model (Single dataset)

```
PE_DIM=140
SCENARIO=2
SEED=1
CUDA_VISIBLE_DEVICES=0 python Main.py\
    --data-name yahoo_music\
    --epochs 40\
    --save-appendix *${PE_DIM}_${SCENARIO}\
    --data-appendix \_${PE_DIM}\
    --pe-dim ${PE_DIM}\
    --ensemble\
    --testing\
    --batch-size 50\
    --lr 0.0008\
    --seed ${SEED}\
    --scenario ${SCENARIO}
```

### 2.3. Train Link prediction model (Multiple datasets at once)

```
python bot.py\
    -g 0\
    --scenario 7\
    --metric L1
```
