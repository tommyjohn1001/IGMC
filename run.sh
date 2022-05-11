#/bin/bash

###################
## 1. Run script for training IGMC
###################
#douban, yahoo_music
# PE_DIM=140
# SCENARIO=2
# SEED=1
# CUDA_VISIBLE_DEVICES=0 python Main.py\
#         --data-name yahoo_music\
#         --epochs 40\
#         --save-appendix _${PE_DIM}_${SCENARIO}\
#         --data-appendix _${PE_DIM}\
#         --pe-dim ${PE_DIM}\
#         --ensemble\
#         --testing\
#         --batch-size 50\
#         --lr 0.0008\
#         --seed ${SEED}\
#         --scenario ${SCENARIO}
        # --max-nodes-per-hop 200\

# CUDA_VISIBLE_DEVICES=2 python Main.py\
#         --data-name ml_100k\
#         --epochs 80\
#         --save-appendix _40_15\
#         --data-appendix _40\
#         --pe-dim 40\
#         --ensemble\
#         --testing\
#         --batch-size 35\
#         --max-nodes-per-hop 200\
#         --scenario 15
#         --dynamic-train\
#         --no-train\

# python bot.py --scenario 1 -g 0 & python bot.py --scenario 2 -g 1
# python bot.py --scenario 3 -g 0 & python bot.py --scenario 4 -g 1
# python bot.py --scenario 5 -g 0 & python bot.py --scenario 6 -g 1
# python bot.py --scenario 7 -g 0 & python bot.py --scenario 8 -g 1

###################
## 1. Run script for training Regularization trick
###################
# python -m regularization.train --dataset yahoo_music --pe_dim 40 -g 0 --n_perm_graphs 6 &\
# python -m regularization.train --dataset yahoo_music --pe_dim 140 -g 2 --n_perm_graphs 6

python bot.py --scenario 1 -g 2 & python bot.py --scenario 2 -g 1
python bot.py --scenario 3 -g 2 & python bot.py --scenario 4 -g 1
python bot.py --scenario 5 -g 2 & python bot.py --scenario 6 -g 1
python bot.py --scenario 7 -g 2 & python bot.py --scenario 8 -g 1
