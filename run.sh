#/bin/bash
#douban, yahoo_music
python train.py\
    --data-name yahoo_music\
    -g 0 --version 2\
    --save-appendix _gatedGCN_20_11\
    --data-appendix _20\
    --pe-dim 20\
    --scenario 15



# CUDA_VISIBLE_DEVICES=1 python bot_samek.py --scenario 1 & CUDA_VISIBLE_DEVICES=2 python bot_samek.py --scenario 3
# CUDA_VISIBLE_DEVICES=1 python bot_diffk.py --scenario 2 & CUDA_VISIBLE_DEVICES=2 python bot_diffk.py --scenario 4
# CUDA_VISIBLE_DEVICES=1 python bot_samek.py --scenario 5 & CUDA_VISIBLE_DEVICES=2 python bot_samek.py --scenario 7
# CUDA_VISIBLE_DEVICES=1 python bot_diffk.py --scenario 6 & CUDA_VISIBLE_DEVICES=2 python bot_diffk.py --scenario 8