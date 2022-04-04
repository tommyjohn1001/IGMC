#/bin/bash
#douban, yahoo_music

# PE_DIM=40
# SCENARIO=7
# SEED=1
# CUDA_VISIBLE_DEVICES=0 python Main.py\
#         --data-name ml_100k\
#         --epochs 40\
#         --save-appendix _${PE_DIM}_${SCENARIO}\
#         --data-appendix _${PE_DIM}\
#         --pe-dim ${PE_DIM}\
#         --ensemble\
#         --testing\
#         --max-nodes-per-hop 200\
#         --batch-size 30\
#         --seed ${SEED}\
#         --scenario ${SCENARIO}

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
        # --dynamic-train\
        # --no-train\

python bot.py --scenario 7
python bot.py --scenario 8
python bot.py --scenario 9
python bot.py --scenario 10