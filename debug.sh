PE_DIM=40
SCENARIO=7
SEED=1
CUDA_VISIBLE_DEVICES=4 python -m pdb -c continue Main.py\
        --data-name yahoo_music\
        --epochs 20\
        --save-appendix _${PE_DIM}_${SCENARIO}\
        --data-appendix _${PE_DIM}\
        --pe-dim ${PE_DIM}\
        --ensemble\
        --testing\
        --batch-size 50\
        --seed ${SEED}\
        --scenario ${SCENARIO}
        --max-nodes-per-hop 200\