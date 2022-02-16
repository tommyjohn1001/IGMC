#/bin/bash
#douban, yahoo_music
python Main.py --data-name yahoo_music\
    --epochs 40\
    --data-appendix _pyg2.0\
    --save-appendix _gatedGCN\
    --testing\
    --ensemble\
    --wandb\
    --pe-dim 20