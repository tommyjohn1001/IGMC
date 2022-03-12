#/bin/bash
#douban, yahoo_music
python Main.py --data-name yahoo_music\
    --epochs 40\
    --testing --ensemble\
    --save-appendix _gatedGCN\
    --data-appendix _20\
    --scenario 1\
    --pe-dim 20\