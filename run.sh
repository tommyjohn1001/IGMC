#/bin/bash
#douban, yahoo_music
python Main.py --data-name yahoo_music\
    --epochs 40\
    --testing --ensemble\
    --save-appendix _gatedGCN_20_6\
    --data-appendix _20\
    --pe-dim 20\
    --scenario 6\