#/bin/bash
#douban, yahoo_music
python Main.py\
        --data-name yahoo_music\
        --epochs 100\
        --save-appendix _gatedGCN_20_7\
        --data-appendix _20\
        --pe-dim 20\
        --ensemble\
        --testing\
        --scenario 7
        # --no-train\