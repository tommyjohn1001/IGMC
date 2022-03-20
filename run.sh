#/bin/bash
#douban, yahoo_music
python Main.py\
        --data-name yahoo_music\
        --epochs 40\
        --save-appendix _rgcn_20_4\
        --data-appendix _20\
        --pe-dim 20\
        --ensemble\
        --testing\
        --scenario 4
        # --no-train\