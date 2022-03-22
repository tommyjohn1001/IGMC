#/bin/bash
#douban, yahoo_music

rm -r results/*

PE_DIM=20
SCENARIO=1
python Main.py\
        --data-name flixster\
        --epochs 40\
        --save-appendix _${SCENARIO}_${PE_DIM}\
        --data-appendix _${PE_DIM}\
        --pe-dim ${PE_DIM}\
        --ensemble\
        --testing\
        --scenario ${SCENARIO}

# python Main.py\
#         --data-name yahoo_music\
#         --epochs 40\
#         --save-appendix _rgcn_20_4\
#         --data-appendix _20\
#         --pe-dim 20\
#         --ensemble\
#         --testing\
#         --scenario 4
        # --no-train\

# python bot.py --scenario 1 & python bot.py --scenario 2
# python bot.py --scenario 3 & python bot.py --scenario 4
# python bot.py --scenario 5 & python bot.py --scenario 6
# python bot.py --scenario 7 & python bot.py --scenario 8
# python bot.py --scenario 9 & python bot.py --scenario 10