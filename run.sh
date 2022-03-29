#/bin/bash
#douban, yahoo_music

# PE_DIM=40
# SCENARIO=9
# python Main.py\
#         --data-name yahoo_music\
#         --epochs 40\
#         --save-appendix _${PE_DIM}_${SCENARIO}\
#         --data-appendix _${PE_DIM}\
#         --pe-dim ${PE_DIM}\
#         --ensemble\
#         --testing\
#         --scenario ${SCENARIO}

# python Main.py\
#         --data-name yahoo_music\
#         --epochs 40\
#         --save-appendix _10_9\
#         --data-appendix _10\
#         --pe-dim 10\
#         --ensemble\
#         --testing\
#         --scenario 9
        # --no-train\

python bot.py --scenario 3 & python bot.py --scenario 4
python bot.py --scenario 5 & python bot.py --scenario 6
python bot.py --scenario 7 & python bot.py --scenario 8
python bot.py --scenario 9 & python bot.py --scenario 10