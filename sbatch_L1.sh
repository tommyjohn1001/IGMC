#!/bin/bash -e
#SBATCH --job-name=L1
#SBATCH --output=/lustre/scratch/client/vinai/users/hoanglh88/IGMC/slurms/May15_L1.out
#SBATCH --error=/lustre/scratch/client/vinai/users/hoanglh88/IGMC/slurms/May15_L1.err
#SBATCH --partition=research
#SBATCH --gres=gpu:2              # gpu count
#SBATCH --ntasks=1                 # total number of tasks across all nodes
#SBATCH --nodes=1                  # node count
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4          # cpu-cores per task (>1 if multi-threaded tasks)


export HTTP_PROXY=http://proxytc.vingroup.net:9090/ \
export HTTPS_PROXY=http://proxytc.vingroup.net:9090/ \
export http_proxy=http://proxytc.vingroup.net:9090/ \
export https_proxy=http://proxytc.vingroup.net:9090/



srun --container-image=/lustre/scratch/client/vinai/users/hoanglh88/utils/dc-miniconda3-py:38-4.10.3-cuda11.4.2-cudnn8-ubuntu20.04.sqsh \
     --container-mounts=/lustre/scratch/client/vinai/users/hoanglh88/:/hoanglh88/ \
     /bin/bash -c \
     "
     export HTTP_PROXY=http://proxytc.vingroup.net:9090/
     export HTTPS_PROXY=http://proxytc.vingroup.net:9090/
     export http_proxy=http://proxytc.vingroup.net:9090/
     export https_proxy=http://proxytc.vingroup.net:9090/

     source /opt/conda/bin/activate

     cd /hoanglh88/utils/
     conda activate pt180/

     cd /hoanglh88/IGMC

     python -m regularization.train --dataset flixster --pe_dim 40 -g 0 --n_perm_graphs 6 --metric L1 &\
     python -m regularization.train --dataset flixster --pe_dim 86 -g 0 --n_perm_graphs 6 --metric L1
     python -m regularization.train --dataset yahoo_music --pe_dim 40 -g 0 --n_perm_graphs 6 --metric L1 &\
     python -m regularization.train --dataset yahoo_music --pe_dim 140 -g 0 --n_perm_graphs 6 --metric L1

     python bot.py --scenario 1 -g 0 --metric L1 --mixer trans_encoder --dataset flix_yah &\
     python bot.py --scenario 2 -g 0 --metric L1 --mixer trans_encoder --dataset flix_yah
     python bot.py --scenario 3 -g 0 --metric L1 --mixer trans_encoder --dataset flix_yah &\
     python bot.py --scenario 4 -g 0 --metric L1 --mixer trans_encoder --dataset flix_yah 
     python bot.py --scenario 5 -g 0 --metric L1 --mixer trans_encoder --dataset flix_yah &\
     python bot.py --scenario 6 -g 0 --metric L1 --mixer trans_encoder --dataset flix_yah 
     python bot.py --scenario 7 -g 0 --metric L1 --mixer trans_encoder --dataset flix_yah &\
     python bot.py --scenario 8 -g 0 --metric L1 --mixer trans_encoder --dataset flix_yah

     python bot.py --scenario 1 -g 0 --metric L1 --mixer hyper_mixer --dataset flix_yah &\
     python bot.py --scenario 2 -g 0 --metric L1 --mixer hyper_mixer --dataset flix_yah
     python bot.py --scenario 3 -g 0 --metric L1 --mixer hyper_mixer --dataset flix_yah &\
     python bot.py --scenario 4 -g 0 --metric L1 --mixer hyper_mixer --dataset flix_yah 
     python bot.py --scenario 5 -g 0 --metric L1 --mixer hyper_mixer --dataset flix_yah &\
     python bot.py --scenario 6 -g 0 --metric L1 --mixer hyper_mixer --dataset flix_yah 
     python bot.py --scenario 7 -g 0 --metric L1 --mixer hyper_mixer --dataset flix_yah &\
     python bot.py --scenario 8 -g 0 --metric L1 --mixer hyper_mixer --dataset flix_yah



     python -m regularization.train --dataset douban --pe_dim 40 -g 0 --n_perm_graphs 6 --metric L1 &\
     python -m regularization.train --dataset douban --pe_dim 115 -g 1 --n_perm_graphs 6 --metric L1
 
     python bot2.py --scenario 9 -g 0 --metric L1 --mixer trans_encoder --dataset douban &\
     python bot2.py --scenario 10 -g 1 --metric L1 --mixer trans_encoder --dataset douban
     python bot2.py --scenario 11 -g 0 --metric L1 --mixer trans_encoder --dataset douban &\
     python bot2.py --scenario 12 -g 1 --metric L1 --mixer trans_encoder --dataset douban 
     python bot2.py --scenario 13 -g 0 --metric L1 --mixer trans_encoder --dataset douban &\
     python bot2.py --scenario 14 -g 1 --metric L1 --mixer trans_encoder --dataset douban 
     python bot2.py --scenario 15 -g 0 --metric L1 --mixer trans_encoder --dataset douban &\
     python bot2.py --scenario 16 -g 1 --metric L1 --mixer trans_encoder --dataset douban

     python bot2.py --scenario 9 -g 0 --metric L1 --mixer hyper_mixer --dataset douban &\
     python bot2.py --scenario 10 -g 1 --metric L1 --mixer hyper_mixer --dataset douban
     python bot2.py --scenario 11 -g 0 --metric L1 --mixer hyper_mixer --dataset douban &\
     python bot2.py --scenario 12 -g 1 --metric L1 --mixer hyper_mixer --dataset douban 
     python bot2.py --scenario 13 -g 0 --metric L1 --mixer hyper_mixer --dataset douban &\
     python bot2.py --scenario 14 -g 1 --metric L1 --mixer hyper_mixer --dataset douban 
     python bot2.py --scenario 15 -g 0 --metric L1 --mixer hyper_mixer --dataset douban &\
     python bot2.py --scenario 16 -g 1 --metric L1 --mixer hyper_mixer --dataset douban
     "