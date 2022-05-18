#!/bin/bash -e
#SBATCH --job-name=IGMC_pretrain_transencoder
#SBATCH --output=/lustre/scratch/client/vinai/users/duynht1/IGMC/slurms/May17_pretrain_transencoder.out
#SBATCH --error=/lustre/scratch/client/vinai/users/duynht1/IGMC/slurms/May17_pretrain_transencoder.err
#SBATCH --partition=applied
#SBATCH --gres=gpu:4              # gpu count
#SBATCH --ntasks=1                 # total number of tasks across all nodes
#SBATCH --nodes=1                  # node count
#SBATCH --mem=256G
#SBATCH --cpus-per-task=12          # cpu-cores per task (>1 if multi-threaded tasks)
export HTTP_PROXY=http://proxytc.vingroup.net:9090/ \
export HTTPS_PROXY=http://proxytc.vingroup.net:9090/ \
export http_proxy=http://proxytc.vingroup.net:9090/ \
export https_proxy=http://proxytc.vingroup.net:9090/
srun --container-image="harbor.vinai-systems.com#research/igmc:localcondapy38" \
     --container-mounts=/lustre/scratch/client/vinai/users/duynht1/:/home/ubuntu/duynht1/:rshared \
     --container-workdir=/home/ubuntu/duynht1/IGMC \
     /bin/bash -c \
    "
    python bot.py --scenario 1 -g 0 --mixer trans_encoder --mode pretraining &\
    python bot.py --scenario 2 -g 1 --mixer trans_encoder --mode pretraining &\
    python bot.py --scenario 3 -g 2 --mixer trans_encoder --mode pretraining &\
    python bot.py --scenario 4 -g 3 --mixer trans_encoder --mode pretraining &\
    python bot.py --scenario 5 -g 0 --mixer trans_encoder --mode pretraining &\
    python bot.py --scenario 6 -g 1 --mixer trans_encoder --mode pretraining &\
    python bot.py --scenario 7 -g 2 --mixer trans_encoder --mode pretraining &\
    python bot.py --scenario 8 -g 3 --mixer trans_encoder --mode pretraining
    python bot.py --scenario 9 -g 0 --mixer trans_encoder --mode pretraining &\
    python bot.py --scenario 10 -g 1 --mixer trans_encoder --mode pretraining &\
    python bot.py --scenario 11 -g 2 --mixer trans_encoder --mode pretraining &\
    python bot.py --scenario 12 -g 3 --mixer trans_encoder --mode pretraining &\
    python bot.py --scenario 13 -g 0 --mixer trans_encoder --mode pretraining &\
    python bot.py --scenario 14 -g 1 --mixer trans_encoder --mode pretraining &\
    python bot.py --scenario 15 -g 2 --mixer trans_encoder --mode pretraining &\
    python bot.py --scenario 16 -g 3 --mixer trans_encoder --mode pretraining
    "