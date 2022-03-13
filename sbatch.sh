#!/bin/bash -e
#SBATCH --job-name=IGMC            # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/hoanglh88/IGMC/slurms/Mar13.out      # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/hoanglh88/IGMC/slurms/Mar13.err       # create a error file
#SBATCH --partition=research
#SBATCH --gres=gpu:2              # gpu count
#SBATCH --ntasks=1                 # total number of tasks across all nodes
#SBATCH --nodes=1                  # node count
#SBATCH --mem=64G
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
     conda activate igmc/

     cd /hoanglh88/IGMC
     CUDA_VISIBLE_DEVICES=0 python bot_samek.py --scenario  9 & CUDA_VISIBLE_DEVICES=1 python bot_samek.py --scenario 11
     CUDA_VISIBLE_DEVICES=0 python bot_samek.py --scenario 13 & CUDA_VISIBLE_DEVICES=1 python bot_samek.py --scenario 15
     CUDA_VISIBLE_DEVICES=0 python bot_diffk.py --scenario 10 & CUDA_VISIBLE_DEVICES=1 python bot_diffk.py --scenario 12
     CUDA_VISIBLE_DEVICES=0 python bot_diffk.py --scenario 13 & CUDA_VISIBLE_DEVICES=1 python bot_diffk.py --scenario 16
     "