#!/bin/bash
#SBATCH --partition=research
#SBATCH --output=/lustre/scratch/client/vinai/users/hoanglh88/manifold_defense/outputs/slurm_log/%x-%j.out
#SBATCH --error=/lustre/scratch/client/vinai/users/hoanglh88/manifold_defense/outputs/slurm_log/%x-%j.out
#SBATCH --job-name=bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=30G
#SBATCH --cpus-per-gpu=16
#SBATCH --ntasks=1

#SBATCH --job-name=IGMC            # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/hoanglh88/IGMC/slurm.out      # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/hoanglh88/IGMC/slurm.err       # create a error file
#SBATCH --partition=research
#SBATCH --gres=gpu:1              # gpu count
#SBATCH --ntasks=1                 # total number of tasks across all nodes
#SBATCH --nodes=1                  # node count
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4          # cpu-cores per task (>1 if multi-threaded tasks)


srun --container-image=/lustre/scratch/client/vinai/users/hoanglh88/dc-miniconda3-py:38-4.10.3-cuda11.4.2-cudnn8-ubuntu20.04.sqsh \
     --container-mounts=/lustre/scratch/client/vinai/users/hoanglh88/:/hoanglh88/ \
     --nodes=1 \
     --gpus=1 \
     --ntasks=1 \
     --mem=64G \
     --cpus-per-task=4 \
     --pty bash

     # module purge
# module load python/miniconda3/miniconda3
# eval "$(conda shell.bash hook)"
# cd /lustre/scratch/client/vinai/users/hoanglh88/utils/
# conda activate igmc/
# conda activate /lustre/scratch/client/vinai/users/hoanglh88/utils/igmc1


cd /lustre/scratch/client/vinai/users/hoanglh88/IGMC

python train.py --data-name flixster --exp_name nrw2_superpod --version 2 --contrastive 0 --wandb --ensemble

