#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=pl_ft
#SBATCH --output=<path_to_logs_dir>/%x.%j.out
#SBATCH --error=<path_to_logs_dir>/%x.%j.err
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --constraint=48GBgpu  

srun python src/finetune/t5/pl_ft.py $@