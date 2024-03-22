#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=pl_predict
#SBATCH --output=<path_to_log_folder>/%x.%j.out
#SBATCH --error=<path_to_log_folder>/%x.%j.err
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --constraint=48GBgpu
#SBATCH --mem-per-cpu=14G
#SBATCH --cpus-per-task=8


srun /nlp/data/artemisp/mambaforge/bin/python pl_predict.py $@
