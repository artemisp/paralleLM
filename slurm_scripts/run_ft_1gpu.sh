#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=pl_ft
#SBATCH --output=<path_to_log_folder>/%x.%j.out
#SBATCH --error=<path_to_log_folder>/%x.%j.err
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --constraint=48GBgpu
#SBATCH --mem-per-cpu=14G
#SBATCH --cpus-per-task=8

cd <path_to_project_folder>

srun /nlp/data/artemisp/mambaforge/bin/python pl_ft.py $@