#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=pl_ft
#SBATCH --output=/nlp/data/artemisp/multigpu-lm-templates/%x.%j.out
#SBATCH --error=/nlp/data/artemisp/multigpu-lm-templates/%x.%j.err
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --constraint=48GBgpu
#SBATCH --mem-per-cpu=14G
#SBATCH --cpus-per-task=8

cd /nlp/data/artemisp/multigpu-lm-templates/

srun /nlp/data/artemisp/mambaforge/bin/python pl_ft.py $@