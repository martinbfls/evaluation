#!/bin/bash
#SBATCH --job-name=xnli
#SBATCH --output=xnli.log
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --constraint="module-miniforge"
#SBATCH --nodelist="dgx-h100-em2"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=martin.beaufils@polytechnique.edu

python /srv/home/users/beaufilsm35cs/evaluation/eval_xnli.py \
    --gpu 0 \
    --split test \
    --max_examples 500 \
    --model_path "CohereLabs/aya-101" \
    --results_csv results/xnli_results_direct.csv
