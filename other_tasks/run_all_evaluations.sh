#!/bin/bash
#SBATCH --job-name=eval_all_datasets
#SBATCH --output=eval_all_datasets.log
#SBATCH --time=48:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --constraint="module-miniforge"
#SBATCH --nodelist="dgx-h100-em2"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=martin.beaufils@polytechnique.edu

MODEL_PATH="CohereLabs/aya-101"
MAX_SAMPLES=500
RESULTS_DIR="/srv/home/users/beaufilsm35cs/evaluation/results"

python /srv/home/users/beaufilsm35cs/evaluation/run_all_evaluations.py \
    --max_samples $MAX_SAMPLES \
    --model_name "$MODEL_PATH" \
    --output_dir "$RESULTS_DIR"
