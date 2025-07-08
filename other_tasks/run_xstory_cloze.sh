#!/bin/bash
#SBATCH --job-name=xstory_cloze
#SBATCH --output=xstory_cloze.log
#SBATCH --time=12:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --constraint="module-miniforge"
#SBATCH --nodelist="dgx-h100-em2"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=martin.beaufils@polytechnique.edu

python /srv/home/users/beaufilsm35cs/evaluation/eval_xstory_cloze.py \
    --gpu 0 \
    --split val \
    --max_samples 500 \
    --model_path "CohereLabs/aya-101" \
    --results_csv results/xstory_cloze_results_val.csv \
    --val_file "/srv/home/users/beaufilsm35cs/evaluation/xstory_cloze/val/cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv"

python /srv/home/users/beaufilsm35cs/evaluation/eval_xstory_cloze.py \
    --gpu 0 \
    --split test \
    --max_samples 500 \
    --model_path "CohereLabs/aya-101" \
    --results_csv results/xstory_cloze_results_test.csv \
    --test_file "/srv/home/users/beaufilsm35cs/evaluation/xstory_cloze/test/cloze_test_test__winter2018-cloze_test_ALL_test - 1.csv"
