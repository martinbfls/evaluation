#!/bin/bash
#SBATCH --job-name=eval_qa
#SBATCH --output=output/eval_qa_%a.out
#SBATCH --time=12:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --constraint="module-miniforge"
#SBATCH --nodelist="dgx-h100-em2"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=martin.beaufils@polytechnique.edu
#SBATCH --array=0-0%3

source /etc/profile.d/modules.sh

export PYTHON_VERSION="3.10"      
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=
export HF_TOKEN=                            # Complete with your tokens

module load miniforge


pip install --upgrade pip
pip install -r /srv/home/users/beaufilsm35cs/evaluation/requirements.txt

huggingface-cli login --token $HF_TOKEN
huggingface-cli whoami

coreset_list=("0.35")  #0.000732, 0.001464, 0.002196, 0.0029275, 0.0036595, 0.004391, 0.0051233, 0.0058552, 0.0065864, "0.0109774" "0.05" "0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4" "0.5" "0.8" "0.9" "1.0"

coreset_rate=${coreset_list[$SLURM_ARRAY_TASK_ID]}

echo "Running for coreset rate: $coreset_rate"

python eval_qa_mt5.py --max_examples 10000 \
                    --model_path martinbfls/mt5-xxl \
                    --results_csv qa_results.csv \
                    --data_path /srv/home/users/beaufilsm35cs/staff/filtered_qa_dataset_70k/test \
                    --coreset_list "$coreset_rate" \
                    --metrics "rougel,loss"