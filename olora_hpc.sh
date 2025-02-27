#!/bin/bash
#SBATCH --account=pr_268_tandon_advanced
#SBATCH --job-name=OPT-2p7B_olora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --output=OPT-2p7B_olora.out
#SBATCH --error=OPT-2p7B_olora.err


export CUDA_VISIBLE_DEVICES=0,1
echo $CUDA_VISIBLE_DEVICES
hostname; date

singularity exec --nv --overlay /scratch/hsk8171/llmContinualLearning/overlay-25GB-500K.ext3:rw /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif /bin/bash <<EOF
source /ext3/env.sh
conda activate nlpHsk8171 # write your environment name if you did not use the name nlp
cd /scratch/$USER/llmContinualLearning/CL4LLMs


# -------- Global Experiment Variables --------
DATE=\`date +%Y-%m-%d\`
LORA_R=16
LORA_A=16
LORA_DP=0.1
task="dbpedia,amazon,yahoo,agnews"

accelerate launch --config_file 'default_config.yaml' codebase/main_olora_acc.py \
    --experiment_dir "OBT2.7B-ACC-O-LORA-\$task-r=\$LORA_R-a=\$LORA_A-dp=\$LORA_DP-\$DATE" \
    --model_name "facebook/opt-2.7b" --train_epochs 1 --warm_up_epochs 0 \
    --scheduler_type "constant" --bs 4 --lr 1e-4 --wd 1e-2 \
    --tasks "\$task" --use_lora --lora_r "\$LORA_R" --lora_alpha "\$LORA_A" \
    --lora_dropout "\$LORA_DP" --save_model --normalized_lora
EOF
