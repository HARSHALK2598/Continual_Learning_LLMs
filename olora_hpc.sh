#!/bin/bash
#
#SBATCH --partition=rtx8000
#SBATCH --job-name=OPT-2p7B_olora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --output=OPT-2p7B_olora.out
#SBATCH --error=OPT-2p7B_olora.err


export CUDA_VISIBLE_DEVICES=0,1,2,3
echo $CUDA_VISIBLE_DEVICES
hostname; date

module purge
source /scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate nlp # write your environment name if you did not use the name nlp
module load gcc/10.2.0
module load nccl/cuda11.6/2.12.12

cd /scratch/$USER/NLU-Project

# -------- Global Experiment Variables --------
DATE=`date +%Y-%m-%d`

LORA_R=16
LORA_A=16
LORA_DP=0.1
task="dbpedia,amazon,yahoo,agnews"
accelerate launch  --config_file 'default_config.yaml' codebase/main_olora_acc.py --experiment_dir "OBT2.7B-ACC-O-LORA-$task-r=$LORA_R-a=$LORA_A-dp=$LORA_DP-$DATE" \
                                --model_name "facebook/opt-2.7b" --train_epochs 1 --warm_up_epochs 0 --scheduler_type "constant" --bs 8 --lr 1e-4 --wd 1e-2 --tasks $task \
                                --use_lora --lora_r $LORA_R --lora_alpha $LORA_A --lora_dropout $LORA_DP  --save_model 