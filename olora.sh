#!/bin/bash 
# -------- Global Experiment Variables --------
DATE=`date +%Y-%m-%d`

LORA_R=16
LORA_A=16
LORA_DP=0.1
task="dbpedia,amazon,yahoo,agnews"
for llambda in 0.5 0.25 0.1; do
    accelerate launch  --config_file 'default_config.yaml' codebase/main_olora_acc.py --experiment_dir "CORR-OBT2.7B-ACC-O-LORA-lambda=$llambda-$task-r=$LORA_R-a=$LORA_A-dp=$LORA_DP-$DATE" \
                                    --model_name "facebook/opt-2.7b" --train_epochs 1 --warm_up_epochs 0 --scheduler_type "constant" --bs 4 --lr 5e-5 --wd 1e-2 --tasks $task \
                                    --llambda $llambda --use_lora --lora_r $LORA_R --lora_alpha $LORA_A --lora_dropout $LORA_DP  --save_model 
done






