#!/bin/bash

# dataset=$1
# sub_dataset=${2:-''}

# dataset_lst=("Cora" "CiteSeer" "PubMed" "CS")
dataset_lst=("fb100")

lr_lst=(0.01 0.001)
weight_decay_lst=(0.001 0.0005)
hidden_channels_lst=(8 16 32 128)
power_lst=(2 5 10)
epochs=1000

for dataset in "${dataset_lst[@]}"; do
    if [ "$dataset" = "fb100" ]; then
        sub_dataset="Penn94"
    else
        sub_dataset="None"
    fi
    echo $sub_dataset
    
    if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
        directed="--directed"
    else
        directed=""
    fi
    
    for lr in "${lr_lst[@]}"; do
        for weight_decay in "${weight_decay_lst[@]}"; do
            for hidden_channels in "${hidden_channels_lst[@]}"; do
                for power in "${power_lst[@]}"; do
                
                    python main_bilevel_optim.py --dataset $dataset --sub_dataset $sub_dataset \
                                --method gcn_strc --runs 5 --display_step 25 --epochs $epochs \
                                --lr $lr --weight_decay $weight_decay $directed \
                                --hidden_channels $hidden_channels --power $power  
                done
            done
        done
    done
done