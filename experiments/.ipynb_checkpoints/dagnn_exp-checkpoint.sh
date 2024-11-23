#!/bin/bash

# dataset=$1
# sub_dataset=${2:-''}
# dataset_lst=("Cora" "CiteSeer" "PubMed" "CS" "Physics" "Computers" "Photo" "fb100" "arxiv-year" "genius" "twitch-gamer")
# dataset_lst=("Cora" "CiteSeer" "PubMed" "CS" "Physics" "Computers" "Photo")
dataset_lst=("fb100" "arxiv-year" "genius" "twitch-gamer")

method="dagnn"
lr_lst=(0.01 0.001)
weight_decay_lst=(0.001 0.0005)
hidden_channels_lst=(64 128)
power_lst=(2 5 10 20)
norm_lst=("True" "False")
epochs=1000


for dataset in "${dataset_lst[@]}"; do


    if [ "$dataset" = "fb100" ]; then
        sub_dataset="Penn94"
    else
        sub_dataset="None"
    fi
    
    if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
        directed="--directed"
    else
        directed=""
    fi

    

    
    for lr in "${lr_lst[@]}"; do
        for weight_decay in "${weight_decay_lst[@]}"; do
            for hidden_channels in "${hidden_channels_lst[@]}"; do
                for power in "${power_lst[@]}"; do
                    for norm in "${norm_lst[@]}"; do
                    
                        if [ "$norm" = "True" ]; then
                            use_norm="--normalize_feature"
                        else
                            use_norm=""
                        fi
                        
                        python main.py --dataset $dataset --sub_dataset $sub_dataset \
                                        --method $method --runs 5 --display_step 25 --epochs $epochs \
                                        --lr $lr --weight_decay $weight_decay $directed \
                                        --hidden_channels $hidden_channels --power $power $use_norm
                        
                       
                    done 
                done 
            done 
        done 
    done
done 