#!/bin/bash

# dataset=$1

dataset_lst=("Cora" "CiteSeer" "PubMed" "CS" "Physics" "Computers" "Photo" "fb100" "arxiv-year" "genius" "twitch-gamer")
hidden_channels_lst=(16 32 64 128 256)
lr_lst=(0.01 0.05 0.002)
alpha_lst=(0.1 0.2 0.5 0.9)
norm_lst=("True" "False")

for dataset in "${dataset_lst[@]}"; do

    if [ "$dataset" == "fb100" ]; then
        sub_dataset="Penn94"
    else
        sub_dataset=""
    fi
    
    echo "-------------"
    echo $sub_dataset
    echo "-------------"


    for hidden_channels in "${hidden_channels_lst[@]}"; do
        for lr in "${lr_lst[@]}"; do
            for alpha in "${alpha_lst[@]}"; do
                for norm in "${norm_lst[@]}"; do
                
                    if [[ "$norm" == "True" ]]; then
                        if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                            python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method appnp --lr $lr --hidden_channels $hidden_channels --gpr_alpha $alpha --display_step 25 --runs 5 --directed --normalize_feature
                        else
                            python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method appnp --lr $lr --hidden_channels $hidden_channels --gpr_alpha $alpha --display_step 25 --runs 5 --normalize_feature
                        fi
                    else
                        if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                            python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method appnp --lr $lr --hidden_channels $hidden_channels --gpr_alpha $alpha --display_step 25 --runs 5 --directed
                        else
                            python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method appnp --lr $lr --hidden_channels $hidden_channels --gpr_alpha $alpha --display_step 25 --runs 5
                        fi
                    fi
                
                done # for norm
            done # for alpha
        done #for lr
    done # for hidden channels
done # for dataset