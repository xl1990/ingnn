#!/bin/bash

# dataset=$1
# sub_dataset=${2:-''}


hidden_channels_lst=(16 32 64 128 256)
lr_lst=(0.01 0.05 0.002)
dataset_lst=("CiteSeer" "PubMed" "CS" "Physics" "Computers" "Photo" "fb100" "arxiv-year" "genius" "twitch-gamer")

for dataset in "${dataset_lst[@]}"; do
    if [ "$dataset" = "fb100" ]; then
        sub_dataset="Penn94"
    else
        sub_dataset="None"
    fi
    
    for hidden_channels in "${hidden_channels_lst[@]}"; do
        for lr in "${lr_lst[@]}"; do
                if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                    python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method gprgnn --lr $lr --hidden_channels $hidden_channels  --display_step 25 --runs 5 --directed
                else
                    python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method gprgnn --lr $lr --hidden_channels $hidden_channels  --display_step 25 --runs 5
                fi
        done
    done
done
