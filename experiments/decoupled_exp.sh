#!/bin/bash

# dataset=$1
# sub_dataset=${2:-''}

# dataset_lst=("fb100" "arxiv-year" "snap-patents" "pokec" "genius" "twitch-gamer") 
dataset_lst=("snap-patents" "pokec") 
sub_dataset="Penn94" # Only fb100 uses sub_datase

num_layers_lst=(2 3 5 10)
for dataset in "${dataset_lst[@]}"; do
    for num_layers in "${num_layers_lst[@]}"; do
        if [ "$dataset" = "fb100" ]; then
            python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method decoupled --lr 0.01 --num_layers $num_layers --hidden_channels 64  --dropout 0.5 --display_step 100 --runs 5
        elif [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
            python main.py --dataset $dataset --method decoupled --lr 0.01 --num_layers $num_layers --hidden_channels 64 --dropout 0.5 --display_step 100 --runs 5 --directed
        else
            python main.py --dataset $dataset --method decoupled --lr 0.01 --num_layers $num_layers --hidden_channels 64  --dropout 0.5 --display_step 100 --runs 5
        fi
    done
done
