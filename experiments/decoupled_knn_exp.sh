#!/bin/bash

dataset=$1
sub_dataset=${2:-''}

num_layers_lst=(2 3 5 10 15 20)

for num_layers in "${num_layers_lst[@]}"; do
    if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
        python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method decoupled_knn --lr 0.01 --num_layers $num_layers --hidden_channels 64 --dropout 0.5 --k 30 --display_step 25 --runs 5 --directed
    else
        python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method decoupled_knn --lr 0.01 --num_layers $num_layers --hidden_channels 64  --dropout 0.5 --k 30 --display_step 25 --runs 5
    fi
done
