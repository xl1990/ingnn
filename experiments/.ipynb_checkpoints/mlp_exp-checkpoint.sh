#!/bin/bash

# dataset=$1

# dataset_lst=("Cora" "CiteSeer" "PubMed" "CS" "Physics" "Computers" "Photo")
dataset_lst=("syn-cora")

hidden_channels_lst=(16 32 64 128 256)
num_layers_lst=(2 3)

for dataset in "${dataset_lst[@]}"; do
    for num_layers in "${num_layers_lst[@]}"; do
        for hidden_channels in "${hidden_channels_lst[@]}"; do
            python main.py --dataset $dataset --sub_dataset ${2:-''} --method mlp \
            --num_layers $num_layers --hidden_channels $hidden_channels --display_step 25 --runs 5\
            --synh 0.5 --synr 1
        done
    done
done
