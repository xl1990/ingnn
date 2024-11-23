#!/bin/bash

dataset=$1
synr=$2
synh_lst=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

hidden_channels_lst=(8 16 32 64)

num_layers_lst=(1 2)

dropout_lst=(0 .5)

for synh in "${synh_lst[@]}"; do
    for hidden_channels in "${hidden_channels_lst[@]}"; do
        for num_layers in "${num_layers_lst[@]}"; do
            for dropout in "${dropout_lst[@]}"; do
                python main.py --dataset $dataset \
                --method h2gcn --num_layers $num_layers --hidden_channels $hidden_channels \
                --display_step 25 --runs 5 --dropout $dropout \
                --synh $synh --synr $synr
            done
        done            
    done
done
