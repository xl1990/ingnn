#!/bin/bash

dataset=$1
synr=$2
synh_lst=(0.0)

hidden_channels_lst=(16 32 64 128 256)
num_layers_lst=(2 3)

for synh in "${synh_lst[@]}"; do
    for num_layers in "${num_layers_lst[@]}"; do
        for hidden_channels in "${hidden_channels_lst[@]}"; do
            python main.py --dataset $dataset --method mlp \
            --num_layers $num_layers --hidden_channels $hidden_channels --display_step 25 --runs 5\
            --synh $synh --synr $synr
        done
    done
done
