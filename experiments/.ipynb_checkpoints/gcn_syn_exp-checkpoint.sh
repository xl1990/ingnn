#!/bin/bash

dataset=$1
synr=$2
synh_lst=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

hidden_channels_lst=(16 32 64)

lr_lst=(0.1 0.01 0.001)

for synh in "${synh_lst[@]}"; do
    for hidden_channels in "${hidden_channels_lst[@]}"; do
        for lr in "${lr_lst[@]}"; do
                python main.py --dataset $dataset \
                --method gcn --num_layers 2 --hidden_channels $hidden_channels \
                --lr $lr  --display_step 25 --runs 5 --epochs 500\
                --synh $synh --synr $synr
        done            
    done
done
