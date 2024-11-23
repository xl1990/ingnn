#!/bin/bash

dataset=$1
synr=$2
synh_lst=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

hidden_channels_lst=(16 32 64 128 256)
lr_lst=(0.01 0.05 0.002)

for synh in "${synh_lst[@]}"; do
    for hidden_channels in "${hidden_channels_lst[@]}"; do
        for lr in "${lr_lst[@]}"; do
                if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                    python main.py --dataset $dataset --method gprgnn --lr $lr\
                    --hidden_channels $hidden_channels  --display_step 25 --runs 5 --directed \
                    --synh $synh --synr $synr
                else
                    python main.py --dataset $dataset --method gprgnn --lr $lr\
                    --hidden_channels $hidden_channels  --display_step 25 --runs 5 \
                    --synh $synh --synr $synr
                fi
        done
    done
done
