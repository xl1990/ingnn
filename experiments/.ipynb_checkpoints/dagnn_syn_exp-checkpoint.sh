#!/bin/bash

dataset=$1
synr=$2
synh_lst=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for synh in "${synh_lst[@]}"; do
    if [ "$dataset" = "syn-cora" ]; then
        # hidden_channels, power1, power2, lr, weight_decay, epochs
        args_lst=(128 10 2 0.01 0.0005 2000)
    elif [ "$dataset" = "syn-products" ]; then
        # hidden_channels, power1, power2, lr, weight_decay, epochs
        args_lst=(128 10 2 0.01 0.0005 2000)
    fi


    hidden_channels=${args_lst[0]}
    power1=${args_lst[1]}
    power2=${args_lst[2]}
    lr=${args_lst[3]}
    weight_decay=${args_lst[4]}
    epochs=${args_lst[5]}
    preheat=50
    
    power=${args_lst[1]}


    python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method dagnn --num_layers 1 \
    --hidden_channels $hidden_channels --power $power --power1 $power1 --power2 $power2 --lr $lr --weight_decay $weight_decay --policy_update_freq 20 \
    --policy_update_iter 10 --preheat $preheat --epochs $epochs --display_step 100 --runs 5 --normalize_feature\
    --synh $synh --synr $synr
done