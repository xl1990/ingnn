#!/bin/bash

dataset=$1
sub_dataset=${2:-''}
# dataset_lst=("Cora" "CiteSeer" "PubMed" "CS" "Physics" "Computers" "Photo" "fb100" "arxiv-year" "genius" "twitch-gamer")
# dataset_lst=("Cora")

# for dataset in "${dataset_lst[@]}"; do
    if [ "$dataset" = "Cora" ]; then
        # hidden_channels, power1, power2, lr, weight_decay, epochs
        args_lst=(128 20 5 0.01 0.0005 2000)
    elif [ "$dataset" = "CiteSeer" ]; then
        # hidden_channels, power1, power2, lr, weight_decay, epochs
        args_lst=(64 10 2 0.01 0.0005 2000)
    elif [ "$dataset" = "PubMed" ]; then
        # hidden_channels, power1, power2, lr, weight_decay, epochs
        args_lst=(64 20 5 0.01 0.0005 3000)
    elif [ "$dataset" = "CS" ]; then
        # hidden_channels, power1, power2, lr, weight_decay, epochs
        args_lst=(64 10 2 0.01 0.0005 1000)
    elif [ "$dataset" = "Physics" ]; then
        # hidden_channels, power1, power2, lr, weight_decay, epochs
        args_lst=(64 20 10 0.01 0.001 1000)
    elif [ "$dataset" = "Computers" ]; then
        # hidden_channels, power1, power2, lr, weight_decay, epochs
        args_lst=(64 10 32 0.01 0.0005 3000)
    elif [ "$dataset" = "Photo" ]; then
        # hidden_channels, power1, power2, lr, weight_decay, epochs
        args_lst=(64 10 32 0.01 0.0005 1000)
    elif [ "$dataset" = "fb100" ]; then
        # hidden_channels, power1, power2, lr, weight_decay, epochs
        args_lst=(64 5 2 0.01 0.0005 1000)
    elif [ "$dataset" = "arxiv-year" ]; then
        # hidden_channels, power1, power2, lr, weight_decay, epochs
        args_lst=(128 2 2 0.01 0.001 1000)
    elif [ "$dataset" = "genius" ]; then
        # hidden_channels, power1, power2, lr, weight_decay, epochs
        args_lst=(64 5 128 0.005 0.001 1000)
    elif [ "$dataset" = "twitch-gamer" ]; then
        # hidden_channels, power1, power2, lr, weight_decay, epochs
        args_lst=(128 2 5 0.001 0.0005 1000)
    elif [ "$dataset" = "syn-cora" ]; then
        # hidden_channels, power1, power2, lr, weight_decay, epochs
        args_lst=(128 10 2 0.01 0.0005 2000)
    elif [ "$dataset" = "syn-products" ]; then
        # hidden_channels, power1, power2, lr, weight_decay, epochs
        args_lst=(128 10 2 0.01 0.0005 2000)
    fi
    
    if [ "$dataset" = "fb100" ]; then
        sub_dataset="Penn94"
    else
        sub_dataset=""
    fi

    hidden_channels=${args_lst[0]}
    power1=${args_lst[1]}
    power2=${args_lst[2]}
    lr=${args_lst[3]}
    weight_decay=${args_lst[4]}
    epochs=${args_lst[5]}
    preheat=50
    synh=0.5
    synr=1
    
    power=${args_lst[1]}


    if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
        echo "Running $dataset "
        python main_DF.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method linkdex_df2 --num_layers 1 \
        --hidden_channels $hidden_channels --power $power --power1 $power1 --power2 $power2 --lr $lr --weight_decay $weight_decay --policy_update_freq 20 \
        --policy_update_iter 10 --preheat $preheat --epochs $epochs --display_step 100 --runs 5 --normalize_feature --directed --synh 0.5 --synr 1
    else
        python main_DF.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method linkdex_df2 --num_layers 1 \
        --hidden_channels $hidden_channels --power $power --power1 $power1 --power2 $power2 --lr $lr --weight_decay $weight_decay --policy_update_freq 20 \
        --policy_update_iter 10 --preheat $preheat --epochs $epochs --display_step 100 --runs 5 --normalize_feature --synh 0.5 --synr 1
    fi
# done