#!/bin/bash

# dataset=$1
# sub_dataset=${2:-'None'}

# dataset_lst=("Physics" "Computers" "Photo" "fb100" "arxiv-year" "genius" "twitch-gamer") 
# dataset_lst=("chameleon" "cornell" "film" "squirrel" "texas" "wisconsin") 

# dataset_lst=("Cora" "CiteSeer" "PubMed" "CS" "chameleon" "cornell" "film" "squirrel" "texas" "wisconsin")
# dataset_lst=("Cora" "CiteSeer" "PubMed" "CS")
# dataset_lst=("Cora")
dataset_lst=("Cora" "CiteSeer" "PubMed")


lr_lst=(0.01 0.001)
# lr_lst=(0.001)
weight_decay_lst=(0.0005)
hidden_channels_lst=(128)
num_layers_lst=(2)
dropout_lst=(0.5)
alpha_lst=(0.0 0.1)
beta_lst=(0.1 1.0)
delta_lst=(0.1 0.5)
gamma_lst=(0.5 0.7)
orders_lst=(1 2 3)

epochs=3000



for dataset in "${dataset_lst[@]}"; do
    if [ "$dataset" = "fb100" ]; then
        sub_dataset="Penn94"
    else
        sub_dataset="None"
    fi
    echo $sub_dataset
    
    if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
        directed="--directed"
    else
        directed=""
    fi
    
    if [ "$dataset" = "Cora" ]; then
        lr_lst=(0.01)
        weight_decay_lst=(5e-5)
        hidden_channels_lst=(64)
        num_layers_lst=(2)
        dropout_lst=(0.8)
        alpha_lst=(0.0)
        beta_lst=(800.0)
        delta_lst=(0.9)
        gamma_lst=(0.8)
        orders_lst=(4)
    fi

    if [ "$dataset" = "CiteSeer" ]; then
        lr_lst=(0.01)
        weight_decay_lst=(1e-5)
        hidden_channels_lst=(64)
        num_layers_lst=(2)
        dropout_lst=(0.8)
        alpha_lst=(1.0)
        beta_lst=(1000.0)
        delta_lst=(1.0)
        gamma_lst=(0.8)
        orders_lst=(3)
    fi

    if [ "$dataset" = "PubMed" ]; then
        lr_lst=(0.01)
        weight_decay_lst=(5e-5)
        hidden_channels_lst=(64)
        num_layers_lst=(2)
        dropout_lst=(0.6)
        alpha_lst=(0.0001)
        beta_lst=(20000.0)
        delta_lst=(1.0)
        gamma_lst=(0.5)
        orders_lst=(3)
    fi
    
    for lr in "${lr_lst[@]}"; do
        for weight_decay in "${weight_decay_lst[@]}"; do
            for hidden_channels in "${hidden_channels_lst[@]}"; do
                for num_layers in "${num_layers_lst[@]}"; do
                    for dropout in "${dropout_lst[@]}"; do
                        for alpha in "${alpha_lst[@]}"; do
                            for beta in "${beta_lst[@]}"; do
                                for delta in "${delta_lst[@]}"; do
                                    for gamma in "${gamma_lst[@]}"; do
                                        for orders in "${orders_lst[@]}"; do
                
                                            python main.py --dataset $dataset --sub_dataset $sub_dataset \
                                            --method glognn --runs 5 --display_step 25 --epochs $epochs \
                                            --lr $lr --weight_decay $weight_decay $directed \
                                            --hidden_channels $hidden_channels --num_layers $num_layers --dropout $dropout $variant\
                                            --alpha $alpha --beta $beta --delta $delta --gamma $gamma --orders $orders
                                        done
                                    done       
                                done
                            done            
                        done
                    done
                done            
            done
        done
    done
done