#!/bin/bash

# dataset=$1
# sub_dataset=${2:-'None'}

# dataset_lst=("Cora" "CiteSeer" "PubMed" "CS" "Physics" "Computers" "Photo" "fb100" "arxiv-year" "genius" "twitch-gamer") 
# dataset_lst=("chameleon" "cornell" "film" "squirrel" "texas" "wisconsin" "Cora" "CiteSeer" "PubMed" "CS" "Physics" "fb100" "arxiv-year" "genius" "twitch-gamer") 

# dataset_lst=("Cora" "CiteSeer" "PubMed" "CS" "chameleon" "cornell" "film" "squirrel" "texas" "wisconsin")
# dataset_lst=("Cora" "CiteSeer" "PubMed" "CS")
dataset_lst=("Cora" "PubMed")

lr_lst=(0.01 0.001)
weight_decay_lst=(0.001 0.0005)
hidden_channels_lst=(32 64 128)
num_layers_lst=(1 2)
dropout_lst=(0.0 0.5)
norm_lst=("True" "False")
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
    
    for lr in "${lr_lst[@]}"; do
        for weight_decay in "${weight_decay_lst[@]}"; do
            for hidden_channels in "${hidden_channels_lst[@]}"; do
                for num_layers in "${num_layers_lst[@]}"; do
                    for dropout in "${dropout_lst[@]}"; do
                        for norm in "${norm_lst[@]}"; do
                            if [ "$norm" = "True" ]; then
                                use_norm="--normalize_feature"
                            else
                                use_norm=""
                            fi
                            
                            python main_bilevel_optim.py --dataset $dataset --sub_dataset $sub_dataset \
                                    --method h2gcn_strc --runs 5 --display_step 25 --epochs $epochs \
                                    --lr $lr --weight_decay $weight_decay $directed \
                                    --hidden_channels $hidden_channels --num_layers $num_layers --dropout $dropout $use_norm
                        done
                    done
                done            
            done
        done
    done
done