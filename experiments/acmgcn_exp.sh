#!/bin/bash

# dataset=$1
# sub_dataset=${2:-'None'}

# dataset_lst=("Physics" "Computers" "Photo" "fb100" "arxiv-year" "genius" "twitch-gamer") 
# dataset_lst=("chameleon" "cornell" "film" "squirrel" "texas" "wisconsin") 

# dataset_lst=("Cora" "CiteSeer" "PubMed" "CS" "chameleon" "cornell" "film" "squirrel" "texas" "wisconsin")
# dataset_lst=("Cora" "CiteSeer" "PubMed" "CS")
# dataset_lst=("Cora")
dataset_lst=("Cora" "CiteSeer" "PubMed" "CS")


# lr_lst=(0.01 0.001)
lr_lst=(0.001)
weight_decay_lst=(0.0005 0.0)
hidden_channels_lst=(64 128)
dropout_lst=(0.5 0.8)
variant_lst=("--variant" "")

epochs=1000

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
                for dropout in "${dropout_lst[@]}"; do
                    for variant in "${variant_lst[@]}"; do
                
                        python main.py --dataset $dataset --sub_dataset $sub_dataset \
                                    --method acmgcn --runs 5 --display_step 25 --epochs $epochs \
                                    --lr $lr --weight_decay $weight_decay $directed \
                                    --hidden_channels $hidden_channels --dropout $dropout $variant
                    done
                done            
            done
        done
    done
done