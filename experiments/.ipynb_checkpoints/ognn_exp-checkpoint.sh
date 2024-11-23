#!/bin/bash

# dataset=$1
# sub_dataset=${2:-''}

dataset_lst=("Cora" "CiteSeer" "PubMed" "CS" "Physics" "Computers" "Photo")
# dataset_lst=("syn-cora")

method="ognn"

lr_lst=(0.01 0.001)
weight_decay_lst=(0.001 0.0005)
hidden_channels_lst=(32 64 128)
power1_lst=(2 5 10 20)
power2_lst=(1 2 5)
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
    
    # keep this block if not loop over norm_lst
    set +e #otherwise the script will exit on error
    containsElement () {
      local e match="$1"
      shift
      for e; do [[ "$e" == "$match" ]] && return 0; done
      return 1
    }
    need_norm=("Cora" "CiteSeer" "PubMed")
    if containsElement $dataset "${need_norm[@]}"; then
        use_norm="--normalize_feature"
    else
        use_norm=""
    fi
    echo use_norm
    
    for lr in "${lr_lst[@]}"; do
        for weight_decay in "${weight_decay_lst[@]}"; do
            for hidden_channels in "${hidden_channels_lst[@]}"; do
                for power1 in "${power1_lst[@]}"; do
                    for power2 in "${power2_lst[@]}"; do
                        # for norm in "${norm_lst[@]}"; do
                        #     if [ "$norm" = "True" ]; then
                        #         use_norm="--normalize_feature"
                        #     else
                        #         use_norm=""
                        #     fi
                
                            python main_bilevel_optim.py --dataset $dataset --sub_dataset $sub_dataset \
                                        --method $method --runs 5 --display_step 25 --epochs $epochs \
                                        --lr $lr --weight_decay $weight_decay $directed \
                                        --hidden_channels $hidden_channels --power1 $power1 --power2 $power2 $use_norm
                        # done
                    done
                done
            done
        done
    done
done