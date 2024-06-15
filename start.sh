#!/bin/bash

complete_flag=0
while [ $complete_flag -eq 0 ]; do
    # make user choose "train" or "backtesting"
    echo "Please choose the mode: train or backtesting"
    read mode
    if [ "$mode" == "train" ]; then
        echo "You choose train mode."
        # select the model
        while true; do
            echo "Please choose the model: mamba, transformer"
            read model
            if [ "$model" == "mamba" ]; then
                echo "You choose mamba model."
                python train.py --model mamba
                elif [ "$model" == "transformer" ]; then
                echo "You choose transformer model."
                python train.py --model transformer
            else
                echo "Invalid model, please choose again."
                continue
            fi
            complete_flag=1
            break
        done
        
        elif [ "$mode" == "backtesting" ]; then
        echo "You choose backtesting mode."
        # select the model
        model=""
        while true; do
            echo "Please choose the model: mamba, transformer"
            read model
            if [ "$model" == "mamba" ]; then
                echo "You choose mamba model."
                $model = "mamba"
                elif [ "$model" == "transformer" ]; then
                echo "You choose transformer model."
                $model = "transformer"
            else
                echo "Invalid model, please choose again."
                continue
            fi
            break
        done
        # set default weight
        if [ "$model" == "mamba" ]; then
            weight="./default_weight/default_mamba/weight.pth"
            elif [ "$model" == "transformer" ]; then
            weight="./default_weight/default_transformer/weight.pth"
        fi
        # select the weight
        while true; do
            echo "Please choose the weight: (default: $weight)"
            echo "If you want to use the default weight, please press enter."
            read user_weight
            if [ -z "$user_weight" ]; then
                echo "You choose the default weight."
                elif [ -f "$user_weight" ]; then
                echo "You choose the weight: $user_weight"
                weight=$user_weight
            else
                echo "Invalid weight, please choose again."
                continue
            fi
            break;
        done
        python backtesting.py --model $model --weight $weight
        complete_flag=1
        
    else
        echo "Invalid mode, please choose again."
    fi
    
    sleep 1
done