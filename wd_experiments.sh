#!/usr/bin/env bash

DEVICE="cuda:0"

for LR in 0.001 0.005 0.01 0.05
do
    for WD in 2. 4. 6. 8. 10.
    do
        python grokking_experiments.py --lr $LR --weight_decay $WD --num_epochs 401 --log_frequency 1 --device $DEVICE --train_fraction 0.4 --use_embedding --input_size 100 --hidden_sizes 200 200 200
        python grokking_experiments.py --lr $LR --weight_decay $WD --num_epochs 401 --log_frequency 1 --device $DEVICE --train_fraction 0.4 --hidden_sizes 200 200 200
        python grokking_experiments.py --lr $LR --weight_decay $WD --num_epochs 401 --log_frequency 1 --device $DEVICE --train_fraction 0.4 --use_embedding --input_size 100
        python grokking_experiments.py --lr $LR --weight_decay $WD --num_epochs 401 --log_frequency 1 --device $DEVICE --train_fraction 0.4
        python grokking_experiments.py --lr $LR --weight_decay $WD --num_epochs 401 --log_frequency 1 --device $DEVICE --train_fraction 0.4 --use_embedding --input_size 100 --hidden_sizes 200
        python grokking_experiments.py --lr $LR --weight_decay $WD --num_epochs 10001 --log_frequency 1 --device $DEVICE --train_fraction 0.4 --hidden_sizes 200
    done

    python grokking_experiments.py --lr $LR --orthogonal_gradients --num_epochs 401 --log_frequency 1 --device $DEVICE --train_fraction 0.4 --use_embedding --input_size 100 --hidden_sizes 200 200 200
    python grokking_experiments.py --lr $LR --orthogonal_gradients --num_epochs 401 --log_frequency 1 --device $DEVICE --train_fraction 0.4 --hidden_sizes 200 200 200
    python grokking_experiments.py --lr $LR --orthogonal_gradients --num_epochs 401 --log_frequency 1 --device $DEVICE --train_fraction 0.4 --use_embedding --input_size 100
    python grokking_experiments.py --lr $LR --orthogonal_gradients --num_epochs 401 --log_frequency 1 --device $DEVICE --train_fraction 0.4
    python grokking_experiments.py --lr $LR --orthogonal_gradients --num_epochs 401 --log_frequency 1 --device $DEVICE --train_fraction 0.4 --use_embedding --input_size 100 --hidden_sizes 200
    python grokking_experiments.py --lr $LR --orthogonal_gradients --num_epochs 10001 --log_frequency 1 --device $DEVICE --train_fraction 0.4 --hidden_sizes 200
done
