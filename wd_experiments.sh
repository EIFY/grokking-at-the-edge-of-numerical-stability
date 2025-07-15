#!/usr/bin/env bash

DEVICE="cuda:0"

for AR in "--all_reg" ""
do
    HS=200

    for LR in 0.001 0.002 0.005 0.01 0.02 0.05 0.1
    do
        for WD in 2. 4. 6. 8. 10.
        do
            python grokking_experiments.py --lr $LR --weight_decay $WD --num_epochs 10001 --log_frequency 1 --device $DEVICE --train_fraction 0.4 --hidden_sizes $HS $AR
        done
        python grokking_experiments.py --lr $LR --orthogonal_gradients --num_epochs 10001 --log_frequency 1 --device $DEVICE --train_fraction 0.4 --hidden_sizes $HS $AR
        # The PyTorch default that all the official orthogonal AdamW experiments have been running
        python grokking_experiments.py --lr $LR --orthogonal_gradients --num_epochs 10001 --log_frequency 1 --device $DEVICE --train_fraction 0.4 --hidden_sizes $HS $AR --beta2 0.999 --adam_epsilon 1e-08
    done

    EM="--use_embedding --input_size 100"

    for LR in 0.001 0.002 0.005 0.01 0.02 0.05 0.1
    do
        for WD in 2. 4. 6. 8. 10.
        do
            python grokking_experiments.py --lr $LR --weight_decay $WD --num_epochs 401 --log_frequency 1 --device $DEVICE --train_fraction 0.4 $EM --hidden_sizes $HS $AR
        done
        python grokking_experiments.py --lr $LR --orthogonal_gradients --num_epochs 401 --log_frequency 1 --device $DEVICE --train_fraction 0.4 $EM --hidden_sizes $HS $AR
        # The PyTorch default that all the official orthogonal AdamW experiments have been running
        python grokking_experiments.py --lr $LR --orthogonal_gradients --num_epochs 401 --log_frequency 1 --device $DEVICE --train_fraction 0.4 $EM --hidden_sizes $HS $AR --beta2 0.999 --adam_epsilon 1e-08
    done

    for HS in "200 200" "200 200 200"
    do
        for EM in "--use_embedding --input_size 100" ""
        do
            for LR in 0.001 0.002 0.005 0.01 0.02 0.05 0.1
            do
                for WD in 2. 4. 6. 8. 10.
                do
                    python grokking_experiments.py --lr $LR --weight_decay $WD --num_epochs 401 --log_frequency 1 --device $DEVICE --train_fraction 0.4 $EM --hidden_sizes $HS $AR
                done
                python grokking_experiments.py --lr $LR --orthogonal_gradients --num_epochs 401 --log_frequency 1 --device $DEVICE --train_fraction 0.4 $EM --hidden_sizes $HS $AR
                # The PyTorch default that all the official orthogonal AdamW experiments have been running
                python grokking_experiments.py --lr $LR --orthogonal_gradients --num_epochs 401 --log_frequency 1 --device $DEVICE --train_fraction 0.4 $EM --hidden_sizes $HS $AR --beta2 0.999 --adam_epsilon 1e-08
            done
        done
    done
done
