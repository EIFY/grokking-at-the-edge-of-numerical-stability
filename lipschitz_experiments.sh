#!/usr/bin/env bash

DEVICE="cuda:0"

AR="--all_reg"
EM="--use_embedding"

for op in add_mod product_mod
do
    for i in {0..63}
    do
        for LR in 0.001 0.002 0.005 0.01 0.02 0.05 0.1
        do
            for WD in 2. 4. 6. 8. 10.
            do
                python grokking_experiments.py --binary_operation $op --lr $LR --weight_decay $WD --num_epochs 1001 --log_frequency 1 --device $DEVICE --train_fraction 0.4 --train_dtype bfloat16 --cross_entropy_dtype bfloat16 --seed $i $EM $AR
            done
        done
    done
done
