#!/usr/bin/env bash

DEVICE="cuda:0"

for WD in 2. 4. 6. 8.
do
    python grokking_experiments.py --lr 0.01 --weight_decay $WD --num_epochs 1001 --log_frequency 10 --device $DEVICE --train_fraction 0.4
done