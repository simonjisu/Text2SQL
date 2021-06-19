#!/bin/bash
nohup python -u main.py \
    --num_train 25 \
    --num_workers 8 \
    --num_gpus 4 \
    --task TEXT2SQL_v1 \
    --seed 703 \
    --log_every_n_steps 25 \
    --train_batch_size 64 \
    --eval_batch_size 64 > train.log &
    