nohup python -u main.py \
    --num_train 25 \
    --num_workers 8 \
    --num_gpus 4 \
    --task TEXT2SQL_v3 \
    --seed 703 \
    --train_batch_size 64 \
    --eval_batch_size 64 > train.log &

    