nohup python -u main.py \
    --num_train 30 \
    --num_gpus 4 \
    --task TEXT2SQL_v1 \
    --seed 703 \
    --train_batch_size 32 \
    --eval_batch_size 32 > train.log &

    