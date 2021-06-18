nohup python -u main.py \
    --num_train 30 \
    --num_gpus 4 \
    --task TEXT2SQL_v2 \
    --seed 86 \
    --train_batch_size 48 \
    --eval_batch_size 48 > train.log &

    