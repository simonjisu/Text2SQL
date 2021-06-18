nohup python -u main.py \
    --num_train 25 \
    --num_workers 16 \
    --num_gpus 3 \
    --task TEXT2SQL_v1 \
    --seed 703 \
    --log_every_n_steps 30 \
    --train_batch_size 64 \
    --eval_batch_size 64 > train.log &

    