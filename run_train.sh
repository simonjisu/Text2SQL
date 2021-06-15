nohup python -u main.py \
    --num_train 30 \
    --num_gpus 3 \
    --wo_penalty 1 \
    --wv_penalty 1 \
    --task TEXT2SQL_v2 \
    --seed 703 \
    --train_batch_size 32 \
    --eval_batch_size 32 > train.log &