nohup python -u main.py \
    --num_train 30 \
    --tsk TEXT2SQL_v2 \
    --train_batch_size 32 \
    --eval_batch_size 32 > train.log &