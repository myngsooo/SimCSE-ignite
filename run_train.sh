export CUDA_VISIBLE_DEVICES=1

TRAIN_PATH='/home/user/lg3/simcse/codes/data/wiki1m_for_simcse.txt'
STS_DEV_PATH='/home/user/lg3/simcse/codes/data/sts-dev.tsv'
MODEL_PATH='bert-base-uncased'

python codes/train.py \
    --train_fn ${TRAIN_PATH} \
    --valid_fn ${STS_DEV_PATH} \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ./codes/result/model.pt \
    --batch_size 64 \
    --do_eval_step \
    --eval_step 250 \
    --lr 3e-5 \
    --n_epochs 1 \
    --max_length 32 \
    --pooler_type cls \
    --temp 0.05 \
    --mixed_precision \