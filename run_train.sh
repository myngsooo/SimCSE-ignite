TRAIN_PATH=''
STS_DEV_PATH=''
MODEL_PATH='bert-base-uncased' # huggingface transformer embedding model
OUTPUT_DIR=''

python codes/train.py \
    --train_fn ${TRAIN_PATH} \
    --valid_fn ${STS_DEV_PATH} \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 64 \
    --do_eval_step \
    --eval_step 250 \
    --lr 3e-5 \
    --n_epochs 1 \
    --max_length 32 \
    --pooler_type cls \
    --temp 0.05 \
    --mixed_precision \