MODEL_PATH=''

python codes/test.py \
    --model_name_or_path ${MODEL_PATH} \
    --pooler cls_before_pooler \
    --mode test \
    --task_set sts \
    --gpu_id 3
