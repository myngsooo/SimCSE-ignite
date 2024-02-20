## SimCSE: Simple Contrastive Learning of Sentence Embeddings

This repo can be an alternative to the official SimCSE, which is deemed overly complex with numerous features beyond users' expectations. Focused solely on the unsupervised version of SimCSE, this repo provides a minimal feature set for developing the unsupervised SimCSE methodology. Therfore, I hope that this repo can serve as an ideal starting point for individuals seeking simplicity without unnecessary features. 

Please note that some of the code remains unchanged.

## Requirements
First, install PyTorch by following the instructions from [the official website](https://pytorch.org). 

```bash
pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

If you instead use **CUDA** `<11` or **CPU**, install PyTorch by the following command,

```bash
pip install torch==1.12.1
```

Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```

### Download the training/dev dataset
```
cd data
bash download_wiki.sh
bash download_stsb.sh
```

### Download the downstream dataset
```
cd SentEval/data/
bash download.sh
```

## Training
```bash
TRAIN_PATH=''
STS_DEV_PATH=''
MODEL_PATH='bert-base-uncased'
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
    --gpu_id 0 \
```

## Evaluation
You can run the commands below for evaluation after using the repo to train a model:

```bash
MODEL_PATH=''

python codes/test.py \
    --model_name_or_path ${MODEL_PATH} \ # OUTPUT_DIR arguments within Training
    --pooler cls_before_pooler \
    --mode test \
    --task_set sts \
    --gpu_id 0

```

## References
[SimCSE's GitHub repo](https://github.com/princeton-nlp/SimCSE).
