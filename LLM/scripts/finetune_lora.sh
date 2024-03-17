#! /usr/bin/env bash

set -ex

LR=1e-4
NUM_GPUS=1
LORA_RANK=8
LORA_ALPHA=32
LORA_DROUPOUT=0.1

MAX_SOURCE_LEN=2048
MAX_TARGET_LEN=128
DEV_BATCH_SIZE=1
GRAD_ACCUMULARION_STEPS=1
NUM_TRAIN_EPOCHS=4
#MAX_STEP=5000
SAVE_INTERVAL=2000
MAX_SEQ_LEN=2048

RUN_NAME=CEE_4epoch
BASE_MODEL_PATH=xxx
DATASET_PATH=xxx/train.json
DATESTR=`date +%Y%m%d-%H%M%S`
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${LR}
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

mkdir -p $OUTPUT_DIR

#torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
python finetune.py \
      --train_format input-output \
      --train_file $DATASET_PATH \
      --lora_rank $LORA_RANK \
      --lora_alpha $LORA_ALPHA \
      --lora_dropout $LORA_DROUPOUT \
      --max_seq_length $MAX_SEQ_LEN \
      --preprocessing_num_workers 1 \
      --model_name_or_path $BASE_MODEL_PATH \
      --output_dir $OUTPUT_DIR \
      --per_device_train_batch_size $DEV_BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
      --num_train_epochs $NUM_TRAIN_EPOCHS \
      --logging_steps 1 \
      --save_steps $SAVE_INTERVAL \
      --learning_rate $LR 2>&1 | tee ${OUTPUT_DIR}/train.log
