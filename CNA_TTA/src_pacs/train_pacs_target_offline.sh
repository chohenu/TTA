#!/bin/bash
PORT=17019
CUDA=2,3
SRC_DOMAINS=(art_painting art_painting art_painting photo        photo   photo)
TGT_DOMAINS=(cartoon      photo        sketch       art_painting cartoon sketch)
SRC_MODEL_DIR="/opt/tta/CNA_TTA/output/pacs/source"
MEMO="pacs_online_constant_lr"
SUB_MEMO="online_train_constant_lr"

for SEED in 2021 2022 2023
do
    for i in "${!SRC_DOMAINS[@]}"
    do  
        CUDA_VISIBLE_DEVICES=${CUDA} python ../main.py \
        seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="pacs" \
        data.data_root="/mnt/data" data.workers=8 \
        data.dataset="pacs" data.source_domains="[${SRC_DOMAINS[i]}]" data.target_domains="[${TGT_DOMAINS[i]}]" \
        model_src.arch="resnet18" \
        model_tta.src_log_dir=${SRC_MODEL_DIR} \
        data.batch_size=32 \
        learn.epochs=50 \
        optim.lr=2e-4 \
    done
done