#!/bin/bash
PORT=17019
CUDA=6,7
SRC_DOMAINS=(art_painting art_painting art_painting photo        photo   photo)
TGT_DOMAINS=(cartoon      photo        sketch       art_painting cartoon sketch)
PORTS=(${PORT} ${PORT} ${PORT} ${PORT} ${PORT} ${PORT} ${PORT})
CUDAS=(${CUDA} ${CUDA} ${CUDA} ${CUDA} ${CUDA} ${CUDA} ${CUDA})
SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/pacs/source"
MEMO="pacs_online"
SUB_MEMO="online_train"

for SEED in 2023
do
    for i in "${!SRC_DOMAINS[@]}"
    do  
        CUDA_VISIBLE_DEVICES=${CUDAS[i]} python ../main_adacontrast.py \
        seed=${SEED} port=${PORTS[i]} memo=${MEMO} sub_memo=${SUB_MEMO} project="pacs" \
        data.data_root="/mnt/data" data.workers=8 \
        data.dataset="pacs" data.source_domains="[${SRC_DOMAINS[i]}]" data.target_domains="[${TGT_DOMAINS[i]}]" \
        model_src.arch="resnet18" \
        model_tta.src_log_dir=${SRC_MODEL_DIR} \
        learn=targetv1.yaml \
        learn.epochs=1 \
        optim.time=1 \
        optim.lr=2e-4 \
        data.batch_size=32 \
        ckpt_path=False
    done
done