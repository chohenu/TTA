#!/bin/bash

SRC_DOMAINS=(art_painting art_painting art_painting photo        photo   photo)
TGT_DOMAINS=(cartoon      photo        sketch       art_painting cartoon sketch)
PORTS=(10021 10022 10023 10024 10025 10026)
CUDAS=(0,1 2,3 4,5 4,5 6,7 6,7)
SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/pacs/source"

PORT=10027
MEMO="Confidence"
SUB_MEMO="EPOCH15_B64_LR0e5"
N_PROCESS=6
for i in "${!SRC_DOMAINS[@]}"
do  
    CUDA_VISIBLE_DEVICES=${CUDAS[i]} python main_adacontrast.py \
    seed=2021 port=${PORTS[i]} memo=${MEMO} sub_memo=${SUB_MEMO} project="pacs" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="pacs" data.source_domains="[${SRC_DOMAINS[i]}]" data.target_domains="[${TGT_DOMAINS[i]}]" \
    model_src.arch="resnet18" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.cos=False \
    optim.exp=false \
    optim.no_sch=True \
    optim.lr=1e-5 \
    learn.epochs=100 \
    learn=targetv2.yaml &
    if [[ $(jobs -r -p | wc -l) -ge $N_PROCESS ]]; then wait -n; fi
done;
wait