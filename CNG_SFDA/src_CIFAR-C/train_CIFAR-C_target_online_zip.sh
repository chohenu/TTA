#!/bin/bash
    
PORT=17019
CUDA=1,2,3,4
SRC_DOMAINS=(origin origin origin origin origin origin origin origin origin origin origin origin origin origin origin origin )
# TGT_DOMAINS=(gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression)

TGT_DOMAINS=(gaussian_noise)


PORTS=(${PORT} ${PORT} ${PORT} ${PORT} ${PORT} ${PORT} ${PORT})
CUDAS=(${CUDA} ${CUDA} ${CUDA} ${CUDA} ${CUDA} ${CUDA} ${CUDA})
SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/cifar/source"


MEMO="cifar10_online"
SUB_MEMO="cifar10_train"

for SEED in 2021
do
    for i in "${!TGT_DOMAINS[@]}"
    do  
        CUDA_VISIBLE_DEVICES=${CUDAS[i]}\
        python ../main.py \
        seed=${SEED}\
        port=${PORTS[i]}\
        memo=${MEMO}\
        sub_memo=${SUB_MEMO}_${i} \
        project="cifar10" \
        data.data_root="/mnt/data" \
        data.dataset="cifar10" \
        data.source_domains="${SRC_DOMAINS[i]}" \
        data.target_domains="${TGT_DOMAINS[i]}" \
        model_src.arch="Standard" \
        model_tta.src_log_dir=${SRC_MODEL_DIR} \
        learn=target.yaml \
        learn.epochs=1 \
        optim.time=1 \
        optim.lr=2e-5 \
        data.batch_size=32 \
        ckpt_path=False
    done
done