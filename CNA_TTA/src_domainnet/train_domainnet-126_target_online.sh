PORT=17014
CUDA=4,5
SRC_MODEL_DIR="/opt/tta/CNA_TTA/output/domainnet-126/source"
SRC_DOMAINS=(real    real    real     sketch    clipart painting)
TGT_DOMAINS=(sketch  clipart painting painting  sketch  real    )
MEMO="domainnet-126"
SUB_MEMO="online"

for SEED in 2021 2022 2023
do
    for i in "${!SRC_DOMAINS[@]}"
    do
        CUDA_VISIBLE_DEVICES=${CUDA} python ../main.py \
        seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="domainnet-126" \
        data.data_root="/mnt/data" data.workers=8 \
        data.dataset="domainnet-126" data.source_domains="[${SRC_DOMAINS[i]}]" data.target_domains="[${TGT_DOMAINS[i]}]" \
        model_src.arch="resnet50" \
        model_tta.src_log_dir=${SRC_MODEL_DIR} \
        learn.epochs=1 \
    done
done 