SRC_DOMAIN="sketch"
TGT_DOMAIN="painting"
SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/domainnet-126/source"

PORT=10005
MEMO="ours"
SUB_MEMO="PL_ours+Mixup_ours+KL+Contrast_ours"

for SEED in 2020
do
    python main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="domainnet-126" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="domainnet-126" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
    model_src.arch="resnet50" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.lr=2e-4
done