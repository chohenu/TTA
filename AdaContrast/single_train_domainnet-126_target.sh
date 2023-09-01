SRC_DOMAIN="real"
TGT_DOMAIN="clipart"
SRC_MODEL_DIR="/opt/tta/AdaContrast/checkpoint/domainnet-126/"

PORT=10000
MEMO="target"

for SEED in 2020
do
    CUDA_VISIBLE_DEVICES=0,1 python main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} project="domainnet-126" \
    data.data_root="/mnt/data/" data.workers=8 \
    data.dataset="domainnet-126" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
    data.batch_size=128 \
    model_src.arch="resnet50" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.lr=2e-4
done