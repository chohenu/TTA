SRC_DOMAIN="real"
TGT_DOMAIN="clipart"
SRC_MODEL_DIR="/opt/tta/CNA_TTA/output/domainnet-126/source"

PORT=10019
MEMO="VIS_MIXUP"
SUB_MEMO="vis_MIXUP"

for SEED in 2022
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python ../main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="domainnet-126" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="domainnet-126" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
    learn=targetv1.yaml \
    ckpt_path=False \
    model_src.arch="resnet50" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.lr=2e-4 
done 