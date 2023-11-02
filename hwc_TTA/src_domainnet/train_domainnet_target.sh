SRC_DOMAIN="real"
TGT_DOMAIN="sketch"
SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/domainnet-126/source"

PORT=10019
MEMO="VIS_NOISE_ACC"
SUB_MEMO="vis_NOISE_ACC"

for SEED in 2022
do
    CUDA_VISIBLE_DEVICES=0,1,2,4 python ../main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="domainnet-126" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="domainnet-126" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
    learn=targetv1.yaml \
    ckpt_path=False \
    model_src.arch="resnet50" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.lr=2e-4 
done 