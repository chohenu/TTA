SRC_DOMAIN="real"
SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/domainnet-126/source"

PORT=10003
MEMO="Confidence"
SUB_MEMO="CE-KL+PSL_CLS_FIRST_CONFI+PL"

for TGT_DOMAIN in "clipart" "painting" "infograph" "sketch"
do
    TARGET_MEMO=${SUB_MEMO}_${TGT_DOMAIN}
    CUDA_VISIBLE_DEVICES=4,5 python main_adacontrast.py \
    seed=2020 port=${PORT} memo=${MEMO} sub_memo=${TARGET_MEMO} project="domainnet-126" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="domainnet-126" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
    model_src.arch="resnet50" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    learn.use_confidence_instance_loss=False \
    learn.use_distance_instance_loss=False \
    learn.refine_method='nearest_neighbors' \
    optim.lr=2e-4
done