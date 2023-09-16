SRC_DOMAIN="Art"
TGT_DOMAIN="Clipart"
SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/OfficeHome/source"

PORT=10013
MEMO="Confidence"
SUB_MEMO="CE+B_NCONFv2_PROTONCE_ALL_B_CENTER_NCE"

for SEED in 2020
do
    CUDA_VISIBLE_DEVICES=0,5 python main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="OfficeHome" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="OfficeHome" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
    model_src.arch="resnet50" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    learn.use_confidence_instance_loss=True \
    learn.use_distance_instance_loss=False \
    learn.ignore_instance_loss=False \
    learn.proto_instance_loss=False \
    learn.refine_method='nearest_neighbors' \
    learn.sep_gmm=False\
    learn.use_conf_filter=False \
    optim.lr=2e-4
done 