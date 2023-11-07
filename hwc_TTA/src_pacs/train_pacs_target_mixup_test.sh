SRC_DOMAIN="photo"
TGT_DOMAIN="cartoon"
SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/pacs/source"

PORT=10027
MEMO="Confidence"
SUB_MEMO="EPOCH15_B64_LR0e5"

for SEED in 2022
do
    CUDA_VISIBLE_DEVICES=0,5 python main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="pacs" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="pacs" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[[${TGT_DOMAIN}]]" \
    model_src.arch="resnet18" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    learn.use_confidence_instance_loss=True \
    learn.use_distance_instance_loss=False \
    learn.ignore_instance_loss=False \
    learn.proto_instance_loss=False \
    learn.refine_method='nearest_neighbors' \
    learn.sep_gmm=False\
    learn.use_conf_filter=False \
    learn.epochs=15 \
    learn.use_ce_weight=False \
    data.batch_size=64 \
    optim.lr=2e-5
done 