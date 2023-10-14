SRC_DOMAIN=$1
SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/OfficeHome/source"

PORT=$2
MEMO="Confidence"
SUB_MEMO="TARGET_OURS_TEST"

for TGT_DOMAIN in Art Clipart Product RealWorld
do
    if [ "$TGT_DOMAIN" == "$SRC_DOMAIN" ]; then 
        echo "pass trainig"
    else
        CUDA_VISIBLE_DEVICES=2,3 python main_adacontrast.py \
        seed=2021 port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="OfficeHome" \
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
        learn.epochs=15 \
        data.batch_size=128 \
        optim.lr=2e-4
    fi
done 