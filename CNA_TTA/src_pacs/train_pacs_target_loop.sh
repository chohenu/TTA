SRC_DOMAIN="photo"
SRC_MODEL_DIR="/opt/tta/CNA-TTA/output/pacs/source"

PORT=10035
MEMO="Confidence"
SUB_MEMO="2021EPOCH100_GUIDING"

for TGT_DOMAIN in sketch art_painting cartoon 
do
    if [ "$TGT_DOMAIN" == "$SRC_DOMAIN" ]; then 
        echo "pass trainig"
    else
        CUDA_VISIBLE_DEVICES=1,2 python main_adacontrast.py \
        seed=2021 port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="pacs" \
        data.data_root="/mnt/data" data.workers=8 \
        data.dataset="pacs" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
        data.batch_size=64 \
        model_src.arch="resnet18" \
        model_tta.src_log_dir=${SRC_MODEL_DIR} \
        optim.cos=False \
        optim.exp=false \
        optim.no_sch=True \
        optim.lr=0.01 \
        learn.epochs=100 \
        learn=targetv2.yaml
    fi
done 