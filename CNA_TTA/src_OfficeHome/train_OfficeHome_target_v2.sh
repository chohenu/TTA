SEED=2021
SRC_MODEL_DIR="/opt/tta/CNA_TTA/output/OfficeHome/source"

PORT=10013
MEMO="OfficeHome_ORIGIN"
SUB_MEMO="V2"

for SRC_DOMAIN in Product RealWorld
do
    for TGT_DOMAIN in Art Clipart Product RealWorld
    do
        if [ "$TGT_DOMAIN" == "$SRC_DOMAIN" ]; then 
            echo "pass trainig"
        else
            CUDA_VISIBLE_DEVICES=4,5 python ../main_adacontrast.py \
            seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="office-home" \
            data.data_root="/mnt/data" data.workers=8 \
            data.dataset="OfficeHome" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
            learn=targetv2.yaml \
            learn.epochs=100 \
            model_src.arch="resnet50" \
            model_tta.src_log_dir=${SRC_MODEL_DIR} \
            optim.lr=2e-4 
        fi
    done
done
