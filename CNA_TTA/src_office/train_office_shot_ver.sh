SEED=2021
SRC_MODEL_DIR="/opt/tta/CNA_TTA/output/seed${SEED}/office/"

PORT=10021
MEMO="office"
SUB_MEMO="V2_SHOT"

for SRC_DOMAIN in amazon dslr webcam 
do
    for TGT_DOMAIN in amazon dslr webcam
    do
        if [ "$TGT_DOMAIN" == "$SRC_DOMAIN" ]; then 
            echo "pass trainig"
        else
            CUDA_VISIBLE_DEVICES=1,2 python ../main_adacontrast.py \
            seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="office" \
            data.data_root="/mnt/data" data.workers=8 \
            data.dataset="office" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
            model_src=shot.yaml \
            learn=targetv2.yaml \
            learn.epochs=100 \
            model_src.arch="resnet50" \
            model_tta.src_log_dir=${SRC_MODEL_DIR} \
            optim.lr=2e-4 
        fi
    done
done
