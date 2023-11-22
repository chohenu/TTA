SRC_MODEL_DIR="/opt/tta/CNA-TTA/output/pacs/source"

PORT=10027
MEMO="PACS"
SUB_MEMO="V3_SEED"

for SEED in 2021 2022 2023
do  
    for TGT_DOMAIN in art_painting cartoon photo sketch
    do
        for SRC_DOMAIN in art_painting cartoon photo sketch
        do
            if [ "$TGT_DOMAIN" == "$SRC_DOMAIN" ]; then 
                echo "pass trainig"
            else
                CUDA_VISIBLE_DEVICES=0,5 python ../main_adacontrast.py \
                seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="pacs" \
                data.data_root="/mnt/data" data.workers=8 \
                data.dataset="pacs" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
                model_src.arch="resnet18" \
                model_tta.src_log_dir=${SRC_MODEL_DIR} \
                learn=targetv3.yaml
            fi
        done
    done
done 