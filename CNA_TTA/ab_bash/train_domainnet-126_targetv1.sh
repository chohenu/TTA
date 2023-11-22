SRC_MODEL_DIR="/opt/tta/CNA-TTA/output/domainnet-126/source"

PORT=10013
MEMO="DOMAINNET"
SUB_MEMO="V1_SEED"

for SEED in 2021 2022 2023
do  
    for TGT_DOMAIN in real sketch clipart painting
    do
        for SRC_DOMAIN in real sketch clipart painting
        do
            if [ "$TGT_DOMAIN" == "$SRC_DOMAIN" ]; then 
                echo "pass trainig"
            else
                CUDA_VISIBLE_DEVICES=0,5 python ../main_adacontrast.py \
                seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="domainnet-126" \
                data.data_root="/mnt/data" data.workers=8 \
                data.dataset="domainnet-126" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
                model_src.arch="resnet50" \
                model_tta.src_log_dir=${SRC_MODEL_DIR} \
                learn=targetv1.yaml
            fi
        done
    done
done 