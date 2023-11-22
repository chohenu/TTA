SRC_DOMAINS=(art_painting_cartoon_photo art_painting_cartoon_sketch art_painting_photo_sketch cartoon_photo_sketch)
TGT_DOMAINS=(sketch photo cartoon art_painting)

SRC_MODEL_DIR="/opt/tta/CNA_TTA/output/pacs/multi_source"

PORT=10017
MEMO="MSDA_online"
SUB_MEMO="MSDA_online"

for SEED in 2021 2022 2023
do
    for i in "${!SRC_DOMAINS[@]}"
    do  
        CUDA_VISIBLE_DEVICES=2,3 python ../main_adacontrast.py \
        seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="pacs" \
        data.data_root="/mnt/data" data.workers=8 \
        data.dataset="pacs" data.source_domains="[${SRC_DOMAINS[i]}]" data.target_domains="[${TGT_DOMAINS[i]}]" \
        model_src.arch="resnet18" \
        model_tta.src_log_dir=${SRC_MODEL_DIR} \
        learn=targetv1.yaml \
        learn.epochs=1 \
        optim.time=1 \
        optim.lr=2e-4 
        data.batch_size=32 \
        ckpt_path=False
    done
done 