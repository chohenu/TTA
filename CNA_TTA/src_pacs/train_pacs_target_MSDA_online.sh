SRC_DOMAIN="photo"
TGT_DOMAIN="art_painting,cartoon,sketch"
SRC_MODEL_DIR="/opt/tta/CNA_TTA/output/pacs/source"

PORT=10017
MEMO="Confidence"
SUB_MEMO="MULTI_TARGET_NOSCH"

for SEED in 2022
do
    CUDA_VISIBLE_DEVICES=3,6 python main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="pacs" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="pacs" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[[${TGT_DOMAIN}]]" \
    model_src.arch="resnet18" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    learn.epochs=1 \
    optim.lr=2e-4 \
done 