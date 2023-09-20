SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/VISDA-C/source"

PORT=10011
MEMO="VISDAC"
SUB_MEMO="V2_SEED"

for SEED in 2021 2022 2023
do
    CUDA_VISIBLE_DEVICES=3,4 python ../main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="VISDA-C" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
    learn=targetv2.yaml \
    model_src.arch="resnet101" \
    model_tta.src_log_dir=${SRC_MODEL_DIR}
done
