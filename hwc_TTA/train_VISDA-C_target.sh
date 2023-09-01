SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/VISDA-C/source"

PORT=10006
MEMO="Ablation_1"
SUB_MEMO="CE+div+PL_ada+KL+Mixup_w+Contrast_ours"

# for SEED in 2020 2021 2022
for SEED in 2020
do
    python main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="VISDA-C" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
    model_src.arch="resnet101" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.lr=2e-4
done
