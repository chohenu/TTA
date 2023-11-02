SEED=2021
SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/VISDA-C/source/"

PORT=10014
MEMO="VISDAC_online_fix"
SUB_MEMO="V2_online_fix"

for SEED in 2021 2022 2023
do
    CUDA_VISIBLE_DEVICES=1,2,3,4 python ../main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="VISDA-C" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
    learn.epochs=1 \
    learn=targetv1.yaml \
    ckpt_path=False \
    model_src.arch="resnet101" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.lr=2e-4 
done