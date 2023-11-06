SEED=2021
SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/VISDA-C/source/"

PORT=10014
MEMO="VISDAC_VIS_NOISE_ACC_RECALL"
SUB_MEMO="V2_VIS_NOISE_ACC_RECALL"

for SEED in 2021 2022 2023
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python ../main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="VISDA-C" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
    ckpt_path=False \
    model_src.arch="resnet101" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    learn=targetv1.yaml \
    learn.epochs=30 \
    optim.time=1 \
    optim.lr=2e-4 
done

