SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/VISDA-C/source/"

PORT=10019
MEMO="VISDAC_online_ab_legth"
SUB_MEMO="V2_online_ab_legth"

for LEN in 8 16
do
    CUDA_VISIBLE_DEVICES=4,5 python ../main_adacontrast.py \
    seed=2021 port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO}_${LEN} project="VISDA-C" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
    learn.epochs=1 \
    learn=targetv1.yaml \
    learn.online_length=${LEN} \
    ckpt_path=False \
    model_src.arch="resnet101" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.time=1 \
    optim.lr=2e-4 
done