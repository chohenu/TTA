SEED=2021
SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/VISDA-C/source/"

PORT=10014
MEMO="VISDAC_online_ab_qu"
SUB_MEMO="V2_online_ab_qu"

for QU in 128 256 512 1024 2048 4096 8192 16384 
do
    CUDA_VISIBLE_DEVICES=4,5 python ../main_adacontrast.py \
    seed=2021 port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO}_${QU} project="VISDA-C" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
    learn.epochs=1 \
    learn=targetv1.yaml \
    model_tta.queue_size=${QU} \
    ckpt_path=False \
    model_src.arch="resnet101" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.time=1 \
    optim.lr=2e-4 
done