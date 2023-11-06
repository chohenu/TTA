SEED=2021
SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/VISDA-C/source/"

PORT=10014
MEMO="VISDAC_online_ab_conf"
SUB_MEMO="V2_online_ab_conf"

for CONF in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    CUDA_VISIBLE_DEVICES=1,2,3,4 python ../main_adacontrast.py \
    seed=2021 port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="VISDA-C" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
    learn.epochs=1 \
    learn=targetv1.yaml \
    learn.conf_filter=${CONF} \
    ckpt_path=False \
    model_src.arch="resnet101" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.lr=2e-4 
done