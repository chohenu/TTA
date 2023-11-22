SRC_MODEL_DIR="/opt/tta/CNA-TTA/output/VISDA-C/source"

PORT=10013
MEMO="VISDAC"
SUB_MEMO="online"

for SEED in 2021
do
    CUDA_VISIBLE_DEVICES=1,3 python ../main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="VISDA-C" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
    data.batch_size=64 \
    learn=targetv3.yaml \
    learn.epochs=1 \
    model_src.arch="resnet101" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.lr=2e-4 
done
