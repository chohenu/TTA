SRC_MODEL_DIR="/opt/tta/AdaContrast/checkpoint/VISDA-C/"

PORT=10002
MEMO="target"

for SEED in 2020
do
    CUDA_VISIBLE_DEVICES=1,2,3,4 python main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} project="VISDA-C" \
    data.data_root="/mnt/data/" data.workers=8 \
    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
    data.batch_size=128 \
    model_src.arch="resnet101" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.lr=2e-4
done
