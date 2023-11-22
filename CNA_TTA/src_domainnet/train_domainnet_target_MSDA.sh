SRC_DOMAIN="sketch_clipart_painting_quickdraw_infograph"
TGT_DOMAIN="real"
SRC_MODEL_DIR="/opt/tta/CNA_TTA/output/domainnet/multi_source"

PORT=10029
MEMO="MSDA_domainnet"
SUB_MEMO="MSDA"

for SEED in 2021
do
    CUDA_VISIBLE_DEVICES=1,2,6,7 python ../main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="domainnet" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="domainnet" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    model_src.arch="resnet50" \
    learn=targetv1.yaml \
    learn.epochs=15 \
    optim.lr=2e-4 \
    optim.time=1 \
    optim.cos=False \
    optim.exp=False \
    optim.no_sch=True\
    ckpt_path=False
done 