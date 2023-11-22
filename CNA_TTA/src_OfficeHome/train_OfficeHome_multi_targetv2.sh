SRC_DOMAIN="Art"
TGT_DOMAIN="Clipart,Product,RealWorld"
SRC_MODEL_DIR="/opt/tta/CNA-TTA/output/OfficeHome/source"

PORT=10015
MEMO="Confidence"
SUB_MEMO="MULTI_TARGET_TEST"

for SEED in 2020
do
    CUDA_VISIBLE_DEVICES=1,2 python main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="OfficeHome" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="OfficeHome" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[[${TGT_DOMAIN}]]" \
    model_src.arch="resnet50" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    learn.use_confidence_instance_loss=True \
    optim.lr=2e-4
done 