SEED=2021
SRC_MODEL_DIR="/opt/tta/CNA-TTA/output/seed${SEED}/VISDA-C/"

PORT=10014
MEMO="VISDAC_SHOT_LR"
SUB_MEMO="V2_SHOT"

for SEED in 2021
do
    CUDA_VISIBLE_DEVICES=1,2,3 python main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="VISDA-C" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
    model_src=shot.yaml \
    learn=targetv2.yaml \
    model_src.arch="resnet101" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.lr=2e-4 
done
