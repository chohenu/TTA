SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/VISDA-C/source"

PORT=10005
MEMO="VISDAC"
SUB_MEMO="CE+PROTONCE_ALL_B_CENTER_NCE_CLEAN_CONFI"


# for SEED in 2020
for SEED in 2020
do
    CUDA_VISIBLE_DEVICES=3,4 python main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="VISDA-C" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
    model_src.arch="resnet101" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    learn.use_confidence_instance_loss=True \
    learn.use_distance_instance_loss=False \
    learn.ignore_instance_loss=False \
    learn.proto_instance_loss=False \
    learn.refine_method='nearest_neighbors' \
    learn.sep_gmm=False \
    optim.lr=2e-4 

done
