SRC_MODEL_DIR="/opt/tta/CNA-TTA/output/VISDA-C/source"

PORT=10018
MEMO="VISDAC"
SUB_MEMO="CE+B_CONF_PROTONCE_ALL_B_CENTER_NCE_SEED"


# for SEED in 2020
# for SEED in 
# for TYPE in "True" "False"
for SEED in 2021 2022
do
    CUDA_VISIBLE_DEVICES=0,5 python main_adacontrast.py \
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
    # learn.use_conf_filter=${TYPE} \

done
