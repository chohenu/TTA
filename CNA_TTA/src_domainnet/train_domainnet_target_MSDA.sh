PORT=10029
CUDA=1,2,6,7
SRC_DOMAINS=(real_clipart_painting_quickdraw_infograph 
            real_sketch_painting_quickdraw_infograph 
            real_sketch_clipart_quickdraw_infograph 
            real_sketch_clipart_quickdraw_infograph 
            real_sketch_clipart_painting_infograph)
TGT_DOMAINS=(sketch clipart painting quickdraw infograph)

MEMO="MSDA_domainnet"
SUB_MEMO="offline"
SRC_MODEL_DIR="/opt/tta/CNA_TTA/output/domainnet/multi_source"

for SEED in 2021
do
    for i in "${!SRC_DOMAINS[@]}"
    do
        CUDA_VISIBLE_DEVICES=${CUDA} python ../main_adacontrast.py \
        seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="domainnet" \
        data.data_root="/mnt/data" data.workers=8 \
        data.dataset="domainnet" data.source_domains="[${SRC_DOMAINS[i]}]" data.target_domains="[${TGT_DOMAINS[i]}]" \
        model_tta.src_log_dir=${SRC_MODEL_DIR} \
        model_src.arch="resnet50" \
        learn.epochs=15 \
    done
done 