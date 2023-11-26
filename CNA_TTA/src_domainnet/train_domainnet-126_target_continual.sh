PORT=10029
CUDA=1,2,6,7
SRC_DOMAINS=(real clipart painting sketch)
TGT_DOMAINS=("clipart,painting,sketch"  
            "real,painting,sketch" 
            "real,clipart,sketch" 
            "real,clipart,painting" )

MEMO="MSDA_domainnet"
SUB_MEMO="offline"
SRC_MODEL_DIR="/opt/tta/CNA_TTA/output/domainnet/multi_source"

for SEED in 2021
do
    for i in "${!TGT_DOMAINS[@]}"
    do
        CUDA_VISIBLE_DEVICES=${CUDA} python ../main.py \
        seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="domainnet" \
        data.data_root="/mnt/data" data.workers=8 \
        data.dataset="domainnet" data.source_domains="[${SRC_DOMAINS[i]}]" data.target_domains="[${TGT_DOMAINS[i]}]" \
        model_tta.src_log_dir=${SRC_MODEL_DIR} \
        model_src.arch="resnet50" \
        learn.epochs=1 \
    done
done 