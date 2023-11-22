PORT=17011
CUDA=1,2,3,4
SRC_MODEL_DIR="/opt/tta/CNA-TTA/output/domainnet-126/source"
SRC_DOMAINS=(real    real    real     sketch    clipart painting painting)
TGT_DOMAINS=(sketch  clipart painting painting  sketch  real     clipart)

SRC_DOMAINS=(painting)
TGT_DOMAINS=(clipart)
PORTS=(${PORT} ${PORT} ${PORT} ${PORT} ${PORT} ${PORT} ${PORT} ${PORT})
CUDAS=(${CUDA} ${CUDA} ${CUDA} ${CUDA} ${CUDA} ${CUDA} ${CUDA} ${PORT})
MEMO="domainnet_online"
SUB_MEMO="online_component"
COMPONENT=(pr cr ccp div inst all)

for SEED in 2022 2023
do
    for i in "${!SRC_DOMAINS[@]}"
    do
        for j in "${!COMPONENT[@]}"
        do
            CUDA_VISIBLE_DEVICES=${CUDAS[i]} python ../main_adacontrast.py \
            seed=${SEED} port=${PORTS[i]} memo=${MEMO} sub_memo=${SUB_MEMO}_${COMPONENT[j]} project="domainnet-126" \
            data.data_root="/mnt/data" data.workers=8 \
            data.dataset="domainnet-126" data.source_domains="[${SRC_DOMAINS[i]}]" data.target_domains="[${TGT_DOMAINS[i]}]" \
            learn.epochs=1 \
            learn=online_targetv1.yaml \
            learn.component=${COMPONENT[j]} \
            ckpt_path=False \
            model_src.arch="resnet50" \
            model_tta.src_log_dir=${SRC_MODEL_DIR} \
            optim.time=1 \
            optim.lr=2e-4
        done
    done
done 