PORT=17012
CUDA=0,5,6,7
SRC_MODEL_DIR="/opt/tta/hwc_TTA/output/domainnet-126/source"
SRC_DOMAINS=(clipart)
TGT_DOMAINS=(sketch)
PORTS=(${PORT} ${PORT} ${PORT} ${PORT} ${PORT} ${PORT} ${PORT})
CUDAS=(${CUDA} ${CUDA} ${CUDA} ${CUDA} ${CUDA} ${CUDA} ${CUDA})
MEMO="domain_net_noise_acc_epoch"
SUB_MEMO="noise_acc_step_epoch"
for SEED in 2023
do
    for i in "${!SRC_DOMAINS[@]}"
    do
        CUDA_VISIBLE_DEVICES=${CUDAS[i]} python ../main_adacontrast.py \
        seed=${SEED} port=${PORTS[i]} memo=${MEMO} sub_memo=${SUB_MEMO} project="domainnet-126" \
        data.data_root="/mnt/data" data.workers=8 \
        data.dataset="domainnet-126" data.source_domains="[${SRC_DOMAINS[i]}]" data.target_domains="[${TGT_DOMAINS[i]}]" \
        ckpt_path=False \
        model_src.arch="resnet50" \
        model_tta.src_log_dir=${SRC_MODEL_DIR} \
        learn=targetv1.yaml \
        data.batch_size=64 \
        learn.epochs=15 \
        optim.time=1 \
        optim.lr=2e-4
    done
done