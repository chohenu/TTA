SRC_DOMAIN="sketch,clipart,painting"

PORT=10000
MEMO="source"

for SEED in 2021 2022 2023
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python ../main_adacontrast.py train_source=true learn=source \
    seed=${SEED} port=${PORT} memo=${MEMO} project="domainnet-126" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="domainnet-126" data.source_domains="[[${SRC_DOMAIN}]]" data.target_domains="[real,sketch,clipart,painting]" \
    learn.epochs=60 \
    model_src.arch="resnet50" \
    optim.lr=2e-4
done