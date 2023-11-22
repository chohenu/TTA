SRC_DOMAIN="sketch,clipart,painting,quickdraw,infograph"
# SRC_DOMAIN="real,clipart,painting,quickdraw,infograph"
# SRC_DOMAIN="real,sketch,painting,quickdraw,infograph"
# SRC_DOMAIN="real,sketch,clipart,quickdraw,infograph"
# SRC_DOMAIN="real,sketch,clipart,painting,infograph"
# SRC_DOMAIN="real,sketch,clipart,painting,quickdraw"

PORT=10089
MEMO="multi_source"

for SEED in 2022 2023
do
    CUDA_VISIBLE_DEVICES=1,2 python ../main_adacontrast.py train_source=true learn=source \
    seed=${SEED} port=${PORT} memo=${MEMO} project="domainnet" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="domainnet" data.source_domains="[[${SRC_DOMAIN}]]" data.target_domains="[sketch,clipart,painting,quickdraw,infograph]" \
    learn.epochs=30 \
    model_src.arch="resnet50" \
    optim.time=1 \
    optim.lr=2e-4
done