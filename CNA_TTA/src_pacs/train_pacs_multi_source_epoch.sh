# SRC_DOMAIN="cartoon,photo,sketch"
# SRC_DOMAIN="art_painting,photo,sketch"
# SRC_DOMAIN="art_painting,cartoon,sketch"
SRC_DOMAIN="art_painting,cartoon,photo"

PORT=10024
MEMO="multi_source_epoch"

for SEED in 2021 2022 2023
do
    CUDA_VISIBLE_DEVICES=6,7 python ../main_adacontrast.py train_source=true learn=source \
    seed=${SEED} port=${PORT} memo=${MEMO} project="pacs" \
    data.data_root="/mnt/data/" data.workers=8 \
    data.dataset="pacs" data.source_domains="[[${SRC_DOMAIN}]]" data.target_domains="[art_painting,cartoon,photo,sketch]" \
    learn.epochs=300 \
    model_src.arch="resnet18" \
    optim.lr=2e-4
done