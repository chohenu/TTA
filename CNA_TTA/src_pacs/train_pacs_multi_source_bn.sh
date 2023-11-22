SRC_DOMAIN="cartoon,photo,sketch"
# SRC_DOMAIN="art_painting,photo,sketch"
# SRC_DOMAIN="art_painting,cartoon,sketch"
# SRC_DOMAIN="art_painting,cartoon,photo"

PORT=10017
MEMO="multi_source_ab_batch"

for SEED in 2021 2022 2023
do
    CUDA_VISIBLE_DEVICES=1,2 python ../main_adacontrast.py train_source=true learn=source \
    seed=${SEED} port=${PORT} memo=${MEMO} project="pacs" \
    data.data_root="/mnt/data/" data.workers=8 \
    data.dataset="pacs" data.source_domains="[[${SRC_DOMAIN}]]" data.target_domains="[art_painting,cartoon,photo,sketch]" \
    data.batch_size=32 \
    learn.epochs=300 \
    model_src.arch="resnet18" \
    optim.time=1 \
    optim.lr=2e-4
done