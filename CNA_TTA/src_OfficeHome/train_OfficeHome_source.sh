SRC_DOMAIN=$1

PORT=$2
MEMO="source"

for SEED in 2020 2021 2022
do
    CUDA_VISIBLE_DEVICES=0,5 python main_adacontrast.py train_source=true learn=source \
    seed=${SEED} port=${PORT} memo=${MEMO} project="OfficeHome" \
    data.data_root="/mnt/data/" data.workers=8 \
    data.dataset="OfficeHome" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[Art,Clipart,Product,RealWorld]" \
    learn.epochs=60 \
    model_src.arch="resnet50" \
    optim.lr=2e-4
done