PORT=10000
CUDA=4,5
SRC_DOMAINS=(real    real    real     sketch    clipart painting)
MEMO="source"

for SEED in 2020 2021 2022
do  
    for i in "${!SRC_DOMAINS[@]}"
    do
        CUDA_VISIBLE_DEVICES=${CUDA} python main.py train_source=true learn=source \
        seed=${SEED} port=${PORT} memo=${MEMO} project="domainnet-126" \
        data.data_root="${PWD}/datasets" data.workers=8 \
        data.dataset="domainnet-126" data.source_domains="[${SRC_DOMAINS[i]}]" data.target_domains="[real,sketch,clipart,painting]" \
        model_src.arch="resnet50" \
        learn.epochs=60 \
        optim.lr=2e-4
    done
done