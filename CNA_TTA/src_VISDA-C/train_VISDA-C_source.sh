PORT=10005
CUDA=1,2,3,4
MEMO="source"

for SEED in 2020 2021 2022
do
    CUDA_VISIBLE_DEVICES=${CUDA} python main.py train_source=true learn=source \
    seed=${SEED} port=${PORT} memo=${MEMO} project="VISDA-C" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
    model_src.arch="resnet101" \
    learn.epochs=10 \
    optim.lr=2e-4 
done