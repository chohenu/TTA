CUDA=(3,4)
PORT=16001
SRC_DOMAINS=("cartoon,photo,sketch" 
            "art_painting,photo,sketch" 
            "art_painting,cartoon,sketch" 
            "art_painting,cartoon,photo")
MEMO="source"

for SEED in 2021 2021 2022
do
    for i in "${!SRC_DOMAINS[@]}"
    do
        CUDA_VISIBLE_DEVICES=${CUDA} python main.py train_source=true learn=source \
        seed=${SEED} port=${PORT} memo=${MEMO} project="pacs" \
        data.data_root="/mnt/data/" data.workers=8 \
        data.dataset="pacs" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[art_painting,cartoon,photo,sketch]" \
        model_src.arch="resnet18" \
        learn.epochs=100 \
        optim.lr=2e-4
    done
done