
PORT=10030
SRC_DOMAINS=(sketch,clipart,painting,quickdraw,infograph 
            real_clipart_painting_quickdraw_infograph 
            real_sketch_painting_quickdraw_infograph 
            real_sketch_clipart_quickdraw_infograph 
            real_sketch_clipart_quickdraw_infograph 
            real_sketch_clipart_painting_infograph)

MEMO="MSDA_source"

for SEED in 2021 2022 2023
do
    for i in "${!SRC_DOMAINS[@]}"
    do  
        CUDA_VISIBLE_DEVICES=1,2 python ../main.py train_source=true learn=source \
        seed=${SEED} port=${PORT} memo=${MEMO} project="domainnet" \
        data.data_root="/mnt/data" data.workers=8 \
        data.dataset="domainnet" data.source_domains="[[${SRC_DOMAINS[i]}]]" data.target_domains="[sketch,clipart,painting,quickdraw,infograph]" \
        learn.epochs=30 \
        model_src.arch="resnet50" \
        optim.time=1 \
        optim.lr=2e-4
    done
done