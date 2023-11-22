SRC_DOMAIN="real"
TGT_DOMAIN="clipart"
SRC_MODEL_DIR="/opt/tta/CNA-TTA/output/domainnet-126/source"

PORT=10042
MEMO="VIS_MIXUP_INFER"
SUB_MEMO="vis_MIXUP_INFER"

for SEED in 2021
do
    CUDA_VISIBLE_DEVICES=1,2,3,4 python ../main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="domainnet-126" \
    data.data_root="/mnt/data" data.workers=8 \
    data.dataset="domainnet-126" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
    learn=targetv1.yaml \
    ckpt_path=False \
    model_src.arch="resnet50" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.time=1 \
    optim.lr=2e-4 
done 
    # ckpt_path=/opt/tta/CNA-TTA/src_domainnet/output/domainnet-126/VIS_MIXUP_TEST/checkpoint_0014_real-clipart-vis_MIXUP_TEST_2022.pth.tar \
    # do_inference=True \
