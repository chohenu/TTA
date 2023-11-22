SEED=2021
SRC_MODEL_DIR="/opt/tta/CNA_TTA/output/VISDA-C/source/"

PORT=10014
MEMO="VISDAC_inference"
SUB_MEMO="inference"

CUDA_VISIBLE_DEVICES=1,2,3,4 python ../main_adacontrast.py \
                seed=2021 port=${PORT} memo=${MEMO} sub_memo=${SUB_MEMO} project="VISDA-C" \
                data.data_root="/mnt/data" data.workers=8 \
                data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
                ckpt_path="/opt/tta/thrid_party/checkpoint_0014_train-validation-None_2021.pth.tar" \
                model_src.arch="resnet101" \
                model_tta.src_log_dir=${SRC_MODEL_DIR} \
                do_inference=True \
                learn=targetv1.yaml \
                learn.epochs=30 \
                optim.time=1 \
                optim.lr=2e-4 \
                use_wandb=True 